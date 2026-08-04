"""V7 frozen-model historical backtest (2024-2026 ETHUSDT)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
V7_DIR = ROOT / "server_data" / "ETH_V7_Live_Bot_Complete"
sys.path.insert(0, str(V7_DIR))
import v7_feature_engine as fe  # noqa: E402

CACHE_DIR = ROOT / "server_data" / "cache"
BEIJING_TZ = timezone(timedelta(hours=8))
FIVE_MINUTES_MS = fe.FIVE_MINUTES_MS
INTERVALS = {"5m": 5 * 60 * 1000, "15m": 15 * 60 * 1000, "1h": 60 * 60 * 1000}


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int,
                 proxy_url: str | None = None) -> list[list[float]]:
    rows: list[list[float]] = []
    proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None
    cursor = start_ms
    while cursor < end_ms:
        data = None
        for attempt in range(3):
            try:
                resp = requests.get(
                    "https://api.binance.com/api/v3/klines",
                    params={"symbol": symbol, "interval": interval,
                            "startTime": cursor, "endTime": end_ms, "limit": 1000},
                    proxies=proxies, timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        if not data:
            break
        rows.extend([float(v) for v in r[:6]] for r in data)
        cursor = int(data[-1][0]) + 1
        time.sleep(0.15)
    return rows


def load_or_fetch(symbol: str, interval: str, start_ms: int, end_ms: int,
                  proxy_url: str | None = None) -> list[list[float]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{symbol}_{interval}_{start_ms}_{end_ms}.csv"
    if path.exists():
        return pd.read_csv(path).values.tolist()
    rows = fetch_klines(symbol, interval, start_ms, end_ms, proxy_url)
    pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"]
                 ).to_csv(path, index=False)
    return rows


def build_aligned_features(df5: pd.DataFrame, df15: pd.DataFrame,
                           df1h: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray]:
    """全历史向量化构建特征矩阵，与实盘逐行 build_latest_feature_row 等价。

    返回 (F, col_names, finite)：F 为 n×len(col_names) float 矩阵（无效行为 NaN），
    col_names 与 models/v7/eth_v7_balanced_config.json 的 features 顺序完全一致，
    finite 为 n 维 bool 掩码（15m/1h 已对齐完整且全部特征有限）。
    """
    cfg = json.loads((V7_DIR / "models/v7/eth_v7_balanced_config.json").read_text())
    col_names = list(cfg["features"])
    feats5 = fe._build_base_features(df5)
    feats15 = fe._add_timeframe_features(df15, "m15")
    feats1h = fe._add_timeframe_features(df1h, "h1")
    base_cols = [c for c in col_names if c in feats5.columns]
    m15_cols = [c for c in col_names if c.startswith("m15_")]
    h1_cols = [c for c in col_names if c.startswith("h1_")]
    if len(base_cols) + len(m15_cols) + len(h1_cols) != len(col_names):
        unknown = [c for c in col_names
                   if c not in base_cols and c not in m15_cols and c not in h1_cols]
        raise ValueError(f"config features 包含特征引擎不认识的列: {unknown}")

    # 信号时刻 = 5m K线收盘时刻；15m/1h 取"已完整收盘"的最后一根
    close_ts = df5["open_time"].to_numpy() + FIVE_MINUTES_MS
    open15 = df15["open_time"].to_numpy() + fe.FIFTEEN_MINUTES_MS
    open1h = df1h["open_time"].to_numpy() + fe.ONE_HOUR_MS
    i15 = np.searchsorted(open15, close_ts, side="right") - 1
    i1h = np.searchsorted(open1h, close_ts, side="right") - 1
    n = len(df5)
    base_np = feats5[base_cols].to_numpy(dtype=float)
    m15_np = feats15[m15_cols].to_numpy(dtype=float)
    h1_np = feats1h[h1_cols].to_numpy(dtype=float)

    # 实盘要求 5m>=120、15m>=60、1h>=60（1h 60 根需 60 小时 → 最早有效 5m 索引 720）
    valid = (np.arange(n) >= 119) & (i15 >= 59) & (i1h >= 59)
    out = np.full((n, len(col_names)), np.nan)
    idx = np.arange(n)[valid]
    pos = {c: j for j, c in enumerate(col_names)}
    out[np.ix_(idx, [pos[c] for c in base_cols])] = base_np[idx]
    out[np.ix_(idx, [pos[c] for c in m15_cols])] = m15_np[i15[idx]]
    out[np.ix_(idx, [pos[c] for c in h1_cols])] = h1_np[i1h[idx]]
    finite = valid & np.isfinite(out).all(axis=1)
    return out, col_names, finite


def beijing_day(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=BEIJING_TZ).strftime("%Y%m%d")


def simulate(F: np.ndarray, col_names: list[str], finite: np.ndarray,
             model, threshold: float, df5: pd.DataFrame, cooldown: int = 1,
             min_atrp: float = 0.0005, max_daily_signals: int = 50,
             max_daily_loss: float = -75.0, stake: float = 25.0,
             payout: float = 0.80, start_ms: int | None = None,
             end_ms: int | None = None) -> list[dict]:
    atrp_col = col_names.index("atrp")
    open_ts = df5["open_time"].to_numpy()
    open_arr = df5["open"].to_numpy()
    close_arr = df5["close"].to_numpy()
    n = len(df5)
    valid_rows = [i for i in range(n) if finite[i]
                  and (start_ms is None or open_ts[i] + FIVE_MINUTES_MS >= start_ms)
                  and (end_ms is None or open_ts[i] + FIVE_MINUTES_MS <= end_ms)]
    prob_of: dict[int, float] = {}
    if valid_rows:
        probs = model.predict(F[valid_rows], num_iteration=model.num_trees())
        prob_of = dict(zip(valid_rows, probs))

    trades: list[dict] = []
    pending: list[dict] = []
    last_signal_idx = -10**9
    daily: dict[str, dict] = {}

    for i in range(n):
        ts = open_ts[i] + FIVE_MINUTES_MS
        if start_ms is not None and ts < start_ms:
            continue
        if end_ms is not None and ts > end_ms:
            break
        day = beijing_day(ts)
        d = daily.setdefault(day, {"pnl": 0.0, "count": 0, "breaker": False})

        for p in list(pending):  # 先结算（与实盘 process_candle 一致）
            bars = i - p["signal_idx"]
            if p["entry"] is None and bars >= 1:
                p["entry"] = open_arr[i]
                p["entry_ts"] = open_ts[i]
            if p["entry"] is not None and bars >= 2:
                won = close_arr[i] > p["entry"] if p["direction"] == "up" else close_arr[i] < p["entry"]
                pnl = stake * payout if won else -stake
                trades.append({
                    "signal_idx": p["signal_idx"],
                    "signal_time": datetime.fromtimestamp(p["ts"] / 1000, tz=BEIJING_TZ)
                        .strftime("%Y-%m-%d %H:%M:%S"),
                    "beijing_day": beijing_day(p["ts"]),
                    "direction": p["direction"],
                    "prob": p["prob"],
                    "entry": p["entry"],
                    "exit": close_arr[i],
                    "result": "WIN" if won else "LOSS",
                    "pnl": pnl,
                })
                d["pnl"] += pnl
                pending.remove(p)

        # 熔断锁存：任一限额首次触发后当日永久拦截（与实盘 v7_main.py _risk_blocked
        # 一致——_circuit_breaker 一旦置 True 仅跨日复位；即使后续在途单 WIN 使
        # pnl 回升到限额之上，当日也不再恢复出信号）。检查位置不变：结算之后、出信号之前。
        if d["pnl"] <= max_daily_loss or d["count"] >= max_daily_signals:
            d["breaker"] = True
        if d["breaker"]:
            continue
        if i not in prob_of or i - last_signal_idx < cooldown:
            continue
        prob_up = float(prob_of[i])
        if prob_up >= threshold:
            direction = "up"
        elif prob_up <= 1.0 - threshold:
            direction = "down"
        else:
            continue
        if F[i, atrp_col] < min_atrp:
            continue
        last_signal_idx = i
        d["count"] += 1
        pending.append({"signal_idx": i, "ts": ts, "direction": direction,
                        "prob": prob_up, "entry": None, "entry_ts": None})
    return trades


def load_model() -> lgb.Booster:
    cfg = json.loads((V7_DIR / "models/v7/eth_v7_balanced_config.json").read_text(encoding="utf-8"))
    model_file = V7_DIR / "models/v7/eth_v7_balanced_model.txt"
    try:
        model = lgb.Booster(model_file=str(model_file))
    except lgb.basic.LightGBMError:
        # Windows 下 LightGBM C API 无法打开含非 ASCII 字符的路径（中文目录），
        # 回退为从字符串加载（模型为 ASCII 文本，166KB）。
        model = lgb.Booster(model_str=model_file.read_text(encoding="utf-8"))
    assert list(model.feature_name()) == cfg["features"], "模型特征顺序与配置不一致"
    return model


def compare_live(backtest_trades: list[dict], live_rows: list[dict]) -> str:
    """live_rows: 从实盘结算 CSV 解析的 dict 列表（signal_time 北京时间字符串）。"""
    bt_by_day = {}
    for t in backtest_trades:
        bt_by_day.setdefault(t["signal_time"], t)
    live_by_day = {r["signal_time"]: r for r in live_rows}
    shared = set(bt_by_day) & set(live_by_day)
    only_bt = set(bt_by_day) - set(live_by_day)
    only_live = set(live_by_day) - set(bt_by_day)

    def wr(ts, rows):
        wins = sum(1 for r in rows if r["result"] == "WIN")
        return f"{wins}/{len(rows)} ({wins / len(rows) * 100:.1f}%)" if rows else "-"

    bt_l = list(bt_by_day.values())
    live_l = list(live_by_day.values())
    mismatch = []
    for ts in sorted(shared):
        a, b = bt_by_day[ts], live_by_day[ts]
        if a["direction"] != b["direction"] or a["result"] != b["result"]:
            mismatch.append((ts, a["direction"], a["result"], b["direction"], b["result"]))
    return (
        f"回测: {len(bt_l)}笔 胜率{wr(ts, bt_l)}  实盘: {len(live_l)}笔 胜率{wr(ts, live_l)}\n"
        f"交集: {len(shared)}  仅回测: {sorted(only_bt)[:5]}{'...' if len(only_bt) > 5 else ''}  "
        f"仅实盘: {sorted(only_live)[:5]}{'...' if len(only_live) > 5 else ''}\n"
        f"方向/结果不一致: {len(mismatch)} 笔 {mismatch[:5]}"
    )


def _bj_to_ms(s: str) -> int:
    """北京时间字符串 '%Y-%m-%d %H:%M:%S' → UTC epoch ms。"""
    return int(datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
               .replace(tzinfo=BEIJING_TZ).timestamp() * 1000)


def _read_live_settlements() -> list[dict]:
    rows = []
    for path in sorted((ROOT / "server_data" / "logs").glob("v7_settlements_*.csv")):
        rows += pd.read_csv(path).to_dict("records")
    return rows


def _load_frames(start: datetime, end: datetime, proxy_url: str | None):
    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    cols = ["open_time", "open", "high", "low", "close", "volume"]
    return tuple(
        pd.DataFrame(load_or_fetch("ETHUSDT", iv, start_ms, end_ms, proxy_url), columns=cols)
        for iv in ["5m", "15m", "1h"]
    )


def run_validate(proxy_url: str | None) -> None:
    model = load_model()
    live_rows = _read_live_settlements()
    if not live_rows:
        raise SystemExit("未找到实盘结算 CSV: server_data/logs/v7_settlements_*.csv")
    # 验证窗口从实盘 CSV 推导，不硬编码：服务器首次启动 2026-07-29 18:10 北京时间
    # （=10:10 UTC），首笔信号 19:00 北京时间（=11:00 UTC），末笔结算 08-03 15:30 北京时间。
    min_signal_ms = min(_bj_to_ms(r["signal_time"]) for r in live_rows)
    max_settle_ms = max(_bj_to_ms(r["settle_time"]) for r in live_rows)
    # 提前 60h 保证 1h 特征 warmup（有效行要求 1h 索引 >=59）；末尾留 12h 余量
    start_ms = min_signal_ms - 60 * 60 * 60 * 1000
    end_ms = max_settle_ms + 12 * 60 * 60 * 1000
    frames = _load_frames(
        datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc),
        datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc),
        proxy_url,
    )
    F, names, finite = build_aligned_features(*frames)
    # stake=10.0：config.yaml 写 25.0 但服务器从未热加载，实盘 CSV pnl WIN+8.0/LOSS-10.0
    # （payout 0.80）才是事实来源。start/end 限定只生成实盘窗口内信号。
    trades = simulate(F, names, finite, model, 0.555, frames[0],
                      stake=10.0, start_ms=min_signal_ms,
                      end_ms=max_settle_ms + 10 * 60 * 1000)
    print(compare_live(trades, live_rows))


def run_full(start: str, end: str, proxy_url: str | None) -> None:
    raise NotImplementedError("Task 5: 全量回测待实现")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2026-08-03")
    ap.add_argument("--validate", action="store_true", help="只跑 7/29-8/3 并与实盘对比")
    ap.add_argument("--proxy", default="http://127.0.0.1:7892", help="代理或空串禁用")
    args = ap.parse_args()
    proxy = args.proxy or None
    if args.validate:
        run_validate(proxy)
    else:
        run_full(args.start, args.end, proxy)


if __name__ == "__main__":
    main()
