"""V7 frozen-model historical backtest (2024-2026 ETHUSDT)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(description="V7 历史回测数据获取与缓存")
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--start", help="起始时间, 如 2024-01-01 或 unix ms")
    parser.add_argument("--end", help="结束时间, 如 2026-08-01 或 unix ms")
    parser.add_argument("--no-proxy", action="store_true", help="禁用代理直连 Binance")
    args = parser.parse_args()

    def to_ms(s: str | None, default_dt: datetime) -> int:
        if not s:
            return int(default_dt.timestamp() * 1000)
        if s.isdigit():
            return int(s)
        return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=BEIJING_TZ).timestamp() * 1000)

    start_ms = to_ms(args.start, datetime(2024, 1, 1, tzinfo=BEIJING_TZ))
    end_ms = to_ms(args.end, datetime(2026, 8, 1, tzinfo=BEIJING_TZ))
    proxy_url = None if args.no_proxy else "http://127.0.0.1:7892"
    rows = load_or_fetch(args.symbol, args.interval, start_ms, end_ms, proxy_url)
    print(f"candles: {len(rows)}")
    if rows:
        print("first:", rows[0])
        print("last:", rows[-1])


if __name__ == "__main__":
    main()
