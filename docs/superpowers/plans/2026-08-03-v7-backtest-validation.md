# V7 模型历史回测验证 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用冻结的 V7.0-causal-ml-balanced LightGBM 模型对 2024-01-01 至 2026-08-03 的 ETHUSDT 历史数据做完整回测，验证"UP 方向系统性失效"假设并确定最优阈值/方向过滤配置。

**Architecture:** 单个独立脚本 `v7_backtest.py`，复用 `server_data/ETH_V7_Live_Bot_Complete/v7_feature_engine.py`（通过 sys.path import，不复制代码）。由于所有特征（EMA/RMA/rolling）都是因果的，可以在完整历史上一次性向量化计算，再按 open_time 用 searchsorted 对齐 15m/1h 特征，与实盘逐行调用 `build_latest_feature_row` 完全等价。交易模拟严格复刻 `v7_main.py` + `v7_live_tracker.py` 的规则。

**Tech Stack:** Python 3.10+（检查本地版本）、pandas、numpy、lightgbm、requests、pytest

## Global Constraints

- 特征必须 import `v7_feature_engine` 原函数，禁止复制粘贴实现
- 交易模拟必须复刻实盘规则：信号于 5m 收盘时产生 → 下一根 5m 开盘入场（bars_since_signal>=1）→ 再下一根收盘结算（bars_since_signal>=2）；平局（exit==entry）算 LOSS
- cooldown=1 表示信号索引差 `i - last_signal_idx >= 1`（实盘代码 `current_idx - last_idx < cooldown → return`）
- 日内熔断按北京时间：`daily_pnl <= -75` 或 `daily_count >= 50` 后当日不再出信号；结算先于风险检查（先更新 daily_pnl 再检查熔断）
- 特征 NaN（min_periods 不足 / 非有限值）的行必须跳过（实盘 `FeatureBuildError` 行为）
- 验证步骤（7/29-8/3 段与实盘 CSV 对比）必须先通过，才能跑全量
- 模型文件: `server_data/ETH_V7_Live_Bot_Complete/models/v7/eth_v7_balanced_model.txt` + `eth_v7_balanced_config.json`
- 实盘结算 CSV 在 `server_data/logs/v7_settlements_*.csv`

---

### Task 1: 环境准备 + 数据获取与缓存

**Files:**
- Create: `v7_backtest.py`（数据获取部分）
- Create: `tests/test_v7_backtest.py`（缓存与 fetch 相关测试）
- Modify: `requirements.txt`（追加 lightgbm、pytest）

**Interfaces:**
- Produces: `fetch_klines(symbol, interval, start_ms, end_ms, proxy_url=None) -> list[list[float]]`（每行 `[open_time, open, high, low, close, volume]`）；`load_or_fetch(symbol, interval, start_ms, end_ms, proxy_url=None) -> list[list[float]]`（带 `server_data/cache/{symbol}_{interval}_{start}_{end}.csv` 缓存）；CLI 参数 `--start/--end/--no-proxy`。

- [ ] **Step 1: 检查环境**

```bash
python --version && python -m pip install lightgbm pytest
python -c "import lightgbm, requests, pandas, numpy; print('ok')"
```
预期: python ≥ 3.10，import 无报错。

- [ ] **Step 2: 写失败测试（缓存读写 + fetch 参数）**

```python
# tests/test_v7_backtest.py
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import v7_backtest as bt
import v7_feature_engine as fe

def test_cache_roundtrip(tmp_path, monkeypatch):
    calls = {"n": 0}
    def fake_fetch(symbol, interval, start_ms, end_ms, proxy_url=None):
        calls["n"] += 1
        return [[start_ms + i * 300000, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 10.0 + i]
                for i in range(3)]
    monkeypatch.setattr(bt, "fetch_klines", fake_fetch)
    bt.CACHE_DIR = tmp_path
    r1 = bt.load_or_fetch("ETHUSDT", "5m", 0, 900000)
    r2 = bt.load_or_fetch("ETHUSDT", "5m", 0, 900000)
    assert r1 == r2 and len(r1) == 3
    assert calls["n"] == 1  # 第二次命中缓存
```

- [ ] **Step 3: 运行测试确认失败**

Run: `python -m pytest tests/test_v7_backtest.py::test_cache_roundtrip -v`
预期: FAIL（v7_backtest 模块不存在）

- [ ] **Step 4: 实现数据获取**

```python
"""V7 frozen-model historical backtest (2024-2026 ETHUSDT)."""
from __future__ import annotations

import argparse
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
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_v7_backtest.py -v`
预期: PASS

- [ ] **Step 6: 手动连通性验证（拉 2 天数据）**

```bash
python -c "
import sys; sys.path.insert(0, '.')
import v7_backtest as bt
rows = bt.fetch_klines('ETHUSDT', '5m', 1753920000000, 1754092800000, 'http://127.0.0.1:7892')
print('candles:', len(rows)); print('first:', rows[0]); print('last:', rows[-1])"
```
预期: 576 根，open_time 与首末时间戳吻合。若直连失败，改用代理；若两者都失败，与用户确认网络方案（不要跳过此步）。

- [ ] **Step 7: 提交**

```bash
git add v7_backtest.py tests/test_v7_backtest.py requirements.txt
git commit -m "feat: V7回测-数据获取与缓存（Binance REST + 代理支持）"
```

---

### Task 2: 向量化特征构建与对齐

**Files:**
- Modify: `v7_backtest.py`（新增 `build_aligned_features`）
- Modify: `tests/test_v7_backtest.py`（新增等价性测试）

**Interfaces:**
- Consumes: Task 1 的 `load_or_fetch`；`fe._build_base_features`、`fe._add_timeframe_features`（来自 v7_feature_engine）
- Produces: `build_aligned_features(df5: pd.DataFrame, df15: pd.DataFrame, df1h: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray]`，返回 `(F, col_names, finite)`：F 为 n×102 float 矩阵（无效行为 NaN），col_names 与模型 config 的 features 顺序一致，finite 为 n 维 bool 掩码（对齐完整且无 NaN）

- [ ] **Step 1: 写失败测试（向量化 vs 实盘逐行函数等价）**

```python
def _synth(rows, period_ms):
    rng = np.random.default_rng(7)
    t0 = 1704067200000
    closes = 1000 + np.cumsum(rng.normal(0, 0.5, rows))
    out = []
    for i in range(rows):
        o = closes[i - 1] if i else 1000.0
        h = max(o, closes[i]) + rng.random()
        l = min(o, closes[i]) - rng.random()
        v = 10 + rng.random() * 5
        out.append([t0 + i * period_ms, o, h, l, closes[i], v])
    return out

def test_alignment_matches_live_row_builder():
    c5 = _synth(800, 300000)    # 5m: 66.7 小时
    c15 = _synth(300, 900000)   # 15m: 75 小时（≥60 根已收盘）
    c1h = _synth(80, 3600000)   # 1h: 80 小时（≥60 根已收盘）
    df5 = pd.DataFrame(c5, columns=["open_time", "open", "high", "low", "close", "volume"])
    df15 = pd.DataFrame(c15, columns=["open_time", "open", "high", "low", "close", "volume"])
    df1h = pd.DataFrame(c1h, columns=["open_time", "open", "high", "low", "close", "volume"])
    F, names, finite = bt.build_aligned_features(df5, df15, df1h)
    cfg = json.loads((bt.V7_DIR / "models/v7/eth_v7_balanced_config.json").read_text())
    assert names == cfg["features"], "列顺序必须与模型配置一致"
    # 1h 需 60 根已收盘（60 小时）→ 最早有效索引为 720；只抽样 i>=720
    for i in [720, 740, 760, 799]:
        live_row, _ = fe.build_latest_feature_row(c5[:i + 1], c15, c1h,
                                                  c5[i][0] + 300000, names)
        assert finite[i]
        for j, col in enumerate(names):
            a, b = F[i, j], float(live_row.iloc[0][col])
            assert np.isnan(a) == np.isnan(b), f"NaN 不一致 idx={i} col={col}"
            if not np.isnan(a):
                assert abs(a - b) < 1e-9, f"值不一致 idx={i} col={col}: {a} vs {b}"
    # 早期行（15m/1h 不足）必须标记无效
    assert not finite[119] and not finite[700]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_v7_backtest.py::test_alignment_matches_live_row_builder -v`
预期: FAIL（build_aligned_features 不存在）

- [ ] **Step 3: 实现对齐**

```python
def build_aligned_features(df5: pd.DataFrame, df15: pd.DataFrame,
                           df1h: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray]:
    feats5 = fe._build_base_features(df5)
    feats15 = fe._add_timeframe_features(df15, "m15")
    feats1h = fe._add_timeframe_features(df1h, "h1")
    m15_cols = [c for c in feats15.columns if c.startswith("m15_")]
    h1_cols = [c for c in feats1h.columns if c.startswith("h1_")]
    col_names = list(feats5.columns) + m15_cols + h1_cols

    close_ts = df5["open_time"].to_numpy() + FIVE_MINUTES_MS
    open15 = df15["open_time"].to_numpy() + fe.FIFTEEN_MINUTES_MS
    open1h = df1h["open_time"].to_numpy() + fe.ONE_HOUR_MS
    i15 = np.searchsorted(open15, close_ts, side="right") - 1
    i1h = np.searchsorted(open1h, close_ts, side="right") - 1
    n = len(df5)
    base_np = feats5.to_numpy(dtype=float)
    m15_np = feats15[m15_cols].to_numpy(dtype=float)
    h1_np = feats1h[h1_cols].to_numpy(dtype=float)

    valid = (np.arange(n) >= 119) & (i15 >= 59) & (i1h >= 59)
    out = np.full((n, len(col_names)), np.nan)
    idx = np.arange(n)[valid]
    out[idx, :len(feats5.columns)] = base_np[idx]
    out[idx, len(feats5.columns):len(feats5.columns) + len(m15_cols)] = m15_np[i15[idx]]
    out[idx, len(feats5.columns) + len(m15_cols):] = h1_np[i1h[idx]]
    finite = valid & np.isfinite(out).all(axis=1)
    return out, col_names, finite
```

注意：`i15`/`i1h` 可能为 -1（无已收盘K线），`valid` 掩码保证不会索引到 -1；`valid` 已排除 i<119（实盘要求 len(base)>=120）与不足 60 根 15m/1h 的行。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_v7_backtest.py -v`
预期: PASS（含 Task 1 测试）

- [ ] **Step 5: 提交**

```bash
git add v7_backtest.py tests/test_v7_backtest.py
git commit -m "feat: V7回测-向量化特征对齐，与实盘逐行构建等价"
```

---

### Task 3: 交易模拟（含风控）

**Files:**
- Modify: `v7_backtest.py`（新增 `simulate`、`beijing_day`）
- Modify: `tests/test_v7_backtest.py`（新增 3 个模拟测试）

**Interfaces:**
- Consumes: Task 2 的 `build_aligned_features` 输出
- Produces: `simulate(F, col_names, finite, model, threshold, df5, cooldown=1, min_atrp=0.0005, max_daily_signals=50, max_daily_loss=-75.0, stake=25.0, payout=0.80, start_ms=None, end_ms=None) -> list[dict]`，每个 trade 含 `{signal_idx, signal_time, beijing_day, direction, prob, entry, exit, result, pnl}`（signal_time 为北京时间字符串，便于与实盘 CSV 对齐）

- [ ] **Step 1: 写失败测试**

```python
def _model_stub(probs):
    class M:
        def num_trees(self): return 1
        def predict(self, rows, num_iteration=None):
            return np.array([probs.get(i, 0.5) for i in range(len(rows))])
    return M()

def _sim_inputs(falling: bool = False):
    n = 130
    t0 = 1704067200000
    if falling:  # 严格递减价格 → UP 必亏、DOWN 必赢（熔断测试用）
        closes = 1000 - np.arange(n) * 0.5
    else:
        rng = np.random.default_rng(3)
        closes = 1000 + np.cumsum(rng.normal(0, 0.5, n))
    rows = []
    for i in range(n):
        o = closes[i - 1] if i else 1000.0
        rows.append([t0 + i * 300000, o, max(o, closes[i]), min(o, closes[i]), closes[i], 10.0])
    df5 = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    F = np.full((n, 3), 0.5)      # [atrp占位, x, y]
    F[:, 0] = 0.01                 # atrp 足够大
    return df5, F, ["atrp", "x", "y"], np.ones(n, dtype=bool)

def test_settle_timing():
    df5, F, names, finite = _sim_inputs()
    model = _model_stub({5: 0.9, 6: 0.1})  # 索引5:up, 索引6:down
    trades = bt.simulate(F, names, finite, model, 0.555, df5)
    assert len(trades) == 2
    t_up = trades[0]
    assert t_up["signal_idx"] == 5 and t_up["direction"] == "up"
    assert t_up["entry"] == df5.iloc[6]["open"]       # 下一根开盘
    assert t_up["exit"] == df5.iloc[7]["close"]       # 再下一根收盘
    assert t_up["signal_time"].startswith("2024-01-01")

def test_tie_is_loss():
    df5, F, names, finite = _sim_inputs()
    df5["open"] = 1000.0  # 常数价格 → entry == exit
    df5["high"] = 1000.0
    df5["low"] = 1000.0
    df5["close"] = 1000.0
    model = _model_stub({5: 0.9})
    trades = bt.simulate(F, names, finite, model, 0.555, df5)
    assert len(trades) == 1
    assert trades[0]["result"] == "LOSS" and trades[0]["pnl"] == -25.0

def test_cooldown():
    df5, F, names, finite = _sim_inputs()
    model = _model_stub({i: 0.9 for i in range(5, 10)})  # 连续 up
    trades = bt.simulate(F, names, finite, model, 0.555, df5)
    idxs = [t["signal_idx"] for t in trades]
    assert all(b - a >= 1 for a, b in zip(idxs, idxs[1:]))

def test_daily_loss_breaker():
    df5, F, names, finite = _sim_inputs(falling=True)  # 严格递减 → UP 全亏
    model = _model_stub({i: 0.9 for i in range(5, 60)})  # 全是 up
    trades = bt.simulate(F, names, finite, model, 0.555, df5,
                         max_daily_loss=-75.0, stake=25.0, payout=0.8)
    days = set(t["beijing_day"] for t in trades)
    assert len(days) == 1  # 130 根 5m 只覆盖一天
    assert len(trades) == 3  # 第3笔结算后 pnl=-75 → 熔断，之后不再出信号
    assert all(t["result"] == "LOSS" for t in trades)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_v7_backtest.py -v`
预期: FAIL（simulate 不存在）

- [ ] **Step 3: 实现模拟**

```python
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

        if d["breaker"] or d["pnl"] <= max_daily_loss or d["count"] >= max_daily_signals:
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_v7_backtest.py -v`
预期: PASS（4 个测试全绿）

- [ ] **Step 5: 提交**

```bash
git add v7_backtest.py tests/test_v7_backtest.py
git commit -m "feat: V7回测-交易模拟（入场/结算/冷却/日内熔断与实盘一致）"
```

---

### Task 4: 模型接入 + 验证段对齐实盘

**Files:**
- Modify: `v7_backtest.py`（新增 `load_model`、`compare_live`、CLI `--validate` 分支）

**Interfaces:**
- Consumes: Task 2/3 全部；`server_data/logs/v7_settlements_*.csv`
- Produces: `load_model() -> lgb.Booster`；`compare_live(backtest_trades, live_rows) -> str` 输出对比报告

- [ ] **Step 1: 实现 load_model + compare_live**

```python
import json
import lightgbm as lgb


def load_model() -> lgb.Booster:
    cfg = json.loads((V7_DIR / "models/v7/eth_v7_balanced_config.json").read_text(encoding="utf-8"))
    model = lgb.Booster(model_file=str(V7_DIR / "models/v7/eth_v7_balanced_model.txt"))
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
```

- [ ] **Step 2: 实现 validate CLI 分支 + 主入口**

```python
def _read_live_settlements() -> list[dict]:
    rows = []
    for path in sorted((ROOT / "server_data" / "logs").glob("v7_settlements_*.csv")):
        rows += pd.read_csv(path).to_dict("records")
    return rows


def run_validate(proxy_url: str | None) -> None:
    model = load_model()
    start = datetime(2026, 7, 29, 18, 30, tzinfo=timezone.utc)  # 服务器首次启动后
    end = datetime(2026, 8, 4, 0, 0, tzinfo=timezone.utc)
    frames = _load_frames(start, end, proxy_url)
    F, names, finite = build_aligned_features(*frames)
    trades = simulate(F, names, finite, model, 0.555, frames[0])
    print(compare_live(trades, _read_live_settlements()))


def _load_frames(start: datetime, end: datetime, proxy_url: str | None):
    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    cols = ["open_time", "open", "high", "low", "close", "volume"]
    return tuple(
        pd.DataFrame(load_or_fetch("ETHUSDT", iv, start_ms, end_ms, proxy_url), columns=cols)
        for iv in ["5m", "15m", "1h"]
    )


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
```

`run_full` 在 Task 5 实现；先在文件中留占位函数体（`raise NotImplementedError`）保证 `main()` 可解析。

- [ ] **Step 3: 运行验证段**

```bash
python v7_backtest.py --validate
```
预期: 先拉取 5m/15m/1h 数据（约 7 天量）→ 输出对比报告。**接受标准：**
- 回测与实盘笔数差 ≤ 2（实盘 139 笔；可能因服务器启动/重启边界差 1-2 笔）
- 交集覆盖 ≥ 95% 且方向/结果不一致 = 0
- 若笔数差大：先检查服务器 config.yaml 的 stake/payout（实盘 CSV pnl +8/-10 表明 stake=10），确认模拟参数；再检查验证窗口是否覆盖服务器全部运行时间（日志确认 7/29 18:10/18:26 两次启动）

- [ ] **Step 4: 修复任何不一致后重新运行**

直到满足 Step 3 的接受标准才继续。此步禁止跳过。

- [ ] **Step 5: 提交**

```bash
git add v7_backtest.py
git commit -m "feat: V7回测-验证段与实盘139笔对齐"
```

---

### Task 5: 全量回测 + 报告

**Files:**
- Modify: `v7_backtest.py`（实现 `run_full` + 报告函数）

**Interfaces:**
- Consumes: Task 4 的 `_load_frames`；输出全量报告到 stdout

- [ ] **Step 1: 实现报告函数**

```python
def report_sweep(combos: list[tuple[str, list[dict]]]) -> str:
    lines = [f"{'方案':<28}{'笔数':<7}{'胜率':<8}{'PnL':<9}"]
    for name, sel in combos:
        if not sel:
            continue
        wins = sum(1 for t in sel if t["result"] == "WIN")
        pnl = sum(t["pnl"] for t in sel)
        lines.append(f"{name:<28}{len(sel):<7}{wins / len(sel) * 100:<8.1f}{pnl:<+9.1f}")
    return "\n".join(lines)


def report_by_period(trades: list[dict]) -> str:
    rows = []
    for t in trades:
        half = t["signal_time"][:7]
        y, m = int(half[:4]), int(half[5:7])
        rows.append((t, f"{y}H{1 if m <= 6 else 2}"))
    out = [f"{'半年':<8}{'方向':<6}{'笔数':<6}{'胜率':<8}"]
    for half in sorted({h for _, h in rows}):
        for d in ["up", "down"]:
            sel = [t for t, h in rows if h == half and t["direction"] == d]
            if not sel:
                continue
            wins = sum(1 for t in sel if t["result"] == "WIN")
            out.append(f"{half:<8}{d:<6}{len(sel):<6}{wins / len(sel) * 100:<8.1f}")
    return "\n".join(out)


def report_by_hour(trades: list[dict]) -> str:
    from collections import defaultdict
    by_h = defaultdict(list)
    for t in trades:
        by_h[int(t["signal_time"][11:13])].append(t)
    lines = [f"{'小时(北京)':<10}{'笔数':<6}{'胜率':<8}"]
    for h in sorted(by_h):
        sel = by_h[h]
        wins = sum(1 for t in sel if t["result"] == "WIN")
        lines.append(f"{h:02d}时{'':<6}{len(sel):<6}{wins / len(sel) * 100:<8.1f}")
    return "\n".join(lines)
```

- [ ] **Step 2: 实现 run_full**

```python
def run_full(start: str, end: str, proxy_url: str | None) -> None:
    model = load_model()
    s = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    e = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    frames = _load_frames(s, e, proxy_url)
    df5 = frames[0]
    print(f"数据: {len(df5)} 根5mK线 {start} ~ {end}")
    F, names, finite = build_aligned_features(*frames)
    print(f"有效特征行: {finite.sum()} / {len(df5)}")

    base_trades = simulate(F, names, finite, model, 0.555, df5)
    print(f"\n=== 基准 (阈值0.555, 与实盘相同) ===")
    print(report_sweep([("全部", base_trades)]))

    print(f"\n=== 阈值 × 方向扫描 ===")
    scan = [(f"UP≥{t} DOWN≥{t}", simulate(F, names, finite, model, t, df5))
            for t in [0.555, 0.56, 0.57, 0.58, 0.59, 0.60]]
    print(report_sweep(scan))

    print(f"\n=== 分方向 ===")
    ups = [t for t in base_trades if t["direction"] == "up"]
    downs = [t for t in base_trades if t["direction"] == "down"]
    print(report_sweep([("UP", ups), ("DOWN", downs)]))

    print(f"\n=== 组合方案 ===")
    down_hi = [t for t in downs if t["prob"] >= 0.57]
    up_hi = [t for t in ups if t["prob"] >= 0.58]
    print(report_sweep([
        ("只DOWN", downs),
        ("只DOWN 置信≥0.57", down_hi),
        ("UP≥0.58 + DOWN", up_hi + downs),
    ]))

    print(f"\n=== 分半年稳定性 ===")
    print(report_by_period(base_trades))

    print(f"\n=== 分时段(北京时间) ===")
    print(report_by_hour(base_trades))
```

注意：扫描中的 `simulate` 重复跑 6 次会重复结算相同 pending 逻辑——无副作用，结果一致（模拟为纯函数）。为提速可在后续优化为一次预测多次阈值，本阶段保持简单正确。

- [ ] **Step 3: 运行全量回测**

```bash
python v7_backtest.py --start 2024-01-01 --end 2026-08-03
```
预期: 首次运行拉取约 27 万根 5m + 9 万根 15m + 2.2 万根 1h K线（几分钟），随后输出各报告。将输出保存到 `optimize_output_v7.txt` 备用。

- [ ] **Step 4: 分析结果并更新 memory**

- 若 UP 在 ≥4 个半年期胜率 ≤55% → 确认方向过滤有效，记录到 memory（`strategy-optimization.md` 新增 V7 小节）
- 记录：各阈值×方向的笔数/胜率、最优组合、分半年稳定性、分时段特征
- 输出文件保存为 `server_data/v7_backtest_report.txt`

- [ ] **Step 5: 提交**

```bash
git add v7_backtest.py server_data/v7_backtest_report.txt
git commit -m "feat: V7回测-全量报告（阈值扫描/分半年/分时段）"
```
