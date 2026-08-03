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
