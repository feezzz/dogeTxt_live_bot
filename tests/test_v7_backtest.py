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
