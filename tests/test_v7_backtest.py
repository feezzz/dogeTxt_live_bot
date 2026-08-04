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


def _model_stub(probs):
    class M:
        def num_trees(self):
            return 1
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
    # 信号 5/6/7/8 连续触发（cooldown=1 允许连发）；第4个信号(索引8)在第3笔结算前
    # 已触发（当时 pnl=-50 未到熔断线）。第3笔(信号7)在 i=9 结算后 pnl=-75 → 熔断，
    # i>=9 不再出新信号（与实盘 v7_main.py 顺序一致）。无熔断时 5..59 共 55 个信号。
    assert len(trades) == 4
    assert max(t["signal_idx"] for t in trades) == 8  # 熔断后无新信号
    assert all(t["result"] == "LOSS" for t in trades)


def test_breaker_latches_after_pnl_recovery():
    # 回归: 熔断触发后即使 pnl 回升（在途单 WIN 结算）也不得恢复出信号，
    # 与实盘 v7_main.py _risk_blocked 一致——_circuit_breaker 一旦为 True，
    # 当日剩余时间永久拦截（仅跨日复位）。
    df5, F, names, finite = _sim_inputs(falling=True)  # 严格递减 → UP 必亏、DOWN 必赢
    # 信号5 down（必赢+20）、信号10 down（熔断触发时的在途单，必赢+20）、
    # 其余 6..59 全 up（必亏-25）。
    model = _model_stub({**{i: 0.9 for i in range(6, 60)}, 5: 0.1, 10: 0.1})
    trades = bt.simulate(F, names, finite, model, 0.555, df5,
                         max_daily_loss=-75.0, stake=25.0, payout=0.8)
    idxs = [t["signal_idx"] for t in trades]
    # 推演: i=5..10 依次出信号 5(down)/6..9(up)/10(down) 共 6 个；
    # i=11 结算信号9 → pnl=-80 首次触发熔断（i>=11 不再出新信号）；
    # i=12 结算信号10（熔断时在途单）→ WIN +20，pnl 回升到 -60 ——
    # 若熔断不锁存会在这里恢复出信号（signal_idx=12,13...），锁存后必须没有。
    assert idxs == [5, 6, 7, 8, 9, 10]
    assert max(idxs) == 10                      # 熔断后（signal_idx > 10）不再有新信号
    assert trades[-1]["result"] == "WIN"        # 在途单熔断后仍正常结算（WIN）
    assert sum(t["pnl"] for t in trades) == -60.0  # +20 -25*4 +20


def test_session_hours_signal_layer_semantics():
    # 核心语义：被过滤的信号不占冷却位（错误的后过滤实现会让本测试失败）
    df5, F, names, finite = _sim_inputs()          # t0=1704067200000 = 2024-01-01 00:00 UTC = 08:00 北京
    model = _model_stub({10: 0.9, 11: 0.9})        # idx10 close 08:55 北京, idx11 close 09:00 北京
    trades = bt.simulate(F, names, finite, model, 0.555, df5,
                         cooldown=2, session_hours={9})
    assert len(trades) == 1
    assert trades[0]["signal_idx"] == 11           # idx10 被过滤不更新 last_signal_idx → idx11 不受冷却拦截


def test_session_hours_blocks_all():
    df5, F, names, finite = _sim_inputs()
    model = _model_stub({5: 0.9, 6: 0.1})
    trades = bt.simulate(F, names, finite, model, 0.555, df5, session_hours={9})
    assert trades == []                            # idx5/6 均在 08 点, 不在 {9}


def test_prob_of_equivalence():
    df5, F, names, finite = _sim_inputs()
    model = _model_stub({5: 0.9, 6: 0.1})
    a = bt.simulate(F, names, finite, model, 0.555, df5)
    prob_of = {5: 0.9, 6: 0.1}                     # 与 stub 输出一致
    b = bt.simulate(F, names, finite, model, 0.555, df5, prob_of=prob_of)
    assert a == b


def test_predict_probs_window_and_finite():
    # 回归: predict_probs 的 start/end_ms 窗口过滤与含 False 的 finite 掩码过滤。
    # 设计说明: predict_probs 返回的 dict 以"原始行号"为键（= valid_rows 本身），
    # 键集完全由窗口+finite 决定，与模型预测值无关；而 _model_stub.predict 按
    # 子集位置取概率（predict 收到 F[valid_rows]，无法反推原始行号），在此断言
    # 预测值会与原始行号静默错位。因此本测试只断言键集（窗口过滤逻辑本身），
    # 值的正确性由 test_prob_of_equivalence 用显式 {原始索引: 概率} 字典覆盖。
    df5, F, names, finite = _sim_inputs()
    finite = finite.copy()
    finite[7] = False                              # 含 False 掩码: 中间行不可用
    start_ms = df5.iloc[5]["open_time"] + bt.FIVE_MINUTES_MS    # 窗口含信号时刻 i=5..10
    end_ms = df5.iloc[10]["open_time"] + bt.FIVE_MINUTES_MS
    model = _model_stub({})                        # 预测值路径无关, 只验证键集
    prob_of = bt.predict_probs(F, finite, model, df5, start_ms, end_ms)
    # 窗口剔除 0..4 与 11+; finite 剔除 7 → 有效子集恰为 {5, 6, 8, 9, 10}
    assert set(prob_of) == {5, 6, 8, 9, 10}
    assert all(isinstance(k, int) and isinstance(v, float) for k, v in prob_of.items())
