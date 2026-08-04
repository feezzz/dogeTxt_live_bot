"""
Grid search using ensemble_v3.py's ADVANCED filters to find 65%+ WR.
Tests V4.2-style, capitulation, divergence, momentum exhaustion, etc.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from event_backtest.ensemble_v3 import backtest_ensemble_v3

PERIOD = ('2026-01-01', '2026-07-01')
SYM = 'ETHUSDT'

def test(label, **kwargs):
    """Run single backtest and print result."""
    r = backtest_ensemble_v3(SYM, PERIOD[0], PERIOD[1],
        capital=500, amount=25, cooldown=2, max_daily=50,
        min_atr_pct=0.08, use_time_filter=True, skip_low_vol_hours=True,
        **kwargs)
    print(f"  {label:<55} {r['trades']:>5} trades | WR {r['win_rate']:>5.1f}% | PnL ${r['pnl']:>+8.0f}")
    return r

print(f"\n{'='*100}")
print(f"  ADVANCED FILTER GRID SEARCH — {SYM} {PERIOD[1]}")
print(f"  Goal: Find any config reaching 60%+ WR (step 1), then 65%+")
print(f"{'='*100}")

# ===== Round 1: V4.2-style filters (momentum-based capitulation) =====
print(f"\n--- R1: V4.2-style momentum filters ---")
print(f"  {'Config':<55} {'Trades':>5} {'WR':>7} {'PnL':>10}")

# Baseline: no extra filters, just BB + 15m + agree=1
test("R1_baseline", score_threshold=4.0)

# V4.2 with different abs_score_min
for abs_min in [3.5, 4.0, 4.2, 4.5, 5.0, 5.5]:
    for adx_max in [35, 40, 45, 50, 55]:
        test(f"R1_v42_as={abs_min}_adx={adx_max}",
             score_threshold=3.0, use_v42_style=True,
             v42_abs_score_min=abs_min, v42_adx_max=adx_max,
             v42_ret1_min=0.02, v42_ret3_min=0.10,
             indicator_agree_min=1)

# ===== Round 2: Capitulation ret1 + ret3 standalone =====
print(f"\n--- R2: Capitulation filters (ret1=current bar momentum, ret3=3-bar momentum) ---")

for th in [3.0, 3.5, 4.0]:
    for r1 in [0.01, 0.02, 0.03]:
        for r3 in [0.05, 0.10, 0.15]:
            test(f"R2_th={th}_r1={r1}_r3={r3}",
                 score_threshold=th, use_cap_ret1=True, cap_ret1_min=r1,
                 use_cap_ret3=True, cap_ret3_min=r3,
                 indicator_agree_min=1)

# ===== Round 3: RSI divergence =====
print(f"\n--- R3: RSI divergence ---")

for th in [3.0, 3.5, 4.0, 4.5, 5.0]:
    test(f"R3_th={th}_rsi_div", score_threshold=th, use_rsi_divergence=True,
         indicator_agree_min=1)

# ===== Round 4: Momentum exhaustion =====
print(f"\n--- R4: Momentum exhaustion ---")

for th in [3.0, 3.5, 4.0, 4.5, 5.0]:
    test(f"R4_th={th}_mom_exh", score_threshold=th, use_momentum_exhaustion=True,
         indicator_agree_min=1)

# ===== Round 5: Trend continuation =====
print(f"\n--- R5: Trend continuation ---")

for th in [3.0, 3.5, 4.0, 4.5, 5.0]:
    test(f"R5_th={th}_trend_cont", score_threshold=th, use_trend_continuation=True,
         indicator_agree_min=1)

# ===== Round 6: Breakout trending =====
print(f"\n--- R6: Breakout trending ---")

for th in [3.0, 3.5, 4.0, 4.5, 5.0]:
    test(f"R6_th={th}_breakout", score_threshold=th, use_breakout_trending=True,
         indicator_agree_min=1)

# ===== Round 7: Extreme cluster =====
print(f"\n--- R7: Extreme cluster ---")

for th in [3.0, 3.5, 4.0]:
    for cl in [2, 3, 4]:
        test(f"R7_th={th}_cluster={cl}", score_threshold=th,
             extreme_cluster_min=cl, indicator_agree_min=1)

# ===== Round 8: Triple extreme =====
print(f"\n--- R8: Triple extreme (RSI + BB + Stoch) ---")

for th in [3.0, 3.5, 4.0, 4.5, 5.0]:
    test(f"R8_th={th}_triple", score_threshold=th, triple_extreme_only=True,
         indicator_agree_min=1)

# ===== Round 9: V4.2 + divergence + momentum exhaustion combo =====
print(f"\n--- R9: Best combos ---")

# Best individual filters combined
for th in [3.5, 4.0, 4.5]:
    for adx_max in [40, 45, 50]:
        test(f"R9_th={th}_adx={adx_max}_v42+div+mom",
             score_threshold=th, use_v42_style=True,
             v42_abs_score_min=4.0, v42_adx_max=adx_max,
             v42_ret1_min=0.02, v42_ret3_min=0.10,
             use_rsi_divergence=True, use_momentum_exhaustion=True,
             indicator_agree_min=1)

# ===== Round 10: Edge depth + wick filter combos =====
print(f"\n--- R10: Edge depth + wick ---")

for th in [3.5, 4.0, 4.5]:
    for ed in [0.05, 0.07, 0.09]:
        for wm in [5, 10]:
            test(f"R10_th={th}_ed={ed}_wm={wm}",
                 score_threshold=th, use_edge_depth=True, edge_depth_min=ed,
                 use_wick_max=True, wick_max_pct=wm,
                 indicator_agree_min=1)

print(f"\n{'='*100}")
print(f"  DONE")
print(f"{'='*100}")
