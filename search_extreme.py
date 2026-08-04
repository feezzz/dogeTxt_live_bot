"""
Test extreme score thresholds + filter combos not in the main grid search.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from event_backtest.ensemble_v3 import backtest_ensemble_v3

PERIOD = ('2026-01-01', '2026-07-01')

def test(sym, label, **kwargs):
    r = backtest_ensemble_v3(sym, PERIOD[0], PERIOD[1],
        capital=500, amount=25, cooldown=2, max_daily=50,
        min_atr_pct=0.08 if sym == 'ETHUSDT' else 0.03,
        use_time_filter=True, skip_low_vol_hours=True,
        **kwargs)
    wr = r['win_rate']; pnl = r['pnl']; n = r['trades']
    star = ' ***' if wr >= 65 else (' **' if wr >= 60 else '')
    print(f"  {sym} {label:<55} {n:>5} trades | WR {wr:>5.1f}% | PnL ${pnl:>+8.0f}{star}")
    return r

print(f"\n{'='*100}")
print(f"  EXTREME THRESHOLD + COMBO SEARCH")
print(f"{'='*100}")

# Very high thresholds + no filters
print(f"\n--- High thresholds (no filters beyond default agree=1) ---")
for sym in ['ETHUSDT', 'BTCUSDT']:
    for th in [6.0, 7.0, 8.0, 9.0, 10.0]:
        test(sym, f"th={th}_bare", score_threshold=th, indicator_agree_min=1)

# Very high thresholds with strong_score_min
print(f"\n--- Strong score filter ---")
for sym in ['ETHUSDT', 'BTCUSDT']:
    for th in [3.0, 4.0, 5.0]:
        for ss in [3.0, 4.0, 5.0, 6.0]:
            test(sym, f"th={th}_strong={ss}", score_threshold=th,
                 strong_score_min=ss, indicator_agree_min=1)

# Dynamic threshold
print(f"\n--- Dynamic threshold (trend_penalty, range_bonus) ---")
for sym in ['ETHUSDT', 'BTCUSDT']:
    for th in [3.0, 4.0, 5.0]:
        for tp in [1.0, 1.5, 2.0]:
            for rb in [0.5, 1.0]:
                test(sym, f"th={th}_tp={tp}_rb={rb}", score_threshold=th,
                     dynamic_threshold=True, trend_penalty=tp, range_bonus=rb,
                     indicator_agree_min=1)

# Extreme cluster + RSI divergence
print(f"\n--- Extreme cluster + RSI divergence combo ---")
for sym in ['ETHUSDT', 'BTCUSDT']:
    for th in [3.0, 4.0, 5.0]:
        for cl in [2, 3]:
            test(sym, f"th={th}_cl={cl}_div", score_threshold=th,
                 extreme_cluster_min=cl, use_rsi_divergence=True,
                 indicator_agree_min=1)

# V4.2 tight settings
print(f"\n--- V4.2 tight: high abs_score, low adx_max, tight ret ---")
for sym in ['ETHUSDT', 'BTCUSDT']:
    for abs_s in [4.5, 5.0, 5.5, 6.0]:
        for adx in [30, 35, 40]:
            for r3 in [0.10, 0.15]:
                test(sym, f"v42_as={abs_s}_adx={adx}_r3={r3}", score_threshold=3.0,
                     use_v42_style=True, v42_abs_score_min=abs_s,
                     v42_adx_max=adx, v42_ret1_min=0.02, v42_ret3_min=r3,
                     indicator_agree_min=1)

# Require reversal candle
print(f"\n--- Require reversal candle ---")
for sym in ['ETHUSDT', 'BTCUSDT']:
    for th in [3.0, 4.0, 5.0, 6.0]:
        test(sym, f"th={th}_rev_candle", score_threshold=th,
             require_reversal_candle=True, indicator_agree_min=1)

print(f"\nDONE")
