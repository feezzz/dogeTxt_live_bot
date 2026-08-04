"""
Quick verify: ensemble_v3.py baseline WR at different thresholds.
Runs the EXACT same call as run_full_backtest.py's compare_thresholds().
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from event_backtest.ensemble_v3 import backtest_ensemble_v3

symbols = ['ETHUSDT', 'BTCUSDT']
period = ('2026-01-01', '2026-07-01', '2026H1')

print(f"\n{'='*90}")
print(f"  ENSEMBLE V3 VERIFICATION — Default params (agree_min=1, BB=0.10/0.90, 15m=35/65)")
print(f"  ETHUSDT + BTCUSDT | {period[2]}")
print(f"{'='*90}")

for th in [3.0, 3.5, 4.0, 4.5, 5.0, 6.0]:
    print(f"\n  --- th={th:.1f} ---")
    for sym in symbols:
        r = backtest_ensemble_v3(
            sym, period[0], period[1],
            capital=500, amount=25,
            cooldown=2, max_daily=50,
            min_atr_pct=0.08 if sym == 'ETHUSDT' else 0.03,
            score_threshold=th,
            use_time_filter=True, skip_low_vol_hours=True,
        )
        print(f"  {sym}: {r['trades']:>5} trades | WR {r['win_rate']:>5.1f}% | PnL ${r['pnl']:>+8.0f}")

# Also test WITHOUT the 15m and BB filters
print(f"\n{'='*90}")
print(f"  NO FILTERS (agree_min=1, BB=1.0/0.0, 15m=0/100) — essentially unfiltered")
print(f"{'='*90}")

for th in [3.0, 4.0, 5.0]:
    print(f"\n  --- th={th:.1f} ---")
    for sym in symbols:
        r = backtest_ensemble_v3(
            sym, period[0], period[1],
            capital=500, amount=25,
            cooldown=2, max_daily=50,
            min_atr_pct=0.08 if sym == 'ETHUSDT' else 0.03,
            score_threshold=th,
            use_time_filter=True, skip_low_vol_hours=True,
            indicator_agree_min=1,
            bb_up_threshold=1.0, bb_down_threshold=0.0,  # effectively off
            rsi15_up_max=0, rsi15_down_min=100,  # effectively off
        )
        print(f"  {sym}: {r['trades']:>5} trades | WR {r['win_rate']:>5.1f}% | PnL ${r['pnl']:>+8.0f}")

# Test with agree_min=2 (like live_bot)
print(f"\n{'='*90}")
print(f"  AGREE>=2 + BB + 15m (like live_bot production)")
print(f"{'='*90}")

for th in [3.0, 4.0, 5.0]:
    print(f"\n  --- th={th:.1f} ---")
    for sym in symbols:
        r = backtest_ensemble_v3(
            sym, period[0], period[1],
            capital=500, amount=25,
            cooldown=2, max_daily=50,
            min_atr_pct=0.08 if sym == 'ETHUSDT' else 0.03,
            score_threshold=th,
            use_time_filter=True, skip_low_vol_hours=True,
            indicator_agree_min=2,
            bb_up_threshold=0.10, bb_down_threshold=0.90,
            rsi15_up_max=35, rsi15_down_min=65,
        )
        print(f"  {sym}: {r['trades']:>5} trades | WR {r['win_rate']:>5.1f}% | PnL ${r['pnl']:>+8.0f}")
