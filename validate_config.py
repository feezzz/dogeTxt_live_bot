"""
Validate a promising config across ALL 5 half-year periods for ETHUSDT + BTCUSDT.
Usage: modify CONFIG dict below, then run.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from event_backtest.ensemble_v3 import backtest_ensemble_v3

PERIODS = [
    ('2024-01-01', '2024-07-01', '2024H1'),
    ('2024-07-01', '2025-01-01', '2024H2'),
    ('2025-01-01', '2025-07-01', '2025H1'),
    ('2025-07-01', '2026-01-01', '2025H2'),
    ('2026-01-01', '2026-07-01', '2026H1'),
]

CONFIGS = [
    # Best from verification: th=4.0, no extra filters, agree=1
    ("Baseline th=4.0 agree=1", dict(
        score_threshold=4.0, indicator_agree_min=1,
    )),
    # Best from verification: th=4.5 with defaults
    ("Baseline th=4.5 agree=1", dict(
        score_threshold=4.5, indicator_agree_min=1,
    )),
    # Strong score only
    ("th=4.0 strong=4.0", dict(
        score_threshold=4.0, indicator_agree_min=1,
        strong_score_min=4.0,
    )),
    # V4.2 style with reasonable defaults
    ("V42 as=4.0 adx=45", dict(
        score_threshold=3.0, indicator_agree_min=1,
        use_v42_style=True, v42_abs_score_min=4.0,
        v42_adx_max=45, v42_ret1_min=0.02, v42_ret3_min=0.10,
    )),
    # Extreme cluster
    ("th=3.0 cluster=3", dict(
        score_threshold=3.0, indicator_agree_min=1,
        extreme_cluster_min=3,
    )),
    # Triple extreme only
    ("th=3.0 triple_extreme", dict(
        score_threshold=3.0, indicator_agree_min=1,
        triple_extreme_only=True,
    )),
]

def validate(configs=None):
    if configs is None:
        configs = CONFIGS
    print(f"\n{'='*120}")
    print(f"  5-PERIOD VALIDATION — ETHUSDT + BTCUSDT")
    print(f"  Breakeven WR: 55.56% | 10m contracts (80% payout) | $25/trade")
    print(f"{'='*120}")

    for label, kwargs in configs:
        print(f"\n{'─'*120}")
        print(f"  [{label}]")
        print(f"  {'Period':<10} {'ETH Trades':>10} {'ETH WR':>8} {'ETH PnL':>10} {'BTC Trades':>10} {'BTC WR':>8} {'BTC PnL':>10}")
        print(f"  {'─'*100}")

        eth_total_t, eth_total_w, eth_total_pnl = 0, 0, 0.0
        btc_total_t, btc_total_w, btc_total_pnl = 0, 0, 0.0

        for start, end, pname in PERIODS:
            r_eth = backtest_ensemble_v3('ETHUSDT', start, end,
                capital=500, amount=25, cooldown=2, max_daily=50,
                min_atr_pct=0.08, use_time_filter=True, skip_low_vol_hours=True,
                **kwargs)
            r_btc = backtest_ensemble_v3('BTCUSDT', start, end,
                capital=500, amount=25, cooldown=2, max_daily=50,
                min_atr_pct=0.03, use_time_filter=True, skip_low_vol_hours=True,
                **kwargs)

            eth_total_t += r_eth['trades']; eth_total_w += r_eth['wins']; eth_total_pnl += r_eth['pnl']
            btc_total_t += r_btc['trades']; btc_total_w += r_btc['wins']; btc_total_pnl += r_btc['pnl']

            eth_ok = 'OK' if r_eth['win_rate'] > 55.56 else '--'
            btc_ok = 'OK' if r_btc['win_rate'] > 55.56 else '--'
            print(f"  {pname:<10} {r_eth['trades']:>10} {r_eth['win_rate']:>7.1f}% ${r_eth['pnl']:>+9.0f} {eth_ok}  "
                  f"{r_btc['trades']:>10} {r_btc['win_rate']:>7.1f}% ${r_btc['pnl']:>+9.0f} {btc_ok}")

        eth_wr = eth_total_w / eth_total_t * 100 if eth_total_t else 0
        btc_wr = btc_total_w / btc_total_t * 100 if btc_total_t else 0
        total_wr = (eth_total_w + btc_total_w) / (eth_total_t + btc_total_t) * 100 if (eth_total_t + btc_total_t) else 0
        total_pnl = eth_total_pnl + btc_total_pnl

        print(f"  {'─'*100}")
        eth_ok_str = 'OK' if eth_wr > 55.56 else '--'
        btc_ok_str = 'OK' if btc_wr > 55.56 else '--'
        print(f"  {'TOTAL':<10} {eth_total_t:>10} {eth_wr:>7.1f}% ${eth_total_pnl:>+9.0f} {eth_ok_str}  "
              f"{btc_total_t:>10} {btc_wr:>7.1f}% ${btc_total_pnl:>+9.0f} {btc_ok_str}")
        star = ' *** 65%+' if total_wr >= 65 else (' ** 60%+' if total_wr >= 60 else '')
        print(f"  COMBINED: {eth_total_t+btc_total_t} trades | {total_wr:.1f}% WR | ${total_pnl:+.0f} PnL{star}")

if __name__ == '__main__':
    validate()
