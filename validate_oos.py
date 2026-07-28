"""
P0: Out-of-sample validation for current production params.
Tests th=5.0 agree=2 on 2026-07 data (NOT used in any grid search).
Compares against 2026H1 baseline to detect overfitting.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from event_backtest.ensemble_v3 import backtest_ensemble_v3


def print_report(symbol, r, label):
    print(f"\n{'='*70}")
    print(f"  {symbol} | {label}")
    print(f"{'='*70}")
    print(f"  总交易: {r['trades']}  |  胜: {r['wins']}  |  负: {r['losses']}  |  胜率: {r['win_rate']:.1f}%")
    print(f"  总盈亏: ${r['pnl']:+.2f}  |  收益率: {r['return']:+.1f}%  |  终值: ${r['final_equity']:.2f}")

    invested = r['trades'] * 25
    roi = r['pnl'] / invested * 100 if invested > 0 else 0
    print(f"  ROI: {roi:+.1f}% (投入${invested})")
    print(f"  盈亏平衡线: 55.6%  |  超额: {r['win_rate'] - 55.6:+.1f}%")

    print(f"\n  行情胜率:")
    for reg in ['trending', 'neutral', 'ranging']:
        w, l = r['regime_stats'].get(reg, [0, 0])
        t = w + l
        if t > 0:
            print(f"    {reg:10s}: {t:>4}笔  WR={w/t*100:.0f}%")

    if 'by_month' in r and r['by_month']:
        print(f"\n  月度:")
        for mk in sorted(r['by_month'].keys()):
            w, l, pnl = r['by_month'][mk]
            t = w + l
            wr = w / t * 100 if t > 0 else 0
            print(f"    {mk}: {t:>4}笔  胜率 {wr:>5.0f}%  PnL ${pnl:>+8.2f}")

    # Direction breakdown
    trades = r.get('_trades', [])
    if trades:
        ups = [t for t in trades if t.get('direction') == 'up']
        dns = [t for t in trades if t.get('direction') == 'down']
        for label, subset in [('做多', ups), ('做空', dns)]:
            if subset:
                w = sum(1 for t in subset if t['win'])
                wr = w / len(subset) * 100
                pnl = sum(t.get('pnl', 20 if t['win'] else -25) for t in subset)
                print(f"    {label}: {w}/{len(subset)} {wr:.0f}% ${pnl:+.0f}")


def backtest_with_live_config(symbol, start, end):
    """Backtest using EXACT production parameters from config.yaml + strategy_engine.py"""
    return backtest_ensemble_v3(
        symbol=symbol,
        start=start,
        end=end,
        capital=500,
        amount=25,
        cooldown=2,
        max_daily=50,
        min_atr_pct=0.05 if symbol == 'ETHUSDT' else 0.03,
        score_threshold=5.0,
        use_time_filter=False,
        skip_low_vol_hours=False,
        indicator_agree_min=2,
        bb_up_threshold=0.10,
        bb_down_threshold=0.90,
        rsi15_up_max=35,
        rsi15_down_min=65,
        require_cci_dir=False,
        extreme_cluster_min=0,
    )


if __name__ == '__main__':
    # ---- In-sample (2026H1, was used in grid search) ----
    print("=" * 70)
    print("  IN-SAMPLE: 2026-01-01 ~ 2026-07-01 (participated in optimization)")
    print("=" * 70)

    for symbol in ['ETHUSDT', 'BTCUSDT']:
        r = backtest_with_live_config(symbol, '2026-01-01', '2026-07-01')
        print_report(symbol, r, "th=5.0 agree=2 (current production)")

    # ---- Out-of-sample (2026-07, NEVER used in optimization) ----
    print("\n" + "=" * 70)
    print("  OUT-OF-SAMPLE: 2026-07-01 ~ 2026-07-28 (NEVER optimized)")
    print("=" * 70)

    for symbol in ['ETHUSDT', 'BTCUSDT']:
        r = backtest_with_live_config(symbol, '2026-07-01', '2026-07-28')
        print_report(symbol, r, "th=5.0 agree=2 (current production)")

    # ---- Summary comparison ----
    print("\n" + "=" * 70)
    print("  OVERFITTING CHECK")
    print("=" * 70)

    for period_name, start, end in [
        ('In-sample (H1)', '2026-01-01', '2026-07-01'),
        ('Out-of-sample (Jul)', '2026-07-01', '2026-07-28'),
    ]:
        total_trades = 0
        total_pnl = 0.0
        total_wins = 0
        for symbol in ['ETHUSDT', 'BTCUSDT']:
            r = backtest_with_live_config(symbol, start, end)
            total_trades += r['trades']
            total_pnl += r['pnl']
            total_wins += r['wins']

        wr = total_wins / total_trades * 100 if total_trades > 0 else 0
        print(f"  {period_name:<22}: {total_trades:>5}笔  WR={wr:>5.1f}%  PnL=${total_pnl:>+9.2f}  Edge={wr-55.6:>+.1f}%")
