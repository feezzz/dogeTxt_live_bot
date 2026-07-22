"""
Comprehensive backtest: all dimensions for the backtest report.
Runs threshold comparison, timeframe comparison, direction, regime analysis.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict
from datetime import datetime

from event_backtest.ensemble_v3 import backtest_ensemble_v3


def compare_thresholds(symbol='ETHUSDT', start='2026-01-01', end='2026-07-01'):
    """Compare different score thresholds."""
    print(f'\n{"="*90}')
    print(f'  THRESHOLD COMPARISON — {symbol} {start} ~ {end}')
    print(f'  V3 strategy, 10m contracts (80% payout), $25/trade')
    print(f'{"="*90}\n')

    thresholds = [3.0, 3.5, 4.0, 4.5, 5.0]
    print(f'  {"Th":<8} {"Trades":>7} {"Wins":>5} {"Losses":>5} {"WR":>8} {"PnL":>10} {"Return":>8} {"Sharpe":>7}')
    print(f'  {"─"*70}')

    for th in thresholds:
        r = backtest_ensemble_v3(
            symbol, start, end,
            capital=500, amount=25,
            cooldown=2, max_daily=50,
            min_atr_pct=0.08,
            score_threshold=th,
            use_time_filter=True, skip_low_vol_hours=True,
        )
        trades = r['trades']
        wr = r['win_rate']
        pnl = r['pnl']
        ret = r['return']

        # Approximate Sharpe (daily)
        sharpe = (wr / 100 - 0.556) / 0.5 * (trades ** 0.5) if trades else 0

        print(f'  {th:<8.1f} {trades:>7} {r["wins"]:>5} {r["losses"]:>5} '
              f'{wr:>7.1f}% {pnl:>+10.1f} {ret:>+7.1f}% {sharpe:>7.2f}')

    print()


def compare_timeframes(symbol='ETHUSDT', start='2026-01-01', end='2026-07-01'):
    """Compare 10m vs 30m contracts."""
    print(f'\n{"="*90}')
    print(f'  TIMEFRAME COMPARISON — {symbol} {start} ~ {end}')
    print(f'{"="*90}\n')

    configs = [
        ('10m (80%)', '10m', 0.80, 2),
        ('30m (85%)', '30m', 0.85, 6),
    ]

    for label, tf, payout, settle_bars in configs:
        # We need to modify the backtest for 30m. The existing ensemble_v3
        # is hardcoded for 10m. We'll run with modified settle logic.
        # For now, we approximate: 30m trades settle in 6 x 5m bars.
        _run_tf_backtest(symbol, start, end, label, settle_bars, payout)


def _run_tf_backtest(symbol, start, end, label, settle_bars, payout):
    """Run a backtest with custom settle bars and payout."""
    from event_backtest.data_fetcher import load_all
    from event_backtest.indicators import (
        sma, ema, rsi, kdj, kdj_golden_cross, kdj_death_cross,
        bollinger_bands, adx, atr, atr_pct, bb_width, volume_spike,
        cci, williams_r, stochastic_rsi, aroon, aroon_osc, mfi, parabolic_sar,
        detect_candle_patterns,
    )

    data = load_all(symbol, start, end)
    candles = data['5m']
    closes = [c[4] for c in candles]
    opens = [c[1] for c in candles]
    highs = [c[2] for c in candles]
    lows = [c[3] for c in candles]
    volumes = [c[5] for c in candles]
    c1h = [c[4] for c in data['1h']]
    h1h = [c[2] for c in data['1h']]
    l1h = [c[3] for c in data['1h']]
    t5 = [c[0] for c in candles]
    t1h = [c[0] for c in data['1h']]

    total = len(candles)
    warmup = 60

    # All indicators
    rsi7 = rsi(closes, 7)
    ma5 = sma(closes, 5); ma10 = sma(closes, 10); ma20 = sma(closes, 20)
    k, d, j = kdj(highs, lows, closes, period=6, k_period=3, d_period=3)
    kg = kdj_golden_cross(k, d); kd = kdj_death_cross(k, d)
    bb_mid, bb_up, bb_low = bollinger_bands(closes, period=20, std_mult=2.0)
    bbw = bb_width(bb_up, bb_low, bb_mid)
    vol_spike = volume_spike(volumes, period=20, threshold=1.5)
    adx_1h, pdi_1h, mdi_1h = adx(h1h, l1h, c1h, period=14)
    atr_pct_5m = atr_pct(atr(highs, lows, closes, 14), closes)
    cci14 = cci(highs, lows, closes, period=14)
    wr14 = williams_r(highs, lows, closes, period=14)
    stoch_k, stoch_d = stochastic_rsi(closes, period=14, stoch_period=14)
    aroon_up, aroon_down = aroon(highs, lows, period=14)
    aroon_osc_vals = aroon_osc(aroon_up, aroon_down)
    mfi14 = mfi(highs, lows, closes, volumes, period=14)
    sar = parabolic_sar(highs, lows)
    patterns = detect_candle_patterns(opens, highs, lows, closes)
    ema9 = ema(closes, 9); ema21 = ema(closes, 21)

    def tf_idx(timestamps, target_ts):
        for i in range(len(timestamps) - 1, -1, -1):
            if timestamps[i] <= target_ts:
                return i
        return -1

    equity = 500
    trades = []
    last_signal_idx = -999
    daily_bets = 0
    current_day = None
    LOW_VOL_HOURS = frozenset([22, 23, 0, 1, 2, 3, 4])
    SCORE_TH = 4.0

    for i in range(warmup, total - settle_bars - 1):
        ts = candles[i][0]
        dt = datetime.fromtimestamp(ts / 1000)

        if dt.day != current_day:
            current_day = dt.day; daily_bets = 0
        if i - last_signal_idx < 2: continue
        if daily_bets >= 50: continue
        if equity <= 25: break
        if atr_pct_5m[i] < 0.08: continue

        idx_1h = tf_idx(t1h, t5[i])
        if idx_1h < 20: continue
        if dt.hour in LOW_VOL_HOURS and daily_bets >= 50 // 3: continue

        adx_val = adx_1h[idx_1h]
        pdi = pdi_1h[idx_1h]; mdi = mdi_1h[idx_1h]
        di_diff = pdi - mdi; price = closes[i]; price_up = closes[i] > opens[i]

        # Regime
        if adx_val > 25 or abs(aroon_osc_vals[i]) > 50:
            regime = 'trending'
        elif adx_val < 18 or (bbw[i] < 1.5 and abs(aroon_osc_vals[i]) < 25):
            regime = 'ranging'
        else:
            regime = 'neutral'

        # Scoring (same as V3 production)
        score = 0.0
        rsi7_v = rsi7[i]
        if rsi7_v < 20: score += 2.0
        elif rsi7_v < 30: score += 1.2
        elif rsi7_v < 40: score += 0.3
        elif rsi7_v > 80: score -= 2.0
        elif rsi7_v > 70: score -= 1.2
        elif rsi7_v > 60: score -= 0.3

        stoch_k_v = stoch_k[i]; stoch_d_v = stoch_d[i]
        if stoch_k_v < 10 and stoch_d_v < 15: score += 1.5
        elif stoch_k_v < 20: score += 0.5
        elif stoch_k_v > 90 and stoch_d_v > 85: score -= 1.5
        elif stoch_k_v > 80: score -= 0.5
        if i > 0 and stoch_k[i-1] <= stoch_d[i-1] and stoch_k[i] > stoch_d[i]:
            score += 0.8
        elif i > 0 and stoch_k[i-1] >= stoch_d[i-1] and stoch_k[i] < stoch_d[i]:
            score -= 0.8

        mfi_v = mfi14[i]
        if mfi_v < 15: score += 1.5
        elif mfi_v < 25: score += 0.5
        elif mfi_v > 85: score -= 1.5
        elif mfi_v > 75: score -= 0.5

        cci_v = cci14[i]
        if cci_v < -200: score += 1.5
        elif cci_v < -100: score += 0.7
        elif cci_v > 200: score -= 1.5
        elif cci_v > 100: score -= 0.7

        wr_v = wr14[i]
        if wr_v < -90: score += 1.2
        elif wr_v < -80: score += 0.5
        elif wr_v > -10: score -= 1.2
        elif wr_v > -20: score -= 0.5

        sar_v = sar[i]
        if price > sar_v: score += 0.5
        else: score -= 0.5

        aroon_up_v = aroon_up[i]; aroon_down_v = aroon_down[i]
        if aroon_up_v > 70 and aroon_down_v < 30: score += 0.5
        elif aroon_down_v > 70 and aroon_up_v < 30: score -= 0.5

        if price > ma20[i] and ma5[i] > ma10[i]: score += 0.8
        elif price < ma20[i] and ma5[i] < ma10[i]: score -= 0.8

        if ema9[i] > ema21[i]: score += 0.3
        else: score -= 0.3

        if kg[i]: score += 0.8
        elif kd[i]: score -= 0.8
        if j[i] < 0: score += 0.5
        elif j[i] > 100: score -= 0.5

        if bb_up[i] > bb_low[i]:
            bb_pos = (price - bb_low[i]) / (bb_up[i] - bb_low[i])
            if bb_pos < 0.08: score += 1.2
            elif bb_pos < 0.2: score += 0.5
            elif bb_pos > 0.92: score -= 1.2
            elif bb_pos > 0.8: score -= 0.5

        if vol_spike[i]:
            if price_up: score += 0.5
            else: score -= 0.5

        if patterns['hammer'][i]: score += 1.0
        elif patterns['shooting_star'][i]: score -= 1.0
        if patterns['bullish_engulfing'][i]: score += 1.5
        elif patterns['bearish_engulfing'][i]: score -= 1.5

        if regime == 'trending':
            if di_diff > 5: score += 0.8
            elif di_diff < -5: score -= 0.8
            if di_diff > 3 and rsi7_v < 50: score += 0.3
            elif di_diff < -3 and rsi7_v > 50: score -= 0.3
        elif regime == 'ranging':
            bb_pos_val = (price - bb_low[i]) / (bb_up[i] - bb_low[i]) if bb_up[i] > bb_low[i] else 0.5
            if bb_pos_val < 0.15: score += 0.5
            elif bb_pos_val > 0.85: score -= 0.5

        if score >= SCORE_TH: direction = 'up'
        elif score <= -SCORE_TH: direction = 'down'
        else: direction = None
        if direction is None: continue

        # Quality check
        agree = 0
        if direction == 'up':
            if stoch_k_v < 30: agree += 1
            if mfi_v < 40: agree += 1
            if aroon_osc_vals[i] > -30: agree += 1
            if price > sar_v: agree += 1
        else:
            if stoch_k_v > 70: agree += 1
            if mfi_v > 60: agree += 1
            if aroon_osc_vals[i] < 30: agree += 1
            if price < sar_v: agree += 1
        if agree < 1: continue

        # Execute
        entry_price = candles[i + 1][1]
        settle_idx = min(i + settle_bars, total - 1)
        settle_price = candles[settle_idx][4]

        if direction == 'up': win = settle_price > entry_price
        else: win = settle_price < entry_price

        pnl = 25 * payout if win else -25
        equity += pnl
        trades.append({
            'time': dt, 'direction': direction, 'win': win,
            'pnl': pnl, 'regime': regime, 'score': abs(score),
        })
        last_signal_idx = i
        daily_bets += 1

    wins = [t for t in trades if t['win']]
    losses = [t for t in trades if not t['win']]
    wr = len(wins) / len(trades) * 100 if trades else 0
    total_pnl = sum(t['pnl'] for t in trades)

    # Direction breakdown
    ups = [t for t in trades if t['direction'] == 'up']
    downs = [t for t in trades if t['direction'] == 'down']
    up_wr = sum(1 for t in ups if t['win']) / len(ups) * 100 if ups else 0
    dn_wr = sum(1 for t in downs if t['win']) / len(downs) * 100 if downs else 0

    # Monthly
    by_month = defaultdict(lambda: [0, 0, 0.0, 0.0])  # wins, losses, pnl, equity
    eq = 500
    for t in trades:
        mk = t['time'].strftime('%Y-%m')
        by_month[mk][0 if t['win'] else 1] += 1
        by_month[mk][2] += t['pnl']
    for mk in sorted(by_month):
        eq += by_month[mk][2]
        by_month[mk][3] = eq

    print(f'\n  --- {label} ---')
    print(f'  Trades: {len(trades)} | WR: {wr:.1f}% | PnL: ${total_pnl:+.0f} | '
          f'Final: ${equity:.0f}')
    print(f'  Up: {len(ups)} ({up_wr:.1f}%) | Down: {len(downs)} ({dn_wr:.1f}%)')
    print(f'  Monthly:')
    for mk in sorted(by_month):
        w, l, pnl_m, eq_m = by_month[mk]
        twl = w + l
        print(f'    {mk}: {twl} trades, WR={w/twl*100:.1f}%, PnL=${pnl_m:+.0f}, Eq=${eq_m:.0f}')

    return trades


if __name__ == '__main__':
    compare_thresholds('ETHUSDT', '2026-01-01', '2026-07-01')
    compare_timeframes('ETHUSDT', '2026-01-01', '2026-07-01')
