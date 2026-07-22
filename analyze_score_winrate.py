"""
Analyze win rate by absolute score bucket (Jan-Jun 2026).
Tests: does higher score → higher win rate?
"""
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from event_backtest.ensemble_v3 import backtest_ensemble_v3


def analyze(symbol: str = 'ETHUSDT', start: str = '2026-01-01',
            end: str = '2026-07-01'):
    """Run backtest with low threshold to capture all score ranges."""

    result = backtest_ensemble_v3(
        symbol, start, end,
        capital=500, amount=25,
        cooldown=2, max_daily=200,
        min_atr_pct=0.06,
        score_threshold=1.0,  # Low threshold to capture all scored signals
        use_time_filter=False,
        skip_low_vol_hours=False,
    )

    # The backtest returns aggregate stats, but we need individual trades.
    # Let's re-run with per-trade collection.
    # Actually, the return dict doesn't include trade list. We need to modify.
    # Let's just run the backtest function and collect trades inline.

    print(f'\n{"="*60}')
    print(f'  Score → Win Rate Analysis: {symbol} {start} ~ {end}')
    print(f'  (using backtest_ensemble_v3 with th=1.0, no filters)')
    print(f'{"="*60}\n')

    # The return from backtest_ensemble_v3 doesn't include the trades list.
    # Show what aggregate backtest gives us at least.
    total_pnl = result['pnl']
    total_trades = result['trades']
    wr = result['win_rate']

    print(f'Total trades (th=1.0): {total_trades}')
    print(f'Overall WR: {wr:.1f}%')
    print(f'Total PnL: ${total_pnl:+.0f}')
    print(f'Return: {result["return"]:+.1f}%')
    print(f'\nMonthly breakdown:')
    for month, (w, l, pnl) in sorted(result['by_month'].items()):
        print(f'  {month}: {w+l} trades, WR={w/(w+l)*100:.1f}%, PnL=${pnl:+.0f}')

    print(f'\nRegime breakdown:')
    for reg, (w, l) in result['regime_stats'].items():
        t = w + l
        print(f'  {reg}: {t} trades, WR={w/t*100:.1f}%')


# ----------------------------------------------------------------
# Direct score-bucket analysis using ensemble_v3 scoring directly
# ----------------------------------------------------------------
def score_bucket_analysis(symbol: str = 'ETHUSDT', start: str = '2026-01-01',
                          end: str = '2026-07-01'):
    """Re-run V3 scoring on all candles and bucket by absolute score."""
    from datetime import datetime
    from event_backtest.data_fetcher import load_all
    from event_backtest.indicators import (
        sma, ema, rsi, kdj, kdj_golden_cross, kdj_death_cross,
        bollinger_bands, adx, atr, atr_pct, bb_width, volume_spike,
        cci, williams_r, stochastic_rsi, aroon, aroon_osc, mfi, parabolic_sar,
        detect_candle_patterns,
    )

    print(f'\n{"="*80}')
    print(f'  Per-Score-Bucket Win Rate: {symbol} {start} ~ {end}')
    print(f'{"="*80}\n')

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

    # Indicators
    rsi7 = rsi(closes, 7)
    ma5 = sma(closes, 5)
    ma10 = sma(closes, 10)
    ma20 = sma(closes, 20)
    k, d, j = kdj(highs, lows, closes, period=6, k_period=3, d_period=3)
    kg = kdj_golden_cross(k, d)
    kd = kdj_death_cross(k, d)
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
    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)

    def tf_idx(timestamps, target_ts):
        for i in range(len(timestamps) - 1, -1, -1):
            if timestamps[i] <= target_ts:
                return i
        return -1

    # Score buckets: [1-3), [3-4), [4-5), [5-6), [6-7), [7+)
    buckets = [
        (1.0, 3.0, '1.0-3.0'),
        (3.0, 4.0, '3.0-4.0'),
        (4.0, 5.0, '4.0-5.0'),
        (5.0, 6.0, '5.0-6.0'),
        (6.0, 7.0, '6.0-7.0'),
        (7.0, 99,  '7.0+'),
    ]

    bucket_data = {label: {'wins': 0, 'losses': 0, 'pnl': 0.0, 'trades': []}
                   for _, _, label in buckets}

    for i in range(warmup, total - 2):
        ts = candles[i][0]
        dt = datetime.fromtimestamp(ts / 1000)

        if atr_pct_5m[i] < 0.06:
            continue

        idx_1h = tf_idx(t1h, t5[i])
        if idx_1h < 20:
            continue

        adx_val = adx_1h[idx_1h]
        pdi = pdi_1h[idx_1h]
        mdi = mdi_1h[idx_1h]
        di_diff = pdi - mdi
        price = closes[i]
        price_up = closes[i] > opens[i]

        # Regime
        if adx_val > 25 or abs(aroon_osc_vals[i]) > 50:
            regime = 'trending'
        elif adx_val < 18 or (bbw[i] < 1.5 and abs(aroon_osc_vals[i]) < 25):
            regime = 'ranging'
        else:
            regime = 'neutral'

        # --- V3 Scoring (same as production) ---
        score = 0.0

        rsi7_v = rsi7[i]
        if rsi7_v < 20:       score += 2.0
        elif rsi7_v < 30:     score += 1.2
        elif rsi7_v < 40:     score += 0.3
        elif rsi7_v > 80:     score -= 2.0
        elif rsi7_v > 70:     score -= 1.2
        elif rsi7_v > 60:     score -= 0.3

        stoch_k_v = stoch_k[i]
        stoch_d_v = stoch_d[i]
        if stoch_k_v < 10 and stoch_d_v < 15:     score += 1.5
        elif stoch_k_v < 20:                       score += 0.5
        elif stoch_k_v > 90 and stoch_d_v > 85:    score -= 1.5
        elif stoch_k_v > 80:                       score -= 0.5

        if i > 0 and stoch_k[i-1] <= stoch_d[i-1] and stoch_k[i] > stoch_d[i]:
            score += 0.8
        elif i > 0 and stoch_k[i-1] >= stoch_d[i-1] and stoch_k[i] < stoch_d[i]:
            score -= 0.8

        mfi_v = mfi14[i]
        if mfi_v < 15:       score += 1.5
        elif mfi_v < 25:     score += 0.5
        elif mfi_v > 85:     score -= 1.5
        elif mfi_v > 75:     score -= 0.5

        cci_v = cci14[i]
        if cci_v < -200:     score += 1.5
        elif cci_v < -100:   score += 0.7
        elif cci_v > 200:    score -= 1.5
        elif cci_v > 100:    score -= 0.7

        wr_v = wr14[i]
        if wr_v < -90:       score += 1.2
        elif wr_v < -80:     score += 0.5
        elif wr_v > -10:     score -= 1.2
        elif wr_v > -20:     score -= 0.5

        sar_v = sar[i]
        if price > sar_v:    score += 0.5
        else:                score -= 0.5

        aroon_up_v = aroon_up[i]
        aroon_down_v = aroon_down[i]
        if aroon_up_v > 70 and aroon_down_v < 30:    score += 0.5
        elif aroon_down_v > 70 and aroon_up_v < 30:  score -= 0.5

        if price > ma20[i] and ma5[i] > ma10[i]:      score += 0.8
        elif price < ma20[i] and ma5[i] < ma10[i]:    score -= 0.8

        if ema9[i] > ema21[i]:   score += 0.3
        else:                     score -= 0.3

        if kg[i]:      score += 0.8
        elif kd[i]:    score -= 0.8
        if j[i] < 0:   score += 0.5
        elif j[i] > 100: score -= 0.5

        if bb_up[i] > bb_low[i]:
            bb_pos = (price - bb_low[i]) / (bb_up[i] - bb_low[i])
            if bb_pos < 0.08:       score += 1.2
            elif bb_pos < 0.2:      score += 0.5
            elif bb_pos > 0.92:     score -= 1.2
            elif bb_pos > 0.8:      score -= 0.5

        if vol_spike[i]:
            if price_up:   score += 0.5
            else:          score -= 0.5

        if patterns['hammer'][i]:           score += 1.0
        elif patterns['shooting_star'][i]:  score -= 1.0
        if patterns['bullish_engulfing'][i]:    score += 1.5
        elif patterns['bearish_engulfing'][i]:  score -= 1.5

        # Regime adjustments
        if regime == 'trending':
            if di_diff > 5:     score += 0.8
            elif di_diff < -5:  score -= 0.8
            if di_diff > 3 and rsi7_v < 50:    score += 0.3
            elif di_diff < -3 and rsi7_v > 50:  score -= 0.3
        elif regime == 'ranging':
            bb_pos_val = (price - bb_low[i]) / (bb_up[i] - bb_low[i]) if bb_up[i] > bb_low[i] else 0.5
            if bb_pos_val < 0.15:    score += 0.5
            elif bb_pos_val > 0.85:  score -= 0.5

        # --- Determine direction ---
        abs_score = abs(score)
        if abs_score < 1.0:
            continue

        direction = 'up' if score >= 0 else 'down'

        # Simulate execution (10m: entry at next candle open, settle after 2 candles)
        entry_idx = i + 1
        settle_idx = min(i + 2, total - 1)
        entry_price = candles[entry_idx][1]  # open
        settle_price = candles[settle_idx][4]  # close

        if direction == 'up':
            win = settle_price > entry_price
        else:
            win = settle_price < entry_price

        pnl = 20 if win else -25  # 80% payout

        # Bucket
        for lo, hi, label in buckets:
            if lo <= abs_score < hi:
                bucket_data[label]['trades'].append({
                    'score': score, 'direction': direction,
                    'win': win, 'pnl': pnl,
                    'entry': entry_price, 'settle': settle_price,
                    'dt': dt,
                })
                if win:
                    bucket_data[label]['wins'] += 1
                else:
                    bucket_data[label]['losses'] += 1
                bucket_data[label]['pnl'] += pnl
                break

    # --- Print results ---
    print(f'  {"Score":<12} {"Trades":>7} {"Win":>5} {"Loss":>5} {"WR":>8} {"PnL":>10} {"Edge":>8}')
    print(f'  {"─"*60}')
    total_trades = 0
    total_wins = 0
    total_pnl = 0.0
    for _, _, label in buckets:
        bd = bucket_data[label]
        n = bd['wins'] + bd['losses']
        total_trades += n
        total_wins += bd['wins']
        total_pnl += bd['pnl']
        if n == 0:
            print(f'  {label:<12} {"—":>7} {"—":>5} {"—":>5} {"—":>8} {"—":>10} {"—":>8}')
            continue
        wr = bd['wins'] / n * 100
        edge = wr - 55.6  # Breakeven for 80% payout
        bar = '█' * int(wr / 5)
        print(f'  {label:<12} {n:>7} {bd["wins"]:>5} {bd["losses"]:>5} '
              f'{wr:>7.1f}% {bd["pnl"]:>+10.1f} {edge:>+7.1f}%  {bar}')

    print(f'  {"─"*60}')
    overall_wr = total_wins / total_trades * 100 if total_trades else 0
    print(f'  {"TOTAL":<12} {total_trades:>7} {total_wins:>5} '
          f'{total_trades-total_wins:>5} {overall_wr:>7.1f}% {total_pnl:>+10.1f}')

    # Also show direction breakdown
    print(f'\n  --- Direction Breakdown ---')
    print(f'  {"Score":<12} {"UP Win%":>10} {"UP Trades":>10} {"DOWN Win%":>12} {"DOWN Trades":>12}')
    for _, _, label in buckets:
        bd = bucket_data[label]
        ups = [t for t in bd['trades'] if t['direction'] == 'up']
        downs = [t for t in bd['trades'] if t['direction'] == 'down']
        up_wr = sum(1 for t in ups if t['win']) / len(ups) * 100 if ups else 0
        dn_wr = sum(1 for t in downs if t['win']) / len(downs) * 100 if downs else 0
        print(f'  {label:<12} {up_wr:>9.1f}% {len(ups):>10} {dn_wr:>11.1f}% {len(downs):>12}')

    print()
    return bucket_data


if __name__ == '__main__':
    # Quick aggregate backtest
    analyze('ETHUSDT', '2026-01-01', '2026-07-01')

    # Detailed score-bucket analysis
    score_bucket_analysis('ETHUSDT', '2026-01-01', '2026-07-01')
