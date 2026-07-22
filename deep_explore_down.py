"""
Deep exploration: DOWN direction with agree>=2, test stacked filters
to push win rate past 60%.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from collections import defaultdict
from event_backtest.data_fetcher import load_all
from event_backtest.indicators import (
    sma, ema, rsi, kdj, kdj_golden_cross, kdj_death_cross,
    bollinger_bands, adx, atr, atr_pct, bb_width, volume_spike,
    cci, williams_r, stochastic_rsi, aroon, aroon_osc, mfi, parabolic_sar,
    detect_candle_patterns,
)


SYMBOL = 'ETHUSDT'
START, END = '2026-01-01', '2026-07-01'

data = load_all(SYMBOL, START, END)
candles = data['5m']
closes = [c[4] for c in candles]; opens = [c[1] for c in candles]
highs = [c[2] for c in candles]; lows = [c[3] for c in candles]
volumes = [c[5] for c in candles]
c1h = [c[4] for c in data['1h']]; h1h = [c[2] for c in data['1h']]
l1h = [c[3] for c in data['1h']]; t5 = [c[0] for c in candles]
t1h = [c[0] for c in data['1h']]; total = len(candles); warmup = 60

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
        if timestamps[i] <= target_ts: return i
    return -1

def score_candle(i):
    """Full V3 scoring for candle i."""
    if atr_pct_5m[i] < 0.06: return None
    idx_1h = tf_idx(t1h, t5[i])
    if idx_1h < 20: return None

    adx_v = adx_1h[idx_1h]; pdi = pdi_1h[idx_1h]; mdi = mdi_1h[idx_1h]
    di_diff = pdi - mdi; price = closes[i]; price_up = closes[i] > opens[i]

    if adx_v > 25 or abs(aroon_osc_vals[i]) > 50:
        regime = 'trending'
    elif adx_v < 18 or (bbw[i] < 1.5 and abs(aroon_osc_vals[i]) < 25):
        regime = 'ranging'
    else:
        regime = 'neutral'

    score = 0.0; rsi7_v = rsi7[i]
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
    if i > 0 and stoch_k[i-1] <= stoch_d[i-1] and stoch_k[i] > stoch_d[i]: score += 0.8
    elif i > 0 and stoch_k[i-1] >= stoch_d[i-1] and stoch_k[i] < stoch_d[i]: score -= 0.8

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

    if aroon_up[i] > 70 and aroon_down[i] < 30: score += 0.5
    elif aroon_down[i] > 70 and aroon_up[i] < 30: score -= 0.5

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

    direction = 'up' if score >= 0 else 'down'
    abs_score = abs(score)

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

    if bb_up[i] > bb_low[i]:
        bb_pos = (price - bb_low[i]) / (bb_up[i] - bb_low[i])
    else:
        bb_pos = 0.5

    return (abs_score, direction, regime, {
        'adx': adx_v, 'di_diff': di_diff, 'rsi': rsi7_v,
        'mfi': mfi_v, 'stoch_k': stoch_k_v, 'cci': cci_v,
        'atr_pct': atr_pct_5m[i], 'agree': agree,
        'hour': datetime.fromtimestamp(t5[i] / 1000).hour,
        'sar_v': sar_v, 'bb_pos': bb_pos,
        'wr': wr_v, 'ma20': ma20[i], 'price': closes[i],
        'vol_spike_v': vol_spike[i],
    })


def simulate(filters):
    """Run simulation with given filters."""
    wins = losses = 0
    pnl = 0.0; last_sig = -999; day_bets = 0; cur_day = None

    th = filters.get('th', 4.0)
    min_agree = filters.get('min_agree', 1)
    min_atr = filters.get('min_atr', 0.06)
    max_atr = filters.get('max_atr', 999)
    min_rsi = filters.get('min_rsi', 0)
    max_rsi = filters.get('max_rsi', 100)
    min_mfi = filters.get('min_mfi', 0)
    max_mfi = filters.get('max_mfi', 100)
    min_cci = filters.get('min_cci', -9999)
    max_cci = filters.get('max_cci', 9999)
    bb_range = filters.get('bb_range', None)  # (min, max) for BB position
    require_vol_spike = filters.get('require_vol_spike', False)
    exclude_hours = filters.get('exclude_hours', set())

    for i in range(warmup, total - 2):
        ts = candles[i][0]; dt = datetime.fromtimestamp(ts / 1000)

        if dt.day != cur_day: cur_day = dt.day; day_bets = 0
        if i - last_sig < 2: continue
        if day_bets >= 200: continue

        result = score_candle(i)
        if result is None: continue

        abs_score, direction, regime, attrs = result
        if direction != 'down': continue  # DOWN ONLY
        if abs_score < th: continue
        if attrs['agree'] < min_agree: continue
        if attrs['atr_pct'] < min_atr or attrs['atr_pct'] > max_atr: continue
        if attrs['rsi'] < min_rsi or attrs['rsi'] > max_rsi: continue
        if attrs['mfi'] < min_mfi or attrs['mfi'] > max_mfi: continue
        if attrs['cci'] < min_cci or attrs['cci'] > max_cci: continue
        if dt.hour in exclude_hours: continue
        if require_vol_spike and not attrs['vol_spike_v']: continue
        if bb_range is not None:
            if attrs['bb_pos'] < bb_range[0] or attrs['bb_pos'] > bb_range[1]:
                continue

        # 10m execute
        entry = candles[i + 1][1]
        settle_idx = min(i + 2, total - 1)
        settle = candles[settle_idx][4]
        win = settle < entry  # DOWN direction
        if win: wins += 1; pnl += 20
        else: losses += 1; pnl -= 25
        last_sig = i; day_bets += 1

    n = wins + losses
    wr = wins / n * 100 if n else 0
    return {'trades': n, 'wins': wins, 'losses': losses, 'wr': wr, 'pnl': pnl}


def analyze_win_vs_loss():
    """Compare indicator values between winning and losing DOWN signals."""
    print(f'\n{"="*80}')
    print(f'  WIN vs LOSS Analysis: What makes a DOWN signal succeed?')
    print(f'  (th=4.0, agree>=1, DOWN only)')
    print(f'{"="*80}\n')

    wins_data = defaultdict(list)
    loss_data = defaultdict(list)

    for i in range(warmup, total - 2):
        result = score_candle(i)
        if result is None: continue
        abs_score, direction, regime, attrs = result
        if direction != 'down': continue
        if abs_score < 4.0: continue
        if attrs['agree'] < 1: continue

        entry = candles[i + 1][1]
        settle_idx = min(i + 2, total - 1)
        settle = candles[settle_idx][4]
        win = settle < entry

        bucket = wins_data if win else loss_data
        for k, v in attrs.items():
            bucket[k].append(v)
        bucket['score'].append(abs_score)

    keys = ['score', 'rsi', 'mfi', 'stoch_k', 'cci', 'atr_pct', 'adx',
            'bb_pos', 'wr', 'agree', 'di_diff']

    print(f'  {"Metric":<12} {"WIN avg":>9} {"LOSS avg":>9} {"Diff":>8} {"Insight"}')
    print(f'  {"─"*65}')
    for k in keys:
        wv = wins_data[k]
        lv = loss_data[k]
        if not wv or not lv: continue
        w_avg = sum(wv) / len(wv)
        l_avg = sum(lv) / len(lv)
        diff = w_avg - l_avg
        insight = ''
        if abs(diff) / max(w_avg, l_avg, 0.01) > 0.1:
            if k == 'bb_pos' and w_avg > l_avg:
                insight = '<- WINs have higher BB position (nearer top)'
            elif k == 'bb_pos' and w_avg < l_avg:
                insight = '<- WINs at lower BB (nearer bottom)'
            elif k == 'rsi' and w_avg > l_avg:
                insight = '<- WINs have higher RSI'
            elif k == 'rsi' and w_avg < l_avg:
                insight = '<- WINs have lower RSI'
            elif k == 'mfi' and w_avg > l_avg:
                insight = '<- WINs have higher MFI'
            elif k == 'stoch_k' and w_avg > l_avg:
                insight = '<- WINs have higher StochK'
            elif k == 'cci' and w_avg > l_avg:
                insight = '<- WINs have higher CCI'
            elif k == 'atr_pct' and w_avg < l_avg:
                insight = '<- WINs have lower ATR'
            elif k == 'adx' and w_avg < l_avg:
                insight = '<- WINs have lower ADX (less trending)'

        print(f'  {k:<12} {w_avg:>9.2f} {l_avg:>9.2f} {diff:>+8.2f} {insight}')

    print(f'\n  WIN signals: {len(wins_data["score"])}, LOSS signals: {len(loss_data["score"])}')


def main():
    # --- Part 1: Win vs Loss comparison ---
    analyze_win_vs_loss()

    # --- Part 2: Stacked filter tests on DOWN only ---
    print(f'\n\n{"="*80}')
    print(f'  Stacked Filter Tests: DOWN only, agree>=2')
    print(f'{"="*80}')
    print(f'  {"Config":<45} {"Trades":>7} {"WR":>7} {"PnL":>9} {"Edge":>7}')
    print(f'  {"─"*75}')

    BASE = {'th': 4.0, 'min_agree': 2}

    configs = [
        ('Baseline: th=4.0 agree>=2 DOWN', dict(BASE)),
        ('+ ATR 0.08-0.20', dict(BASE, min_atr=0.08, max_atr=0.20)),
        ('+ ATR 0.08-0.15', dict(BASE, min_atr=0.08, max_atr=0.15)),
        ('+ RSI >= 70', dict(BASE, min_rsi=70)),
        ('+ RSI >= 75', dict(BASE, min_rsi=75)),
        ('+ RSI >= 80', dict(BASE, min_rsi=80)),
        ('+ MFI >= 80', dict(BASE, min_mfi=80)),
        ('+ CCI >= 150', dict(BASE, min_cci=150)),
        ('+ BB position >= 0.8', dict(BASE, bb_range=(0.8, 99))),
        ('+ BB position >= 0.9', dict(BASE, bb_range=(0.9, 99))),
        # Combinations
        ('+ RSI>=70 + ATR 0.08-0.20', dict(BASE, min_rsi=70, min_atr=0.08, max_atr=0.20)),
        ('+ RSI>=70 + BB>=0.8', dict(BASE, min_rsi=70, bb_range=(0.8, 99))),
        ('+ RSI>=75 + ATR 0.08-0.20', dict(BASE, min_rsi=75, min_atr=0.08, max_atr=0.20)),
        ('+ RSI>=70 + ATR 0.08-0.15', dict(BASE, min_rsi=70, min_atr=0.08, max_atr=0.15)),
        ('+ RSI>=75 + BB>=0.8 + ATR 0.08-0.20', dict(BASE, min_rsi=75, bb_range=(0.8, 99), min_atr=0.08, max_atr=0.20)),
        # Higher threshold variants
        ('th=4.5 agree>=2 + RSI>=70 + ATR 0.08-0.20', dict(BASE, th=4.5, min_rsi=70, min_atr=0.08, max_atr=0.20)),
        ('th=4.5 agree>=2 + RSI>=75 + BB>=0.8', dict(BASE, th=4.5, min_rsi=75, bb_range=(0.8, 99))),
        ('th=5.0 agree>=2 + RSI>=70', dict(BASE, th=5.0, min_rsi=70)),
        # CCI + ATR combo
        ('+ CCI>=150 + MFI>=80', dict(BASE, min_cci=150, min_mfi=80)),
        ('+ CCI>=150 + ATR 0.08-0.20', dict(BASE, min_cci=150, min_atr=0.08, max_atr=0.20)),
        # Time filter combo
        ('+ Excl 22-05 UTC + RSI>=70', dict(BASE, min_rsi=70, exclude_hours={22,23,0,1,2,3,4,5})),
    ]

    results = []
    for label, filters in configs:
        r = simulate(filters)
        edge = r['wr'] - 55.6
        mark = '**' if r['wr'] >= 60.5 else ('*' if r['wr'] >= 60 else ('+' if r['wr'] >= 59 else ''))
        results.append((label, r, mark))
        print(f'  {mark} {label:<43} {r["trades"]:>7} {r["wr"]:>6.1f}% {r["pnl"]:>+9.1f} {edge:>+6.1f}%')

    # Best picks
    print(f'\n  --- Top Picks (WR >= 60% or close) ---')
    for label, r, mark in results:
        if r['wr'] >= 59.5 and r['trades'] >= 100:
            print(f'  {mark} {label}: {r["trades"]} trades, WR={r["wr"]:.1f}%, PnL=${r["pnl"]:+.0f}')

    print()


if __name__ == '__main__':
    main()
