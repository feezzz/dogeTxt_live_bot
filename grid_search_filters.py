"""
Grid search: find filter combinations that improve win rate above baseline (58.3%).
Tests: threshold, agreement level, regime, direction, ATR, ADX, RSI, time filters.
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
BASELINE_TH = 4.0

# Load data once
data = load_all(SYMBOL, START, END)
candles = data['5m']
closes = [c[4] for c in candles]; opens = [c[1] for c in candles]
highs = [c[2] for c in candles]; lows = [c[3] for c in candles]
volumes = [c[5] for c in candles]
c1h = [c[4] for c in data['1h']]
h1h = [c[2] for c in data['1h']]; l1h = [c[3] for c in data['1h']]
t5 = [c[0] for c in candles]; t1h = [c[0] for c in data['1h']]
total = len(candles); warmup = 60

# All indicators (computed once)
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
    """Return (score, direction, regime, attrs_dict) for candle i. None if below ATR."""
    if atr_pct_5m[i] < 0.06: return None

    idx_1h = tf_idx(t1h, t5[i])
    if idx_1h < 20: return None

    adx_v = adx_1h[idx_1h]; pdi = pdi_1h[idx_1h]; mdi = mdi_1h[idx_1h]
    di_diff = pdi - mdi; price = closes[i]; price_up = closes[i] > opens[i]

    # Regime
    if adx_v > 25 or abs(aroon_osc_vals[i]) > 50:
        regime = 'trending'
    elif adx_v < 18 or (bbw[i] < 1.5 and abs(aroon_osc_vals[i]) < 25):
        regime = 'ranging'
    else:
        regime = 'neutral'

    # Full V3 scoring
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

    # Agreement count
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

    return (abs_score, direction, regime, {
        'adx': adx_v, 'di_diff': di_diff, 'rsi': rsi7_v,
        'mfi': mfi_v, 'stoch_k': stoch_k_v, 'cci': cci_v,
        'atr_pct': atr_pct_5m[i], 'agree': agree,
        'hour': datetime.fromtimestamp(t5[i] / 1000).hour,
        'sar_v': sar_v, 'bbw_v': bbw[i],
    })


def simulate(filters, name):
    """Run simulation with given filters. Return (wins, losses, total, wr%, pnl)."""
    wins = losses = 0
    last_sig = -999
    day_bets = 0; cur_day = None
    pnl = 0.0

    th = filters.get('th', 4.0)
    min_agree = filters.get('min_agree', 1)
    regimes = filters.get('regimes', None)       # None=all, or {'trending','ranging','neutral'}
    directions = filters.get('directions', None)  # None=both, or {'up','down'}
    min_atr = filters.get('min_atr', 0.06)
    max_atr = filters.get('max_atr', 999)
    min_adx = filters.get('min_adx', 0)
    max_adx = filters.get('max_adx', 999)
    min_rsi_range = filters.get('min_rsi_range', None)  # tuple (min_rsi_for_up, max_rsi_for_down)
    exclude_hours = filters.get('exclude_hours', set())
    require_bb_extreme = filters.get('require_bb_extreme', False)
    require_cci_confirm = filters.get('require_cci_confirm', False)
    cooldown = filters.get('cooldown', 2)
    max_daily = filters.get('max_daily', 200)

    for i in range(warmup, total - 2):
        ts = candles[i][0]; dt = datetime.fromtimestamp(ts / 1000)

        if dt.day != cur_day:
            cur_day = dt.day; day_bets = 0
        if i - last_sig < cooldown: continue
        if day_bets >= max_daily: continue

        result = score_candle(i)
        if result is None: continue
        abs_score, direction, regime, attrs = result

        if abs_score < th: continue
        if attrs['agree'] < min_agree: continue
        if regimes is not None and regime not in regimes: continue
        if directions is not None and direction not in directions: continue
        if attrs['atr_pct'] < min_atr or attrs['atr_pct'] > max_atr: continue
        if attrs['adx'] < min_adx or attrs['adx'] > max_adx: continue
        if dt.hour in exclude_hours: continue

        if require_bb_extreme:
            price = closes[i]
            if bb_up[i] > bb_low[i]:
                bb_pos = (price - bb_low[i]) / (bb_up[i] - bb_low[i])
                if direction == 'up' and bb_pos > 0.3: continue
                if direction == 'down' and bb_pos < 0.7: continue

        if require_cci_confirm:
            if direction == 'up' and attrs['cci'] > -50: continue
            if direction == 'down' and attrs['cci'] < 50: continue

        # Execute (10m, 80% payout)
        entry = candles[i + 1][1]
        settle_idx = min(i + 2, total - 1)
        settle = candles[settle_idx][4]

        if direction == 'up': win = settle > entry
        else: win = settle < entry

        if win: wins += 1; pnl += 20
        else: losses += 1; pnl -= 25

        last_sig = i; day_bets += 1

    n = wins + losses
    wr = wins / n * 100 if n else 0
    return {'trades': n, 'wins': wins, 'losses': losses, 'wr': wr, 'pnl': pnl}


def main():
    # Pre-score all candles
    print('Scoring all candles...')
    all_signals = []
    for i in range(warmup, total - 2):
        result = score_candle(i)
        if result is None: continue
        abs_score, direction, regime, attrs = result
        if abs_score < 1.0: continue
        all_signals.append((i, abs_score, direction, regime, attrs))
    print(f'Total scorable candles: {len(all_signals)}')

    # ========================================================
    # Test 1: Threshold + Agreement combinations
    # ========================================================
    print(f'\n{"="*90}')
    print(f'  Test 1: Score Threshold × Indicator Agreement')
    print(f'{"="*90}')
    print(f'  {"Config":<30} {"Trades":>7} {"WR":>7} {"PnL":>10} {"Edge":>7}')
    print(f'  {"─"*65}')

    results = []
    for th in [3.5, 4.0, 4.5, 5.0, 5.5]:
        for agree in [1, 2, 3]:
            r = simulate({'th': th, 'min_agree': agree, 'max_daily': 999},
                         f'th={th},agree={agree}')
            label = f'th={th:.1f} agree≥{agree}'
            edge = r['wr'] - 55.6
            results.append((label, r))
            mark = '*' if r['wr'] >= 60 else ('+' if r['wr'] >= 58.3 else '')
            print(f'  {mark} {label:<28} {r["trades"]:>7} {r["wr"]:>6.1f}% {r["pnl"]:>+10.1f} {edge:>+6.1f}%')

    # ========================================================
    # Test 2: Regime filter (baseline th=4.0, agree=1)
    # ========================================================
    print(f'\n{"="*90}')
    print(f'  Test 2: Regime Filter (th=4.0, agree≥1)')
    print(f'{"="*90}')
    print(f'  {"Config":<30} {"Trades":>7} {"WR":>7} {"PnL":>10} {"Edge":>7}')
    print(f'  {"─"*65}')

    for reg_label, regs in [('All', None), ('Trending only', {'trending'}),
                              ('Ranging only', {'ranging'}), ('No Neutral', {'trending','ranging'}),
                              ('Neutral only', {'neutral'})]:
        r = simulate({'th': 4.0, 'min_agree': 1, 'regimes': regs, 'max_daily': 999}, reg_label)
        edge = r['wr'] - 55.6
        mark = '*' if r['wr'] >= 60 else ''
        print(f'  {mark} {reg_label:<28} {r["trades"]:>7} {r["wr"]:>6.1f}% {r["pnl"]:>+10.1f} {edge:>+6.1f}%')

    # ========================================================
    # Test 3: Direction bias (th=4.0, agree=1)
    # ========================================================
    print(f'\n{"="*90}')
    print(f'  Test 3: Direction Bias (th=4.0, agree≥1)')
    print(f'{"="*90}')
    print(f'  {"Config":<30} {"Trades":>7} {"WR":>7} {"PnL":>10} {"Edge":>7}')
    print(f'  {"─"*65}')

    for dir_label, dirs in [('Both', None), ('UP only', {'up'}), ('DOWN only', {'down'})]:
        r = simulate({'th': 4.0, 'min_agree': 1, 'directions': dirs, 'max_daily': 999}, dir_label)
        edge = r['wr'] - 55.6
        mark = '*' if r['wr'] >= 60 else ''
        print(f'  {mark} {dir_label:<28} {r["trades"]:>7} {r["wr"]:>6.1f}% {r["pnl"]:>+10.1f} {edge:>+6.1f}%')

    # ========================================================
    # Test 4: ATR range filter (th=4.0, agree=1)
    # ========================================================
    print(f'\n{"="*90}')
    print(f'  Test 4: ATR Filter Range (th=4.0, agree≥1)')
    print(f'{"="*90}')
    print(f'  {"Config":<30} {"Trades":>7} {"WR":>7} {"PnL":>10} {"Edge":>7}')
    print(f'  {"─"*65}')

    for atr_label, (mn, mx) in [
        ('All', (0.06, 999)),
        ('ATR 0.08-0.20', (0.08, 0.20)),
        ('ATR 0.10-0.25', (0.10, 0.25)),
        ('ATR ≥ 0.10', (0.10, 999)),
        ('ATR ≥ 0.12', (0.12, 999)),
        ('ATR ≤ 0.15', (0.06, 0.15)),
    ]:
        r = simulate({'th': 4.0, 'min_agree': 1, 'min_atr': mn, 'max_atr': mx, 'max_daily': 999}, atr_label)
        edge = r['wr'] - 55.6
        mark = '*' if r['wr'] >= 60 else ''
        print(f'  {mark} {atr_label:<28} {r["trades"]:>7} {r["wr"]:>6.1f}% {r["pnl"]:>+10.1f} {edge:>+6.1f}%')

    # ========================================================
    # Test 5: Time filter (th=4.0, agree=1)
    # ========================================================
    print(f'\n{"="*90}')
    print(f'  Test 5: Time of Day Filter (th=4.0, agree≥1)')
    print(f'{"="*90}')
    print(f'  {"Config":<30} {"Trades":>7} {"WR":>7} {"PnL":>10} {"Edge":>7}')
    print(f'  {"─"*65}')

    for time_label, excluded in [
        ('No filter', set()),
        ('Excl 22-05 UTC', {22, 23, 0, 1, 2, 3, 4, 5}),
        ('Excl 0-8 BJ = 16-00 UTC', {16, 17, 18, 19, 20, 21, 22, 23, 0}),
        ('Only 8-16 UTC (BJ 16-24)', set(range(0, 8)) | set(range(16, 24))),
        ('Only 12-20 UTC (BJ 20-04)', set(range(0, 12)) | set(range(20, 24))),
    ]:
        r = simulate({'th': 4.0, 'min_agree': 1, 'exclude_hours': excluded, 'max_daily': 999}, time_label)
        edge = r['wr'] - 55.6
        mark = '*' if r['wr'] >= 60 else ''
        print(f'  {mark} {time_label:<28} {r["trades"]:>7} {r["wr"]:>6.1f}% {r["pnl"]:>+10.1f} {edge:>+6.1f}%')

    # ========================================================
    # Test 6: Best combinations
    # ========================================================
    print(f'\n{"="*90}')
    print(f'  Test 6: Best Filter Combinations')
    print(f'{"="*90}')
    print(f'  {"Config":<40} {"Trades":>7} {"WR":>7} {"PnL":>10} {"Edge":>7}')
    print(f'  {"─"*70}')

    combos = [
        # (label, filters_dict)
        ('Baseline: th=4.0 agree≥1', {'th': 4.0, 'min_agree': 1}),
        ('th=4.0 agree≥2', {'th': 4.0, 'min_agree': 2}),
        ('th=4.0 agree≥2 + DOWN only', {'th': 4.0, 'min_agree': 2, 'directions': {'down'}}),
        ('th=4.5 agree≥2', {'th': 4.5, 'min_agree': 2}),
        ('th=4.5 agree≥2 + DOWN only', {'th': 4.5, 'min_agree': 2, 'directions': {'down'}}),
        ('th=5.0 agree≥1', {'th': 5.0, 'min_agree': 1}),
        ('th=5.0 agree≥2', {'th': 5.0, 'min_agree': 2}),
        ('th=4.0 agree≥2 ATR≥0.10', {'th': 4.0, 'min_agree': 2, 'min_atr': 0.10}),
        ('th=4.5 agree≥2 ATR≥0.10 DOWN', {'th': 4.5, 'min_agree': 2, 'min_atr': 0.10, 'directions': {'down'}}),
        ('th=5.0 agree≥2 No Neutral', {'th': 5.0, 'min_agree': 2, 'regimes': {'trending', 'ranging'}}),
        ('th=4.0 agree≥2 + CCI confirm', {'th': 4.0, 'min_agree': 2, 'require_cci_confirm': True}),
        ('th=4.5 agree≥2 + BB extreme', {'th': 4.5, 'min_agree': 2, 'require_bb_extreme': True}),
        ('th=4.0 agree≥2 + BB extreme', {'th': 4.0, 'min_agree': 2, 'require_bb_extreme': True}),
        ('th=4.5 agree≥3', {'th': 4.5, 'min_agree': 3}),
    ]

    for label, filters in combos:
        filters['max_daily'] = 999
        r = simulate(filters, label)
        edge = r['wr'] - 55.6
        mark = '**' if r['wr'] >= 60.5 else ('*' if r['wr'] >= 59 else '')
        print(f'  {mark} {label:<38} {r["trades"]:>7} {r["wr"]:>6.1f}% {r["pnl"]:>+10.1f} {edge:>+6.1f}%')

    print()


if __name__ == '__main__':
    main()
