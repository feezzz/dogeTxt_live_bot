"""
Deep loss analysis: three angles to push win rate past 60.6%.

Angle 1: 1-candle checkpoint — were losing trades winning at 5min?
Angle 2: 15m+5m confluence — does higher-TF agreement matter?
Angle 3: Loss pattern mining — what specific indicator combos fail?
Angle 4: Trend strength filter — counter-trend in strong trends fails?
Angle 5: Momentum/mean-reversion conflict — when indicators disagree
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from collections import defaultdict, Counter
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
candles5 = data['5m']; candles15 = data['15m']; candles1h = data['1h']

def ohlc(candles):
    return ([c[4] for c in candles], [c[1] for c in candles],
            [c[2] for c in candles], [c[3] for c in candles], [c[5] for c in candles])

closes, opens, highs, lows, volumes = ohlc(candles5)
closes15, _, highs15, lows15, _ = ohlc(candles15)
closes1h, _, highs1h, lows1h, _ = ohlc(candles1h)
t5 = [c[0] for c in candles5]; t15 = [c[0] for c in candles15]
t1h = [c[0] for c in candles1h]
total = len(candles5); warmup = 60

# ----- All indicators (5m) -----
rsi7 = rsi(closes, 7)
ma5 = sma(closes, 5); ma10 = sma(closes, 10); ma20 = sma(closes, 20)
k, d, j = kdj(highs, lows, closes, period=6, k_period=3, d_period=3)
kg = kdj_golden_cross(k, d); kd = kdj_death_cross(k, d)
bb_mid, bb_up, bb_low = bollinger_bands(closes, period=20, std_mult=2.0)
bbw = bb_width(bb_up, bb_low, bb_mid)
vol_spike = volume_spike(volumes, period=20, threshold=1.5)
adx_1h, pdi_1h, mdi_1h = adx(highs1h, lows1h, closes1h, period=14)
atr_pct_vals = atr_pct(atr(highs, lows, closes, 14), closes)
cci14 = cci(highs, lows, closes, period=14)
wr14 = williams_r(highs, lows, closes, period=14)
stoch_k, stoch_d = stochastic_rsi(closes, period=14, stoch_period=14)
aroon_up, aroon_down = aroon(highs, lows, period=14)
aroon_osc_vals = aroon_osc(aroon_up, aroon_down)
mfi14 = mfi(highs, lows, closes, volumes, period=14)
sar_vals = parabolic_sar(highs, lows)
patterns = detect_candle_patterns(opens, highs, lows, closes)
ema9 = ema(closes, 9); ema21 = ema(closes, 21)

# ----- 15m indicators (for confluence check) -----
rsi7_15m = rsi(closes15, 7)
stoch_k15, stoch_d15 = stochastic_rsi(closes15, period=14, stoch_period=14)

# ----- ATR trend (is volatility expanding or contracting?) -----
atr_vals = atr(highs, lows, closes, 14)
atr_rising = [False] * len(closes)
for i in range(15, len(closes)):
    atr_rising[i] = atr_vals[i] > atr_vals[i-5]  # ATR rising vs 5 candles ago

def tf_idx(ts_list, target_ts):
    for i in range(len(ts_list)-1, -1, -1):
        if ts_list[i] <= target_ts: return i
    return -1

def v3_score_and_attrs(i):
    """Return (abs_score, direction, regime, attrs_dict) or None."""
    if atr_pct_vals[i] < 0.06: return None
    idx_1h = tf_idx(t1h, t5[i])
    idx_15 = tf_idx(t15, t5[i])
    if idx_1h < 20 or idx_15 < 10: return None

    adx_v = adx_1h[idx_1h]; pdi = pdi_1h[idx_1h]; mdi = mdi_1h[idx_1h]
    di_diff = pdi - mdi; price = closes[i]; price_up = closes[i] > opens[i]

    # 15m confluence
    rsi15 = rsi7_15m[idx_15] if idx_15 >= 0 else 50
    stoch15 = stoch_k15[idx_15] if idx_15 >= 0 else 50

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

    sar_v = sar_vals[i]
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
    else:
        bb_pos = 0.5

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

    # Agreement
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

    # 15m confluence: does 15m RSI agree with 5m direction?
    tf_agree = 0
    if direction == 'up' and rsi15 < 45: tf_agree = 1   # 15m also oversold
    if direction == 'up' and rsi15 < 35: tf_agree = 2
    if direction == 'down' and rsi15 > 55: tf_agree = 1  # 15m also overbought
    if direction == 'down' and rsi15 > 65: tf_agree = 2

    # Momentum vs mean reversion conflict
    mom_score = 0  # positive = momentum aligned with direction
    if direction == 'up':
        if price > ma20[i]: mom_score += 1
        if ma5[i] > ma10[i]: mom_score += 1
        if ema9[i] > ema21[i]: mom_score += 1
        if aroon_osc_vals[i] > 0: mom_score += 1
    else:
        if price < ma20[i]: mom_score += 1
        if ma5[i] < ma10[i]: mom_score += 1
        if ema9[i] < ema21[i]: mom_score += 1
        if aroon_osc_vals[i] < 0: mom_score += 1

    return (abs_score, direction, regime, {
        'rsi': rsi7_v, 'mfi': mfi_v, 'stoch_k': stoch_k_v, 'cci': cci_v,
        'atr_pct': atr_pct_vals[i], 'agree': agree, 'adx': adx_v,
        'di_diff': di_diff, 'bb_pos': bb_pos, 'wr': wr_v,
        'hour': datetime.fromtimestamp(t5[i]/1000).hour,
        'rsi15': rsi15, 'tf_agree': tf_agree,
        'mom_score': mom_score,  # 0-4: how many momentum indicators agree
        'atr_rising': atr_rising[i],
        'sar_v': sar_v, 'price': price,
        'vol_spike_v': vol_spike[i],
    })


def main():
    print('Collecting signals with BB filter applied...')

    # Collect all signals with production filters (th=4.0, agree>=1, BB extreme)
    records = []  # (i, abs_score, direction, regime, attrs, win_1c, win_2c)
    for i in range(warmup, total - 3):
        result = v3_score_and_attrs(i)
        if result is None: continue
        abs_score, direction, regime, attrs = result
        if abs_score < 4.0: continue
        if attrs['agree'] < 1: continue

        # BB extreme filter
        if bb_up[i] > bb_low[i]:
            bb_pos = attrs['bb_pos']
            if direction == 'up' and bb_pos > 0.10: continue
            if direction == 'down' and bb_pos < 0.90: continue

        # Check outcomes: entry at i+1 open, settle at i+2 close (=10min contract)
        entry = candles5[i+1][1]
        settle_10m = candles5[i+2][4]  # 10min from signal = i+2 close
        settle_5m = candles5[i+1][4]   # entry candle close = ~5min

        if direction == 'up':
            win10 = settle_10m > entry
            win5 = settle_5m > entry
        else:
            win10 = settle_10m < entry
            win5 = settle_5m < entry

        records.append((i, abs_score, direction, regime, attrs, win5, win10))

    print(f'Total signals: {len(records)}')
    wins = [r for r in records if r[6]]   # win10
    losses = [r for r in records if not r[6]]
    wr = len(wins) / len(records) * 100
    print(f'Win rate: {len(wins)}/{len(losses)} = {wr:.1f}%')

    # ================================================================
    # Angle 1: 1-candle checkpoint
    # ================================================================
    print(f'\n{"="*80}')
    print(f'  ANGLE 1: 1-Candle (5min) Checkpoint Analysis')
    print(f'{"="*80}')

    # Among losses, how many were winning at 5min checkpoint?
    loss_was_winning = sum(1 for r in records if not r[6] and r[5])
    print(f'\n  Losses that were WINNING at entry close (~5min): {loss_was_winning}/{len(losses)} '
          f'({loss_was_winning/len(losses)*100:.1f}%)')

    # Among wins, how many were losing at 5min checkpoint?
    win_was_losing = sum(1 for r in records if r[6] and not r[5])
    print(f'  Wins that were LOSING at entry close (~5min):  {win_was_losing}/{len(wins)} '
          f'({win_was_losing/len(wins)*100:.1f}%)')

    # Hypothetical: if we could take profit at entry-candle close when ahead
    hy_wins = sum(1 for r in records if r[5] or r[6])
    hy_wr = hy_wins / len(records) * 100
    print(f'\n  Hypothetical: "exit at entry-close if winning, else hold 10m":')
    print(f'  WR = {hy_wins}/{len(records)} = {hy_wr:.1f}%')

    # Direction breakdown of 1c reversals
    for d in ['up', 'down']:
        d_recs = [r for r in records if r[2] == d]
        d_wins = [r for r in d_recs if r[6]]
        d_loss = [r for r in d_recs if not r[6]]
        d_rev = [r for r in d_loss if r[5]]  # losses that were winning
        if d_loss:
            print(f'  {d}: {len(d_rev)}/{len(d_loss)} losses were winning at 5min '
                  f'({len(d_rev)/len(d_loss)*100:.1f}%), '
                  f'wins: {len(d_wins)}')

    # ================================================================
    # Angle 2: 15m Confluence
    # ================================================================
    print(f'\n{"="*80}')
    print(f'  ANGLE 2: Multi-Timeframe Confluence (15m RSI agreement)')
    print(f'{"="*80}\n')

    for tf_label, tf_min in [('15m RSI agrees (tf_agree>=1)', 1),
                               ('15m RSI strong agree (tf_agree>=2)', 2)]:
        subset = [r for r in records if r[4]['tf_agree'] >= tf_min]
        sw = sum(1 for r in subset if r[6])
        sl = len(subset) - sw
        swr = sw / len(subset) * 100 if subset else 0
        print(f'  {tf_label:<40} {len(subset):>5} trades  WR={swr:.1f}%')

    # Without 15m agreement vs with
    subset = [r for r in records if r[4]['tf_agree'] == 0]
    sw = sum(1 for r in subset if r[6])
    swr = sw / len(subset) * 100 if subset else 0
    print(f'  {"No 15m agreement":<40} {len(subset):>5} trades  WR={swr:.1f}%')

    # ================================================================
    # Angle 3: Momentum alignment with direction
    # ================================================================
    print(f'\n{"="*80}')
    print(f'  ANGLE 3: Momentum vs Mean Reversion Conflict')
    print(f'  (mom_score 0-4: how many momentum indicators agree with direction)')
    print(f'{"="*80}\n')

    for ms in range(5):
        subset = [r for r in records if r[4]['mom_score'] == ms]
        sw = sum(1 for r in subset if r[6])
        sl = len(subset) - sw
        swr = sw / len(subset) * 100 if subset else 0
        bar = '#' * int(swr) if swr > 0 else ''
        print(f'  {"Momentum conflict" if ms <= 1 else "Neutral" if ms == 2 else "Momentum aligned"}'
              f' (mom={ms}): {len(subset):>5} trades  WR={swr:.1f}%  {bar}')

    # Counter-trend (mean reversion = opposite to momentum)
    conflict = [r for r in records if r[4]['mom_score'] <= 1]
    aligned = [r for r in records if r[4]['mom_score'] >= 3]
    cw = sum(1 for r in conflict if r[6])
    aw = sum(1 for r in aligned if r[6])
    print(f'\n  Counter-trend (mom 0-1): {len(conflict)} trades, WR={cw/len(conflict)*100:.1f}%')
    print(f'  Trend-aligned  (mom 3-4): {len(aligned)} trades, WR={aw/len(aligned)*100:.1f}% '
          f'(but these should be FEW — V3 is counter-trend!)')

    # ================================================================
    # Angle 4: Trend Strength vs Counter-Trend Success
    # ================================================================
    print(f'\n{"="*80}')
    print(f'  ANGLE 4: Counter-Trend Success by Trend Strength')
    print(f'{"="*80}\n')

    for adx_range, label in [((0, 25), 'Weak trend (ADX 0-25)'),
                               ((25, 35), 'Moderate trend (ADX 25-35)'),
                               ((35, 50), 'Strong trend (ADX 35-50)'),
                               ((50, 999), 'Very strong (ADX 50+)')]:
        subset = [r for r in records if adx_range[0] <= r[4]['adx'] < adx_range[1]]
        sw = sum(1 for r in subset if r[6])
        wr_v = sw/len(subset)*100 if subset else 0
        print(f'  {label:<35} {len(subset):>5} trades  WR={wr_v:.1f}%')

    # The key test: when we trade counter-trend in a strong trend
    strong_ct = [r for r in records if r[4]['adx'] > 35 and r[4]['mom_score'] <= 1]
    scw = sum(1 for r in strong_ct if r[6])
    print(f'\n  Counter-trend in STRONG trend (ADX>35 + mom<=1): '
          f'{len(strong_ct)} trades, WR={scw/len(strong_ct)*100:.1f}%' if strong_ct else '')

    # ================================================================
    # Angle 5: ATR Expansion vs Contraction
    # ================================================================
    print(f'\n{"="*80}')
    print(f'  ANGLE 5: Volatility Regime (ATR Rising vs Falling)')
    print(f'{"="*80}\n')

    for atr_label, atr_val in [('ATR falling (contracting)', False),
                                 ('ATR rising (expanding)', True)]:
        subset = [r for r in records if r[4]['atr_rising'] == atr_val]
        sw = sum(1 for r in subset if r[6])
        wr_v = sw/len(subset)*100 if subset else 0
        print(f'  {atr_label:<35} {len(subset):>5} trades  WR={wr_v:.1f}%')

    # ================================================================
    # Angle 6: Specific loss patterns — extreme combos
    # ================================================================
    print(f'\n{"="*80}')
    print(f'  ANGLE 6: Loss Pattern Mining — Top Loss Conditions')
    print(f'{"="*80}\n')

    # Score losses by specific condition combinations
    conditions = [
        ('ADX>35 & mom<=1 (strong counter-trend)', lambda r: r[4]['adx']>35 and r[4]['mom_score']<=1),
        ('ADX>40 & any direction', lambda r: r[4]['adx']>40),
        ('CCI extreme (>250)', lambda r: abs(r[4]['cci'])>250),
        ('ATR% > 0.4 (very high vol)', lambda r: r[4]['atr_pct']>0.4),
        ('ATR% < 0.08 (very low vol)', lambda r: r[4]['atr_pct']<0.08),
        ('No 15m agreement + low mom', lambda r: r[4]['tf_agree']==0 and r[4]['mom_score']<=1),
        ('RSI extreme but others weak', lambda r: abs(r[4]['rsi']-50)>25 and r[4]['agree']<=1),
        ('Momentum ALIGNED (mom>=3) counter-trend failure', lambda r: r[4]['mom_score']>=3),
        ('Volume spike present', lambda r: r[4]['vol_spike_v']),
        ('BB neutral (0.3-0.7) + low score', lambda r: 0.3<r[4]['bb_pos']<0.7 and r[1]<4.5),
    ]

    for label, cond in conditions:
        subset = [r for r in records if cond(r)]
        if len(subset) < 10: continue
        sw = sum(1 for r in subset if r[6])
        sl = len(subset) - sw
        wr_v = sw/len(subset)*100 if subset else 0
        loss_pct = sl/len(subset)*100
        mark = '<<< KILLER' if wr_v < 56 and len(subset) > 20 else ''
        print(f'  {label:<55} {len(subset):>4}t  WR={wr_v:.1f}%  '
              f'loss_rate={loss_pct:.0f}%  {mark}')

    # ================================================================
    # Best combo filters to test
    # ================================================================
    print(f'\n{"="*80}')
    print(f'  FINAL: Best Filter Combinations (simulated)')
    print(f'{"="*80}\n')

    filters_to_test = [
        ('Baseline (BB extreme + agree>=1)', {}),
        ('+ Exclude ADX>35 + mom<=1', {'exclude_strong_ct': True}),
        ('+ Exclude ADX>35 only', {'exclude_adx35': True}),
        ('+ Require 15m agree>=1', {'require_tf_agree': 1}),
        ('+ Exclude ATR rising', {'exclude_atr_rising': True}),
        ('+ Exclude ADX>35 & ATR rising', {'exclude_strong_ct': True, 'exclude_atr_rising': True}),
        ('+ 15m agree + exclude strong CT', {'require_tf_agree': 1, 'exclude_strong_ct': True}),
        ('+ Exclude ADX>35 + ATR rising + no 15m', {'exclude_strong_ct': True, 'exclude_atr_rising': True, 'require_tf_agree': 1}),
        ('+ Exclude volume spike at loss risk', {'exclude_vol_spike': True}),
    ]

    for label, filters in filters_to_test:
        wins_f = losses_f = 0
        for r in records:
            i, abs_score, direction, regime, attrs = r[:5]
            win2 = r[6]

            # Apply extra filters
            if filters.get('exclude_strong_ct') and attrs['adx'] > 35 and attrs['mom_score'] <= 1:
                continue
            if filters.get('exclude_adx35') and attrs['adx'] > 35:
                continue
            if filters.get('require_tf_agree') is not None:
                if attrs['tf_agree'] < filters['require_tf_agree']:
                    continue
            if filters.get('exclude_atr_rising') and attrs['atr_rising']:
                continue
            if filters.get('exclude_vol_spike') and attrs['vol_spike_v']:
                continue

            if win2: wins_f += 1
            else: losses_f += 1

        n = wins_f + losses_f
        wr_f = wins_f/n*100 if n else 0
        edge = wr_f - 55.6
        mark = '**' if wr_f >= 62 else ('*' if wr_f >= 61 else '')
        print(f'  {mark} {label:<55} {n:>4}t  WR={wr_f:.1f}%  edge={edge:+.1f}%')

    print()


if __name__ == '__main__':
    main()
