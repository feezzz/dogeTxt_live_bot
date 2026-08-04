"""
Recent 2-day signal-by-signal backtest.
Uses the exact same scoring logic as full_backtest.py (th=5.0, agree>=2, Optuna weights).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone, timedelta
from collections import defaultdict
from event_backtest.data_fetcher import load_all
from event_backtest.indicators import (
    sma, ema, rsi, kdj, kdj_golden_cross, kdj_death_cross,
    bollinger_bands, adx, atr, atr_pct, bb_width, volume_spike,
    cci, williams_r, stochastic_rsi, aroon, aroon_osc, mfi, parabolic_sar,
    detect_candle_patterns,
)

# ============================================================
# Config — identical to full_backtest.py
# ============================================================
THRESHOLD = 5.0
MIN_AGREE = 2
BB_UP_TH = 0.10
BB_DOWN_TH = 0.90
RSI15_UP_MAX = 35
RSI15_DOWN_MIN = 65
STAKE = 25.0
PAYOUT = 0.80

WEIGHTS = {
    'rsi_extreme': 2.26, 'rsi_moderate': 1.59, 'rsi_mild': 0.29,
    'stoch_extreme': 1.93, 'stoch_moderate': 0.75, 'stoch_cross': 0.83,
    'mfi_extreme': 2.42, 'mfi_moderate': 1.00,
    'cci_extreme': 1.96, 'cci_moderate': 1.17,
    'wr_extreme': 1.72, 'wr_moderate': 0.68,
    'sar': 0.30, 'aroon': 0.42,
    'ma_trend': 0.34, 'ema_cross': 0.31,
    'kdj_cross': 0.41, 'kdj_j': 0.31,
    'bb_extreme': 1.91, 'bb_moderate': 0.74,
    'volume': 0.31,
    'hammer': 1.83, 'engulfing': 1.25,
    'rsi_divergence': 0.73,
    'trend_di': 0.45, 'trend_rsi': 0.56,
    'range_bb': 0.32,
}


def tf_idx(timestamps, target_ts):
    for i in range(len(timestamps) - 1, -1, -1):
        if timestamps[i] <= target_ts:
            return i
    return -1


def load_indicators(symbol, start, end):
    data = load_all(symbol, start, end)
    candles5 = data['5m']
    candles15 = data['15m']
    candles1h = data['1h']

    closes = [c[4] for c in candles5]
    opens = [c[1] for c in candles5]
    highs = [c[2] for c in candles5]
    lows = [c[3] for c in candles5]
    volumes = [c[5] for c in candles5]
    t5 = [c[0] for c in candles5]
    t15 = [c[0] for c in candles15]
    t1h = [c[0] for c in candles1h]

    c1h = [c[4] for c in candles1h]
    h1h = [c[2] for c in candles1h]
    l1h = [c[3] for c in candles1h]

    ind = {}
    ind['rsi7'] = rsi(closes, 7)
    ind['rsi14'] = rsi(closes, 14)
    ind['ma5'] = sma(closes, 5)
    ind['ma10'] = sma(closes, 10)
    ind['ma20'] = sma(closes, 20)
    k, d, j = kdj(highs, lows, closes, period=6, k_period=3, d_period=3)
    ind['k'] = k; ind['d'] = d; ind['j'] = j
    ind['kg'] = kdj_golden_cross(k, d)
    ind['kd'] = kdj_death_cross(k, d)
    bb_mid, bb_up, bb_low = bollinger_bands(closes, period=20, std_mult=2.0)
    ind['bb_up'] = bb_up; ind['bb_low'] = bb_low
    ind['bbw'] = bb_width(bb_up, bb_low, bb_mid)
    ind['vol_spike'] = volume_spike(volumes, period=20, threshold=1.5)
    adx_1h, pdi_1h, mdi_1h = adx(h1h, l1h, c1h, period=14)
    ind['adx_1h'] = adx_1h; ind['pdi_1h'] = pdi_1h; ind['mdi_1h'] = mdi_1h
    ind['atr_pct'] = atr_pct(atr(highs, lows, closes, 14), closes)
    ind['cci14'] = cci(highs, lows, closes, period=14)
    ind['wr14'] = williams_r(highs, lows, closes, period=14)
    stoch_k, stoch_d = stochastic_rsi(closes, period=14, stoch_period=14)
    ind['stoch_k'] = stoch_k; ind['stoch_d'] = stoch_d
    aroon_up, aroon_down = aroon(highs, lows, period=14)
    ind['aroon_osc'] = aroon_osc(aroon_up, aroon_down)
    ind['aroon_up'] = aroon_up; ind['aroon_down'] = aroon_down
    ind['mfi'] = mfi(highs, lows, closes, volumes, period=14)
    ind['sar'] = parabolic_sar(highs, lows)
    ind['ema9'] = ema(closes, 9); ind['ema21'] = ema(closes, 21)
    pat = detect_candle_patterns(opens, highs, lows, closes)
    ind['hammer'] = pat['hammer']
    ind['shooting_star'] = pat['shooting_star']
    ind['bullish_engulfing'] = pat['bullish_engulfing']
    ind['bearish_engulfing'] = pat['bearish_engulfing']
    ind['_closes'] = closes; ind['_opens'] = opens
    ind['_t5'] = t5; ind['_t15'] = t15; ind['_t1h'] = t1h

    # RSI divergence
    ind['rsi_div_bull'] = [False] * len(closes)
    ind['rsi_div_bear'] = [False] * len(closes)
    rsi14 = ind['rsi14']
    for i in range(15, len(closes)):
        pw = closes[i-15:i+1]; rw = rsi14[i-15:i+1]
        pmin_i = pw.index(min(pw)); rmin_i = rw.index(min(rw))
        if pmin_i > len(pw) - 6 and rmin_i < pmin_i - 3:
            if closes[i] <= closes[i-15+pmin_i] * 1.005:
                ind['rsi_div_bull'][i] = True
        pmax_i = pw.index(max(pw)); rmax_i = rw.index(max(rw))
        if pmax_i > len(pw) - 6 and rmax_i < pmax_i - 3:
            if closes[i] >= closes[i-15+pmax_i] * 0.995:
                ind['rsi_div_bear'][i] = True

    # 15m RSI
    c15 = [c[4] for c in candles15]
    rsi15m = rsi(c15, 7)

    return candles5, ind, rsi15m


def score_signal(i, ind, rsi15m, symbol):
    """Identical scoring to full_backtest.py"""
    closes = ind['_closes']; opens = ind['_opens']
    t5 = ind['_t5']; t1h = ind['_t1h']

    price = closes[i]
    atr_pct_v = ind['atr_pct'][i]
    min_atr = 0.05 if symbol == 'ETHUSDT' else 0.03
    if atr_pct_v < min_atr:
        return 0, None, None, {}

    idx_1h = tf_idx(t1h, t5[i] - 55 * 60 * 1000)
    if idx_1h < 20:
        return 0, None, None, {}

    adx_val = ind['adx_1h'][idx_1h]
    di_diff = ind['pdi_1h'][idx_1h] - ind['mdi_1h'][idx_1h]
    aroon_osc_val = ind['aroon_osc'][i]
    bbw_val = ind['bbw'][i]
    if adx_val > 25 or abs(aroon_osc_val) > 50:
        regime = 'trending'
    elif adx_val < 18 or (bbw_val < 1.5 and abs(aroon_osc_val) < 25):
        regime = 'ranging'
    else:
        regime = 'neutral'

    w = WEIGHTS
    score = 0.0
    reasons = []

    rsi7_v = ind['rsi7'][i]
    if rsi7_v < 20: score += w['rsi_extreme']; reasons.append(f'RSI7={rsi7_v:.0f}(超卖)')
    elif rsi7_v < 30: score += w['rsi_moderate']; reasons.append(f'RSI7={rsi7_v:.0f}(低)')
    elif rsi7_v < 40: score += w['rsi_mild']
    elif rsi7_v > 80: score -= w['rsi_extreme']; reasons.append(f'RSI7={rsi7_v:.0f}(超买)')
    elif rsi7_v > 70: score -= w['rsi_moderate']; reasons.append(f'RSI7={rsi7_v:.0f}(高)')
    elif rsi7_v > 60: score -= w['rsi_mild']

    sk = ind['stoch_k'][i]; sd = ind['stoch_d'][i]
    if sk < 10 and sd < 15: score += w['stoch_extreme']; reasons.append(f'Stoch={sk:.0f}(极低)')
    elif sk < 20: score += w['stoch_moderate']
    elif sk > 90 and sd > 85: score -= w['stoch_extreme']; reasons.append(f'Stoch={sk:.0f}(极高)')
    elif sk > 80: score -= w['stoch_moderate']
    if i > 0 and ind['stoch_k'][i-1] <= ind['stoch_d'][i-1] and sk > sd: score += w['stoch_cross']
    elif i > 0 and ind['stoch_k'][i-1] >= ind['stoch_d'][i-1] and sk < sd: score -= w['stoch_cross']

    mfi_v = ind['mfi'][i]
    if mfi_v < 15: score += w['mfi_extreme']; reasons.append(f'MFI={mfi_v:.0f}(超卖)')
    elif mfi_v < 25: score += w['mfi_moderate']
    elif mfi_v > 85: score -= w['mfi_extreme']; reasons.append(f'MFI={mfi_v:.0f}(超买)')
    elif mfi_v > 75: score -= w['mfi_moderate']

    cci_v = ind['cci14'][i]
    if cci_v < -200: score += w['cci_extreme']
    elif cci_v < -100: score += w['cci_moderate']
    elif cci_v > 200: score -= w['cci_extreme']
    elif cci_v > 100: score -= w['cci_moderate']

    wr_v = ind['wr14'][i]
    if wr_v < -90: score += w['wr_extreme']
    elif wr_v < -80: score += w['wr_moderate']
    elif wr_v > -10: score -= w['wr_extreme']
    elif wr_v > -20: score -= w['wr_moderate']

    if price > ind['sar'][i]: score += w['sar']
    else: score -= w['sar']

    if ind['aroon_up'][i] > 70 and ind['aroon_down'][i] < 30: score += w['aroon']
    elif ind['aroon_down'][i] > 70 and ind['aroon_up'][i] < 30: score -= w['aroon']

    if price > ind['ma20'][i] and ind['ma5'][i] > ind['ma10'][i]: score += w['ma_trend']
    elif price < ind['ma20'][i] and ind['ma5'][i] < ind['ma10'][i]: score -= w['ma_trend']

    if ind['ema9'][i] > ind['ema21'][i]: score += w['ema_cross']
    else: score -= w['ema_cross']

    if ind['kg'][i]: score += w['kdj_cross']; reasons.append('KDJ金叉')
    elif ind['kd'][i]: score -= w['kdj_cross']; reasons.append('KDJ死叉')
    if ind['j'][i] < 0: score += w['kdj_j']
    elif ind['j'][i] > 100: score -= w['kdj_j']

    bb_up_v = ind['bb_up'][i]; bb_low_v = ind['bb_low'][i]
    if bb_up_v > bb_low_v:
        bb_pos = (price - bb_low_v) / (bb_up_v - bb_low_v)
        if bb_pos < 0.08: score += w['bb_extreme']; reasons.append(f'BB底({bb_pos:.2f})')
        elif bb_pos < 0.2: score += w['bb_moderate']
        elif bb_pos > 0.92: score -= w['bb_extreme']; reasons.append(f'BB顶({bb_pos:.2f})')
        elif bb_pos > 0.8: score -= w['bb_moderate']

    if ind['vol_spike'][i]:
        if closes[i] > opens[i]: score += w['volume']
        else: score -= w['volume']

    if ind['hammer'][i]: score += w['hammer']; reasons.append('锤子线')
    elif ind['shooting_star'][i]: score -= w['hammer']; reasons.append('射击之星')
    if ind['bullish_engulfing'][i]: score += w['engulfing']; reasons.append('多头吞没')
    elif ind['bearish_engulfing'][i]: score -= w['engulfing']; reasons.append('空头吞没')

    if ind['rsi_div_bull'][i]: score += w['rsi_divergence']; reasons.append('RSI底背离')
    elif ind['rsi_div_bear'][i]: score -= w['rsi_divergence']; reasons.append('RSI顶背离')

    if regime == 'trending':
        if di_diff > 5: score += w['trend_di']
        elif di_diff < -5: score -= w['trend_di']
        if di_diff > 3 and rsi7_v < 50: score += w['trend_rsi']
        elif di_diff < -3 and rsi7_v > 50: score -= w['trend_rsi']
    elif regime == 'ranging':
        bb_pos_val = (price - bb_low_v) / (bb_up_v - bb_low_v) if bb_up_v > bb_low_v else 0.5
        if bb_pos_val < 0.15: score += w['range_bb']
        elif bb_pos_val > 0.85: score -= w['range_bb']

    direction = 'up' if score >= 0 else 'down'

    # Agree filter
    agree = 0
    if direction == 'up':
        if sk < 30: agree += 1
        if mfi_v < 40: agree += 1
        if aroon_osc_val > -30: agree += 1
        if price > ind['sar'][i]: agree += 1
    else:
        if sk > 70: agree += 1
        if mfi_v > 60: agree += 1
        if aroon_osc_val < 30: agree += 1
        if price < ind['sar'][i]: agree += 1
    if agree < MIN_AGREE:
        return 0, None, None, {}

    # BB position filter
    if bb_up_v > bb_low_v:
        bb_pos = (price - bb_low_v) / (bb_up_v - bb_low_v)
        if direction == 'up' and bb_pos > BB_UP_TH: return 0, None, None, {}
        if direction == 'down' and bb_pos < BB_DOWN_TH: return 0, None, None, {}

    # 15m filter
    idx_15 = tf_idx(ind['_t15'], t5[i] - 10 * 60 * 1000)
    if idx_15 >= 0 and idx_15 < len(rsi15m):
        r15 = rsi15m[idx_15]
        if direction == 'up' and r15 < RSI15_UP_MAX: return 0, None, None, {}
        if direction == 'down' and r15 > RSI15_DOWN_MIN: return 0, None, None, {}

    details = {
        'rsi7': rsi7_v, 'mfi': mfi_v, 'stoch_k': sk,
        'cci': cci_v, 'wr': wr_v, 'atr_pct': atr_pct_v,
        'adx': adx_val, 'agree': agree,
        'bb_pos': (price - bb_low_v) / (bb_up_v - bb_low_v) if bb_up_v > bb_low_v else -1,
        'rsi15': rsi15m[idx_15] if (idx_15 >= 0 and idx_15 < len(rsi15m)) else -1,
    }
    return score, direction, regime, details


# ============================================================
# Main
# ============================================================
END = datetime(2026, 7, 28, 23, 59, tzinfo=timezone.utc)
START = END - timedelta(days=4)  # extra for warmup
SYMBOLS = ['ETHUSDT', 'BTCUSDT']

print(f"{'='*130}")
print(f"  RECENT 2-DAY SIGNAL-BY-SIGNAL BACKTEST")
print(f"  Period: 2026-07-27 ~ 2026-07-28  |  th={THRESHOLD}  agree>={MIN_AGREE}  stake=${STAKE}")
print(f"{'='*130}")

grand_trades = 0
grand_wins = 0
grand_pnl = 0.0

for sym in SYMBOLS:
    print(f"\n  Loading {sym}...")
    candles, ind, rsi15m = load_indicators(sym, START.strftime('%Y-%m-%d'), END.strftime('%Y-%m-%d'))

    total = len(candles)
    trades = []
    daily = defaultdict(lambda: {'trades': 0, 'wins': 0, 'loss': 0, 'pnl': 0.0})

    for i in range(60, total - 2):
        score, direction, regime, det = score_signal(i, ind, rsi15m, sym)
        if direction is None: continue
        if abs(score) < THRESHOLD: continue

        ts = candles[i][0]
        dt = datetime.fromtimestamp(ts / 1000)
        day = dt.strftime('%m-%d')

        # Only report Jul 27-28
        if day not in ('07-27', '07-28'):
            continue

        entry = candles[i + 1][1]   # open of next candle
        settle = candles[min(i + 2, total - 1)][4]  # close of i+2

        win = (direction == 'up' and settle > entry) or (direction == 'down' and settle < entry)
        pnl = STAKE * PAYOUT if win else -STAKE

        d = daily[day]
        d['trades'] += 1
        if win: d['wins'] += 1
        else: d['loss'] += 1
        d['pnl'] += pnl

        trades.append({
            'time': dt.strftime('%m-%d %H:%M'),
            'day': day,
            'dir': direction,
            'score': score,
            'entry': entry,
            'settle': settle,
            'win': win,
            'pnl': pnl,
            'regime': regime,
            'reasons': [],  # will fill below
            'det': det,
        })

    # Print per-symbol detail
    print(f"\n{'='*130}")
    print(f"  {sym}  —  {len(trades)} signals")
    print(f"{'='*130}")
    print(f"  {'#':<4} {'Time':<12} {'Dir':<5} {'Score':<7} {'Entry':<10} {'Settle':<10} "
          f"{'Result':<8} {'PnL':<8} {'Rsi7':<6} {'Mfi':<6} {'StochK':<7} {'CCI':<6} "
          f"{'WR':<6} {'BBpos':<7} {'Rsi15':<6} {'ADX':<5} {'Agr':<4} {'Regime':<9}  Reasons")
    print(f"  {'─'*127}")

    sym_wins = sum(1 for t in trades if t['win'])
    sym_loss = len(trades) - sym_wins
    sym_pnl = sum(t['pnl'] for t in trades)

    for j, t in enumerate(trades):
        d = t['det']
        win_str = '[WIN]' if t['win'] else '[LOSS]'

        # Build reasons from strategy engine fields
        reasons = []
        score_abs = abs(t['score'])
        direction = t['dir']

        # Check what contributed to the score based on indicator values
        r7 = d['rsi7']
        if direction == 'up':
            if r7 < 20: reasons.append('RSI超卖')
            elif r7 < 30: reasons.append('RSI低')
        else:
            if r7 > 80: reasons.append('RSI超买')
            elif r7 > 70: reasons.append('RSI高')

        mf = d['mfi']
        if direction == 'up':
            if mf < 15: reasons.append('MFI超卖')
            elif mf < 25: reasons.append('MFI低')
        else:
            if mf > 85: reasons.append('MFI超买')
            elif mf > 75: reasons.append('MFI高')

        skv = d['stoch_k']
        if direction == 'up':
            if skv < 10: reasons.append('Stoch极低')
            elif skv < 20: reasons.append('Stoch低')
        else:
            if skv > 90: reasons.append('Stoch极高')
            elif skv > 80: reasons.append('Stoch高')

        bb = d['bb_pos']
        if direction == 'up':
            if bb < 0.08: reasons.append('BB底')
            elif bb < 0.2: reasons.append('BB低位')
        else:
            if bb > 0.92: reasons.append('BB顶')
            elif bb > 0.8: reasons.append('BB高位')

        c = d['cci']
        if direction == 'up':
            if c < -200: reasons.append('CCI极低')
            elif c < -100: reasons.append('CCI低')
        else:
            if c > 200: reasons.append('CCI极高')
            elif c > 100: reasons.append('CCI高')

        wrv = d['wr']
        if direction == 'up':
            if wrv < -90: reasons.append('WR极低')
            elif wrv < -80: reasons.append('WR低')
        else:
            if wrv > -10: reasons.append('WR极高')
            elif wrv > -20: reasons.append('WR高')

        reasons_str = ','.join(reasons[:5]) if reasons else '-'

        print(f"  {j+1:<4} {t['time']:<12} {t['dir']:<5} {t['score']:+6.1f}  "
              f"{t['entry']:<10.2f} {t['settle']:<10.2f} "
              f"{win_str:<8} ${t['pnl']:+6.1f} "
              f"{r7:<6.0f} {mf:<6.0f} {skv:<7.0f} {c:<6.0f} "
              f"{wrv:<6.0f} {bb:<7.3f} {d['rsi15']:<6.0f} {d['adx']:<5.0f} {d['agree']:<4} "
              f"{t['regime']:<9}  {reasons_str}")

    # Daily summary
    print(f"\n  {sym} daily:")
    for day in sorted(daily):
        d = daily[day]
        wr = d['wins'] / d['trades'] * 100 if d['trades'] else 0
        print(f"    {day}: {d['trades']} trades | {d['wins']}W {d['loss']}L | WR={wr:.1f}% | PnL=${d['pnl']:+.1f}")

    print(f"\n  {sym} TOTAL: {len(trades)} trades | {sym_wins}W {sym_loss}L | "
          f"WR={sym_wins/len(trades)*100:.1f}% | PnL=${sym_pnl:+.1f}")

    grand_trades += len(trades)
    grand_wins += sym_wins
    grand_pnl += sym_pnl

print(f"\n{'='*130}")
print(f"  GRAND TOTAL: {grand_trades} trades | {grand_wins}W {grand_trades-grand_wins}L | "
      f"WR={grand_wins/grand_trades*100:.1f}% | PnL=${grand_pnl:+.1f}")
print(f"{'='*130}")
