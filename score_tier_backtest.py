"""
Per-score-tier backtest with V4 config (th=3.0, Optuna weights, all filters).
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

THRESHOLD = 3.0
MIN_AGREE = 2
BB_UP_TH = 0.10
BB_DOWN_TH = 0.90
RSI15_UP_MAX = 35
RSI15_DOWN_MIN = 65
STAKE_BASE = 25.0  # used as reference

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

# Position tiers for PnL calculation
POSITION_TIERS = [
    (7.0, 2.0, "$50"),
    (6.0, 1.4, "$35"),
    (5.0, 1.0, "$25"),
]

SCORE_BUCKETS = [
    (3.0, 5.0,  "3.0-5.0"),
    (5.0, 6.0,  "5.0-6.0"),
    (6.0, 7.0,  "6.0-7.0"),
    (7.0, 99,   "7.0+"),
]


def tf_idx(timestamps, target_ts):
    for i in range(len(timestamps) - 1, -1, -1):
        if timestamps[i] <= target_ts:
            return i
    return -1


def load_indicators(symbol, start, end):
    data = load_all(symbol, start, end)
    candles5 = data['5m']; candles15 = data['15m']; candles1h = data['1h']

    closes = [c[4] for c in candles5]; opens = [c[1] for c in candles5]
    highs = [c[2] for c in candles5]; lows = [c[3] for c in candles5]
    volumes = [c[5] for c in candles5]
    t5 = [c[0] for c in candles5]; t15 = [c[0] for c in candles15]; t1h = [c[0] for c in candles1h]
    c1h = [c[4] for c in candles1h]; h1h = [c[2] for c in candles1h]; l1h = [c[3] for c in candles1h]

    ind = {}
    ind['rsi7'] = rsi(closes, 7); ind['rsi14'] = rsi(closes, 14)
    ind['ma5'] = sma(closes, 5); ind['ma10'] = sma(closes, 10); ind['ma20'] = sma(closes, 20)
    k, d, j = kdj(highs, lows, closes, period=6, k_period=3, d_period=3)
    ind['k'] = k; ind['d'] = d; ind['j'] = j
    ind['kg'] = kdj_golden_cross(k, d); ind['kd'] = kdj_death_cross(k, d)
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
    ind['hammer'] = pat['hammer']; ind['shooting_star'] = pat['shooting_star']
    ind['bullish_engulfing'] = pat['bullish_engulfing']; ind['bearish_engulfing'] = pat['bearish_engulfing']
    ind['_closes'] = closes; ind['_opens'] = opens
    ind['_t5'] = t5; ind['_t15'] = t15; ind['_t1h'] = t1h

    ind['rsi_div_bull'] = [False] * len(closes); ind['rsi_div_bear'] = [False] * len(closes)
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

    c15 = [c[4] for c in candles15]
    return candles5, ind, rsi(c15, 7)


def score_signal(i, ind, rsi15m, symbol):
    closes = ind['_closes']; opens = ind['_opens']; t5 = ind['_t5']; t1h = ind['_t1h']

    price = closes[i]
    min_atr = 0.05 if symbol == 'ETHUSDT' else 0.03
    if ind['atr_pct'][i] < min_atr:
        return 0, None, None

    idx_1h = tf_idx(t1h, t5[i] - 55 * 60 * 1000)
    if idx_1h < 20:
        return 0, None, None

    adx_val = ind['adx_1h'][idx_1h]
    di_diff = ind['pdi_1h'][idx_1h] - ind['mdi_1h'][idx_1h]
    aroon_osc_val = ind['aroon_osc'][i]; bbw_val = ind['bbw'][i]
    if adx_val > 25 or abs(aroon_osc_val) > 50:
        regime = 'trending'
    elif adx_val < 18 or (bbw_val < 1.5 and abs(aroon_osc_val) < 25):
        regime = 'ranging'
    else:
        regime = 'neutral'

    w = WEIGHTS; score = 0.0
    rsi7_v = ind['rsi7'][i]
    if rsi7_v < 20: score += w['rsi_extreme']
    elif rsi7_v < 30: score += w['rsi_moderate']
    elif rsi7_v < 40: score += w['rsi_mild']
    elif rsi7_v > 80: score -= w['rsi_extreme']
    elif rsi7_v > 70: score -= w['rsi_moderate']
    elif rsi7_v > 60: score -= w['rsi_mild']

    sk = ind['stoch_k'][i]; sd = ind['stoch_d'][i]
    if sk < 10 and sd < 15: score += w['stoch_extreme']
    elif sk < 20: score += w['stoch_moderate']
    elif sk > 90 and sd > 85: score -= w['stoch_extreme']
    elif sk > 80: score -= w['stoch_moderate']
    if i > 0 and ind['stoch_k'][i-1] <= ind['stoch_d'][i-1] and sk > sd: score += w['stoch_cross']
    elif i > 0 and ind['stoch_k'][i-1] >= ind['stoch_d'][i-1] and sk < sd: score -= w['stoch_cross']

    mfi_v = ind['mfi'][i]
    if mfi_v < 15: score += w['mfi_extreme']
    elif mfi_v < 25: score += w['mfi_moderate']
    elif mfi_v > 85: score -= w['mfi_extreme']
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

    if ind['kg'][i]: score += w['kdj_cross']
    elif ind['kd'][i]: score -= w['kdj_cross']
    if ind['j'][i] < 0: score += w['kdj_j']
    elif ind['j'][i] > 100: score -= w['kdj_j']

    bb_up_v = ind['bb_up'][i]; bb_low_v = ind['bb_low'][i]
    if bb_up_v > bb_low_v:
        bb_pos = (price - bb_low_v) / (bb_up_v - bb_low_v)
        if bb_pos < 0.08: score += w['bb_extreme']
        elif bb_pos < 0.2: score += w['bb_moderate']
        elif bb_pos > 0.92: score -= w['bb_extreme']
        elif bb_pos > 0.8: score -= w['bb_moderate']

    if ind['vol_spike'][i]:
        if closes[i] > opens[i]: score += w['volume']
        else: score -= w['volume']

    if ind['hammer'][i]: score += w['hammer']
    elif ind['shooting_star'][i]: score -= w['hammer']
    if ind['bullish_engulfing'][i]: score += w['engulfing']
    elif ind['bearish_engulfing'][i]: score -= w['engulfing']

    if ind['rsi_div_bull'][i]: score += w['rsi_divergence']
    elif ind['rsi_div_bear'][i]: score -= w['rsi_divergence']

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
        return 0, None, None

    # BB position filter
    if bb_up_v > bb_low_v:
        bb_pos = (price - bb_low_v) / (bb_up_v - bb_low_v)
        if direction == 'up' and bb_pos > BB_UP_TH: return 0, None, None
        if direction == 'down' and bb_pos < BB_DOWN_TH: return 0, None, None

    # 15m RSI filter
    idx_15 = tf_idx(ind['_t15'], t5[i] - 10 * 60 * 1000)
    if idx_15 >= 0 and idx_15 < len(rsi15m):
        r15 = rsi15m[idx_15]
        if direction == 'up' and r15 < RSI15_UP_MAX: return 0, None, None
        if direction == 'down' and r15 > RSI15_DOWN_MIN: return 0, None, None

    return score, direction, regime


def get_stake(abs_score):
    for lo, mult, _ in POSITION_TIERS:
        if abs_score >= lo:
            return STAKE_BASE * mult
    return STAKE_BASE


if __name__ == '__main__':
    periods = [
        ("2024H1", "2024-01-01", "2024-07-01"),
        ("2024H2", "2024-07-01", "2025-01-01"),
        ("2025H1", "2025-01-01", "2025-07-01"),
        ("2025H2", "2025-07-01", "2026-01-01"),
        ("2026H1", "2026-01-01", "2026-07-01"),
    ]

    buckets = {label: {'wins': 0, 'losses': 0, 'pnl': 0.0, 'count': 0}
               for _, _, label in SCORE_BUCKETS}
    grand = {'wins': 0, 'losses': 0, 'pnl': 0.0, 'count': 0}
    by_period = defaultdict(lambda: {label: {'wins': 0, 'losses': 0, 'pnl': 0.0}
                                      for _, _, label in SCORE_BUCKETS})

    print("=" * 95)
    print("  Per-Score-Tier Backtest — V4 Config (th=3.0, agree>=2, BB+15m filters)")
    print("=" * 95)

    for pname, start, end in periods:
        for sym in ['ETHUSDT', 'BTCUSDT']:
            print(f"\n  Loading {sym} {pname}...")
            candles, ind, rsi15m = load_indicators(sym, start, end)
            total = len(candles)
            cooldown = 0
            daily_count = defaultdict(int)

            for i in range(60, total - 2):
                if cooldown > 0:
                    cooldown -= 1
                    continue

                score, direction, regime = score_signal(i, ind, rsi15m, sym)
                if direction is None: continue
                if abs(score) < THRESHOLD: continue

                dt = datetime.fromtimestamp(candles[i][0] / 1000)
                if daily_count[dt.strftime('%Y%m%d')] >= 50: continue

                entry = candles[i + 1][1]
                settle = candles[min(i + 2, total - 1)][4]
                win = (direction == 'up' and settle > entry) or (direction == 'down' and settle < entry)

                abs_score = abs(score)
                stake = get_stake(abs_score)
                pnl = stake * 0.80 if win else -stake

                for lo, hi, label in SCORE_BUCKETS:
                    if lo <= abs_score < hi:
                        buckets[label]['count'] += 1
                        if win: buckets[label]['wins'] += 1
                        else: buckets[label]['losses'] += 1
                        buckets[label]['pnl'] += pnl
                        by_period[pname][label]['wins' if win else 'losses'] += 1
                        by_period[pname][label]['pnl'] += pnl
                        break

                if win: grand['wins'] += 1
                else: grand['losses'] += 1
                grand['pnl'] += pnl
                grand['count'] += 1
                daily_count[dt.strftime('%Y%m%d')] += 1
                cooldown = 2

    # Print per-bucket summary
    print(f"\n\n  {'Score':<12} {'Trades':>7} {'Win':>5} {'Loss':>5} {'WR':>8} {'PnL':>12} {'Stake':>6} {'期望/笔':>10}")
    print(f"  {'─'*72}")
    for lo, hi, label in SCORE_BUCKETS:
        b = buckets[label]
        n = b['wins'] + b['losses']
        if n == 0: continue
        wr = b['wins'] / n * 100
        stake_label = "$25"
        for t_lo, _, sl in POSITION_TIERS:
            if lo >= t_lo:
                stake_label = sl
                break
        ev = b['pnl'] / n
        print(f"  {label:<12} {n:>7} {b['wins']:>5} {b['losses']:>5} {wr:>7.1f}% ${b['pnl']:>+11.2f} {stake_label:>6} ${ev:>+9.2f}")

    # Grand total
    n = grand['wins'] + grand['losses']
    wr = grand['wins'] / n * 100 if n else 0
    ev = grand['pnl'] / n if n else 0
    print(f"  {'─'*72}")
    print(f"  {'ALL':<12} {n:>7} {grand['wins']:>5} {grand['losses']:>5} {wr:>7.1f}% ${grand['pnl']:>+11.2f} {'':>6} ${ev:>+9.2f}")

    # Per-period breakdown
    print(f"\n\n  Per-Period Breakdown:")
    print(f"  {'Period':<10}", end="")
    for _, _, label in SCORE_BUCKETS:
        print(f" {'  '+label+'  ':<16}", end="")
    print(f" {'TOTAL':<10}")
    print(f"  {'─'*10}{'─'*16*len(SCORE_BUCKETS)}{'─'*10}")

    for pname, _, _ in periods:
        print(f"  {pname:<10}", end="")
        pt = 0
        for _, _, label in SCORE_BUCKETS:
            b = by_period[pname][label]
            n = b['wins'] + b['losses']
            pt += n
            wr = b['wins'] / n * 100 if n else 0
            pnl = b['pnl']
            print(f" {n:>4}笔 {wr:>5.1f}% {pnl:>+7.0f}", end="")
        print(f" {pt:>5}笔")

    # Direction breakdown per bucket
    print(f"\n\n  Direction by Score Tier:")
    for _, _, label in SCORE_BUCKETS:
        n = buckets[label]['wins'] + buckets[label]['losses']
        pnl = buckets[label]['pnl']
        print(f"  {label}: {n}笔, PnL=${pnl:+.2f}")

    print()
