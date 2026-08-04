"""
Full historical backtest with current optimized parameters.
Covers 2024 H1/H2, 2025 H1/H2, 2026 H1, 2026 July OOS.
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

THRESHOLD = 5.0
MIN_AGREE = 2
BB_UP_TH = 0.10
BB_DOWN_TH = 0.90
RSI15_UP_MAX = 35
RSI15_DOWN_MIN = 65

# Optimized weights from Optuna
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


def load_period(symbol, start, end):
    data = load_all(symbol, start, end)
    candles5 = data['5m']; candles15 = data['15m']; candles1h = data['1h']

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
    ind['patterns'] = detect_candle_patterns(opens, highs, lows, closes)
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

    # 15m RSI for divergence filter
    c15 = [c[4] for c in candles15]
    return candles5, ind, rsi(c15, 7)


def score_signal(i, candles, ind, rsi15m_vals, symbol):
    closes = ind['_closes']; opens = ind['_opens']
    t5 = ind['_t5']; t1h = ind['_t1h']
    total = len(closes)

    if i < 60 or i >= total - 2:
        return 0, None, None

    price = closes[i]
    min_atr = 0.05 if symbol == 'ETHUSDT' else 0.03
    if ind['atr_pct'][i] < min_atr:
        return 0, None, None

    idx_1h = tf_idx(t1h, t5[i] - 55 * 60 * 1000)
    if idx_1h < 20:
        return 0, None, None

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

    pat = ind['patterns']
    if pat['hammer'][i]: score += w['hammer']
    elif pat['shooting_star'][i]: score -= w['hammer']
    if pat['bullish_engulfing'][i]: score += w['engulfing']
    elif pat['bearish_engulfing'][i]: score -= w['engulfing']

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

    if bb_up_v > bb_low_v:
        bb_pos = (price - bb_low_v) / (bb_up_v - bb_low_v)
        if direction == 'up' and bb_pos > BB_UP_TH: return 0, None, None
        if direction == 'down' and bb_pos < BB_DOWN_TH: return 0, None, None

    idx_15 = tf_idx(ind['_t15'], t5[i] - 10 * 60 * 1000)
    if idx_15 >= 0 and idx_15 < len(rsi15m_vals):
        r15 = rsi15m_vals[idx_15]
        if direction == 'up' and r15 < RSI15_UP_MAX: return 0, None, None
        if direction == 'down' and r15 > RSI15_DOWN_MIN: return 0, None, None

    return score, direction, regime


def backtest(symbol, start, end):
    candles, ind, rsi15m = load_period(symbol, start, end)
    total = len(candles)
    wins = 0; losses = 0
    cooldown = 0
    daily_count = defaultdict(int)
    by_month = defaultdict(lambda: [0, 0, 0.0])  # wins, losses, pnl
    by_regime = defaultdict(lambda: [0, 0])
    by_dir = {'up': [0, 0], 'down': [0, 0]}

    for i in range(60, total - 2):
        if cooldown > 0:
            cooldown -= 1
            continue

        score, direction, regime = score_signal(i, candles, ind, rsi15m, symbol)
        if direction is None: continue
        if abs(score) < THRESHOLD: continue

        dt = datetime.fromtimestamp(candles[i][0] / 1000)
        day_key = dt.strftime('%Y%m%d')
        if daily_count[day_key] >= 50: continue

        entry = candles[i + 1][1]
        settle = candles[min(i + 2, total - 1)][4]
        win = (direction == 'up' and settle > entry) or (direction == 'down' and settle < entry)
        pnl = 20 if win else -25

        if win: wins += 1
        else: losses += 1

        month_key = dt.strftime('%Y-%m')
        by_month[month_key][0 if win else 1] += 1
        by_month[month_key][2] += pnl
        by_regime[regime][0 if win else 1] += 1
        by_dir[direction][0 if win else 1] += 1
        daily_count[day_key] += 1
        cooldown = 2

    trades = wins + losses
    wr = wins / trades * 100 if trades > 0 else 0
    return trades, wr, wins, losses, sum(v[2] for v in by_month.values()), by_month, by_regime, by_dir


if __name__ == '__main__':
    periods = [
        ("2024 H1", "2024-01-01", "2024-07-01"),
        ("2024 H2", "2024-07-01", "2025-01-01"),
        ("2025 H1", "2025-01-01", "2025-07-01"),
        ("2025 H2", "2025-07-01", "2026-01-01"),
        ("2026 H1 (train)", "2026-01-01", "2026-07-01"),
        ("2026 Jul (OOS)", "2026-07-01", "2026-07-28"),
    ]

    print("=" * 90)
    print("  FULL HISTORICAL BACKTEST — Optimized V3 Strategy (th=5.0, agree>=2)")
    print("=" * 90)

    all_results = {}
    for label, start, end in periods:
        for sym in ['ETHUSDT', 'BTCUSDT']:
            trades, wr, wins, losses, pnl, by_m, by_r, by_d = backtest(sym, start, end)
            all_results[(label, sym)] = (trades, wr, wins, losses, pnl, by_m, by_r, by_d)

    # --- Overall table ---
    print(f"\n  {'Period':<16} {'Symbol':<8} {'Trades':>6} {'Win':>5} {'Loss':>5} {'WR':>7} {'PnL':>9} {'Edge':>7}")
    print(f"  {'─'*70}")
    grand_trades = 0; grand_wins = 0; grand_pnl = 0.0

    for label, start, end in periods:
        for sym in ['ETHUSDT', 'BTCUSDT']:
            trades, wr, wins, losses, pnl, _, _, _ = all_results[(label, sym)]
            grand_trades += trades; grand_wins += wins; grand_pnl += pnl
            edge = wr - 55.6
            mark = "***" if wr >= 65 else ("**" if wr >= 60 else ("*" if wr >= 55.6 else ""))
            print(f"  {label:<16} {sym:<8} {trades:>6} {wins:>5} {losses:>5} {wr:>6.1f}% ${pnl:>+9.2f} {edge:>+6.1f}% {mark}")
        # Combined row
        eth_t, eth_wr, eth_w, eth_l, eth_p, _, _, _ = all_results[(label, 'ETHUSDT')]
        btc_t, btc_wr, btc_w, btc_l, btc_p, _, _, _ = all_results[(label, 'BTCUSDT')]
        combined_t = eth_t + btc_t
        combined_w = eth_w + btc_w
        combined_wr = combined_w / combined_t * 100 if combined_t else 0
        combined_p = eth_p + btc_p
        edge_c = combined_wr - 55.6
        print(f"  {'  combined':<16} {'':<8} {combined_t:>6} {combined_w:>5} {combined_t-combined_w:>5} {combined_wr:>6.1f}% ${combined_p:>+9.2f} {edge_c:>+6.1f}%")
        print()

    print(f"  {'─'*70}")
    grand_wr = grand_wins / grand_trades * 100 if grand_trades else 0
    grand_edge = grand_wr - 55.6
    print(f"  {'GRAND TOTAL':<16} {'':<8} {grand_trades:>6} {grand_wins:>5} {grand_trades-grand_wins:>5} {grand_wr:>6.1f}% ${grand_pnl:>+9.2f} {grand_edge:>+6.1f}%")

    # --- Regime breakdown ---
    print(f"\n\n  Regime Breakdown (all periods combined):")
    print(f"  {'Regime':<12} {'ETH WR':>8} {'ETH Trades':>10} {'BTC WR':>8} {'BTC Trades':>10}")
    regime_totals = defaultdict(lambda: [0, 0, 0, 0])  # eth_w, eth_t, btc_w, btc_t
    for (label, sym), (_, _, _, _, _, _, by_r, _) in all_results.items():
        for reg, (w, l) in by_r.items():
            idx = 0 if sym == 'ETHUSDT' else 2
            regime_totals[reg][idx] += w
            regime_totals[reg][idx+1] += (w + l)
    for reg in ['trending', 'ranging', 'neutral']:
        r = regime_totals[reg]
        eth_wr = r[0]/r[1]*100 if r[1] else 0
        btc_wr = r[2]/r[3]*100 if r[3] else 0
        print(f"  {reg:<12} {eth_wr:>7.1f}% {r[1]:>10} {btc_wr:>7.1f}% {r[3]:>10}")

    # --- Direction breakdown ---
    print(f"\n\n  Direction Breakdown (all periods combined):")
    print(f"  {'Dir':<8} {'ETH WR':>8} {'ETH Trades':>10} {'BTC WR':>8} {'BTC Trades':>10}")
    dir_totals = defaultdict(lambda: [0, 0, 0, 0])
    for (label, sym), (_, _, _, _, _, _, _, by_d) in all_results.items():
        for d, (w, l) in by_d.items():
            idx = 0 if sym == 'ETHUSDT' else 2
            dir_totals[d][idx] += w
            dir_totals[d][idx+1] += (w + l)
    for d in ['up', 'down']:
        r = dir_totals[d]
        eth_wr = r[0]/r[1]*100 if r[1] else 0
        btc_wr = r[2]/r[3]*100 if r[3] else 0
        print(f"  {d:<8} {eth_wr:>7.1f}% {r[1]:>10} {btc_wr:>7.1f}% {r[3]:>10}")

    # --- Monthly detail ---
    print(f"\n\n  Monthly Detail:")
    for label, start, end in periods:
        for sym in ['ETHUSDT', 'BTCUSDT']:
            _, wr, _, _, _, by_m, _, _ = all_results[(label, sym)]
            for mk in sorted(by_m.keys()):
                w, l, p = by_m[mk]
                t = w + l
                if t > 0:
                    mwr = w/t*100
                    print(f"  {mk} {sym:<8} {t:>4}笔  WR={mwr:>5.1f}%  PnL=${p:>+8.2f}")

    # --- OOS vs In-sample comparison ---
    print(f"\n\n  Overfitting Check:")
    h1_trades = sum(all_results[(l, s)][0] for l in ['2026 H1 (train)'] for s in ['ETHUSDT', 'BTCUSDT'])
    h1_wins = sum(all_results[(l, s)][2] for l in ['2026 H1 (train)'] for s in ['ETHUSDT', 'BTCUSDT'])
    oos_trades = sum(all_results[(l, s)][0] for l in ['2026 Jul (OOS)'] for s in ['ETHUSDT', 'BTCUSDT'])
    oos_wins = sum(all_results[(l, s)][2] for l in ['2026 Jul (OOS)'] for s in ['ETHUSDT', 'BTCUSDT'])
    h1_wr = h1_wins/h1_trades*100 if h1_trades else 0
    oos_wr = oos_wins/oos_trades*100 if oos_trades else 0
    print(f"  In-sample (2026H1):  {h1_trades} trades, WR={h1_wr:.1f}%")
    print(f"  Out-of-sample (Jul): {oos_trades} trades, WR={oos_wr:.1f}%")
    print(f"  Delta: {oos_wr-h1_wr:+.1f}% {'(OOS BETTER — no overfitting)' if oos_wr >= h1_wr else '(WARNING — possible overfitting)'}")
    print()
