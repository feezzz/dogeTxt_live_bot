"""
P3: Optuna Bayesian optimization of V3 strategy scoring weights.
Uses walk-forward validation (train H1, validate July) to prevent overfitting.
Also adds RSI divergence as a new signal component.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from collections import defaultdict
import optuna

from event_backtest.data_fetcher import load_all
from event_backtest.indicators import (
    sma, ema, rsi, kdj, kdj_golden_cross, kdj_death_cross,
    bollinger_bands, adx, atr, atr_pct, bb_width, volume_spike,
    cci, williams_r, stochastic_rsi, aroon, aroon_osc, mfi, parabolic_sar,
    detect_candle_patterns,
)

SYMBOL = 'ETHUSDT'
TRAIN_START, TRAIN_END = '2026-01-01', '2026-07-01'
VAL_START, VAL_END = '2026-07-01', '2026-07-28'
THRESHOLD = 5.0
MIN_AGREE = 2
BB_UP_TH = 0.10
BB_DOWN_TH = 0.90
RSI15_UP_MAX = 35
RSI15_DOWN_MIN = 65

# ---------------------------------------------------------------------------
# Data loading (cached once)
# ---------------------------------------------------------------------------
_train_data = _val_data = None

def _load_period(start, end):
    data = load_all(SYMBOL, start, end)
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

    # Compute all indicators once
    ind = {}
    ind['rsi7'] = rsi(closes, 7)
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
    ind['ema9'] = ema(closes, 9)
    ind['ema21'] = ema(closes, 21)
    ind['patterns'] = detect_candle_patterns(opens, highs, lows, closes)
    ind['_closes'] = closes; ind['_opens'] = opens
    ind['_t5'] = t5; ind['_t15'] = t15; ind['_t1h'] = t1h

    # RSI divergence (15-period lookback)
    ind['rsi_div_bull'] = [False] * len(closes)
    ind['rsi_div_bear'] = [False] * len(closes)
    rsi14 = rsi(closes, 14)
    for i in range(15, len(closes)):
        # Bullish divergence: price lower low, RSI higher low
        price_window = closes[i-15:i+1]
        rsi_window = rsi14[i-15:i+1]
        price_min_idx = price_window.index(min(price_window))
        rsi_min_idx = rsi_window.index(min(rsi_window))
        if price_min_idx > i - 5 and rsi_min_idx < price_min_idx - 3:
            if closes[i] <= closes[price_min_idx] * 1.005:  # near the low
                ind['rsi_div_bull'][i] = True
        # Bearish divergence: price higher high, RSI lower high
        price_max_idx = price_window.index(max(price_window))
        rsi_max_idx = rsi_window.index(max(rsi_window))
        if price_max_idx > i - 5 and rsi_max_idx < price_max_idx - 3:
            if closes[i] >= closes[price_max_idx] * 0.995:  # near the high
                ind['rsi_div_bear'][i] = True

    return candles5, ind


def _tf_idx(timestamps, target_ts):
    for i in range(len(timestamps) - 1, -1, -1):
        if timestamps[i] <= target_ts:
            return i
    return -1


# ---------------------------------------------------------------------------
# Parameterized scoring function
# ---------------------------------------------------------------------------
def score_signal(i, candles, ind, weights, rsi15m_vals=None):
    """V3 scoring with parameterized weights. Returns (score, direction, regime)."""
    closes = ind['_closes']; opens = ind['_opens']
    t5 = ind['_t5']; t1h = ind['_t1h']
    total = len(closes)
    price = closes[i]

    warmup = 60
    if i < warmup or i >= total - 2:
        return 0, None, None

    # ATR filter
    atr_pct_val = ind['atr_pct'][i]
    if atr_pct_val < 0.05:
        return 0, None, None

    idx_1h = _tf_idx(t1h, t5[i] - 55 * 60 * 1000)
    if idx_1h < 20:
        return 0, None, None

    # Regime detection
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

    score = 0.0

    # --- RSI ---
    rsi7_v = ind['rsi7'][i]
    if rsi7_v < 20:       score += weights['rsi_extreme']
    elif rsi7_v < 30:     score += weights['rsi_moderate']
    elif rsi7_v < 40:     score += weights['rsi_mild']
    elif rsi7_v > 80:     score -= weights['rsi_extreme']
    elif rsi7_v > 70:     score -= weights['rsi_moderate']
    elif rsi7_v > 60:     score -= weights['rsi_mild']

    # --- StochRSI ---
    stoch_k_v = ind['stoch_k'][i]; stoch_d_v = ind['stoch_d'][i]
    if stoch_k_v < 10 and stoch_d_v < 15:     score += weights['stoch_extreme']
    elif stoch_k_v < 20:                       score += weights['stoch_moderate']
    elif stoch_k_v > 90 and stoch_d_v > 85:    score -= weights['stoch_extreme']
    elif stoch_k_v > 80:                       score -= weights['stoch_moderate']

    if i > 0 and ind['stoch_k'][i-1] <= ind['stoch_d'][i-1] and stoch_k_v > stoch_d_v:
        score += weights['stoch_cross']
    elif i > 0 and ind['stoch_k'][i-1] >= ind['stoch_d'][i-1] and stoch_k_v < stoch_d_v:
        score -= weights['stoch_cross']

    # --- MFI ---
    mfi_v = ind['mfi'][i]
    if mfi_v < 15:       score += weights['mfi_extreme']
    elif mfi_v < 25:     score += weights['mfi_moderate']
    elif mfi_v > 85:     score -= weights['mfi_extreme']
    elif mfi_v > 75:     score -= weights['mfi_moderate']

    # --- CCI ---
    cci_v = ind['cci14'][i]
    if cci_v < -200:     score += weights['cci_extreme']
    elif cci_v < -100:   score += weights['cci_moderate']
    elif cci_v > 200:    score -= weights['cci_extreme']
    elif cci_v > 100:    score -= weights['cci_moderate']

    # --- Williams %R ---
    wr_v = ind['wr14'][i]
    if wr_v < -90:       score += weights['wr_extreme']
    elif wr_v < -80:     score += weights['wr_moderate']
    elif wr_v > -10:     score -= weights['wr_extreme']
    elif wr_v > -20:     score -= weights['wr_moderate']

    # --- PSAR ---
    if price > ind['sar'][i]:    score += weights['sar']
    else:                        score -= weights['sar']

    # --- Aroon ---
    if ind['aroon_up'][i] > 70 and ind['aroon_down'][i] < 30:    score += weights['aroon']
    elif ind['aroon_down'][i] > 70 and ind['aroon_up'][i] < 30:  score -= weights['aroon']

    # --- MA trend ---
    if price > ind['ma20'][i] and ind['ma5'][i] > ind['ma10'][i]:      score += weights['ma_trend']
    elif price < ind['ma20'][i] and ind['ma5'][i] < ind['ma10'][i]:    score -= weights['ma_trend']

    # --- EMA cross ---
    if ind['ema9'][i] > ind['ema21'][i]:   score += weights['ema_cross']
    else:                                   score -= weights['ema_cross']

    # --- KDJ ---
    if ind['kg'][i]:      score += weights['kdj_cross']
    elif ind['kd'][i]:    score -= weights['kdj_cross']
    if ind['j'][i] < 0:   score += weights['kdj_j']
    elif ind['j'][i] > 100: score -= weights['kdj_j']

    # --- BB position ---
    bb_up_v = ind['bb_up'][i]; bb_low_v = ind['bb_low'][i]
    if bb_up_v > bb_low_v:
        bb_pos = (price - bb_low_v) / (bb_up_v - bb_low_v)
        if bb_pos < 0.08:       score += weights['bb_extreme']
        elif bb_pos < 0.2:      score += weights['bb_moderate']
        elif bb_pos > 0.92:     score -= weights['bb_extreme']
        elif bb_pos > 0.8:      score -= weights['bb_moderate']

    # --- Volume ---
    if ind['vol_spike'][i]:
        if closes[i] > opens[i]:   score += weights['volume']
        else:                       score -= weights['volume']

    # --- Candle patterns ---
    pat = ind['patterns']
    if pat['hammer'][i]:          score += weights['hammer']
    elif pat['shooting_star'][i]: score -= weights['hammer']
    if pat['bullish_engulfing'][i]:    score += weights['engulfing']
    elif pat['bearish_engulfing'][i]:  score -= weights['engulfing']

    # --- RSI Divergence (NEW) ---
    if ind['rsi_div_bull'][i]:
        score += weights['rsi_divergence']
    elif ind['rsi_div_bear'][i]:
        score -= weights['rsi_divergence']

    # --- Regime adjustments ---
    if regime == 'trending':
        if di_diff > 5:     score += weights['trend_di']
        elif di_diff < -5:  score -= weights['trend_di']
        if di_diff > 3 and rsi7_v < 50:    score += weights['trend_rsi']
        elif di_diff < -3 and rsi7_v > 50: score -= weights['trend_rsi']
    elif regime == 'ranging':
        bb_pos_val = (price - bb_low_v) / (bb_up_v - bb_low_v) if bb_up_v > bb_low_v else 0.5
        if bb_pos_val < 0.15:    score += weights['range_bb']
        elif bb_pos_val > 0.85:  score -= weights['range_bb']

    # Determine direction
    direction = 'up' if score >= 0 else 'down'

    # Agree check
    agree = 0
    if direction == 'up':
        if stoch_k_v < 30: agree += 1
        if mfi_v < 40: agree += 1
        if aroon_osc_val > -30: agree += 1
        if price > ind['sar'][i]: agree += 1
    else:
        if stoch_k_v > 70: agree += 1
        if mfi_v > 60: agree += 1
        if aroon_osc_val < 30: agree += 1
        if price < ind['sar'][i]: agree += 1
    if agree < MIN_AGREE:
        return 0, None, None

    # BB extreme filter
    if bb_up_v > bb_low_v:
        bb_pos = (price - bb_low_v) / (bb_up_v - bb_low_v)
        if direction == 'up' and bb_pos > BB_UP_TH:
            return 0, None, None
        if direction == 'down' and bb_pos < BB_DOWN_TH:
            return 0, None, None

    # 15m divergence filter
    idx_15 = _tf_idx(ind['_t15'], t5[i] - 10 * 60 * 1000)
    if idx_15 >= 0 and rsi15m_vals is not None and idx_15 < len(rsi15m_vals):
        r15 = rsi15m_vals[idx_15]
        if direction == 'up' and r15 < RSI15_UP_MAX:
            return 0, None, None
        if direction == 'down' and r15 > RSI15_DOWN_MIN:
            return 0, None, None

    return score, direction, regime


def compute_15m_rsi(candles15):
    """Compute RSI(7) on 15m data for the divergence filter."""
    c15 = [c[4] for c in candles15]
    return rsi(c15, 7)


# ---------------------------------------------------------------------------
# Backtest with given weights
# ---------------------------------------------------------------------------
def backtest_with_weights(weights, candles, ind, candles15):
    """Run backtest using parameterized weights. Returns (win_rate, trades, pnl)."""
    total = len(candles)
    wins = 0; losses = 0; pnl = 0.0
    cooldown = 0  # candles since last signal
    daily_count = defaultdict(int)
    MAX_DAILY = 50

    # Compute 15m RSI for divergence filter
    rsi15m_vals = compute_15m_rsi(candles15)

    for i in range(60, total - 2):
        if cooldown > 0:
            cooldown -= 1
            continue

        score, direction, regime = score_signal(i, candles, ind, weights, rsi15m_vals)
        if direction is None:
            continue

        abs_score = abs(score)
        if abs_score < THRESHOLD:
            continue

        # Daily limit
        dt = datetime.fromtimestamp(candles[i][0] / 1000)
        day_key = dt.strftime('%Y%m%d')
        if daily_count[day_key] >= MAX_DAILY:
            continue

        # Simulate execution: entry at candle i+1 open, settle at i+2 close
        entry = candles[i + 1][1]  # open
        settle = candles[min(i + 2, total - 1)][4]  # close

        if direction == 'up':
            win = settle > entry
        else:
            win = settle < entry

        if win:
            wins += 1
            pnl += 20  # 80% payout on $25
        else:
            losses += 1
            pnl -= 25

        daily_count[day_key] += 1
        cooldown = 2  # skip next candle (10 min cooldown)

    trades = wins + losses
    wr = wins / trades * 100 if trades > 0 else 0
    return wr, trades, pnl


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def objective(trial, candles, ind, candles15):
    """Optuna objective: maximize score = wr * log(trades) to balance quality & quantity."""
    w = {
        'rsi_extreme': trial.suggest_float('rsi_extreme', 1.0, 3.0),
        'rsi_moderate': trial.suggest_float('rsi_moderate', 0.5, 2.0),
        'rsi_mild': trial.suggest_float('rsi_mild', 0.1, 0.8),
        'stoch_extreme': trial.suggest_float('stoch_extreme', 1.0, 2.5),
        'stoch_moderate': trial.suggest_float('stoch_moderate', 0.2, 1.0),
        'stoch_cross': trial.suggest_float('stoch_cross', 0.3, 1.5),
        'mfi_extreme': trial.suggest_float('mfi_extreme', 1.0, 2.5),
        'mfi_moderate': trial.suggest_float('mfi_moderate', 0.2, 1.0),
        'cci_extreme': trial.suggest_float('cci_extreme', 1.0, 2.5),
        'cci_moderate': trial.suggest_float('cci_moderate', 0.3, 1.2),
        'wr_extreme': trial.suggest_float('wr_extreme', 0.8, 2.0),
        'wr_moderate': trial.suggest_float('wr_moderate', 0.2, 1.0),
        'sar': trial.suggest_float('sar', 0.2, 1.0),
        'aroon': trial.suggest_float('aroon', 0.2, 1.0),
        'ma_trend': trial.suggest_float('ma_trend', 0.3, 1.5),
        'ema_cross': trial.suggest_float('ema_cross', 0.1, 0.8),
        'kdj_cross': trial.suggest_float('kdj_cross', 0.3, 1.5),
        'kdj_j': trial.suggest_float('kdj_j', 0.2, 1.0),
        'bb_extreme': trial.suggest_float('bb_extreme', 0.5, 2.0),
        'bb_moderate': trial.suggest_float('bb_moderate', 0.2, 1.0),
        'volume': trial.suggest_float('volume', 0.2, 1.0),
        'hammer': trial.suggest_float('hammer', 0.5, 2.0),
        'engulfing': trial.suggest_float('engulfing', 0.5, 2.5),
        'rsi_divergence': trial.suggest_float('rsi_divergence', 0.3, 2.0),
        'trend_di': trial.suggest_float('trend_di', 0.3, 1.5),
        'trend_rsi': trial.suggest_float('trend_rsi', 0.1, 0.8),
        'range_bb': trial.suggest_float('range_bb', 0.2, 1.0),
    }

    wr, trades, pnl = backtest_with_weights(w, candles, ind, candles15)
    if trades < 50:  # Penalize too-few trades
        return -100.0
    return wr * (1 + 0.3 * (trades ** 0.5 - 1))  # Reward more trades, but sub-linearly


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 70)
    print("  Optuna Weight Optimization — V3 Strategy")
    print(f"  Train: {TRAIN_START} ~ {TRAIN_END} | Validate: {VAL_START} ~ {VAL_END}")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading training data...")
    candles, ind = _load_period(TRAIN_START, TRAIN_END)
    candles15_train = [c for c in load_all(SYMBOL, TRAIN_START, TRAIN_END)['15m']]
    print(f"  {len(candles)} 5m candles loaded")

    # Baseline
    print("\n[2/4] Baseline (current hardcoded weights)...")
    default_weights = {
        'rsi_extreme': 2.0, 'rsi_moderate': 1.2, 'rsi_mild': 0.3,
        'stoch_extreme': 1.5, 'stoch_moderate': 0.5, 'stoch_cross': 0.8,
        'mfi_extreme': 1.5, 'mfi_moderate': 0.5,
        'cci_extreme': 1.5, 'cci_moderate': 0.7,
        'wr_extreme': 1.2, 'wr_moderate': 0.5,
        'sar': 0.5, 'aroon': 0.5,
        'ma_trend': 0.8, 'ema_cross': 0.3,
        'kdj_cross': 0.8, 'kdj_j': 0.5,
        'bb_extreme': 1.2, 'bb_moderate': 0.5,
        'volume': 0.5,
        'hammer': 1.0, 'engulfing': 1.5,
        'rsi_divergence': 0.8,
        'trend_di': 0.8, 'trend_rsi': 0.3,
        'range_bb': 0.5,
    }
    wr_base, trades_base, pnl_base = backtest_with_weights(default_weights, candles, ind, candles15_train)
    print(f"  Baseline: {trades_base} trades, WR={wr_base:.1f}%, PnL=${pnl_base:+.1f}")

    # Optuna optimization
    print("\n[3/4] Running Optuna TPE optimization (100 trials, ~25 min)...")
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    def _objective(trial):
        return objective(trial, candles, ind, candles15_train)

    study.optimize(_objective, n_trials=100, show_progress_bar=True)

    best_weights = study.best_params
    best_wr, best_trades, best_pnl = backtest_with_weights(best_weights, candles, ind, candles15_train)
    print(f"\n  Best trial: {best_trades} trades, WR={best_wr:.1f}%, PnL=${best_pnl:+.1f}")
    print(f"  Improvement: WR {best_wr - wr_base:+.1f}%, Trades {best_trades - trades_base:+d}")

    # Print best weights vs default
    print(f"\n{'─'*60}")
    print(f"  {'Weight':<20} {'Default':>8} {'Optimal':>8} {'Delta':>8}")
    print(f"  {'─'*60}")
    for k in default_weights:
        d = default_weights[k]
        o = best_weights[k]
        delta = o - d
        marker = ' <<<' if abs(delta) > 0.3 else ''
        print(f"  {k:<20} {d:>8.2f} {o:>8.2f} {delta:>+8.2f}{marker}")
    print(f"  {'─'*60}")

    # Out-of-sample validation
    print(f"\n[4/4] Out-of-sample validation ({VAL_START} ~ {VAL_END})...")
    candles_val, ind_val = _load_period(VAL_START, VAL_END)
    candles15_val = [c for c in load_all(SYMBOL, VAL_START, VAL_END)['15m']]

    wr_oos_base, trades_oos_base, pnl_oos_base = backtest_with_weights(
        default_weights, candles_val, ind_val, candles15_val)
    wr_oos_opt, trades_oos_opt, pnl_oos_opt = backtest_with_weights(
        best_weights, candles_val, ind_val, candles15_val)

    print(f"\n  {'':<20} {'Trades':>6} {'WR':>7} {'PnL':>10}")
    print(f"  {'─'*45}")
    print(f"  {'Default (OOS)':<20} {trades_oos_base:>6} {wr_oos_base:>6.1f}% {pnl_oos_base:>+10.1f}")
    print(f"  {'Optimal (OOS)':<20} {trades_oos_opt:>6} {wr_oos_opt:>6.1f}% {pnl_oos_opt:>+10.1f}")
    print(f"  {'Delta':<20} {trades_oos_opt - trades_oos_base:>+6} {wr_oos_opt - wr_oos_base:>+6.1f}% {pnl_oos_opt - pnl_oos_base:>+10.1f}")

    # Recommendation
    print(f"\n{'='*70}")
    if wr_oos_opt >= wr_oos_base and pnl_oos_opt >= pnl_oos_base:
        print("  RESULT: Optimized weights IMPROVE out-of-sample performance.")
        print("  Recommend applying optimized weights to strategy_engine.py.")
    else:
        print("  RESULT: Optimized weights DO NOT improve out-of-sample performance.")
        print("  Stick with current hardcoded weights — they generalize better.")
    print(f"{'='*70}")
