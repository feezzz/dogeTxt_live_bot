"""
New approaches beyond what was tested in Rounds 1-3:
1. Trend-aligned only (only LONG in uptrend, only SHORT in downtrend)
2. Confirmation candle (wait for next candle to confirm before entry)
3. Dynamic exit (exit on RSI recovery instead of fixed 10m)
4. Volume climax filter (declining volume = exhaustion)
5. 15m primary signals with 5m entry timing
6. BB squeeze breakout (opposite paradigm)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from event_backtest.data_fetcher import load_all
from event_backtest.indicators import (
    rsi, bollinger_bands, adx, atr, atr_pct, volume_spike,
    cci, williams_r, stochastic_rsi, mfi, ema, bb_width,
)

def tf_idx(ts, t):
    for i in range(len(ts)-1,-1,-1):
        if ts[i] <= t: return i
    return -1


# ============================================================
# Approach 1: Trend-aligned only
# Only take LONG signals when 1h trend is bullish (DI+ > DI-)
# Only take SHORT when 1h trend is bearish
# ============================================================
def backtest_trend_aligned(symbol, start, end, conds, trend_strength=0):
    """Only trade in direction of 1h trend."""
    data = load_all(symbol, start, end)
    c5 = data['5m']
    cl = [c[4] for c in c5]; op = [c[1] for c in c5]
    hi = [c[2] for c in c5]; lo = [c[3] for c in c5]
    vol = [c[5] for c in c5]
    c1h = [c[4] for c in data['1h']]; h1h = [c[2] for c in data['1h']]; l1h = [c[3] for c in data['1h']]
    t5 = [c[0] for c in c5]; t1h = [c[0] for c in data['1h']]
    total = len(c5); warmup = 60

    rsi_period = conds.get('rsi_period', 7)
    rsi_vals = rsi(cl, rsi_period)
    _, bb_u, bb_l = bollinger_bands(cl, conds.get('bb_period', 20), 2.0)
    sk, sd = stochastic_rsi(cl, 14, 14)
    c14 = cci(hi, lo, cl, 14)
    wr14 = williams_r(hi, lo, cl, 14)
    adx_h, pdi, mdi = adx(h1h, l1h, c1h, 14)
    atr_p = atr_pct(atr(hi, lo, cl, 14), cl)
    vs = volume_spike(vol, 20, conds.get('vol_th', 1.5))
    min_atr = 0.05 if symbol == 'ETHUSDT' else 0.03

    LOWS = frozenset([22,23,0,1,2,3,4])
    eq = 500; trades = []; last_sig = -999; day_bets = 0; cur_day = None

    for i in range(warmup, total-2):
        ts = c5[i][0]; dt = datetime.fromtimestamp(ts/1000)
        if dt.day != cur_day: cur_day = dt.day; day_bets = 0
        if i - last_sig < 2: continue
        if day_bets >= 50: continue
        if eq <= 25: break
        if atr_p[i] < min_atr: continue
        i1h = tf_idx(t1h, t5[i] - 55*60*1000)
        if i1h < 20: continue
        if dt.hour in LOWS and day_bets >= 16: continue

        p = cl[i]; bb_p = (p-bb_l[i])/(bb_u[i]-bb_l[i]) if bb_u[i]>bb_l[i] else 0.5
        r1 = (cl[i]/cl[i-1]-1)*100 if i>0 and cl[i-1]>0 else 0
        r3 = (cl[i]/cl[i-3]-1)*100 if i>=3 and cl[i-3]>0 else 0
        adx_v = adx_h[i1h]; di_d = pdi[i1h]-mdi[i1h]

        rsi_th = conds.get('rsi', 20); bb_th = conds.get('bb', 0.08)
        stoch_th = conds.get('stoch', 10); ret1_th = conds.get('ret1', 0.02)
        adx_th = conds.get('adx', 40); cci_th = conds.get('cci', -150)
        wr_th = conds.get('wr', None); use_vol = conds.get('use_vol', False)

        direction = None

        # Only LONG if 1h trend is bullish
        if di_d > trend_strength:
            if (rsi_vals[i] < rsi_th and bb_p < bb_th and sk[i] < stoch_th
                and r1 < -ret1_th and adx_v < adx_th and c14[i] < cci_th
                and (wr_th is None or wr14[i] < wr_th)
                and (not use_vol or vs[i])):
                direction = 'up'

        # Only SHORT if 1h trend is bearish
        if direction is None and di_d < -trend_strength:
            if (rsi_vals[i] > (100-rsi_th) and bb_p > (1-bb_th) and sk[i] > (100-stoch_th)
                and r1 > ret1_th and adx_v < adx_th and c14[i] > -cci_th
                and (wr_th is None or wr14[i] > -wr_th)
                and (not use_vol or vs[i])):
                direction = 'down'

        if direction is None: continue

        entry = c5[i+1][1]; settle = c5[min(i+2,total-1)][4]
        win = (direction=='up' and settle>entry) or (direction=='down' and settle<entry)
        pnl = 20 if win else -25; eq += pnl
        trades.append({'time':dt,'dir':direction,'win':win,'pnl':pnl})
        last_sig = i; day_bets += 1

    n = len(trades); w = sum(1 for t in trades if t['win'])
    wr = w/n*100 if n else 0; pnl = sum(t['pnl'] for t in trades)
    return n, wr, pnl, eq


# ============================================================
# Approach 2: Confirmation candle
# After signal at candle i, wait for candle i+1 to close.
# Only enter if candle i+1 close confirms direction.
# ============================================================
def backtest_confirm_candle(symbol, start, end, conds):
    """Wait for next candle to confirm direction before entry."""
    data = load_all(symbol, start, end)
    c5 = data['5m']
    cl = [c[4] for c in c5]; op = [c[1] for c in c5]
    hi = [c[2] for c in c5]; lo = [c[3] for c in c5]
    vol = [c[5] for c in c5]
    c1h = [c[4] for c in data['1h']]; h1h = [c[2] for c in data['1h']]; l1h = [c[3] for c in data['1h']]
    t5 = [c[0] for c in c5]; t1h = [c[0] for c in data['1h']]
    total = len(c5); warmup = 60

    rsi_vals = rsi(cl, conds.get('rsi_period', 7))
    _, bb_u, bb_l = bollinger_bands(cl, conds.get('bb_period', 20), 2.0)
    sk, sd = stochastic_rsi(cl, 14, 14)
    c14 = cci(hi, lo, cl, 14)
    wr14 = williams_r(hi, lo, cl, 14)
    adx_h, pdi, mdi = adx(h1h, l1h, c1h, 14)
    atr_p = atr_pct(atr(hi, lo, cl, 14), cl)
    vs = volume_spike(vol, 20, conds.get('vol_th', 1.5))
    min_atr = 0.05 if symbol == 'ETHUSDT' else 0.03

    LOWS = frozenset([22,23,0,1,2,3,4])
    eq = 500; trades = []; last_sig = -999; day_bets = 0; cur_day = None

    for i in range(warmup, total-3):  # need 3 more candles for confirm+entry+exit
        ts = c5[i][0]; dt = datetime.fromtimestamp(ts/1000)
        if dt.day != cur_day: cur_day = dt.day; day_bets = 0
        if i - last_sig < 3: continue  # wider spacing due to delay
        if day_bets >= 50: continue
        if eq <= 25: break
        if atr_p[i] < min_atr: continue
        i1h = tf_idx(t1h, t5[i] - 55*60*1000)
        if i1h < 20: continue
        if dt.hour in LOWS and day_bets >= 16: continue

        p = cl[i]; bb_p = (p-bb_l[i])/(bb_u[i]-bb_l[i]) if bb_u[i]>bb_l[i] else 0.5
        r1 = (cl[i]/cl[i-1]-1)*100 if i>0 and cl[i-1]>0 else 0
        adx_v = adx_h[i1h]; di_d = pdi[i1h]-mdi[i1h]

        rsi_th = conds.get('rsi', 20); bb_th = conds.get('bb', 0.08)
        stoch_th = conds.get('stoch', 10); ret1_th = conds.get('ret1', 0.02)
        adx_th = conds.get('adx', 40); cci_th = conds.get('cci', -150)
        wr_th = conds.get('wr', None); use_vol = conds.get('use_vol', False)

        direction = None

        # Signal detection (same as baseline)
        if (rsi_vals[i] < rsi_th and bb_p < bb_th and sk[i] < stoch_th
            and r1 < -ret1_th and adx_v < adx_th and c14[i] < cci_th
            and (wr_th is None or wr14[i] < wr_th)
            and (not use_vol or vs[i])):
            direction = 'up'
        elif (rsi_vals[i] > (100-rsi_th) and bb_p > (1-bb_th) and sk[i] > (100-stoch_th)
            and r1 > ret1_th and adx_v < adx_th and c14[i] > -cci_th
            and (wr_th is None or wr14[i] > -wr_th)
            and (not use_vol or vs[i])):
            direction = 'down'

        if direction is None: continue

        # CONFIRMATION: candle i+1 must close in signal direction
        confirm_close = cl[i+1]
        if direction == 'up' and confirm_close <= cl[i]: continue  # didn't confirm
        if direction == 'down' and confirm_close >= cl[i]: continue

        # Enter at candle i+2 open, exit at candle i+3 close
        entry = c5[i+2][1]
        settle = c5[min(i+3, total-1)][4]
        win = (direction=='up' and settle>entry) or (direction=='down' and settle<entry)
        pnl = 20 if win else -25; eq += pnl
        trades.append({'time':dt,'dir':direction,'win':win,'pnl':pnl})
        last_sig = i; day_bets += 1

    n = len(trades); w = sum(1 for t in trades if t['win'])
    wr = w/n*100 if n else 0; pnl = sum(t['pnl'] for t in trades)
    return n, wr, pnl, eq


# ============================================================
# Approach 3: Dynamic exit on RSI recovery
# Exit when RSI recovers to neutral zone instead of fixed 10m hold
# ============================================================
def backtest_dynamic_exit(symbol, start, end, conds):
    """Exit on RSI recovery instead of fixed 10 minutes."""
    data = load_all(symbol, start, end)
    c5 = data['5m']
    cl = [c[4] for c in c5]; op = [c[1] for c in c5]
    hi = [c[2] for c in c5]; lo = [c[3] for c in c5]
    vol = [c[5] for c in c5]
    c1h = [c[4] for c in data['1h']]; h1h = [c[2] for c in data['1h']]; l1h = [c[3] for c in data['1h']]
    t5 = [c[0] for c in c5]; t1h = [c[0] for c in data['1h']]
    total = len(c5); warmup = 60

    rsi_period = conds.get('rsi_period', 7)
    rsi_vals = rsi(cl, rsi_period)
    _, bb_u, bb_l = bollinger_bands(cl, conds.get('bb_period', 20), 2.0)
    sk, sd = stochastic_rsi(cl, 14, 14)
    c14 = cci(hi, lo, cl, 14)
    wr14 = williams_r(hi, lo, cl, 14)
    adx_h, pdi, mdi = adx(h1h, l1h, c1h, 14)
    atr_p = atr_pct(atr(hi, lo, cl, 14), cl)
    vs = volume_spike(vol, 20, conds.get('vol_th', 1.5))
    min_atr = 0.05 if symbol == 'ETHUSDT' else 0.03

    LOWS = frozenset([22,23,0,1,2,3,4])
    eq = 500; trades = []; last_sig = -999; day_bets = 0; cur_day = None
    max_bars = conds.get('max_bars', 6)  # max hold time in 5m bars

    for i in range(warmup, total-max_bars-1):
        ts = c5[i][0]; dt = datetime.fromtimestamp(ts/1000)
        if dt.day != cur_day: cur_day = dt.day; day_bets = 0
        if i - last_sig < 2: continue
        if day_bets >= 50: continue
        if eq <= 25: break
        if atr_p[i] < min_atr: continue
        i1h = tf_idx(t1h, t5[i] - 55*60*1000)
        if i1h < 20: continue
        if dt.hour in LOWS and day_bets >= 16: continue

        p = cl[i]; bb_p = (p-bb_l[i])/(bb_u[i]-bb_l[i]) if bb_u[i]>bb_l[i] else 0.5
        r1 = (cl[i]/cl[i-1]-1)*100 if i>0 and cl[i-1]>0 else 0
        adx_v = adx_h[i1h]; di_d = pdi[i1h]-mdi[i1h]

        rsi_th = conds.get('rsi', 20); bb_th = conds.get('bb', 0.08)
        stoch_th = conds.get('stoch', 10); ret1_th = conds.get('ret1', 0.02)
        adx_th = conds.get('adx', 40); cci_th = conds.get('cci', -150)
        wr_th = conds.get('wr', None); use_vol = conds.get('use_vol', False)

        direction = None
        if (rsi_vals[i] < rsi_th and bb_p < bb_th and sk[i] < stoch_th
            and r1 < -ret1_th and adx_v < adx_th and c14[i] < cci_th
            and (wr_th is None or wr14[i] < wr_th)
            and (not use_vol or vs[i])):
            direction = 'up'
        elif (rsi_vals[i] > (100-rsi_th) and bb_p > (1-bb_th) and sk[i] > (100-stoch_th)
            and r1 > ret1_th and adx_v < adx_th and c14[i] > -cci_th
            and (wr_th is None or wr14[i] > -wr_th)
            and (not use_vol or vs[i])):
            direction = 'down'

        if direction is None: continue

        entry = c5[i+1][1]
        rsi_exit = conds.get('rsi_exit', 40)  # RSI level to exit

        # Find exit bar: when RSI recovers to neutral or max_bars reached
        exit_bar = min(i+2, total-1)
        for j in range(i+2, min(i+max_bars+1, total)):
            if direction == 'up' and rsi_vals[j] >= rsi_exit:
                exit_bar = j; break
            if direction == 'down' and rsi_vals[j] <= (100-rsi_exit):
                exit_bar = j; break
            exit_bar = j  # hold to max

        settle = c5[exit_bar][4]
        win = (direction=='up' and settle>entry) or (direction=='down' and settle<entry)
        pnl = 20 if win else -25; eq += pnl
        trades.append({'time':dt,'dir':direction,'win':win,'pnl':pnl,'bars':exit_bar-i})
        last_sig = i; day_bets += 1

    n = len(trades); w = sum(1 for t in trades if t['win'])
    wr = w/n*100 if n else 0; pnl = sum(t['pnl'] for t in trades)
    avg_bars = sum(t.get('bars',2) for t in trades)/n if n else 0
    return n, wr, pnl, eq, avg_bars


# ============================================================
# Approach 4: Volume climax / declining volume
# Only trade when volume is declining (selling/buying exhaustion)
# ============================================================
def backtest_vol_climax(symbol, start, end, conds):
    """Require declining volume (exhaustion) for entry."""
    data = load_all(symbol, start, end)
    c5 = data['5m']
    cl = [c[4] for c in c5]; op = [c[1] for c in c5]
    hi = [c[2] for c in c5]; lo = [c[3] for c in c5]
    vol = [c[5] for c in c5]
    c1h = [c[4] for c in data['1h']]; h1h = [c[2] for c in data['1h']]; l1h = [c[3] for c in data['1h']]
    t5 = [c[0] for c in c5]; t1h = [c[0] for c in data['1h']]
    total = len(c5); warmup = 60

    rsi_vals = rsi(cl, conds.get('rsi_period', 7))
    _, bb_u, bb_l = bollinger_bands(cl, conds.get('bb_period', 20), 2.0)
    sk, sd = stochastic_rsi(cl, 14, 14)
    c14 = cci(hi, lo, cl, 14)
    wr14 = williams_r(hi, lo, cl, 14)
    adx_h, pdi, mdi = adx(h1h, l1h, c1h, 14)
    atr_p = atr_pct(atr(hi, lo, cl, 14), cl)

    # Volume decline: current vol < average of last N bars
    vol_avg = [0.0]*total
    vp = conds.get('vol_period', 5)
    for i in range(vp, total):
        vol_avg[i] = sum(vol[i-vp:i]) / vp
    vol_declining = [vol[i] < vol_avg[i]*conds.get('vol_decline_ratio', 0.7) for i in range(total)]

    min_atr = 0.05 if symbol == 'ETHUSDT' else 0.03
    LOWS = frozenset([22,23,0,1,2,3,4])
    eq = 500; trades = []; last_sig = -999; day_bets = 0; cur_day = None

    for i in range(warmup, total-2):
        ts = c5[i][0]; dt = datetime.fromtimestamp(ts/1000)
        if dt.day != cur_day: cur_day = dt.day; day_bets = 0
        if i - last_sig < 2: continue
        if day_bets >= 50: continue
        if eq <= 25: break
        if atr_p[i] < min_atr: continue
        i1h = tf_idx(t1h, t5[i] - 55*60*1000)
        if i1h < 20: continue
        if dt.hour in LOWS and day_bets >= 16: continue

        p = cl[i]; bb_p = (p-bb_l[i])/(bb_u[i]-bb_l[i]) if bb_u[i]>bb_l[i] else 0.5
        r1 = (cl[i]/cl[i-1]-1)*100 if i>0 and cl[i-1]>0 else 0
        adx_v = adx_h[i1h]; di_d = pdi[i1h]-mdi[i1h]

        rsi_th = conds.get('rsi', 20); bb_th = conds.get('bb', 0.08)
        stoch_th = conds.get('stoch', 10); ret1_th = conds.get('ret1', 0.02)
        adx_th = conds.get('adx', 40); cci_th = conds.get('cci', -150)
        wr_th = conds.get('wr', None)

        # Volume climax check: volume must be declining (exhaustion)
        if not vol_declining[i]: continue

        direction = None
        if (rsi_vals[i] < rsi_th and bb_p < bb_th and sk[i] < stoch_th
            and r1 < -ret1_th and adx_v < adx_th and c14[i] < cci_th
            and (wr_th is None or wr14[i] < wr_th)):
            direction = 'up'
        elif (rsi_vals[i] > (100-rsi_th) and bb_p > (1-bb_th) and sk[i] > (100-stoch_th)
            and r1 > ret1_th and adx_v < adx_th and c14[i] > -cci_th
            and (wr_th is None or wr14[i] > -wr_th)):
            direction = 'down'

        if direction is None: continue

        entry = c5[i+1][1]; settle = c5[min(i+2,total-1)][4]
        win = (direction=='up' and settle>entry) or (direction=='down' and settle<entry)
        pnl = 20 if win else -25; eq += pnl
        trades.append({'time':dt,'dir':direction,'win':win,'pnl':pnl})
        last_sig = i; day_bets += 1

    n = len(trades); w = sum(1 for t in trades if t['win'])
    wr = w/n*100 if n else 0; pnl = sum(t['pnl'] for t in trades)
    return n, wr, pnl, eq


# ============================================================
# Approach 5: 15m primary signal + 5m entry timing
# Signal on 15m extreme, enter on 5m confirmation
# ============================================================
def backtest_15m_primary(symbol, start, end, conds):
    """Use 15m candles for signal, 5m for entry timing."""
    data = load_all(symbol, start, end)
    c5 = data['5m']; c15 = data['15m']
    cl5 = [c[4] for c in c5]
    cl15 = [c[4] for c in c15]; hi15 = [c[2] for c in c15]; lo15 = [c[3] for c in c15]
    c1h = [c[4] for c in data['1h']]; h1h = [c[2] for c in data['1h']]; l1h = [c[3] for c in data['1h']]
    t5 = [c[0] for c in c5]; t15 = [c[0] for c in c15]; t1h = [c[0] for c in data['1h']]
    total = len(c5); warmup = 60

    rsi15 = rsi(cl15, conds.get('rsi_period', 7))
    _, bb_u15, bb_l15 = bollinger_bands(cl15, 20, 2.0)
    sk15, sd15 = stochastic_rsi(cl15, 14, 14)
    c15_cci = cci(hi15, lo15, cl15, 14)
    wr15 = williams_r(hi15, lo15, cl15, 14)
    adx_h, pdi, mdi = adx(h1h, l1h, c1h, 14)
    atr_p = atr_pct(atr(hi15, lo15, cl15, 14), cl15)
    min_atr = 0.05 if symbol == 'ETHUSDT' else 0.03

    LOWS = frozenset([22,23,0,1,2,3,4])
    eq = 500; trades = []; last_sig = -999; day_bets = 0; cur_day = None

    for i5 in range(warmup, total-2):
        ts = c5[i5][0]; dt = datetime.fromtimestamp(ts/1000)
        if dt.day != cur_day: cur_day = dt.day; day_bets = 0
        if i5 - last_sig < 2: continue
        if day_bets >= 50: continue
        if eq <= 25: break

        # Find corresponding 15m candle
        i15 = tf_idx(t15, t5[i5] - 10*60*1000)  # no future leak
        if i15 < 20: continue
        if atr_p[i15] < min_atr: continue
        i1h = tf_idx(t1h, t5[i5] - 55*60*1000)
        if i1h < 20: continue
        if dt.hour in LOWS and day_bets >= 16: continue

        p15 = cl15[i15]
        bb_p = (p15-bb_l15[i15])/(bb_u15[i15]-bb_l15[i15]) if bb_u15[i15]>bb_l15[i15] else 0.5
        r1 = (cl15[i15]/cl15[i15-1]-1)*100 if i15>0 and cl15[i15-1]>0 else 0
        adx_v = adx_h[i1h]; di_d = pdi[i1h]-mdi[i1h]

        rsi_th = conds.get('rsi', 20); bb_th = conds.get('bb', 0.08)
        stoch_th = conds.get('stoch', 10); ret1_th = conds.get('ret1', 0.02)
        adx_th = conds.get('adx', 40); cci_th = conds.get('cci', -150)

        direction = None
        if (rsi15[i15] < rsi_th and bb_p < bb_th and sk15[i15] < stoch_th
            and r1 < -ret1_th and adx_v < adx_th and c15_cci[i15] < cci_th
            and wr15[i15] < conds.get('wr', -85)):
            direction = 'up'
        elif (rsi15[i15] > (100-rsi_th) and bb_p > (1-bb_th) and sk15[i15] > (100-stoch_th)
            and r1 > ret1_th and adx_v < adx_th and c15_cci[i15] > -cci_th
            and wr15[i15] > -conds.get('wr', -85)):
            direction = 'down'

        if direction is None: continue

        entry = c5[i5+1][1]; settle = c5[min(i5+2,total-1)][4]
        win = (direction=='up' and settle>entry) or (direction=='down' and settle<entry)
        pnl = 20 if win else -25; eq += pnl
        trades.append({'time':dt,'dir':direction,'win':win,'pnl':pnl})
        last_sig = i5; day_bets += 1

    n = len(trades); w = sum(1 for t in trades if t['win'])
    wr = w/n*100 if n else 0; pnl = sum(t['pnl'] for t in trades)
    return n, wr, pnl, eq


# ============================================================
# Approach 6: EMA pullback in trend
# In uptrend: wait for price to pull back to EMA before buying
# In downtrend: wait for price to rally to EMA before selling
# ============================================================
def backtest_ema_pullback(symbol, start, end, conds):
    """Trade pullbacks to EMA in direction of 1h trend."""
    data = load_all(symbol, start, end)
    c5 = data['5m']
    cl = [c[4] for c in c5]; op = [c[1] for c in c5]
    hi = [c[2] for c in c5]; lo = [c[3] for c in c5]
    vol = [c[5] for c in c5]
    c1h = [c[4] for c in data['1h']]; h1h = [c[2] for c in data['1h']]; l1h = [c[3] for c in data['1h']]
    t5 = [c[0] for c in c5]; t1h = [c[0] for c in data['1h']]
    total = len(c5); warmup = 60

    rsi_vals = rsi(cl, 7)
    ema20 = ema(cl, 20)
    adx_h, pdi, mdi = adx(h1h, l1h, c1h, 14)
    atr_p = atr_pct(atr(hi, lo, cl, 14), cl)
    min_atr = 0.05 if symbol == 'ETHUSDT' else 0.03

    LOWS = frozenset([22,23,0,1,2,3,4])
    eq = 500; trades = []; last_sig = -999; day_bets = 0; cur_day = None

    for i in range(warmup, total-2):
        ts = c5[i][0]; dt = datetime.fromtimestamp(ts/1000)
        if dt.day != cur_day: cur_day = dt.day; day_bets = 0
        if i - last_sig < 2: continue
        if day_bets >= 50: continue
        if eq <= 25: break
        if atr_p[i] < min_atr: continue
        i1h = tf_idx(t1h, t5[i] - 55*60*1000)
        if i1h < 20: continue
        if dt.hour in LOWS and day_bets >= 16: continue

        adx_v = adx_h[i1h]; di_d = pdi[i1h]-mdi[i1h]
        p = cl[i]; dist_ema = (p-ema20[i])/ema20[i]*100

        direction = None

        # LONG: 1h uptrend, price pulled back to/below EMA, RSI oversold
        if di_d > conds.get('di_min', 3) and dist_ema < conds.get('ema_dist', 0.5):
            if rsi_vals[i] < conds.get('rsi_buy', 35):
                direction = 'up'

        # SHORT: 1h downtrend, price rallied to/above EMA, RSI overbought
        if direction is None and di_d < -conds.get('di_min', 3):
            if dist_ema > -conds.get('ema_dist', 0.5):
                if rsi_vals[i] > conds.get('rsi_sell', 65):
                    direction = 'down'

        if direction is None: continue

        entry = c5[i+1][1]; settle = c5[min(i+2,total-1)][4]
        win = (direction=='up' and settle>entry) or (direction=='down' and settle<entry)
        pnl = 20 if win else -25; eq += pnl
        trades.append({'time':dt,'dir':direction,'win':win,'pnl':pnl})
        last_sig = i; day_bets += 1

    n = len(trades); w = sum(1 for t in trades if t['win'])
    wr = w/n*100 if n else 0; pnl = sum(t['pnl'] for t in trades)
    return n, wr, pnl, eq


# ============================================================
# Run all approaches on 2026H1 first, then validate best on all 5 periods
# ============================================================
PERIODS = [
    ('2024-01-01', '2024-07-01', '2024H1'),
    ('2024-07-01', '2025-01-01', '2024H2'),
    ('2025-01-01', '2025-07-01', '2025H1'),
    ('2025-07-01', '2026-01-01', '2025H2'),
    ('2026-01-01', '2026-07-01', '2026H1'),
]

BASELINE = dict(rsi=20, bb=0.08, stoch=10, ret1=0.02, adx=40, cci=-150)

def validate_5p(backtest_fn, symbol, conds, label, **extra):
    """Run across all 5 periods."""
    total_n = 0; total_w = 0; total_pnl = 0.0
    print(f"\n  [{label}] {symbol}")
    for start, end, pn in PERIODS:
        result = backtest_fn(symbol, start, end, conds, **extra)
        if len(result) == 5:
            n, wr, pnl, eq, extra_info = result
        else:
            n, wr, pnl, eq = result
        total_n += n; total_w += n*wr/100 if n else 0; total_pnl += pnl
        print(f"    {pn}: {n:>4d} trades  {wr:>5.1f}% WR  ${pnl:>+8.0f}")
    total_wr = total_w/total_n*100 if total_n else 0
    star = ' ***' if total_wr >= 65 else (' **' if total_wr >= 60 else '')
    print(f"    TOTAL: {total_n} trades  {total_wr:.1f}% WR  ${total_pnl:+.0f}{star}")
    return total_n, total_wr, total_pnl


print(f"\n{'='*100}")
print(f"  NEW APPROACHES — Finding 65%+ WR")
print(f"  First: Quick scan on 2026H1 | Then: 5-period validation for best")
print(f"{'='*100}")

SYM = 'ETHUSDT'; P = ('2026-01-01', '2026-07-01')

# ============================================================
# Phase 1: Quick scan on 2026H1
# ============================================================
print(f"\n{'─'*100}")
print(f"  PHASE 1: Quick scan on 2026H1 ETHUSDT")
print(f"{'─'*100}")

# Baseline reference
n, wr, pnl, eq = backtest_trend_aligned(SYM, P[0], P[1], BASELINE, trend_strength=0)
print(f"  {'Baseline (no trend filter)':<45} {n:>5} {wr:>6.1f}% ${pnl:>+9.0f}")

# --- Approach 1: Trend-aligned ---
print(f"\n  --- Trend-Aligned (only trade in 1h trend direction) ---")
for ts in [0, 2, 5, 8]:
    n, wr, pnl, eq = backtest_trend_aligned(SYM, P[0], P[1], BASELINE, trend_strength=ts)
    star = ' ***' if wr >= 65 else (' **' if wr >= 60 else '')
    print(f"  {'  trend_strength=' + str(ts):<45} {n:>5} {wr:>6.1f}% ${pnl:>+9.0f}{star}")

# --- Approach 2: Confirmation candle ---
print(f"\n  --- Confirmation Candle (wait for next candle close) ---")
n, wr, pnl, eq = backtest_confirm_candle(SYM, P[0], P[1], BASELINE)
star = ' ***' if wr >= 65 else (' **' if wr >= 60 else '')
print(f"  {'  confirm_candle':<45} {n:>5} {wr:>6.1f}% ${pnl:>+9.0f}{star}")

# With WR filter
for wr_th in [-85, -90]:
    conds = {**BASELINE, 'wr': wr_th}
    n, wr, pnl, eq = backtest_confirm_candle(SYM, P[0], P[1], conds)
    star = ' ***' if wr >= 65 else (' **' if wr >= 60 else '')
    print(f"  {'  confirm_candle +WR<' + str(wr_th):<45} {n:>5} {wr:>6.1f}% ${pnl:>+9.0f}{star}")

# --- Approach 3: Dynamic exit ---
print(f"\n  --- Dynamic Exit (RSI recovery) ---")
for rsi_exit in [35, 40, 45, 50]:
    for max_bars in [3, 5, 8]:
        n, wr, pnl, eq, avg_bars = backtest_dynamic_exit(SYM, P[0], P[1],
            {**BASELINE, 'rsi_exit': rsi_exit, 'max_bars': max_bars})
        if n < 10: continue
        star = ' ***' if wr >= 65 else (' **' if wr >= 60 else '')
        print(f"  {'  rsi_exit=' + str(rsi_exit) + ' max=' + str(max_bars):<45} {n:>5} {wr:>6.1f}% ${pnl:>+9.0f} avg{avg_bars:.1f}bar{star}")

# --- Approach 4: Volume climax ---
print(f"\n  --- Volume Climax (declining volume = exhaustion) ---")
for ratio in [0.5, 0.7, 0.9]:
    for vp in [3, 5]:
        conds = {**BASELINE, 'vol_decline_ratio': ratio, 'vol_period': vp}
        n, wr, pnl, eq = backtest_vol_climax(SYM, P[0], P[1], conds)
        star = ' ***' if wr >= 65 else (' **' if wr >= 60 else '')
        print(f"  {'  vol<' + str(ratio) + '*avg' + str(vp):<45} {n:>5} {wr:>6.1f}% ${pnl:>+9.0f}{star}")

# Add WR to best vol climax
conds = {**BASELINE, 'wr': -85, 'vol_decline_ratio': 0.7, 'vol_period': 5}
n, wr, pnl, eq = backtest_vol_climax(SYM, P[0], P[1], conds)
star = ' ***' if wr >= 65 else (' **' if wr >= 60 else '')
print(f"  {'  +WR<-85 vol<0.7*avg5':<45} {n:>5} {wr:>6.1f}% ${pnl:>+9.0f}{star}")

# --- Approach 5: 15m primary ---
print(f"\n  --- 15m Primary Signal ---")
for rsi_th in [20, 25]:
    for wr_th in [-85, -90]:
        conds = dict(rsi=rsi_th, bb=0.08, stoch=10, ret1=0.02, adx=40, cci=-150, wr=wr_th)
        n, wr, pnl, eq = backtest_15m_primary(SYM, P[0], P[1], conds)
        star = ' ***' if wr >= 65 else (' **' if wr >= 60 else '')
        print(f"  {'  rsi<' + str(rsi_th) + ' WR<' + str(wr_th):<45} {n:>5} {wr:>6.1f}% ${pnl:>+9.0f}{star}")

# --- Approach 6: EMA pullback ---
print(f"\n  --- EMA Pullback ---")
for di_min in [3, 5, 8]:
    for rsi_buy in [30, 35, 40]:
        conds = dict(di_min=di_min, ema_dist=0.5, rsi_buy=rsi_buy, rsi_sell=100-rsi_buy)
        n, wr, pnl, eq = backtest_ema_pullback(SYM, P[0], P[1], conds)
        if n < 10: continue
        star = ' ***' if wr >= 65 else (' **' if wr >= 60 else '')
        print(f"  {'  di>' + str(di_min) + ' rsi<' + str(rsi_buy) + ' pullback':<45} {n:>5} {wr:>6.1f}% ${pnl:>+9.0f}{star}")

print(f"\n{'='*100}")
print(f"  PHASE 1 COMPLETE — Now validate best approaches across all 5 periods")
print(f"{'='*100}")
