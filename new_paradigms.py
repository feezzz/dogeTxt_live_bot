"""
New strategy paradigms for 10-min event contracts:
1. ADAPTIVE: trend-following in trends, mean reversion in ranges
2. Triple-timeframe RSI: 5m+15m+1h all extreme
3. Keltner Channel (ATR-based) mean reversion
4. BB Squeeze breakout: low vol contraction -> expansion entry
5. VWAP extreme deviation with RSI confirmation
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from event_backtest.data_fetcher import load_all
from event_backtest.indicators import (
    rsi, bollinger_bands, adx, atr, atr_pct, volume_spike,
    cci, williams_r, stochastic_rsi, mfi, ema, sma, bb_width, vwap,
)

def tf_idx(ts, t):
    for i in range(len(ts)-1,-1,-1):
        if ts[i] <= t: return i
    return -1

PERIODS = [
    ('2024-01-01', '2024-07-01', '2024H1'),
    ('2024-07-01', '2025-01-01', '2024H2'),
    ('2025-01-01', '2025-07-01', '2025H1'),
    ('2025-07-01', '2026-01-01', '2025H2'),
    ('2026-01-01', '2026-07-01', '2026H1'),
]


def keltner(highs, lows, closes, period=20, atr_mult=2.0):
    """Keltner Channel: EMA middle, ATR-based bands."""
    n = len(closes)
    middle = ema(closes, period)
    atr_vals = atr(highs, lows, closes, period)
    upper = [middle[i] + atr_mult * atr_vals[i] for i in range(n)]
    lower = [middle[i] - atr_mult * atr_vals[i] for i in range(n)]
    return middle, upper, lower


# ================================================================
# Approach 1: ADAPTIVE — trend-following vs mean-reversion by regime
# ================================================================
def backtest_adaptive(symbol, start, end, conds):
    """If 1h ADX>30 and trend clear: trend-following. Else: mean reversion."""
    data = load_all(symbol, start, end)
    c5 = data['5m']
    cl5 = [c[4] for c in c5]; op5 = [c[1] for c in c5]
    hi5 = [c[2] for c in c5]; lo5 = [c[3] for c in c5]
    vol5 = [c[5] for c in c5]
    c1h = [c[4] for c in data['1h']]; h1h = [c[2] for c in data['1h']]; l1h = [c[3] for c in data['1h']]
    t5 = [c[0] for c in c5]; t1h = [c[0] for c in data['1h']]
    total = len(c5); warmup = 60

    rsi7 = rsi(cl5, 7)
    _, bb_u, bb_l = bollinger_bands(cl5, 20, 2.0)
    sk, sd = stochastic_rsi(cl5, 14, 14)
    c14 = cci(hi5, lo5, cl5, 14)
    wr14 = williams_r(hi5, lo5, cl5, 14)
    adx_h, pdi, mdi = adx(h1h, l1h, c1h, 14)
    atr_p5 = atr_pct(atr(hi5, lo5, cl5, 14), cl5)
    ema20_5 = ema(cl5, 20)
    ema50_5 = ema(cl5, 50)
    min_atr = 0.05 if symbol == 'ETHUSDT' else 0.03

    LOWS = frozenset([22,23,0,1,2,3,4])
    eq = 500; trades = []; last_sig = -999; day_bets = 0; cur_day = None

    for i in range(warmup, total-2):
        ts = c5[i][0]; dt = datetime.fromtimestamp(ts/1000)
        if dt.day != cur_day: cur_day = dt.day; day_bets = 0
        if i - last_sig < 2: continue
        if day_bets >= 50: continue
        if eq <= 25: break
        if atr_p5[i] < min_atr: continue
        i1h = tf_idx(t1h, t5[i] - 55*60*1000)
        if i1h < 20: continue
        if dt.hour in LOWS and day_bets >= 16: continue

        p = cl5[i]; bb_p = (p-bb_l[i])/(bb_u[i]-bb_l[i]) if bb_u[i]>bb_l[i] else 0.5
        adx_v = adx_h[i1h]; di_d = pdi[i1h]-mdi[i1h]
        r1 = (cl5[i]/cl5[i-1]-1)*100 if i>0 and cl5[i-1]>0 else 0

        direction = None
        trend_strength = abs(di_d)
        trend_adx = conds.get('trend_adx', 30)

        if adx_v > trend_adx and trend_strength > conds.get('trend_di', 8):
            # === TRENDING: trend-following ===
            # Buy pullbacks in uptrend, sell rallies in downtrend
            if di_d > conds.get('trend_di', 8):
                # Uptrend: buy pullback to EMA20 with RSI oversold
                if p <= ema20_5[i]*1.002 and rsi7[i] < conds.get('trend_rsi', 40):
                    if cl5[i] > op5[i]:
                        direction = 'up'
            elif di_d < -conds.get('trend_di', 8):
                # Downtrend: sell rally to EMA20 with RSI overbought
                if p >= ema20_5[i]*0.998 and rsi7[i] > conds.get('trend_rsi_sell', 60):
                    if cl5[i] < op5[i]:
                        direction = 'down'
        else:
            # === RANGING: mean reversion ===
            mr_rsi = conds.get('mr_rsi', 20); mr_bb = conds.get('mr_bb', 0.08)
            mr_stoch = conds.get('mr_stoch', 10); mr_ret1 = conds.get('mr_ret1', 0.02)
            mr_cci = conds.get('mr_cci', -150)

            if (rsi7[i] < mr_rsi and bb_p < mr_bb and sk[i] < mr_stoch
                and r1 < -mr_ret1 and adx_v < conds.get('mr_adx', 40)
                and c14[i] < mr_cci and wr14[i] < conds.get('mr_wr', -85)):
                direction = 'up'
            elif (rsi7[i] > (100-mr_rsi) and bb_p > (1-mr_bb) and sk[i] > (100-mr_stoch)
                and r1 > mr_ret1 and adx_v < conds.get('mr_adx', 40)
                and c14[i] > -mr_cci and wr14[i] > -conds.get('mr_wr', -85)):
                direction = 'down'

        if direction is None: continue

        entry = c5[i+1][1]; settle = c5[min(i+2,total-1)][4]
        win = (direction=='up' and settle>entry) or (direction=='down' and settle<entry)
        pnl = 20 if win else -25; eq += pnl
        trades.append({'time':dt,'dir':direction,'win':win,'pnl':pnl,'regime':'trend' if adx_v>trend_adx else 'range'})
        last_sig = i; day_bets += 1

    n = len(trades); w = sum(1 for t in trades if t['win'])
    wr = w/n*100 if n else 0; pnl = sum(t['pnl'] for t in trades)
    trend_n = sum(1 for t in trades if t.get('regime')=='trend')
    return n, wr, pnl, eq, trend_n


# ================================================================
# Approach 2: Triple-TF RSI extreme
# RSI must be extreme on 5m, 15m, AND 1h simultaneously
# ================================================================
def backtest_triple_rsi(symbol, start, end, conds):
    """Require RSI extreme on all three timeframes."""
    data = load_all(symbol, start, end)
    c5 = data['5m']; c15 = data['15m']
    cl5 = [c[4] for c in c5]
    cl15 = [c[4] for c in c15]; cl1h = [c[4] for c in data['1h']]
    hi5 = [c[2] for c in c5]; lo5 = [c[3] for c in c5]
    h1h = [c[2] for c in data['1h']]; l1h = [c[3] for c in data['1h']]
    t5 = [c[0] for c in c5]; t15 = [c[0] for c in c15]; t1h = [c[0] for c in data['1h']]
    total = len(c5); warmup = 60

    rsi5 = rsi(cl5, conds.get('rsi_period', 7))
    rsi15 = rsi(cl15, conds.get('rsi_period', 7))
    rsi1h = rsi(cl1h, conds.get('rsi_period', 7))

    # Additional 5m confirmations
    _, bb_u, bb_l = bollinger_bands(cl5, 20, 2.0)
    sk, sd = stochastic_rsi(cl5, 14, 14)
    c14 = cci(hi5, lo5, cl5, 14)
    wr14 = williams_r(hi5, lo5, cl5, 14)
    adx_h, pdi, mdi = adx(h1h, l1h, cl1h, 14)
    atr_p5 = atr_pct(atr(hi5, lo5, cl5, 14), cl5)
    min_atr = 0.05 if symbol == 'ETHUSDT' else 0.03

    LOWS = frozenset([22,23,0,1,2,3,4])
    eq = 500; trades = []; last_sig = -999; day_bets = 0; cur_day = None

    for i in range(warmup, total-2):
        ts = c5[i][0]; dt = datetime.fromtimestamp(ts/1000)
        if dt.day != cur_day: cur_day = dt.day; day_bets = 0
        if i - last_sig < 2: continue
        if day_bets >= 50: continue
        if eq <= 25: break
        if atr_p5[i] < min_atr: continue

        i15 = tf_idx(t15, t5[i] - 10*60*1000)
        i1h = tf_idx(t1h, t5[i] - 55*60*1000)
        if i15 < 20 or i1h < 20: continue
        if dt.hour in LOWS and day_bets >= 16: continue

        p = cl5[i]; bb_p = (p-bb_l[i])/(bb_u[i]-bb_l[i]) if bb_u[i]>bb_l[i] else 0.5
        r1 = (cl5[i]/cl5[i-1]-1)*100 if i>0 and cl5[i-1]>0 else 0
        adx_v = adx_h[i1h]; di_d = pdi[i1h]-mdi[i1h]

        direction = None
        rsi_ext = conds.get('rsi_ext', 25)
        rsi_ext_high = 100 - rsi_ext

        # All 3 timeframes must show oversold
        if (rsi5[i] < rsi_ext and rsi15[i15] < rsi_ext and rsi1h[i1h] < rsi_ext
            and bb_p < conds.get('bb', 0.12) and sk[i] < conds.get('stoch', 15)
            and adx_v < conds.get('adx', 45)):
            direction = 'up'
        elif (rsi5[i] > rsi_ext_high and rsi15[i15] > rsi_ext_high and rsi1h[i1h] > rsi_ext_high
            and bb_p > (1-conds.get('bb', 0.12)) and sk[i] > (100-conds.get('stoch', 15))
            and adx_v < conds.get('adx', 45)):
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


# ================================================================
# Approach 3: Keltner Channel mean reversion
# ================================================================
def backtest_keltner(symbol, start, end, conds):
    """Keltner Channel based mean reversion."""
    data = load_all(symbol, start, end)
    c5 = data['5m']
    cl = [c[4] for c in c5]; op = [c[1] for c in c5]
    hi = [c[2] for c in c5]; lo = [c[3] for c in c5]
    vol = [c[5] for c in c5]
    c1h = [c[4] for c in data['1h']]; h1h = [c[2] for c in data['1h']]; l1h = [c[3] for c in data['1h']]
    t5 = [c[0] for c in c5]; t1h = [c[0] for c in data['1h']]
    total = len(c5); warmup = 60

    kc_mid, kc_up, kc_low = keltner(hi, lo, cl,
        period=conds.get('kc_period', 20), atr_mult=conds.get('kc_mult', 2.0))
    rsi7 = rsi(cl, 7)
    sk, sd = stochastic_rsi(cl, 14, 14)
    adx_h, pdi, mdi = adx(h1h, l1h, c1h, 14)
    atr_p5 = atr_pct(atr(hi, lo, cl, 14), cl)
    min_atr = 0.05 if symbol == 'ETHUSDT' else 0.03

    # KC position
    kc_pos = [(cl[i] - kc_low[i]) / (kc_up[i] - kc_low[i]) if kc_up[i] > kc_low[i] else 0.5
              for i in range(total)]
    # KC width (volatility measure)
    kc_width = [(kc_up[i] - kc_low[i]) / kc_mid[i] * 100 if kc_mid[i] > 0 else 0
                for i in range(total)]

    LOWS = frozenset([22,23,0,1,2,3,4])
    eq = 500; trades = []; last_sig = -999; day_bets = 0; cur_day = None

    for i in range(warmup, total-2):
        ts = c5[i][0]; dt = datetime.fromtimestamp(ts/1000)
        if dt.day != cur_day: cur_day = dt.day; day_bets = 0
        if i - last_sig < 2: continue
        if day_bets >= 50: continue
        if eq <= 25: break
        if atr_p5[i] < min_atr: continue
        i1h = tf_idx(t1h, t5[i] - 55*60*1000)
        if i1h < 20: continue
        if dt.hour in LOWS and day_bets >= 16: continue

        adx_v = adx_h[i1h]; di_d = pdi[i1h]-mdi[i1h]

        direction = None

        # LONG: price at/below KC lower band + RSI oversold
        if (kc_pos[i] < conds.get('kc_low', 0.05) and rsi7[i] < conds.get('rsi', 25)
            and sk[i] < conds.get('stoch', 15) and adx_v < conds.get('adx', 40)):
            direction = 'up'
        # SHORT: price at/above KC upper band + RSI overbought
        elif (kc_pos[i] > conds.get('kc_high', 0.95) and rsi7[i] > conds.get('rsi_sell', 75)
            and sk[i] > conds.get('stoch_sell', 85) and adx_v < conds.get('adx', 40)):
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


# ================================================================
# Approach 4: BB Squeeze breakout
# Entry when BB narrows (low vol), then price breaks out with volume
# ================================================================
def backtest_bb_squeeze(symbol, start, end, conds):
    """BB squeeze: enter on breakout after low volatility contraction."""
    data = load_all(symbol, start, end)
    c5 = data['5m']
    cl = [c[4] for c in c5]; op = [c[1] for c in c5]
    hi = [c[2] for c in c5]; lo = [c[3] for c in c5]
    vol = [c[5] for c in c5]
    c1h = [c[4] for c in data['1h']]; h1h = [c[2] for c in data['1h']]; l1h = [c[3] for c in data['1h']]
    t5 = [c[0] for c in c5]; t1h = [c[0] for c in data['1h']]
    total = len(c5); warmup = 60

    _, bb_u, bb_l = bollinger_bands(cl, conds.get('bb_period', 20), conds.get('bb_std', 2.0))
    bbw = bb_width(bb_u, bb_l, sma(cl, conds.get('bb_period', 20)))
    rsi7 = rsi(cl, 7)
    adx_h, pdi, mdi = adx(h1h, l1h, c1h, 14)
    atr_p5 = atr_pct(atr(hi, lo, cl, 14), cl)
    vs = volume_spike(vol, 20, conds.get('vol_th', 1.3))
    ema20 = ema(cl, 20)
    min_atr = 0.05 if symbol == 'ETHUSDT' else 0.03

    # BB squeeze: BBW at N-period low
    squeeze_period = conds.get('squeeze_lookback', 20)
    is_squeeze = [False]*total
    for i in range(squeeze_period, total):
        recent_bbw = bbw[i-squeeze_period:i+1]
        is_squeeze[i] = bbw[i] <= sorted(recent_bbw)[int(len(recent_bbw)*conds.get('squeeze_pct', 0.15))]

    LOWS = frozenset([22,23,0,1,2,3,4])
    eq = 500; trades = []; last_sig = -999; day_bets = 0; cur_day = None

    for i in range(warmup, total-2):
        ts = c5[i][0]; dt = datetime.fromtimestamp(ts/1000)
        if dt.day != cur_day: cur_day = dt.day; day_bets = 0
        if i - last_sig < 2: continue
        if day_bets >= 50: continue
        if eq <= 25: break
        if atr_p5[i] < min_atr: continue
        i1h = tf_idx(t1h, t5[i] - 55*60*1000)
        if i1h < 20: continue
        if dt.hour in LOWS and day_bets >= 16: continue

        di_d = pdi[i1h]-mdi[i1h]; p = cl[i]

        direction = None

        # LONG: squeeze + bullish candle breaking above EMA20 + volume
        if (is_squeeze[i] and cl[i] > op[i] and cl[i] > ema20[i]
            and vs[i] and di_d > conds.get('di_min', -3)):
            direction = 'up'
        # SHORT: squeeze + bearish candle breaking below EMA20 + volume
        elif (is_squeeze[i] and cl[i] < op[i] and cl[i] < ema20[i]
            and vs[i] and di_d < conds.get('di_max', 3)):
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


# ================================================================
# Approach 5: VWAP extreme deviation + RSI
# ================================================================
def backtest_vwap_extreme(symbol, start, end, conds):
    """Enter when price is far from VWAP AND RSI confirms extreme."""
    data = load_all(symbol, start, end)
    c5 = data['5m']
    cl = [c[4] for c in c5]; op = [c[1] for c in c5]
    hi = [c[2] for c in c5]; lo = [c[3] for c in c5]
    vol = [c[5] for c in c5]
    c1h = [c[4] for c in data['1h']]; h1h = [c[2] for c in data['1h']]; l1h = [c[3] for c in data['1h']]
    t5 = [c[0] for c in c5]; t1h = [c[0] for c in data['1h']]
    total = len(c5); warmup = 60

    vwap_vals = vwap(hi, lo, cl, vol)
    rsi7 = rsi(cl, 7)
    sk, sd = stochastic_rsi(cl, 14, 14)
    adx_h, pdi, mdi = adx(h1h, l1h, c1h, 14)
    atr_p5 = atr_pct(atr(hi, lo, cl, 14), cl)
    min_atr = 0.05 if symbol == 'ETHUSDT' else 0.03

    # VWAP deviation %
    vwap_dev = [(cl[i] - vwap_vals[i]) / vwap_vals[i] * 100 if vwap_vals[i] > 0 else 0
                for i in range(total)]

    LOWS = frozenset([22,23,0,1,2,3,4])
    eq = 500; trades = []; last_sig = -999; day_bets = 0; cur_day = None

    for i in range(warmup, total-2):
        ts = c5[i][0]; dt = datetime.fromtimestamp(ts/1000)
        if dt.day != cur_day: cur_day = dt.day; day_bets = 0
        if i - last_sig < 2: continue
        if day_bets >= 50: continue
        if eq <= 25: break
        if atr_p5[i] < min_atr: continue
        i1h = tf_idx(t1h, t5[i] - 55*60*1000)
        if i1h < 20: continue
        if dt.hour in LOWS and day_bets >= 16: continue

        adx_v = adx_h[i1h]; di_d = pdi[i1h]-mdi[i1h]

        direction = None

        # LONG: price significantly below VWAP + RSI oversold
        if (vwap_dev[i] < -conds.get('vwap_dev', 0.5) and rsi7[i] < conds.get('rsi', 25)
            and sk[i] < conds.get('stoch', 15) and adx_v < conds.get('adx', 40)):
            direction = 'up'
        # SHORT: price significantly above VWAP + RSI overbought
        elif (vwap_dev[i] > conds.get('vwap_dev', 0.5) and rsi7[i] > conds.get('rsi_sell', 75)
            and sk[i] > conds.get('stoch_sell', 85) and adx_v < conds.get('adx', 40)):
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


def quick_test(symbol, period, label, fn, conds, **extra):
    """Run a single test and print result."""
    if len(extra) > 0:
        result = fn(symbol, period[0], period[1], conds, **extra)
    else:
        result = fn(symbol, period[0], period[1], conds)
    n, wr, pnl = result[0], result[1], result[2]
    extra_str = ''
    if len(result) >= 5:
        extra_str = f'  (extra: {result[4]})'
    star = ' ***' if wr >= 65 else (' **' if wr >= 60 else '')
    print(f"  {label:<50} {n:>5} {wr:>6.1f}% ${pnl:>+9.0f}{star}{extra_str}")
    return n, wr, pnl


print(f"\n{'='*100}")
print(f"  NEW PARADIGMS — Finding 65%+ WR")
print(f"  Phase 1: 2026H1 ETHUSDT quick scan")
print(f"{'='*100}")

SYM = 'ETHUSDT'
P2026 = ('2026-01-01', '2026-07-01')

best_results = []

# === 1. Adaptive (trend-following + mean reversion) ===
print(f"\n--- 1. ADAPTIVE: Trend-following in trends, MR in ranges ---")
for trend_adx in [25, 30, 35]:
    for trend_di in [5, 8, 10]:
        for trend_rsi in [35, 40, 45]:
            conds = dict(trend_adx=trend_adx, trend_di=trend_di, trend_rsi=trend_rsi,
                        trend_rsi_sell=100-trend_rsi, mr_rsi=20, mr_bb=0.08, mr_stoch=10,
                        mr_ret1=0.02, mr_adx=40, mr_cci=-150, mr_wr=-85)
            n, wr, pnl = quick_test(SYM, P2026,
                f"adx>{trend_adx}_di>{trend_di}_rsi<{trend_rsi}/mr_rsi<20_wr<-85",
                backtest_adaptive, conds)
            if n >= 10:
                best_results.append((wr, n, pnl, f"ADAPTIVE adx>{trend_adx}_di>{trend_di}_rsi<{trend_rsi}"))

# === 2. Triple RSI ===
print(f"\n--- 2. TRIPLE TIMEFRAME RSI ---")
for rsi_ext in [20, 25, 30]:
    for bb in [0.10, 0.12, 0.15]:
        for stoch in [12, 15, 20]:
            conds = dict(rsi_ext=rsi_ext, bb=bb, stoch=stoch, adx=45, rsi_period=7)
            n, wr, pnl = quick_test(SYM, P2026,
                f"3xTF_RSI<{rsi_ext}_bb<{bb}_stoch<{stoch}",
                backtest_triple_rsi, conds)
            if n >= 10:
                best_results.append((wr, n, pnl, f"TRIPLE_RSI rsi<{rsi_ext}_bb<{bb}_st<{stoch}"))

# RSI(14) variant
for rsi_ext in [25, 30]:
    conds = dict(rsi_ext=rsi_ext, bb=0.15, stoch=20, adx=45, rsi_period=14)
    n, wr, pnl = quick_test(SYM, P2026,
        f"3xTF_RSI14<{rsi_ext}_bb<.15",
        backtest_triple_rsi, conds)
    if n >= 10:
        best_results.append((wr, n, pnl, f"TRIPLE_RSI14 rsi<{rsi_ext}"))

# === 3. Keltner Channel ===
print(f"\n--- 3. KELTNER CHANNEL ---")
for kc_mult in [2.0, 2.5, 3.0]:
    for kc_period in [15, 20, 30]:
        for rsi_th in [20, 25, 30]:
            conds = dict(kc_period=kc_period, kc_mult=kc_mult, kc_low=0.05, kc_high=0.95,
                        rsi=rsi_th, rsi_sell=100-rsi_th, stoch=15, stoch_sell=85, adx=40)
            n, wr, pnl = quick_test(SYM, P2026,
                f"KC{kc_period}x{kc_mult}_rsi<{rsi_th}",
                backtest_keltner, conds)
            if n >= 10:
                best_results.append((wr, n, pnl, f"KC p{kc_period}x{kc_mult}_rsi<{rsi_th}"))

# === 4. BB Squeeze ===
print(f"\n--- 4. BB SQUEEZE BREAKOUT ---")
for squeeze_pct in [0.10, 0.15, 0.20]:
    for vol_th in [1.3, 1.5, 2.0]:
        for bb_period in [15, 20]:
            conds = dict(bb_period=bb_period, bb_std=2.0, squeeze_lookback=20,
                        squeeze_pct=squeeze_pct, vol_th=vol_th, di_min=-5, di_max=5)
            n, wr, pnl = quick_test(SYM, P2026,
                f"SQZ{squeeze_pct}_vol>{vol_th}_bb{bb_period}",
                backtest_bb_squeeze, conds)
            if n >= 10:
                best_results.append((wr, n, pnl, f"BB_SQZ pct{squeeze_pct}_vol>{vol_th}"))

# === 5. VWAP extreme ===
print(f"\n--- 5. VWAP EXTREME DEVIATION ---")
for vwap_dev in [0.3, 0.5, 0.8, 1.0]:
    for rsi_th in [20, 25]:
        conds = dict(vwap_dev=vwap_dev, rsi=rsi_th, rsi_sell=100-rsi_th,
                    stoch=15, stoch_sell=85, adx=40)
        n, wr, pnl = quick_test(SYM, P2026,
            f"VWAP_dev>{vwap_dev}%_rsi<{rsi_th}",
            backtest_vwap_extreme, conds)
        if n >= 10:
            best_results.append((wr, n, pnl, f"VWAP dev>{vwap_dev}%_rsi<{rsi_th}"))

# Sort and show best
best_results.sort(key=lambda x: x[0], reverse=True)
print(f"\n{'='*100}")
print(f"  TOP 10 — 2026H1 ETHUSDT")
print(f"{'='*100}")
for wr, n, pnl, label in best_results[:10]:
    star = ' *** 65%+' if wr >= 65 else (' ** 60%+' if wr >= 60 else '')
    print(f"  {label:<55} {n:>5} {wr:>6.1f}% ${pnl:>+9.0f}{star}")
