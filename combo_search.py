"""
Combo search: Test specific high-promise combinations within 10-min constraint.
Ideas from new_approaches.py results:
- Dynamic exit proved signals are good (66.4% with longer hold)
- Need to improve WR within 10-min window
- Test: session filters, double timeframe, 4h trend, early exit
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from event_backtest.data_fetcher import load_all
from event_backtest.indicators import (
    rsi, bollinger_bands, adx, atr, atr_pct, volume_spike,
    cci, williams_r, stochastic_rsi, mfi, ema, aroon, aroon_osc,
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

BASELINE = dict(rsi=20, bb=0.08, stoch=10, ret1=0.02, adx=40, cci=-150)


def backtest_and(symbol, start, end, conds):
    """Standard AND-condition backtest with 10-min hold."""
    data = load_all(symbol, start, end)
    c5 = data['5m']
    cl = [c[4] for c in c5]; op = [c[1] for c in c5]
    hi = [c[2] for c in c5]; lo = [c[3] for c in c5]
    vol = [c[5] for c in c5]
    c1h = [c[4] for c in data['1h']]; h1h = [c[2] for c in data['1h']]; l1h = [c[3] for c in data['1h']]
    t5 = [c[0] for c in c5]; t1h = [c[0] for c in data['1h']]
    c15 = data.get('15m')
    if c15:
        cl15 = [c[4] for c in c15]; hi15 = [c[2] for c in c15]; lo15 = [c[3] for c in c15]
        t15 = [c[0] for c in c15]
    else:
        cl15 = hi15 = lo15 = t15 = None
    c4h = data.get('4h')
    if c4h:
        c4hc = [c[4] for c in c4h]; h4h = [c[2] for c in c4h]; l4h = [c[3] for c in c4h]
        t4h = [c[0] for c in c4h]
    else:
        c4hc = h4h = l4h = t4h = None
    total = len(c5); warmup = 60

    rsi_period = conds.get('rsi_period', 7)
    rsi_vals = rsi(cl, rsi_period)
    _, bb_u, bb_l = bollinger_bands(cl, conds.get('bb_period', 20), 2.0)
    sk, sd = stochastic_rsi(cl, 14, 14)
    c14 = cci(hi, lo, cl, 14)
    wr14 = williams_r(hi, lo, cl, 14)
    mf = mfi(hi, lo, cl, vol, 14)
    adx_h, pdi, mdi = adx(h1h, l1h, c1h, 14)
    atr_p = atr_pct(atr(hi, lo, cl, 14), cl)
    vs = volume_spike(vol, 20, conds.get('vol_th', 1.5))
    ema20 = ema(cl, 20)

    # 15m indicators (if available)
    if cl15:
        rsi15 = rsi(cl15, 7)
        sk15, sd15 = stochastic_rsi(cl15, 14, 14)
    # 4h indicators (if available)
    if c4hc:
        adx_4h, pdi4, mdi4 = adx(h4h, l4h, c4hc, 14)

    # Volume decline
    vol_avg5 = [0.0]*total
    for i in range(5, total):
        vol_avg5[i] = sum(vol[i-5:i]) / 5
    vol_decl = [vol[i] < vol_avg5[i]*0.7 for i in range(total)]

    min_atr = 0.05 if symbol == 'ETHUSDT' else 0.03
    LOWS = frozenset([22,23,0,1,2,3,4])
    HIGH_VOL = frozenset([7,8,9,10,11,12,13,14,15,16])  # UTC high-vol hours (EU+US overlap)
    ASIA = frozenset([0,1,2,3,4,5,6,7])
    EU = frozenset([7,8,9,10,11,12,13,14,15])
    US = frozenset([12,13,14,15,16,17,18,19,20,21])

    eq = 500; trades = []; last_sig = -999; day_bets = 0; cur_day = None

    for i in range(warmup, total-2):
        ts = c5[i][0]; dt = datetime.fromtimestamp(ts/1000)
        if dt.day != cur_day: cur_day = dt.day; day_bets = 0
        if i - last_sig < conds.get('cooldown', 2): continue
        if day_bets >= conds.get('max_daily', 50): continue
        if eq <= 25: break
        if atr_p[i] < min_atr: continue
        i1h = tf_idx(t1h, t5[i] - 55*60*1000)
        if i1h < 20: continue
        if dt.hour in LOWS and day_bets >= 16: continue

        # Session filter
        session = conds.get('session', None)
        if session == 'high_vol' and dt.hour not in HIGH_VOL: continue
        if session == 'us' and dt.hour not in US: continue
        if session == 'eu_us' and dt.hour not in (EU | US): continue
        if session == 'not_asia' and dt.hour in ASIA: continue

        p = cl[i]; bb_p = (p-bb_l[i])/(bb_u[i]-bb_l[i]) if bb_u[i]>bb_l[i] else 0.5
        r1 = (cl[i]/cl[i-1]-1)*100 if i>0 and cl[i-1]>0 else 0
        r3 = (cl[i]/cl[i-3]-1)*100 if i>=3 and cl[i-3]>0 else 0
        adx_v = adx_h[i1h]; di_d = pdi[i1h]-mdi[i1h]

        # 15m index
        if t15 and conds.get('use_15m', False):
            i15 = tf_idx(t15, t5[i] - 10*60*1000)
        else:
            i15 = -1
        # 4h index
        if t4h and conds.get('use_4h', False):
            i4h = tf_idx(t4h, t5[i] - 55*60*1000)
        else:
            i4h = -1

        direction = None

        rsi_th = conds.get('rsi', 20); bb_th = conds.get('bb', 0.08)
        stoch_th = conds.get('stoch', 10); ret1_th = conds.get('ret1', 0.02)
        adx_th = conds.get('adx', 40); cci_th = conds.get('cci', -150)
        wr_th = conds.get('wr', None); use_vol = conds.get('use_vol', False)
        use_vol_decl = conds.get('use_vol_decl', False)
        mfi_th = conds.get('mfi', None)
        ret3_th = conds.get('ret3', None)
        use_trend = conds.get('use_trend', False)
        trend_dir = conds.get('trend_dir', 0)  # min DI diff for trend alignment
        ema_req = conds.get('ema_req', False)  # require price near EMA

        # LONG conditions
        ok = True
        if rsi_vals[i] >= rsi_th: ok = False
        if bb_p >= bb_th: ok = False
        if sk[i] >= stoch_th: ok = False
        if r1 >= -ret1_th: ok = False
        if adx_v >= adx_th: ok = False
        if c14[i] >= cci_th: ok = False
        if wr_th is not None and wr14[i] >= wr_th: ok = False
        if mfi_th is not None and mf[i] >= mfi_th: ok = False
        if ret3_th is not None and r3 >= -ret3_th: ok = False
        if use_vol and not vs[i]: ok = False
        if use_vol_decl and not vol_decl[i]: ok = False
        if use_trend and di_d <= trend_dir: ok = False  # must be in uptrend
        if ema_req and p > ema20[i]: ok = False  # price must be below EMA for LONG

        # Double timeframe: 15m must also be extreme
        if ok and i15 >= 0 and conds.get('double_tf', False):
            if rsi15[i15] >= conds.get('rsi15_th', 30): ok = False

        # 4h trend filter
        if ok and i4h >= 0 and conds.get('use_4h', False):
            if adx_4h[i4h] >= conds.get('adx4h', 30): ok = False
            di4 = pdi4[i4h] - mdi4[i4h]
            if conds.get('align_4h', False) and di4 <= 0: ok = False

        if ok: direction = 'up'

        # SHORT conditions
        if direction is None:
            ok = True
            if rsi_vals[i] <= (100-rsi_th): ok = False
            if bb_p <= (1-bb_th): ok = False
            if sk[i] <= (100-stoch_th): ok = False
            if r1 <= ret1_th: ok = False
            if adx_v >= adx_th: ok = False
            if c14[i] <= -cci_th: ok = False
            if wr_th is not None and wr14[i] <= -wr_th: ok = False
            if mfi_th is not None and mf[i] <= (100-mfi_th): ok = False
            if ret3_th is not None and r3 <= ret3_th: ok = False
            if use_vol and not vs[i]: ok = False
            if use_vol_decl and not vol_decl[i]: ok = False
            if use_trend and di_d >= -trend_dir: ok = False
            if ema_req and p < ema20[i]: ok = False

            if ok and i15 >= 0 and conds.get('double_tf', False):
                if rsi15[i15] <= (100-conds.get('rsi15_th', 30)): ok = False

            if ok and i4h >= 0 and conds.get('use_4h', False):
                if adx_4h[i4h] >= conds.get('adx4h', 30): ok = False
                di4 = pdi4[i4h] - mdi4[i4h]
                if conds.get('align_4h', False) and di4 >= 0: ok = False

            if ok: direction = 'down'

        if direction is None: continue

        entry = c5[i+1][1]; settle = c5[min(i+2,total-1)][4]
        win = (direction=='up' and settle>entry) or (direction=='down' and settle<entry)
        pnl = 20 if win else -25; eq += pnl
        trades.append({'time':dt,'dir':direction,'win':win,'pnl':pnl})
        last_sig = i; day_bets += 1

    n = len(trades); w = sum(1 for t in trades if t['win'])
    wr = w/n*100 if n else 0; pnl = sum(t['pnl'] for t in trades)
    return n, wr, pnl, eq


def validate_5p(conds, label):
    """Run across all 5 periods for both ETH and BTC."""
    total_n = 0; total_w = 0; total_pnl = 0.0
    print(f"\n  [{label}]")
    print(f"  {'Period':<10} {'ETH Trades':>10} {'ETH WR':>8} {'ETH PnL':>10} {'BTC Trades':>10} {'BTC WR':>8} {'BTC PnL':>10}")

    for start, end, pn in PERIODS:
        r_eth = backtest_and('ETHUSDT', start, end, conds)
        r_btc = backtest_and('BTCUSDT', start, end, conds)
        total_n += r_eth[0] + r_btc[0]
        total_w += (r_eth[0]*r_eth[1]/100 if r_eth[0] else 0) + (r_btc[0]*r_btc[1]/100 if r_btc[0] else 0)
        total_pnl += r_eth[2] + r_btc[2]
        eth_ok = 'OK' if r_eth[1] > 55.56 else '--'
        btc_ok = 'OK' if r_btc[1] > 55.56 else '--'
        print(f"  {pn:<10} {r_eth[0]:>10} {r_eth[1]:>7.1f}% ${r_eth[2]:>+9.0f} {eth_ok}  {r_btc[0]:>10} {r_btc[1]:>7.1f}% ${r_btc[2]:>+9.0f} {btc_ok}")

    total_wr = total_w/total_n*100 if total_n else 0
    star = ' *** 65%+' if total_wr >= 65 else (' ** 60%+' if total_wr >= 60 else '')
    print(f"  {'TOTAL':<10} {total_n:>10} {total_wr:>7.1f}% ${total_pnl:>+9.0f}{star}")
    return total_n, total_wr, total_pnl


print(f"\n{'='*110}")
print(f"  COMBO SEARCH — 5-PERIOD VALIDATION")
print(f"  Testing: session filters, double TF, 4h trend, volume decline, EMA filter")
print(f"  Breakeven: 55.56% | 10m (80%) | $25/trade")
print(f"{'='*110}")

configs = []

# R1 baseline reference
configs.append(("R1 baseline (6 conds)", {**BASELINE}))

# Best from R3: baseline + WR<-85
configs.append(("R1 +WR<-85", {**BASELINE, 'wr': -85}))

# Session filters
configs.append(("R1 +WR<-85 US session only", {**BASELINE, 'wr': -85, 'session': 'us'}))
configs.append(("R1 +WR<-85 EU+US only", {**BASELINE, 'wr': -85, 'session': 'eu_us'}))
configs.append(("R1 +WR<-85 high-vol only", {**BASELINE, 'wr': -85, 'session': 'high_vol'}))
configs.append(("R1 +WR<-85 not-asia", {**BASELINE, 'wr': -85, 'session': 'not_asia'}))

# Volume decline
configs.append(("R1 +vol_decline(<0.7*avg5)", {**BASELINE, 'use_vol_decl': True}))

# Double timeframe (5m + 15m both extreme)
configs.append(("R1 +double_tf(rsi15<30)", {**BASELINE, 'double_tf': True, 'rsi15_th': 30}))
configs.append(("R1 +WR<-85 +double_tf(rsi15<35)", {**BASELINE, 'wr': -85, 'double_tf': True, 'rsi15_th': 35}))

# EMA filter (price must be below EMA for LONG)
configs.append(("R1 +WR<-85 +EMA_req", {**BASELINE, 'wr': -85, 'ema_req': True}))

# Volume spike
configs.append(("R1 +vol_spike>1.5", {**BASELINE, 'use_vol': True, 'vol_th': 1.5}))

# Trend aligned
configs.append(("R1 +WR<-85 +trend(dir>3)", {**BASELINE, 'wr': -85, 'use_trend': True, 'trend_dir': 3}))
configs.append(("R1 +WR<-85 +trend(dir>5)", {**BASELINE, 'wr': -85, 'use_trend': True, 'trend_dir': 5}))

# 4h trend filter
configs.append(("R1 +WR<-85 +4h_adx<30", {**BASELINE, 'wr': -85, 'use_4h': True, 'adx4h': 30}))
configs.append(("R1 +WR<-85 +4h_align_bull", {**BASELINE, 'wr': -85, 'use_4h': True, 'adx4h': 35, 'align_4h': True}))

# Combined: session + trend + WR
configs.append(("R1 +WR<-85 +US_only +trend>3", {**BASELINE, 'wr': -85, 'session': 'us', 'use_trend': True, 'trend_dir': 3}))
configs.append(("R1 +WR<-85 +not_asia +trend>5", {**BASELINE, 'wr': -85, 'session': 'not_asia', 'use_trend': True, 'trend_dir': 5}))

# Triple combo: WR + trend + double TF
configs.append(("R1 +WR<-85 +trend>3 +double15m<35", {**BASELINE, 'wr': -85, 'use_trend': True, 'trend_dir': 3, 'double_tf': True, 'rsi15_th': 35}))

# Tighter baseline variants
configs.append(("Tighter: rsi<18 +WR<-85", dict(rsi=18, bb=0.05, stoch=8, ret1=0.03, adx=35, cci=-200, wr=-85)))
configs.append(("Tighter: rsi<18 +WR<-90 +trend>3", dict(rsi=18, bb=0.05, stoch=8, ret1=0.03, adx=35, cci=-200, wr=-90, use_trend=True, trend_dir=3)))

# Ret3
configs.append(("R1 +WR<-85 +ret3<-.15", {**BASELINE, 'wr': -85, 'ret3': 0.15}))

best_wr = 0; best_label = ''
for label, conds in configs:
    n, wr, pnl = validate_5p(conds, label)
    if wr > best_wr:
        best_wr = wr; best_label = label

print(f"\n{'='*110}")
print(f"  BEST: {best_label} — {best_wr:.1f}% WR")
print(f"{'='*110}")
