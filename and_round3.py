"""
Round 3: Test best configs from Round 2 across ALL 5 periods.
Best findings: +WR<-85 = 63.3% WR, +vol_spike>1.5 = 63.1% WR.
Combine WR + vol, change RSI period, test 1h trend filter.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from event_backtest.data_fetcher import load_all
from event_backtest.indicators import (
    rsi, bollinger_bands, adx, atr, atr_pct, volume_spike,
    cci, williams_r, stochastic_rsi, mfi,
)

def tf_idx(ts, t):
    for i in range(len(ts)-1,-1,-1):
        if ts[i] <= t: return i
    return -1

def backtest(symbol, start, end, conds, amount=25):
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
    mf = mfi(hi, lo, cl, vol, 14)
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
        if eq <= amount: break
        if atr_p[i] < min_atr: continue
        i1h = tf_idx(t1h, t5[i] - 55*60*1000)
        if i1h < 20: continue
        if dt.hour in LOWS and day_bets >= 16: continue

        p = cl[i]; bb_p = (p-bb_l[i])/(bb_u[i]-bb_l[i]) if bb_u[i]>bb_l[i] else 0.5
        r1 = (cl[i]/cl[i-1]-1)*100 if i>0 and cl[i-1]>0 else 0
        r3 = (cl[i]/cl[i-3]-1)*100 if i>=3 and cl[i-3]>0 else 0
        adx_v = adx_h[i1h]; di_d = pdi[i1h]-mdi[i1h]

        direction = None

        rsi_th = conds.get('rsi', 20)
        bb_th = conds.get('bb', 0.08)
        stoch_th = conds.get('stoch', 15)
        ret1_th = conds.get('ret1', 0.02)
        adx_th = conds.get('adx', 40)
        cci_th = conds.get('cci', -150)
        wr_th = conds.get('wr', None)
        mfi_th = conds.get('mfi', None)
        ret3_th = conds.get('ret3', None)
        use_vol = conds.get('use_vol', False)
        use_di = conds.get('use_di', False)
        di_max = conds.get('di_max', 3)

        # LONG
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
        if use_di and di_d <= di_max: ok = False  # must have bullish trend on 1h
        if ok: direction = 'up'

        # SHORT
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
            if use_di and di_d >= -di_max: ok = False
            if ok: direction = 'down'

        if direction is None: continue

        entry = c5[i+1][1]; settle = c5[min(i+2,total-1)][4]
        win = (direction=='up' and settle>entry) or (direction=='down' and settle<entry)
        pnl = amount*0.80 if win else -amount; eq += pnl
        trades.append({'time':dt,'dir':direction,'win':win,'pnl':pnl})
        last_sig = i; day_bets += 1

    n = len(trades); w = sum(1 for t in trades if t['win'])
    wr = w/n*100 if n else 0; pnl = sum(t['pnl'] for t in trades)
    return n, wr, pnl, eq


def validate(conds, name):
    """Run across all 5 periods."""
    periods = [
        ('2024-01-01', '2024-07-01', '2024H1'),
        ('2024-07-01', '2025-01-01', '2024H2'),
        ('2025-01-01', '2025-07-01', '2025H1'),
        ('2025-07-01', '2026-01-01', '2025H2'),
        ('2026-01-01', '2026-07-01', '2026H1'),
    ]
    total_n = 0; total_w = 0; total_pnl = 0.0
    print(f"\n  [{name}]")
    print(f"  {'Period':<10} {'ETH Trades':>10} {'ETH WR':>8} {'ETH PnL':>10} {'BTC Trades':>10} {'BTC WR':>8} {'BTC PnL':>10}")

    for start, end, pn in periods:
        r_eth = backtest('ETHUSDT', start, end, conds)
        r_btc = backtest('BTCUSDT', start, end, conds)
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
print(f"  ROUND 3: 5-PERIOD VALIDATION OF BEST AND-CONDITION CONFIGS")
print(f"  Breakeven: 55.56% | 10m (80%) | $25/trade")
print(f"{'='*110}")

configs = [
    # Baseline from R1
    ("R1_best (rsi20_bb08_st10_r1.03_adx35_cci-200)", dict(
        rsi=20, bb=0.08, stoch=10, ret1=0.03, adx=35, cci=-200)),
    # Best from R2: WR
    ("R2 +WR<-85", dict(
        rsi=20, bb=0.08, stoch=10, ret1=0.03, adx=35, cci=-200, wr=-85)),
    # Best from R2: vol
    ("R2 +vol_spike>1.5", dict(
        rsi=20, bb=0.08, stoch=10, ret1=0.03, adx=35, cci=-200, use_vol=True, vol_th=1.5)),
    # Combine WR + vol
    ("R2 +WR<-85 +vol>1.5", dict(
        rsi=20, bb=0.08, stoch=10, ret1=0.03, adx=35, cci=-200, wr=-85, use_vol=True, vol_th=1.5)),
    # Add ret3 to best
    ("R2 +WR<-85 +ret3<-.10", dict(
        rsi=20, bb=0.08, stoch=10, ret1=0.03, adx=35, cci=-200, wr=-85, ret3=0.10)),
    # Vol + ret3
    ("R2 +vol>1.5 +ret3<-.10", dict(
        rsi=20, bb=0.08, stoch=10, ret1=0.03, adx=35, cci=-200, use_vol=True, vol_th=1.5, ret3=0.10)),
    # RSI14 instead of RSI7
    ("R2 +WR<-85 RSI14", dict(
        rsi=20, bb=0.08, stoch=10, ret1=0.03, adx=35, cci=-200, wr=-85, rsi_period=14)),
    # Tighter ADX
    ("R2 +WR<-85 adx<30", dict(
        rsi=20, bb=0.08, stoch=10, ret1=0.03, adx=30, cci=-200, wr=-85)),
]

best_wr = 0; best_label = ''
for label, conds in configs:
    n, wr, pnl = validate(conds, label)
    if wr > best_wr:
        best_wr = wr; best_label = label

print(f"\n{'='*110}")
print(f"  BEST: {best_label} — {best_wr:.1f}% WR")
print(f"{'='*110}")
