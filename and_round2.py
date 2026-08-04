"""
Round 2: Add more AND conditions to push WR toward 65%.
Adds: MFI, Williams %R, volume spike, ret3 (3-bar momentum), candle patterns.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from event_backtest.data_fetcher import load_all
from event_backtest.indicators import (
    rsi, bollinger_bands, adx, atr, atr_pct, volume_spike,
    cci, williams_r, stochastic_rsi, mfi, parabolic_sar,
    detect_candle_patterns,
)

def tf_idx(ts, t):
    for i in range(len(ts)-1,-1,-1):
        if ts[i] <= t: return i
    return -1

def run(symbol, start, end, name, **conds):
    """Run AND-condition backtest with given conditions."""
    data = load_all(symbol, start, end)
    c5 = data['5m']
    cl = [c[4] for c in c5]; op = [c[1] for c in c5]
    hi = [c[2] for c in c5]; lo = [c[3] for c in c5]
    vol = [c[5] for c in c5]
    c1h = [c[4] for c in data['1h']]; h1h = [c[2] for c in data['1h']]; l1h = [c[3] for c in data['1h']]
    t5 = [c[0] for c in c5]; t1h = [c[0] for c in data['1h']]
    total = len(c5); warmup = 60

    rsi7 = rsi(cl, 7); rsi14 = rsi(cl, 14)
    _, bb_u, bb_l = bollinger_bands(cl, 20, 2.0)
    sk, sd = stochastic_rsi(cl, 14, 14)
    c14 = cci(hi, lo, cl, 14)
    wr14 = williams_r(hi, lo, cl, 14)
    mf = mfi(hi, lo, cl, vol, 14)
    sar = parabolic_sar(hi, lo)
    adx_h, pdi, mdi = adx(h1h, l1h, c1h, 14)
    atr_p = atr_pct(atr(hi, lo, cl, 14), cl)
    vs = volume_spike(vol, 20, conds.get('vol_th', 1.5))
    pat = detect_candle_patterns(op, hi, lo, cl)

    LOWS = frozenset([22,23,0,1,2,3,4])
    eq = 500; trades = []; last_sig = -999; day_bets = 0; cur_day = None

    for i in range(warmup, total-2):
        ts = c5[i][0]; dt = datetime.fromtimestamp(ts/1000)
        if dt.day != cur_day: cur_day = dt.day; day_bets = 0
        if i - last_sig < 2: continue
        if day_bets >= 50: continue
        if eq <= 25: break
        if atr_p[i] < 0.05: continue
        i1h = tf_idx(t1h, t5[i] - 55*60*1000)
        if i1h < 20: continue
        if dt.hour in LOWS and day_bets >= 16: continue

        p = cl[i]; bb_p = (p-bb_l[i])/(bb_u[i]-bb_l[i]) if bb_u[i]>bb_l[i] else 0.5
        r1 = (cl[i]/cl[i-1]-1)*100 if i>0 and cl[i-1]>0 else 0
        r3 = (cl[i]/cl[i-3]-1)*100 if i>=3 and cl[i-3]>0 else 0
        adx_v = adx_h[i1h]; di_d = pdi[i1h]-mdi[i1h]

        direction = None

        # LONG
        if (rsi7[i] < conds.get('rsi', 20) and bb_p < conds.get('bb', 0.08)
            and sk[i] < conds.get('stoch', 15) and r1 < -conds.get('ret1', 0.02)
            and adx_v < conds.get('adx', 40) and di_d > -8
            and c14[i] < conds.get('cci', -150)
            and (not conds.get('use_mfi', False) or mf[i] < conds.get('mfi', 20))
            and (not conds.get('use_wr', False) or wr14[i] < conds.get('wr', -90))
            and (not conds.get('use_ret3', False) or r3 < -conds.get('ret3', 0.10))
            and (not conds.get('use_vol', False) or vs[i])
            and (not conds.get('use_hammer', False) or pat['hammer'][i])):
            direction = 'up'

        # SHORT
        if direction is None:
            if (rsi7[i] > (100-conds.get('rsi', 20))
                and bb_p > (1-conds.get('bb', 0.08))
                and sk[i] > (100-conds.get('stoch', 15))
                and r1 > conds.get('ret1', 0.02)
                and adx_v < conds.get('adx', 40) and di_d < 8
                and c14[i] > -conds.get('cci', -150)
                and (not conds.get('use_mfi', False) or mf[i] > (100-conds.get('mfi', 20)))
                and (not conds.get('use_wr', False) or wr14[i] > -conds.get('wr', -90))
                and (not conds.get('use_ret3', False) or r3 > conds.get('ret3', 0.10))
                and (not conds.get('use_vol', False) or vs[i])
                and (not conds.get('use_hammer', False) or pat['shooting_star'][i])):
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


print(f"\n{'='*100}")
print(f"  ROUND 2: ADDING MORE AND-CONDITIONS — ETHUSDT 2026H1")
print(f"  Goal: 65%+ WR by adding MFI, WR, ret3, volume spike, candle patterns")
print(f"{'='*100}")

SYM = 'ETHUSDT'; P = ('2026-01-01', '2026-07-01')

# Baseline from round 1 best
print(f"\n--- Baseline (best from Round 1) ---")
n, wr, pnl, eq = run(SYM, P[0], P[1], "baseline", rsi=20, bb=0.08, stoch=10, ret1=0.03, adx=35, cci=-200)
print(f"  {'baseline_rsi20_bb0.08_st10_r1.03_adx35_cci-200':<55} {n:>5} {wr:>6.1f}% ${pnl:>+9.0f}")

# Add MFI
print(f"\n--- Adding MFI ---")
for mfi_th in [15, 20, 25]:
    n, wr, pnl, eq = run(SYM, P[0], P[1], f"mfi<{mfi_th}", rsi=20, bb=0.08, stoch=10, ret1=0.03, adx=35, cci=-200, use_mfi=True, mfi=mfi_th)
    star = ' ***' if wr >= 65 else (' **' if wr >= 60 else '')
    print(f"  +MFI<{mfi_th:<3}                                                  {n:>5} {wr:>6.1f}% ${pnl:>+9.0f}{star}")

# Add Williams %R
print(f"\n--- Adding Williams %R ---")
for wr_th in [-95, -90, -85]:
    n, wr, pnl, eq = run(SYM, P[0], P[1], f"wr<{wr_th}", rsi=20, bb=0.08, stoch=10, ret1=0.03, adx=35, cci=-200, use_wr=True, wr=wr_th)
    star = ' ***' if wr >= 65 else (' **' if wr >= 60 else '')
    print(f"  +WR<{wr_th:<4}                                                  {n:>5} {wr:>6.1f}% ${pnl:>+9.0f}{star}")

# Add ret3
print(f"\n--- Adding ret3 (3-bar momentum) ---")
for r3 in [0.05, 0.10, 0.15]:
    n, wr, pnl, eq = run(SYM, P[0], P[1], f"ret3<-{r3}", rsi=20, bb=0.08, stoch=10, ret1=0.03, adx=35, cci=-200, use_ret3=True, ret3=r3)
    star = ' ***' if wr >= 65 else (' **' if wr >= 60 else '')
    print(f"  +ret3<-{r3:.2f}                                                {n:>5} {wr:>6.1f}% ${pnl:>+9.0f}{star}")

# Add volume spike
print(f"\n--- Adding volume spike ---")
for vol_th in [1.3, 1.5, 2.0]:
    n, wr, pnl, eq = run(SYM, P[0], P[1], f"vol>{vol_th}", rsi=20, bb=0.08, stoch=10, ret1=0.03, adx=35, cci=-200, use_vol=True, vol_th=vol_th)
    star = ' ***' if wr >= 65 else (' **' if wr >= 60 else '')
    print(f"  +vol_spike>{vol_th}                                              {n:>5} {wr:>6.1f}% ${pnl:>+9.0f}{star}")

# Add candle patterns
print(f"\n--- Adding candle patterns ---")
n, wr, pnl, eq = run(SYM, P[0], P[1], "hammer", rsi=20, bb=0.08, stoch=10, ret1=0.03, adx=35, cci=-200, use_hammer=True)
star = ' ***' if wr >= 65 else (' **' if wr >= 60 else '')
print(f"  +hammer/shooting_star                                          {n:>5} {wr:>6.1f}% ${pnl:>+9.0f}{star}")

# COMBOS: best individual additions combined
print(f"\n--- COMBOS: MFI + WR + ret3 + vol ---")
for mfi_th in [15, 20]:
    for wr_th in [-95, -90]:
        n, wr, pnl, eq = run(SYM, P[0], P[1], f"MFI<{mfi_th}+WR<{wr_th}+ret3",
            rsi=20, bb=0.08, stoch=10, ret1=0.03, adx=35, cci=-200,
            use_mfi=True, mfi=mfi_th, use_wr=True, wr=wr_th, use_ret3=True, ret3=0.10)
        star = ' *** 65%+' if wr >= 65 else (' ** 60%+' if wr >= 60 else '')
        print(f"  MFI<{mfi_th}_WR<{wr_th}_ret3<-.10                                   {n:>5} {wr:>6.1f}% ${pnl:>+9.0f}{star}")

# Full combo: all 4 additions
print(f"\n--- FULL COMBO: MFI + WR + ret3 + vol + candle ---")
n, wr, pnl, eq = run(SYM, P[0], P[1], "FULL",
    rsi=20, bb=0.05, stoch=10, ret1=0.03, adx=35, cci=-200,
    use_mfi=True, mfi=20, use_wr=True, wr=-90, use_ret3=True, ret3=0.10,
    use_vol=True, vol_th=1.3)
star = ' *** 65%+' if wr >= 65 else (' ** 60%+' if wr >= 60 else '')
print(f"  FULL_COMBO_(rsi20_bb5_st10_mfi20_wr-90_ret3.10_vol)            {n:>5} {wr:>6.1f}% ${pnl:>+9.0f}{star}")

# Even tighter RSI
print(f"\n--- Tighter RSI ---")
for rsi_th in [15, 18]:
    n, wr, pnl, eq = run(SYM, P[0], P[1], f"rsi<{rsi_th}",
        rsi=rsi_th, bb=0.05, stoch=10, ret1=0.03, adx=35, cci=-200,
        use_mfi=True, mfi=20, use_ret3=True, ret3=0.10)
    star = ' *** 65%+' if wr >= 65 else (' ** 60%+' if wr >= 60 else '')
    print(f"  RSI<{rsi_th}+MFI<20+ret3<-.10                                      {n:>5} {wr:>6.1f}% ${pnl:>+9.0f}{star}")

# Tighter BB
print(f"\n--- Tighter BB ---")
for bb_th in [0.03, 0.05]:
    n, wr, pnl, eq = run(SYM, P[0], P[1], f"bb<{bb_th}",
        rsi=20, bb=bb_th, stoch=10, ret1=0.03, adx=35, cci=-200,
        use_mfi=True, mfi=20, use_ret3=True, ret3=0.10)
    star = ' *** 65%+' if wr >= 65 else (' ** 60%+' if wr >= 60 else '')
    print(f"  BB<{bb_th}+MFI<20+ret3<-.10                                        {n:>5} {wr:>6.1f}% ${pnl:>+9.0f}{star}")

# Tightest combo: RSI<18, BB<0.03, Stoch<10, MFI<15, WR<-95, ret1<-0.05, ret3<-0.15, CCI<-250
print(f"\n--- ULTRA-TIGHT combo ---")
n, wr, pnl, eq = run(SYM, P[0], P[1], "ULTRA",
    rsi=18, bb=0.03, stoch=8, ret1=0.05, adx=30, cci=-250,
    use_mfi=True, mfi=15, use_wr=True, wr=-95, use_ret3=True, ret3=0.15)
star = ' *** 65%+' if wr >= 65 else (' ** 60%+' if wr >= 60 else '')
print(f"  ULTRA_TIGHT_(rsi18_bb3_st8_mfi15_wr-95_r1.05_r3.15_cci-250)   {n:>5} {wr:>6.1f}% ${pnl:>+9.0f}{star}")

# Moderate combo aiming for balance
print(f"\n--- BALANCED combo (trades vs WR) ---")
for stoch_th in [10, 12]:
    n, wr, pnl, eq = run(SYM, P[0], P[1], f"balanced_st{stoch_th}",
        rsi=20, bb=0.08, stoch=stoch_th, ret1=0.02, adx=35, cci=-150,
        use_mfi=True, mfi=25)
    star = ' *** 65%+' if wr >= 65 else (' ** 60%+' if wr >= 60 else '')
    print(f"  rsi20_bb8_st{stoch_th}_r1.02_adx35_cci-150_mfi25                  {n:>5} {wr:>6.1f}% ${pnl:>+9.0f}{star}")

print(f"\nDONE")
