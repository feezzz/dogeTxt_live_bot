"""
Optimize: relax AND conditions with Consec<=2 filter to increase trade count
while maintaining 65%+ WR.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from event_backtest.data_fetcher import load_all
from event_backtest.indicators import (
    rsi, bollinger_bands, adx, atr, atr_pct, cci, williams_r, stochastic_rsi,
)

PERIODS = [
    ('2024-01-01', '2024-07-01', '2024H1'),
    ('2024-07-01', '2025-01-01', '2024H2'),
    ('2025-01-01', '2025-07-01', '2025H1'),
    ('2025-07-01', '2026-01-01', '2025H2'),
    ('2026-01-01', '2026-07-01', '2026H1'),
]

def tf_idx(ts, t):
    for i in range(len(ts)-1, -1, -1):
        if ts[i] <= t: return i
    return -1

def backtest_relaxed(symbol, start, end, conds):
    """Backtest with relaxed AND + Consec<=2 filter."""
    data = load_all(symbol, start, end)
    c5 = data['5m']
    cl = [c[4] for c in c5]; op = [c[1] for c in c5]
    hi = [c[2] for c in c5]; lo = [c[3] for c in c5]
    c1h = [c[4] for c in data['1h']]; h1h = [c[2] for c in data['1h']]; l1h = [c[3] for c in data['1h']]
    t5 = [c[0] for c in c5]; t1h = [c[0] for c in data['1h']]
    total = len(c5); warmup = 60

    bb_mid, bb_up, bb_low = bollinger_bands(cl, 20, 2.0)
    bb_pos = [(cl[i] - bb_low[i]) / (bb_up[i] - bb_low[i]) if bb_up[i] > bb_low[i] else 0.5
              for i in range(total)]

    rsi7 = rsi(cl, 7)
    sk, sd = stochastic_rsi(cl, 14, 14)
    c14 = cci(hi, lo, cl, 14)
    wr14 = williams_r(hi, lo, cl, 14)
    adx_h, pdi, mdi = adx(h1h, l1h, c1h, 14)
    atr_p5 = atr_pct(atr(hi, lo, cl, 14), cl)

    min_atr = 0.05 if symbol == 'ETHUSDT' else 0.03
    eq = 500; trades = []; last_sig = -999; day_bets = 0; cur_day = None

    rsi_th = conds.get('rsi', 20)
    bb_th = conds.get('bb', 0.08)
    stoch_th = conds.get('stoch', 10)
    ret1_th = conds.get('ret1', 0.02)
    adx_th = conds.get('adx', 40)
    cci_th = conds.get('cci', -150)
    wr_th = conds.get('wr', None)
    max_consec = conds.get('max_consec', 2)
    session_hours = conds.get('session', None)

    for i in range(warmup, total-2):
        ts = c5[i][0]; dt = datetime.fromtimestamp(ts/1000)
        if dt.day != cur_day: cur_day = dt.day; day_bets = 0
        if i - last_sig < 2: continue
        if day_bets >= 50: continue
        if eq <= 25: break
        if atr_p5[i] < min_atr: continue
        i1h = tf_idx(t1h, t5[i] - 55*60*1000)
        if i1h < 20: continue
        if session_hours and dt.hour not in session_hours: continue

        adx_v = adx_h[i1h]
        r1 = (cl[i]/cl[i-1]-1)*100 if i>0 and cl[i-1]>0 else 0

        consec_down = 0
        for j in range(i, max(0, i-20), -1):
            if j > 0 and cl[j] < cl[j-1]: consec_down += 1
            else: break
        consec_up = 0
        for j in range(i, max(0, i-20), -1):
            if j > 0 and cl[j] > cl[j-1]: consec_up += 1
            else: break

        direction = None
        wr_ok = wr_th is None or wr14[i] < wr_th

        if (bb_pos[i] < bb_th and rsi7[i] < rsi_th and sk[i] < stoch_th
            and r1 < -ret1_th and adx_v < adx_th and c14[i] < cci_th
            and wr_ok and consec_down <= max_consec):
            direction = 'up'
        elif (bb_pos[i] > (1-bb_th) and rsi7[i] > (100-rsi_th) and sk[i] > (100-stoch_th)
            and r1 > ret1_th and adx_v < adx_th and c14[i] > -cci_th
            and wr_ok and consec_up <= max_consec):
            direction = 'down'

        if direction is None: continue

        entry = c5[i+1][1]; settle = c5[min(i+2, total-1)][4]
        win = (direction == 'up' and settle > entry) or (direction == 'down' and settle < entry)
        pnl = 20 if win else -25; eq += pnl
        consec = consec_down if direction == 'up' else consec_up
        trades.append({'time': dt, 'dir': direction, 'win': win, 'pnl': pnl, 'consec': consec})
        last_sig = i; day_bets += 1

    n = len(trades); w = sum(1 for t in trades if t['win'])
    wr = w/n*100 if n else 0; pnl = sum(t['pnl'] for t in trades)
    return n, wr, pnl, eq, trades


def validate_relaxed(label, conds):
    total_n = 0; total_w = 0; total_pnl = 0.0
    for start, end, pn in PERIODS:
        r_eth = backtest_relaxed('ETHUSDT', start, end, conds)
        r_btc = backtest_relaxed('BTCUSDT', start, end, conds)
        n_eth, wr_eth, pnl_eth = r_eth[0], r_eth[1], r_eth[2]
        n_btc, wr_btc, pnl_btc = r_btc[0], r_btc[1], r_btc[2]
        total_n += n_eth + n_btc
        total_w += (n_eth*wr_eth/100 if n_eth else 0) + (n_btc*wr_btc/100 if n_btc else 0)
        total_pnl += pnl_eth + pnl_btc
        eth_ok = 'OK' if wr_eth > 55.56 else '--'
        btc_ok = 'OK' if wr_btc > 55.56 else '--'
        print(f"  {pn:<8} ETH:{n_eth:>4}t {wr_eth:>5.1f}% ${pnl_eth:>+6.0f} {eth_ok}  BTC:{n_btc:>4}t {wr_btc:>5.1f}% ${pnl_btc:>+6.0f} {btc_ok}")

    total_wr = total_w/total_n*100 if total_n else 0
    star = ' *** 65%+' if total_wr >= 65 else (' ** 60%+' if total_wr >= 60 else '')
    print(f"  TOTAL:  {total_n:>4}t {total_wr:>5.1f}% ${total_pnl:>+7.0f} ({total_n//30}/mo){star}")
    return total_n, total_wr, total_pnl

# Baseline tight with consec<=2
BASELINE = dict(rsi=18, bb=0.05, stoch=8, ret1=0.03, adx=35, cci=-200, wr=-85, max_consec=2)

print("="*100)
print("  RELAXED AND + CONSEC<=2 — Optimizing for trade volume at 65%+ WR")
print("="*100)

configs = [
    # Baseline
    ("Tight(18/5/8/3/35/-200/-85) +consec<=2", BASELINE),
    # Relax RSI: 18→20
    ("RSI<20 (18→20)", dict(rsi=20, bb=0.05, stoch=8, ret1=0.03, adx=35, cci=-200, wr=-85, max_consec=2)),
    # Relax BB: 0.05→0.08
    ("BB<0.08 (0.05→0.08)", dict(rsi=18, bb=0.08, stoch=8, ret1=0.03, adx=35, cci=-200, wr=-85, max_consec=2)),
    # Relax Stoch: 8→10
    ("Stoch<10 (8→10)", dict(rsi=18, bb=0.05, stoch=10, ret1=0.03, adx=35, cci=-200, wr=-85, max_consec=2)),
    # Relax ret1: 0.03→0.02
    ("ret1<-0.02 (0.03→0.02)", dict(rsi=18, bb=0.05, stoch=8, ret1=0.02, adx=35, cci=-200, wr=-85, max_consec=2)),
    # Relax ADX: 35→40
    ("ADX<40 (35→40)", dict(rsi=18, bb=0.05, stoch=8, ret1=0.03, adx=40, cci=-200, wr=-85, max_consec=2)),
    # Relax CCI: -200→-150
    ("CCI<-150 (-200→-150)", dict(rsi=18, bb=0.05, stoch=8, ret1=0.03, adx=35, cci=-150, wr=-85, max_consec=2)),
    # Relax WR: -85→-80
    ("WR<-80 (-85→-80)", dict(rsi=18, bb=0.05, stoch=8, ret1=0.03, adx=35, cci=-200, wr=-80, max_consec=2)),
    # R1 baseline with consec<=2
    ("R1(20/8/10/2/40/-150) +consec<=2", dict(rsi=20, bb=0.08, stoch=10, ret1=0.02, adx=40, cci=-150, max_consec=2)),
    # R1 baseline + WR
    ("R1+WR<-85 +consec<=2", dict(rsi=20, bb=0.08, stoch=10, ret1=0.02, adx=40, cci=-150, wr=-85, max_consec=2)),
    # Moderate relaxation
    ("RSI<20,BB<0.06,Stoch<9,ADX<38 +consec<=2", dict(rsi=20, bb=0.06, stoch=9, ret1=0.03, adx=38, cci=-200, wr=-85, max_consec=2)),
    # Drop WR requirement
    ("No WR filter +consec<=2", dict(rsi=18, bb=0.05, stoch=8, ret1=0.03, adx=35, cci=-200, max_consec=2)),
]

best_wr = 0; best_n = 0; best_label = ''
for label, conds in configs:
    print(f"\n  [{label}]")
    n, wr, pnl = validate_relaxed(label, conds)
    score = wr * min(n, 500)  # reward both WR and volume
    if score > best_wr * min(best_n, 500):
        best_wr = wr; best_n = n; best_label = label

print(f"\n{'='*100}")
print(f"  BEST BALANCE: {best_label} — {best_n}t, {best_wr:.1f}% WR")
print(f"{'='*100}")
