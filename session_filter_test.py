"""
Test session filters + consec bar filters based on loss analysis findings.
Findings from entry_timing.py: Asia(0-7) 77.8%, Night(22-24) 80%, consec<=2 75-89% WR
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

def backtest_with_filters(symbol, start, end, session_hours, max_consec):
    """Backtest AND-condition strategy with session + consec bar filters."""
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

    for i in range(warmup, total-2):
        ts = c5[i][0]; dt = datetime.fromtimestamp(ts/1000)
        if dt.day != cur_day: cur_day = dt.day; day_bets = 0
        if i - last_sig < 2: continue
        if day_bets >= 50: continue
        if eq <= 25: break
        if atr_p5[i] < min_atr: continue
        i1h = tf_idx(t1h, t5[i] - 55*60*1000)
        if i1h < 20: continue

        # Session filter
        if dt.hour not in session_hours: continue

        adx_v = adx_h[i1h]
        r1 = (cl[i]/cl[i-1]-1)*100 if i>0 and cl[i-1]>0 else 0

        # Count consecutive bars in same direction (down for LONG, up for SHORT)
        consec_down = 0
        for j in range(i, max(0, i-20), -1):
            if cl[j] < cl[j-1] if j > 0 else False:
                consec_down += 1
            else:
                break
        consec_up = 0
        for j in range(i, max(0, i-20), -1):
            if cl[j] > cl[j-1] if j > 0 else False:
                consec_up += 1
            else:
                break

        # Tighter AND conditions (best config)
        direction = None
        if (bb_pos[i] < 0.05 and rsi7[i] < 18 and sk[i] < 8
            and r1 < -0.03 and adx_v < 35 and c14[i] < -200
            and wr14[i] < -85 and consec_down <= max_consec):
            direction = 'up'
        elif (bb_pos[i] > 0.95 and rsi7[i] > 82 and sk[i] > 92
            and r1 > 0.03 and adx_v < 35 and c14[i] > 200
            and wr14[i] > -15 and consec_up <= max_consec):
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


def validate_config(label, session_hours, max_consec, session_label, consec_label):
    """Cross-period validation for a config."""
    total_n = 0; total_w = 0; total_pnl = 0.0
    results = []

    for start, end, pn in PERIODS:
        r_eth = backtest_with_filters('ETHUSDT', start, end, session_hours, max_consec)
        r_btc = backtest_with_filters('BTCUSDT', start, end, session_hours, max_consec)
        n_eth, wr_eth, pnl_eth = r_eth[0], r_eth[1], r_eth[2]
        n_btc, wr_btc, pnl_btc = r_btc[0], r_btc[1], r_btc[2]
        total_n += n_eth + n_btc
        total_w += (n_eth*wr_eth/100 if n_eth else 0) + (n_btc*wr_btc/100 if n_btc else 0)
        total_pnl += pnl_eth + pnl_btc
        eth_ok = 'OK' if wr_eth > 55.56 else '--'
        btc_ok = 'OK' if wr_btc > 55.56 else '--'
        results.append(f"  {pn:<8} ETH:{n_eth:>4}t {wr_eth:>5.1f}% ${pnl_eth:>+6.0f} {eth_ok}  BTC:{n_btc:>4}t {wr_btc:>5.1f}% ${pnl_btc:>+6.0f} {btc_ok}")

    total_wr = total_w/total_n*100 if total_n else 0
    star = ' *** 65%+' if total_wr >= 65 else (' ** 60%+' if total_wr >= 60 else '')
    print(f"\n  [{label}] Session={session_label} Consec<={consec_label}")
    for r in results:
        print(r)
    print(f"  {'TOTAL':<8} {total_n:>5}t {total_wr:>5.1f}% ${total_pnl:>+7.0f}/mo avg {total_n//30}t{star}")
    return total_n, total_wr, total_pnl


# Session hour sets
ASIA = frozenset([0,1,2,3,4,5,6,7])
ASIA_NIGHT = frozenset([0,1,2,3,4,5,6,7,22,23])
EU_US = frozenset([7,8,9,10,11,12,13,14,15,16,17])
ALL_HOURS = frozenset(range(24))

print("="*100)
print("  SESSION + CONSEC BAR FILTER VALIDATION")
print("  Base config: rsi<18, bb<.05, stoch<8, ret1<-.03, adx<35, cci<-200, wr<-85")
print("="*100)

configs = [
    # Baseline: all hours, no consec limit
    ("Baseline (no filter)", ALL_HOURS, 99, "All", "none"),
    # Session filters
    ("Asia only", ASIA, 99, "0-7", "none"),
    ("Asia+Night", ASIA_NIGHT, 99, "0-7+22-23", "none"),
    ("Not EU/US", ALL_HOURS - EU_US, 99, "!EU/US", "none"),
    # Consec bar filters
    ("Consec<=3", ALL_HOURS, 3, "All", "3"),
    ("Consec<=2", ALL_HOURS, 2, "All", "2"),
    # Session + Consec combos
    ("Asia only + Consec<=3", ASIA, 3, "0-7", "3"),
    ("Asia only + Consec<=2", ASIA, 2, "0-7", "2"),
    ("Asia+Night + Consec<=3", ASIA_NIGHT, 3, "0-7+22-23", "3"),
    ("Asia+Night + Consec<=2", ASIA_NIGHT, 2, "0-7+22-23", "2"),
    ("!EU/US + Consec<=3", ALL_HOURS - EU_US, 3, "!EU/US", "3"),
    # Aggressive: Asia only + Consec<=4
    ("Asia+Consec<=4", ASIA, 4, "0-7", "4"),
]

best_wr = 0; best_label = ''
for label, hours, max_c, sl, cl in configs:
    n, wr, pnl = validate_config(label, hours, max_c, sl, cl)
    if wr > best_wr:
        best_wr = wr; best_label = label

print(f"\n{'='*100}")
print(f"  BEST: {best_label} — {best_wr:.1f}% WR")
print(f"{'='*100}")
