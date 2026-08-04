"""
Entry timing improvements to bridge 62% → 65% WR:
1. RSI re-cross: wait for RSI to recover above threshold before entry
2. Exhaustion count: require N consecutive bars in signal direction
3. Limit entry: use limit order at signal close instead of market open
4. Loss analysis: find patterns in losing trades to filter them out
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from collections import Counter
from event_backtest.data_fetcher import load_all
from event_backtest.indicators import (
    rsi, bollinger_bands, adx, atr, atr_pct,
    cci, williams_r, stochastic_rsi,
)

def tf_idx(ts, t):
    for i in range(len(ts)-1,-1,-1):
        if ts[i] <= t: return i
    return -1

# Tight config (best from combo_search)
TIGHT = dict(rsi=18, bb=0.05, stoch=8, ret1=0.03, adx=35, cci=-200, wr=-85)

PERIODS = [
    ('2024-01-01', '2024-07-01', '2024H1'),
    ('2024-07-01', '2025-01-01', '2024H2'),
    ('2025-01-01', '2025-07-01', '2025H1'),
    ('2025-07-01', '2026-01-01', '2025H2'),
    ('2026-01-01', '2026-07-01', '2026H1'),
]


def backtest_entry_variants(symbol, start, end, conds):
    """Test multiple entry variants in one pass, return detailed trade data."""
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
    min_atr = 0.05 if symbol == 'ETHUSDT' else 0.03

    LOWS = frozenset([22,23,0,1,2,3,4])

    # Results for different entry methods
    results = {
        'baseline': {'trades': [], 'eq': 500},
        'rsi_recross': {'trades': [], 'eq': 500},
        'limit_entry': {'trades': [], 'eq': 500},
        'exhaustion': {'trades': [], 'eq': 500},
    }

    last_sig = {k: -999 for k in results}
    day_bets = {k: 0 for k in results}
    cur_day = None

    for i in range(warmup, total-2):
        ts = c5[i][0]; dt = datetime.fromtimestamp(ts/1000)
        if dt.day != cur_day:
            cur_day = dt.day
            for k in day_bets: day_bets[k] = 0

        p = cl[i]; bb_p = (p-bb_l[i])/(bb_u[i]-bb_l[i]) if bb_u[i]>bb_l[i] else 0.5
        r1 = (cl[i]/cl[i-1]-1)*100 if i>0 and cl[i-1]>0 else 0
        i1h = tf_idx(t1h, t5[i] - 55*60*1000)
        if i1h < 20: continue
        adx_v = adx_h[i1h]; di_d = pdi[i1h]-mdi[i1h]

        rsi_th = conds.get('rsi', 20); bb_th = conds.get('bb', 0.08)
        stoch_th = conds.get('stoch', 10); ret1_th = conds.get('ret1', 0.02)
        adx_th = conds.get('adx', 40); cci_th = conds.get('cci', -150)
        wr_th = conds.get('wr', None)

        signal_dir = None  # 'up' or 'down'

        # Check if baseline signal fires
        if (rsi_vals[i] < rsi_th and bb_p < bb_th and sk[i] < stoch_th
            and r1 < -ret1_th and adx_v < adx_th and c14[i] < cci_th
            and (wr_th is None or wr14[i] < wr_th)):
            signal_dir = 'up'
        elif (rsi_vals[i] > (100-rsi_th) and bb_p > (1-bb_th) and sk[i] > (100-stoch_th)
            and r1 > ret1_th and adx_v < adx_th and c14[i] > -cci_th
            and (wr_th is None or wr14[i] > -wr_th)):
            signal_dir = 'down'

        if signal_dir is None: continue

        # Count consecutive bars in signal direction before this signal
        consec = 0
        for j in range(i-1, max(i-10, 0), -1):
            if signal_dir == 'up' and cl[j] <= cl[j+1]: break
            if signal_dir == 'down' and cl[j] >= cl[j+1]: break
            consec += 1

        # ============================================
        # 1. BASELINE: enter at next open (reference)
        # ============================================
        key = 'baseline'
        if (i - last_sig[key] >= 2 and day_bets[key] < 50
            and results[key]['eq'] > 25 and atr_p[i] >= min_atr
            and not (dt.hour in LOWS and day_bets[key] >= 16)):
            entry = c5[i+1][1]; settle = c5[min(i+2,total-1)][4]
            win = (signal_dir=='up' and settle>entry) or (signal_dir=='down' and settle<entry)
            pnl = 20 if win else -25; results[key]['eq'] += pnl
            results[key]['trades'].append({
                'time': dt, 'dir': signal_dir, 'win': win, 'pnl': pnl,
                'rsi': rsi_vals[i], 'r1': r1, 'consec': consec,
                'adx': adx_v, 'di_d': di_d, 'hour': dt.hour,
                'bb_p': bb_p, 'cci': c14[i], 'wr': wr14[i],
                'vol_chg': vol[i]/vol[i-1] if i>0 and vol[i-1]>0 else 1,
            })
            last_sig[key] = i; day_bets[key] += 1

        # ============================================
        # 2. RSI RE-CROSS: wait for RSI to recover past threshold
        # ============================================
        key = 'rsi_recross'
        re_cross_th = conds.get('recross_th', 25)
        if signal_dir == 'up':
            # RSI was below threshold (already verified), now check if it crossed back up
            rsi_crossed_up = (i > 0 and rsi_vals[i] >= re_cross_th and rsi_vals[i-1] < re_cross_th)
        else:
            rsi_crossed_up = (i > 0 and rsi_vals[i] <= (100-re_cross_th) and rsi_vals[i-1] > (100-re_cross_th))

        if rsi_crossed_up:
            if (i - last_sig[key] >= 2 and day_bets[key] < 50
                and results[key]['eq'] > 25 and atr_p[i] >= min_atr
                and not (dt.hour in LOWS and day_bets[key] >= 16)):
                entry = c5[i+1][1]; settle = c5[min(i+2,total-1)][4]
                win = (signal_dir=='up' and settle>entry) or (signal_dir=='down' and settle<entry)
                pnl = 20 if win else -25; results[key]['eq'] += pnl
                results[key]['trades'].append({
                    'time': dt, 'dir': signal_dir, 'win': win, 'pnl': pnl,
                    'rsi': rsi_vals[i], 'consec': consec,
                })
                last_sig[key] = i; day_bets[key] += 1

        # ============================================
        # 3. LIMIT ENTRY: use signal close instead of next open
        # ============================================
        key = 'limit_entry'
        if (i - last_sig[key] >= 2 and day_bets[key] < 50
            and results[key]['eq'] > 25 and atr_p[i] >= min_atr
            and not (dt.hour in LOWS and day_bets[key] >= 16)):
            # Enter at signal candle close (better price for mean reversion)
            entry = cl[i]  # signal close, not next open
            settle = c5[min(i+2,total-1)][4]
            win = (signal_dir=='up' and settle>entry) or (signal_dir=='down' and settle<entry)
            pnl = 20 if win else -25; results[key]['eq'] += pnl
            results[key]['trades'].append({
                'time': dt, 'dir': signal_dir, 'win': win, 'pnl': pnl,
                'consec': consec, 'rsi': rsi_vals[i],
            })
            last_sig[key] = i; day_bets[key] += 1

        # ============================================
        # 4. EXHAUSTION: require >= N consecutive bars in signal direction
        # ============================================
        key = 'exhaustion'
        min_consec = conds.get('min_consec', 2)
        if consec >= min_consec:
            if (i - last_sig[key] >= 2 and day_bets[key] < 50
                and results[key]['eq'] > 25 and atr_p[i] >= min_atr
                and not (dt.hour in LOWS and day_bets[key] >= 16)):
                entry = c5[i+1][1]; settle = c5[min(i+2,total-1)][4]
                win = (signal_dir=='up' and settle>entry) or (signal_dir=='down' and settle<entry)
                pnl = 20 if win else -25; results[key]['eq'] += pnl
                results[key]['trades'].append({
                    'time': dt, 'dir': signal_dir, 'win': win, 'pnl': pnl,
                    'consec': consec, 'rsi': rsi_vals[i],
                })
                last_sig[key] = i; day_bets[key] += 1

    # Summarize
    summary = {}
    for key in results:
        trades = results[key]['trades']
        n = len(trades)
        w = sum(1 for t in trades if t['win'])
        wr = w/n*100 if n else 0
        pnl = sum(t['pnl'] for t in trades)
        summary[key] = {'n': n, 'wr': wr, 'pnl': pnl, 'eq': results[key]['eq'],
                        'trades': trades}
    return summary


def analyze_losers(trades):
    """Find patterns that distinguish winners from losers."""
    winners = [t for t in trades if t['win']]
    losers = [t for t in trades if not t['win']]

    if not losers: return

    print(f"\n  --- Loss Analysis ({len(winners)}W / {len(losers)}L) ---")

    # Hour distribution
    win_hours = Counter(t['hour'] for t in winners)
    loss_hours = Counter(t['hour'] for t in losers)
    print(f"  Hour WR by session:")
    for session, hours in [('Asia(0-7)', range(0,8)), ('EU(7-12)', range(7,13)),
                            ('US(12-17)', range(12,18)), ('Eve(17-22)', range(17,23)),
                            ('Night(22-24)', range(22,24))]:
        s_w = sum(win_hours.get(h,0) for h in hours)
        s_l = sum(loss_hours.get(h,0) for h in hours)
        s_wr = s_w/(s_w+s_l)*100 if (s_w+s_l) else 0
        print(f"    {session}: {s_w}W/{s_l}L = {s_wr:.1f}%")

    # RSI of losers vs winners
    if 'rsi' in winners[0]:
        avg_rsi_w = sum(t['rsi'] for t in winners)/len(winners)
        avg_rsi_l = sum(t['rsi'] for t in losers)/len(losers)
        print(f"  Avg RSI: Winners={avg_rsi_w:.1f} Losers={avg_rsi_l:.1f}")

    # Consecutive bars
    if 'consec' in winners[0]:
        avg_cons_w = sum(t['consec'] for t in winners)/len(winners)
        avg_cons_l = sum(t['consec'] for t in losers)/len(losers)
        print(f"  Avg Consec bars: Winners={avg_cons_w:.1f} Losers={avg_cons_l:.1f}")

    # ADX of losers vs winners
    if 'adx' in winners[0]:
        avg_adx_w = sum(t['adx'] for t in winners)/len(winners)
        avg_adx_l = sum(t['adx'] for t in losers)/len(losers)
        print(f"  Avg ADX: Winners={avg_adx_w:.1f} Losers={avg_adx_l:.1f}")

    # DI diff
    if 'di_d' in winners[0]:
        avg_di_w = sum(abs(t['di_d']) for t in winners)/len(winners)
        avg_di_l = sum(abs(t['di_d']) for t in losers)/len(losers)
        print(f"  Avg |DI diff|: Winners={avg_di_w:.1f} Losers={avg_di_l:.1f}")

    # BB position
    if 'bb_p' in winners[0]:
        avg_bb_w = sum(t['bb_p'] for t in winners)/len(winners)
        avg_bb_l = sum(t['bb_p'] for t in losers)/len(losers)
        print(f"  Avg BB pos: Winners={avg_bb_w:.3f} Losers={avg_bb_l:.3f}")

    # Volume change
    if 'vol_chg' in winners[0]:
        avg_vol_w = sum(t['vol_chg'] for t in winners)/len(winners)
        avg_vol_l = sum(t['vol_chg'] for t in losers)/len(losers)
        print(f"  Avg Vol change: Winners={avg_vol_w:.2f}x Losers={avg_vol_l:.2f}x")

    # Consec distribution for WR
    if 'consec' in winners[0]:
        print(f"  WR by consecutive bars:")
        for c in sorted(set(t['consec'] for t in trades)):
            c_w = sum(1 for t in winners if t['consec'] == c)
            c_l = sum(1 for t in losers if t['consec'] == c)
            c_wr = c_w/(c_w+c_l)*100 if (c_w+c_l) else 0
            print(f"    consec={c}: {c_w}W/{c_l}L = {c_wr:.1f}%")


print(f"\n{'='*100}")
print(f"  ENTRY TIMING IMPROVEMENTS + LOSS ANALYSIS")
print(f"  Config: rsi<18, bb<.05, stoch<8, ret1<-.03, adx<35, cci<-200, wr<-85")
print(f"{'='*100}")

SYM = 'ETHUSDT'

# Phase 1: Deep test on 2026H1
print(f"\n--- Phase 1: 2026H1 Deep Test ---")
summary = backtest_entry_variants(SYM, '2026-01-01', '2026-07-01', TIGHT)

for key in ['baseline', 'rsi_recross', 'limit_entry', 'exhaustion']:
    s = summary[key]
    star = ' ***' if s['wr'] >= 65 else (' **' if s['wr'] >= 60 else '')
    print(f"  {key:<20} {s['n']:>5} trades  {s['wr']:>5.1f}% WR  ${s['pnl']:>+8.0f}{star}")

# Loss analysis on baseline
analyze_losers(summary['baseline']['trades'])

# Phase 2: Test exhaustion filter variations
print(f"\n--- Exhaustion Filter Variations ---")
for min_cons in [1, 2, 3, 4]:
    conds = {**TIGHT, 'min_consec': min_cons}
    s = backtest_entry_variants(SYM, '2026-01-01', '2026-07-01', conds)
    ex = s['exhaustion']
    star = ' ***' if ex['wr'] >= 65 else (' **' if ex['wr'] >= 60 else '')
    print(f"  min_consec>={min_cons:<5} {ex['n']:>5} trades  {ex['wr']:>5.1f}% WR  ${ex['pnl']:>+8.0f}{star}")

# Phase 3: RSI re-cross threshold variations
print(f"\n--- RSI Re-Cross Threshold Variations ---")
for re_th in [20, 22, 25, 28, 30]:
    conds = {**TIGHT, 'recross_th': re_th}
    s = backtest_entry_variants(SYM, '2026-01-01', '2026-07-01', conds)
    rc = s['rsi_recross']
    star = ' ***' if rc['wr'] >= 65 else (' **' if rc['wr'] >= 60 else '')
    print(f"  recross>{re_th:<5} {rc['n']:>5} trades  {rc['wr']:>5.1f}% WR  ${rc['pnl']:>+8.0f}{star}")

# Phase 4: Best combo — exhaustion + limit entry
print(f"\n--- Combo: Exhaustion + Limit Entry ---")
for min_cons in [2, 3]:
    for recross_th in [22, 25]:
        conds = {**TIGHT, 'min_consec': min_cons, 'recross_th': recross_th}
        s = backtest_entry_variants(SYM, '2026-01-01', '2026-07-01', conds)
        ex = s['exhaustion']; rc = s['rsi_recross']; li = s['limit_entry']
        print(f"  consec>={min_cons} recross>{recross_th}:")
        print(f"    exhaustion:  {ex['n']:>4} trades  {ex['wr']:>5.1f}% WR  ${ex['pnl']:>+8.0f}")
        print(f"    rsi_recross: {rc['n']:>4} trades  {rc['wr']:>5.1f}% WR  ${rc['pnl']:>+8.0f}")
        print(f"    limit_entry: {li['n']:>4} trades  {li['wr']:>5.1f}% WR  ${li['pnl']:>+8.0f}")

# Phase 5: Validate best approaches across 5 periods
print(f"\n{'='*100}")
print(f"  Phase 5: 5-Period Validation of Best Approaches")
print(f"{'='*100}")

best_approaches = [
    ("baseline", {}),
    ("exhaustion>=3", {'min_consec': 3}),
    ("limit_entry", {}),
    ("exhaust>=3+limit", {'min_consec': 3}),
]

for label, extra in best_approaches:
    conds = {**TIGHT, **extra}
    total_n = 0; total_w = 0; total_pnl = 0.0
    print(f"\n  [{label}]")
    key = 'exhaustion' if 'min_consec' in extra else ('limit_entry' if 'limit_entry' in label else 'baseline')
    key = 'exhaustion' if 'exhaust' in label else key

    for start, end, pn in PERIODS:
        s = backtest_entry_variants('ETHUSDT', start, end, conds)
        r = s[key]
        total_n += r['n']; total_w += r['n']*r['wr']/100 if r['n'] else 0; total_pnl += r['pnl']
        ok = 'OK' if r['wr'] > 55.56 else '--'
        print(f"    {pn}: {r['n']:>4} trades  {r['wr']:>5.1f}% WR  ${r['pnl']:>+8.0f} {ok}")
    total_wr = total_w/total_n*100 if total_n else 0
    star = ' *** 65%+' if total_wr >= 65 else (' ** 60%+' if total_wr >= 60 else '')
    print(f"    TOTAL: {total_n} trades  {total_wr:.1f}% WR  ${total_pnl:+.0f}{star}")
