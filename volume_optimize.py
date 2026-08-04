"""
Find the best balance: WR ≥ 60% with maximum trade volume.
Tests: consec<=2 vs <=3, tight vs relaxed AND, ETH-only vs dual.
"""
import sys, os
sys.path.insert(0, 'D:/code/demo/狗哥的视频知识库/dogeTxt')

from datetime import datetime
from event_backtest.data_fetcher import load_all
from event_backtest.indicators import *

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

def bt(symbol, start, end, conds):
    data = load_all(symbol, start, end)
    c5 = data['5m']
    cl = [c[4] for c in c5]; hi = [c[2] for c in c5]; lo = [c[3] for c in c5]
    c1h = [c[4] for c in data['1h']]; h1h = [c[2] for c in data['1h']]; l1h = [c[3] for c in data['1h']]
    t5 = [c[0] for c in c5]; t1h = [c[0] for c in data['1h']]
    total = len(c5); warmup = 60

    bb_mid, bb_up, bb_low = bollinger_bands(cl, 20, 2.0)
    bb_pos = [(cl[i] - bb_low[i]) / (bb_up[i] - bb_low[i]) if bb_up[i] > bb_low[i] else 0.5 for i in range(total)]
    rsi7 = rsi(cl, 7)
    sk, sd = stochastic_rsi(cl, 14, 14)
    c14 = cci(hi, lo, cl, 14)
    wr14 = williams_r(hi, lo, cl, 14)
    adx_h, pdi, mdi = adx(h1h, l1h, c1h, 14)
    atr_p5 = atr_pct(atr(hi, lo, cl, 14), cl)
    min_atr = 0.05 if symbol == 'ETHUSDT' else 0.03
    eq = 500; last_sig = -999; day_bets = 0; cur_day = None

    rsi_th = conds['rsi']; bb_th = conds['bb']; stoch_th = conds['stoch']
    ret1_th = conds['ret1']; adx_th = conds['adx']; cci_th = conds['cci']
    wr_th = conds.get('wr'); max_c = conds['max_consec']

    wins = 0; total_t = 0; total_pnl = 0.0

    for i in range(warmup, total-2):
        ts = c5[i][0]; dt = datetime.fromtimestamp(ts/1000)
        if dt.day != cur_day: cur_day = dt.day; day_bets = 0
        if i - last_sig < 2: continue
        if day_bets >= 50: continue
        if eq <= 25: break
        if atr_p5[i] < min_atr: continue
        i1h = tf_idx(t1h, t5[i] - 55*60*1000)
        if i1h < 20: continue

        adx_v = adx_h[i1h]
        r1 = (cl[i]/cl[i-1]-1)*100 if i>0 and cl[i-1]>0 else 0

        # Count consecutive bars in each direction
        cd = 0
        for j in range(i, max(0, i-20), -1):
            if j > 0 and cl[j] < cl[j-1]: cd += 1
            else: break
        cu = 0
        for j in range(i, max(0, i-20), -1):
            if j > 0 and cl[j] > cl[j-1]: cu += 1
            else: break

        direction = None
        wr_ok = wr_th is None or wr14[i] < wr_th
        wr_ok_s = wr_th is None or wr14[i] > -wr_th

        if (bb_pos[i] < bb_th and rsi7[i] < rsi_th and sk[i] < stoch_th
            and r1 < -ret1_th and adx_v < adx_th and c14[i] < cci_th
            and wr_ok and cd <= max_c):
            direction = 'up'
        elif (bb_pos[i] > (1-bb_th) and rsi7[i] > (100-rsi_th) and sk[i] > (100-stoch_th)
            and r1 > ret1_th and adx_v < adx_th and c14[i] > -cci_th
            and wr_ok_s and cu <= max_c):
            direction = 'down'

        if direction is None: continue
        entry = c5[i+1][1]; settle = c5[min(i+2, total-1)][4]
        win = (direction == 'up' and settle > entry) or (direction == 'down' and settle < entry)
        pnl = 20 if win else -25; eq += pnl
        wins += 1 if win else 0; total_t += 1; total_pnl += pnl
        last_sig = i; day_bets += 1

    wr = wins/total_t*100 if total_t else 0
    return total_t, wr, total_pnl

# Config presets
TIGHT = dict(rsi=18, bb=0.05, stoch=8, ret1=0.03, adx=35, cci=-200, wr=-85)
TIGHT_NOWR = dict(rsi=18, bb=0.05, stoch=8, ret1=0.03, adx=35, cci=-200)
R1 = dict(rsi=20, bb=0.08, stoch=10, ret1=0.02, adx=40, cci=-150)
R1_WR = dict(rsi=20, bb=0.08, stoch=10, ret1=0.02, adx=40, cci=-150, wr=-85)
MID = dict(rsi=20, bb=0.06, stoch=9, ret1=0.03, adx=38, cci=-200, wr=-85)
MID_NOWR = dict(rsi=20, bb=0.06, stoch=9, ret1=0.03, adx=38, cci=-200)

configs = [
    # (label, base_conds, max_consec, coins)
    # Tight AND variants
    ("Tight+consec<=2 ETH", TIGHT, 2, ['ETHUSDT']),
    ("Tight+consec<=3 ETH", TIGHT, 3, ['ETHUSDT']),
    ("Tight+consec<=2 ETH+BTC", TIGHT, 2, ['ETHUSDT', 'BTCUSDT']),
    ("Tight+consec<=3 ETH+BTC", TIGHT, 3, ['ETHUSDT', 'BTCUSDT']),
    # No WR filter (more signals)
    ("Tight(noWR)+consec<=2 ETH", TIGHT_NOWR, 2, ['ETHUSDT']),
    ("Tight(noWR)+consec<=3 ETH", TIGHT_NOWR, 3, ['ETHUSDT']),
    ("Tight(noWR)+consec<=2 ETH+BTC", TIGHT_NOWR, 2, ['ETHUSDT', 'BTCUSDT']),
    ("Tight(noWR)+consec<=3 ETH+BTC", TIGHT_NOWR, 3, ['ETHUSDT', 'BTCUSDT']),
    # Mid (slightly relaxed)
    ("Mid+consec<=2 ETH", MID, 2, ['ETHUSDT']),
    ("Mid+consec<=3 ETH", MID, 3, ['ETHUSDT']),
    ("Mid+consec<=2 ETH+BTC", MID, 2, ['ETHUSDT', 'BTCUSDT']),
    ("Mid+consec<=3 ETH+BTC", MID, 3, ['ETHUSDT', 'BTCUSDT']),
    # Mid no WR
    ("Mid(noWR)+consec<=3 ETH", MID_NOWR, 3, ['ETHUSDT']),
    ("Mid(noWR)+consec<=3 ETH+BTC", MID_NOWR, 3, ['ETHUSDT', 'BTCUSDT']),
    # R1 baseline (most volume)
    ("R1+consec<=3 ETH", R1, 3, ['ETHUSDT']),
    ("R1+consec<=3 ETH+BTC", R1, 3, ['ETHUSDT', 'BTCUSDT']),
    ("R1+WR+consec<=3 ETH", R1_WR, 3, ['ETHUSDT']),
    ("R1+WR+consec<=3 ETH+BTC", R1_WR, 3, ['ETHUSDT', 'BTCUSDT']),
]

print(f"{'='*110}")
print(f"  VOLUME vs WR OPTIMIZATION — 5-Period Validation")
print(f"  Breakeven: 55.56% | $25/trade | 80% payout")
print(f"{'='*110}")
print(f"  {'Config':<35} {'Total':>6} {'WR':>7} {'PnL/mo':>8} {'T/mo':>6} {'ETH WR':>7} {'BTC WR':>7}")
print(f"  {'-'*105}")

results = []
for label, base, max_c, coins in configs:
    conds = {**base, 'max_consec': max_c}
    total_n = 0; total_w = 0; total_pnl = 0.0
    eth_n = 0; eth_w = 0; btc_n = 0; btc_w = 0

    for start, end, pn in PERIODS:
        for coin in coins:
            n, wr, pnl = bt(coin, start, end, conds)
            total_n += n; total_w += n*wr/100 if n else 0; total_pnl += pnl
            if coin == 'ETHUSDT': eth_n += n; eth_w += n*wr/100 if n else 0
            else: btc_n += n; btc_w += n*wr/100 if n else 0

    total_wr = total_w/total_n*100 if total_n else 0
    eth_wr = eth_w/eth_n*100 if eth_n else 0
    btc_wr = btc_w/btc_n*100 if btc_n else 0
    monthly = total_n // 30

    star = '***' if total_wr >= 65 else ('**' if total_wr >= 60 else '--')
    print(f"  {label:<35} {total_n:>6} {total_wr:>6.1f}% ${total_pnl:>+7.0f} {monthly:>5}t  {eth_wr:>6.1f}% {btc_wr:>6.1f}% {star}")

    results.append((total_n, total_wr, total_pnl, monthly, label, eth_wr, btc_wr, star))

# Top picks by volume at different WR thresholds
print(f"\n{'='*110}")
print(f"  SUMMARY: Best by WR bracket")
print(f"{'='*110}")

for min_wr, label_wr in [(65, '65%+'), (60, '60-65%'), (55.56, '>breakeven')]:
    candidates = [(r[3], r[1], r[0], r[4], r[5], r[6]) for r in results if r[1] >= min_wr]
    if candidates:
        best = max(candidates, key=lambda x: x[0])  # most trades/month
        print(f"  {label_wr}: {best[3]} — {best[0]:.0f}t/mo, {best[1]:.1f}% WR, ETH:{best[4]:.1f}% BTC:{best[5]:.1f}%")
