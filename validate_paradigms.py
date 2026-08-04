"""
Quick 5-period validation of best configs from new paradigms search.
Tests: Keltner Channel, Triple-TF RSI, KC+AND hybrid
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from event_backtest.data_fetcher import load_all
from event_backtest.indicators import (
    rsi, bollinger_bands, adx, atr, atr_pct, volume_spike,
    cci, williams_r, stochastic_rsi, mfi, ema, sma, vwap,
)

def tf_idx(ts, t):
    for i in range(len(ts)-1,-1,-1):
        if ts[i] <= t: return i
    return -1

def keltner(highs, lows, closes, period=20, atr_mult=2.0):
    n = len(closes)
    middle = ema(closes, period)
    atr_vals = atr(highs, lows, closes, period)
    upper = [middle[i] + atr_mult * atr_vals[i] for i in range(n)]
    lower = [middle[i] - atr_mult * atr_vals[i] for i in range(n)]
    return middle, upper, lower

PERIODS = [
    ('2024-01-01', '2024-07-01', '2024H1'),
    ('2024-07-01', '2025-01-01', '2024H2'),
    ('2025-01-01', '2025-07-01', '2025H1'),
    ('2025-07-01', '2026-01-01', '2025H2'),
    ('2026-01-01', '2026-07-01', '2026H1'),
]


def backtest_kc_and_hybrid(symbol, start, end, conds):
    """Keltner Channel + optional AND-condition filters."""
    data = load_all(symbol, start, end)
    c5 = data['5m']
    cl = [c[4] for c in c5]; op = [c[1] for c in c5]
    hi = [c[2] for c in c5]; lo = [c[3] for c in c5]
    vol = [c[5] for c in c5]
    c1h = [c[4] for c in data['1h']]; h1h = [c[2] for c in data['1h']]; l1h = [c[3] for c in data['1h']]
    t5 = [c[0] for c in c5]; t1h = [c[0] for c in data['1h']]
    c15 = data.get('15m')
    cl15 = [c[4] for c in c15] if c15 else None
    t15 = [c[0] for c in c15] if c15 else None
    total = len(c5); warmup = 60

    kc_mid, kc_up, kc_low = keltner(hi, lo, cl,
        period=conds.get('kc_period', 20), atr_mult=conds.get('kc_mult', 2.0))
    kc_pos = [(cl[i] - kc_low[i]) / (kc_up[i] - kc_low[i]) if kc_up[i] > kc_low[i] else 0.5
              for i in range(total)]

    rsi7 = rsi(cl, 7)
    sk, sd = stochastic_rsi(cl, 14, 14)
    c14 = cci(hi, lo, cl, 14)
    wr14 = williams_r(hi, lo, cl, 14)
    adx_h, pdi, mdi = adx(h1h, l1h, c1h, 14)
    atr_p5 = atr_pct(atr(hi, lo, cl, 14), cl)

    # Triple-TF RSI
    if cl15 and conds.get('triple_rsi', False):
        rsi15 = rsi(cl15, 7)
        rsi1h_vals = rsi([c[4] for c in data['1h']], 7)
        t1h_ts = [c[0] for c in data['1h']]

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

        adx_v = adx_h[i1h]; di_d = pdi[i1h]-mdi[i1h]
        r1 = (cl[i]/cl[i-1]-1)*100 if i>0 and cl[i-1]>0 else 0

        direction = None
        use_and = conds.get('use_and', False)

        if use_and:
            # KC + AND-condition filters
            rsi_th = conds.get('rsi', 20); kc_th = conds.get('kc_pos_th', 0.05)
            stoch_th = conds.get('stoch', 10); ret1_th = conds.get('ret1', 0.02)
            adx_th = conds.get('adx', 40); cci_th = conds.get('cci', -150)

            if (kc_pos[i] < kc_th and rsi7[i] < rsi_th and sk[i] < stoch_th
                and r1 < -ret1_th and adx_v < adx_th and c14[i] < cci_th
                and (conds.get('wr') is None or wr14[i] < conds.get('wr'))):
                direction = 'up'
            elif (kc_pos[i] > (1-kc_th) and rsi7[i] > (100-rsi_th) and sk[i] > (100-stoch_th)
                and r1 > ret1_th and adx_v < adx_th and c14[i] > -cci_th
                and (conds.get('wr') is None or wr14[i] > -conds.get('wr'))):
                direction = 'down'
        elif conds.get('triple_rsi', False):
            # Triple-TF RSI
            i15 = tf_idx(t15, t5[i] - 10*60*1000)
            i1h_rsi = tf_idx(t1h_ts, t5[i] - 55*60*1000)
            if i15 < 20 or i1h_rsi < 20: continue
            rsi_ext = conds.get('rsi_ext', 25)

            if (rsi7[i] < rsi_ext and rsi15[i15] < rsi_ext and rsi1h_vals[i1h_rsi] < rsi_ext
                and kc_pos[i] < conds.get('kc_pos_th', 0.12)
                and sk[i] < conds.get('stoch', 15) and adx_v < conds.get('adx', 45)):
                direction = 'up'
            elif (rsi7[i] > (100-rsi_ext) and rsi15[i15] > (100-rsi_ext) and rsi1h_vals[i1h_rsi] > (100-rsi_ext)
                and kc_pos[i] > (1-conds.get('kc_pos_th', 0.12))
                and sk[i] > (100-conds.get('stoch', 15)) and adx_v < conds.get('adx', 45)):
                direction = 'down'
        else:
            # Pure Keltner
            if (kc_pos[i] < conds.get('kc_pos_th', 0.05) and rsi7[i] < conds.get('rsi', 25)
                and sk[i] < conds.get('stoch', 15) and adx_v < conds.get('adx', 40)):
                direction = 'up'
            elif (kc_pos[i] > conds.get('kc_pos_high', 0.95) and rsi7[i] > conds.get('rsi_sell', 75)
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


def validate(configs):
    print(f"\n{'='*120}")
    print(f"  5-PERIOD VALIDATION — Best New Paradigms")
    print(f"  Breakeven: 55.56% | 10m (80%) | $25/trade")
    print(f"{'='*120}")

    best_wr = 0; best_label = ''

    for label, conds in configs:
        total_n = 0; total_w = 0; total_pnl = 0.0
        print(f"\n  [{label}]")
        print(f"  {'Period':<10} {'ETH Trades':>10} {'ETH WR':>8} {'ETH PnL':>10} {'BTC Trades':>10} {'BTC WR':>8} {'BTC PnL':>10}")

        for start, end, pn in PERIODS:
            r_eth = backtest_kc_and_hybrid('ETHUSDT', start, end, conds)
            r_btc = backtest_kc_and_hybrid('BTCUSDT', start, end, conds)
            total_n += r_eth[0] + r_btc[0]
            total_w += (r_eth[0]*r_eth[1]/100 if r_eth[0] else 0) + (r_btc[0]*r_btc[1]/100 if r_btc[0] else 0)
            total_pnl += r_eth[2] + r_btc[2]
            eth_ok = 'OK' if r_eth[1] > 55.56 else '--'
            btc_ok = 'OK' if r_btc[1] > 55.56 else '--'
            print(f"  {pn:<10} {r_eth[0]:>10} {r_eth[1]:>7.1f}% ${r_eth[2]:>+9.0f} {eth_ok}  {r_btc[0]:>10} {r_btc[1]:>7.1f}% ${r_btc[2]:>+9.0f} {btc_ok}")

        total_wr = total_w/total_n*100 if total_n else 0
        star = ' *** 65%+' if total_wr >= 65 else (' ** 60%+' if total_wr >= 60 else '')
        print(f"  {'TOTAL':<10} {total_n:>10} {total_wr:>7.1f}% ${total_pnl:>+9.0f}{star}")

        if total_wr > best_wr:
            best_wr = total_wr; best_label = label

    print(f"\n{'='*120}")
    print(f"  BEST: {best_label} — {best_wr:.1f}% WR")
    print(f"{'='*120}")


configs = [
    # Best Keltner: KC(15,2.0), RSI<30
    ("KC15x2.0 RSI<25", dict(kc_period=15, kc_mult=2.0, kc_pos_th=0.05, kc_pos_high=0.95,
                             rsi=25, rsi_sell=75, stoch=15, stoch_sell=85, adx=40)),
    # Keltner + RSI<20
    ("KC15x2.0 RSI<20", dict(kc_period=15, kc_mult=2.0, kc_pos_th=0.05, kc_pos_high=0.95,
                             rsi=20, rsi_sell=80, stoch=15, stoch_sell=85, adx=40)),
    # KC hybrid: KC + AND conditions
    ("KC15x2.0 +AND(rsi20,st10,r1.02,adx40,cci-150)", dict(
        use_and=True, kc_period=15, kc_mult=2.0, kc_pos_th=0.08,
        rsi=20, stoch=10, ret1=0.02, adx=40, cci=-150)),
    # KC hybrid + WR
    ("KC15x2.0 +AND(rsi20,st10,r1.02,adx40,cci-150) +WR<-85", dict(
        use_and=True, kc_period=15, kc_mult=2.0, kc_pos_th=0.08,
        rsi=20, stoch=10, ret1=0.02, adx=40, cci=-150, wr=-85)),
    # Triple-TF RSI best
    ("TripleRSI<25 KC_pos<0.12 stoch<15 adx<45", dict(
        triple_rsi=True, rsi_ext=25, kc_pos_th=0.12, stoch=15, adx=45)),
    # Triple-TF RSI tighter
    ("TripleRSI<20 KC_pos<0.10 stoch<12 adx<45", dict(
        triple_rsi=True, rsi_ext=20, kc_pos_th=0.10, stoch=12, adx=45)),
    # KC + tighter AND
    ("KC15x2.0 +AND(rsi18,st8,r1.03,adx35,cci-200) +WR<-85", dict(
        use_and=True, kc_period=15, kc_mult=2.0, kc_pos_th=0.05,
        rsi=18, stoch=8, ret1=0.03, adx=35, cci=-200, wr=-85)),
]

validate(configs)
