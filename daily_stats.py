"""
Daily signal statistics report.
Usage: python live_bot/daily_stats.py [--date YYYYMMDD]
Reads today's settlements CSV and prints win/loss stats for 7:00-23:00 Beijing time.
"""
import csv
import os
import sys
from datetime import datetime, timezone, timedelta

BEIJING_TZ = timezone(timedelta(hours=8))
LOGS_DIR = os.path.join(os.path.dirname(__file__), 'logs')


def daily_stats(date_str: str = None):
    """Print daily statistics from settlement CSV."""
    if date_str is None:
        date_str = datetime.now(BEIJING_TZ).strftime('%Y%m%d')

    settle_path = os.path.join(LOGS_DIR, f'settlements_{date_str}.csv')
    signals_path = os.path.join(LOGS_DIR, f'signals_{date_str}.csv')

    print(f"\n{'='*55}")
    print(f"  每日信号统计 — {date_str}")
    print(f"{'='*55}")

    # Read settlements
    settlements = []
    if os.path.exists(settle_path):
        with open(settle_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dt = datetime.strptime(row['signal_time'], '%Y-%m-%d %H:%M:%S')
                dt = dt.replace(tzinfo=BEIJING_TZ)
                # Filter: 7:00-23:00 only
                if 7 <= dt.hour < 23:
                    settlements.append({
                        **row,
                        'dt': dt,
                        'pnl': float(row['pnl']),
                    })
        print(f"\n已结算: {len(settlements)} 笔 (7:00-23:00)")
    else:
        print(f"\n⚠ 结算文件不存在: {settle_path}")

    # Count total signals (including unsettled)
    total_signals = 0
    if os.path.exists(signals_path):
        with open(signals_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dt = datetime.strptime(row['time'], '%Y-%m-%d %H:%M:%S')
                dt = dt.replace(tzinfo=BEIJING_TZ)
                if 7 <= dt.hour < 23:
                    total_signals += 1
        print(f"总信号: {total_signals} 条")

    if not settlements:
        print("\n暂无已结算数据。信号触发后需等待10分钟自动结算。\n")
        return

    # Stats
    wins = [s for s in settlements if s['result'] == 'WIN']
    losses = [s for s in settlements if s['result'] != 'WIN']
    total_pnl = sum(s['pnl'] for s in settlements)
    win_rate = len(wins) / len(settlements) * 100 if settlements else 0

    # By symbol
    eth = [s for s in settlements if s['symbol'] == 'ETHUSDT']
    btc = [s for s in settlements if s['symbol'] == 'BTCUSDT']
    eth_wins = [s for s in eth if s['result'] == 'WIN']
    btc_wins = [s for s in btc if s['result'] == 'WIN']

    print(f"\n{'─'*55}")
    print(f"  胜: {len(wins)}  负: {len(losses)}  胜率: {win_rate:.1f}%")
    print(f"  总盈亏: ${total_pnl:+.2f}")
    if eth:
        print(f"  ETH: {len(eth_wins)}/{len(eth)} 胜率 {len(eth_wins)/len(eth)*100:.1f}%")
    if btc:
        print(f"  BTC: {len(btc_wins)}/{len(btc)} 胜率 {len(btc_wins)/len(btc)*100:.1f}%")

    # By direction
    ups = [s for s in settlements if s['direction'] == 'up']
    dns = [s for s in settlements if s['direction'] == 'down']
    if ups:
        up_wins = [s for s in ups if s['result'] == 'WIN']
        print(f"  做多: {len(up_wins)}/{len(ups)} 胜率 {len(up_wins)/len(ups)*100:.1f}%")
    if dns:
        dn_wins = [s for s in dns if s['result'] == 'WIN']
        print(f"  做空: {len(dn_wins)}/{len(dns)} 胜率 {len(dn_wins)/len(dns)*100:.1f}%")

    # Signal list
    print(f"\n{'─'*55}")
    print(f"  {'时间':<8} {'品种':<8} {'方向':<6} {'入场':>8} {'出场':>8} {'结果':<6} {'盈亏':>7}")
    print(f"  {'─'*55}")
    for s in sorted(settlements, key=lambda x: x['dt']):
        tag = '✅' if s['result'] == 'WIN' else '❌'
        print(f"  {s['dt'].strftime('%H:%M'):<8} {s['symbol']:<8} "
              f"{s['direction']:<6} {float(s['entry_price']):>8.2f} "
              f"{float(s['exit_price']):>8.2f} {tag:<6} {s['pnl']:>+7.2f}")

    # Unsettled reminder
    pending = total_signals - len(settlements)
    if pending > 0:
        print(f"\n  ⏳ 还有 {pending} 条信号等待结算...")

    print()


if __name__ == '__main__':
    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    daily_stats(date_arg)
