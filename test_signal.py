"""
Test signal generation against historical data.
Verifies live bot produces correct signals compared to backtest.
Usage: python live_bot/test_signal.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from collections import Counter

from event_backtest.data_fetcher import load_all
from indicator_engine import IndicatorEngine
from strategy_engine import StrategyEngine
from risk_manager import RiskManager


def test_signals(symbol: str = 'ETHUSDT', start: str = '2026-06-01',
                 end: str = '2026-07-01', threshold: float = 5.0):
    """Run live bot pipeline on historical data and report signals."""

    print(f'\nLoading {symbol} data {start} ~ {end}...')
    data = load_all(symbol, start, end)
    candles_5m = data['5m']
    candles_15m = data['15m']
    candles_1h = data['1h']

    print(f'{len(candles_5m)} 5m candles loaded.')

    strategy = StrategyEngine({
        'score_threshold': threshold, 'min_atr_pct': 0.08, 'low_vol_hours': []
    })
    risk = RiskManager({
        'cooldown_candles': 2, 'max_daily_trades': 30,
        'daily_loss_limit': 75, 'max_consecutive_loss': 3,
        'low_vol_hours': [],
    })
    engine = IndicatorEngine()

    # Compute all indicators once
    print('Computing indicators...')
    engine.update(candles_5m, candles_15m, candles_1h)
    print('Done.\n')

    signals = []
    warmup = 60

    for i in range(warmup, len(candles_5m) - 2):
        risk.increment_candle(symbol)
        risk.check_daily_reset(candles_5m[i][0])

        ts = candles_5m[i][0]
        signal = strategy.evaluate(engine, symbol, ts, idx_5m=i)
        if signal is None:
            continue

        allowed, reason = risk.can_signal(symbol, signal, candles_5m[i][0])
        if not allowed:
            continue

        risk.record_signal(symbol)
        signals.append(signal)

    print(f'{"="*60}')
    print(f'Results: {len(signals)} signals generated (threshold={threshold})')
    print(f'{"="*60}')

    if not signals:
        print('No signals found.')
        return signals

    up = sum(1 for s in signals if s['direction'] == 'up')
    dn = len(signals) - up
    avg_score = sum(s['score'] for s in signals) / len(signals)
    regimes = Counter(s['regime'] for s in signals)

    print(f'UP: {up} | DOWN: {dn}')
    print(f'Avg score: {avg_score:.1f}')
    print(f'Regimes: {dict(regimes)}')

    daily = Counter(datetime.fromtimestamp(s['timestamp'] / 1000).strftime('%m-%d')
                    for s in signals)
    print(f'Daily avg: {sum(daily.values())/len(daily):.1f} signals/day')

    print(f'\n--- Signal log ---')
    for s in signals:
        dt = datetime.fromtimestamp(s['timestamp'] / 1000)
        dir_str = 'LONG ' if s['direction'] == 'up' else 'SHORT'
        print(f'{dt.strftime("%m-%d %H:%M")}  {dir_str}  score={s["score"]:+.1f}  '
              f'{s["regime"]:9s}  RSI={s["rsi7"]:.0f}  MFI={s["mfi"]:.0f}  '
              f'StochK={s["stoch_k"]:.0f}  price={s["price"]:.2f}  {s.get("reasons", [])}')

    return signals


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', default='ETHUSDT')
    p.add_argument('--start', default='2026-06-01')
    p.add_argument('--end', default='2026-07-01')
    p.add_argument('--threshold', type=float, default=5.0)
    args = p.parse_args()
    test_signals(args.symbol, args.start, args.end, args.threshold)
