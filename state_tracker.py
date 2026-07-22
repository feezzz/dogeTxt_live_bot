"""
State tracker: CSV logging, daily signal history, trade records,
and automatic settlement for 10-min event contracts.
"""
import csv
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import List, Optional

logger = logging.getLogger(__name__)

BEIJING_TZ = timezone(timedelta(hours=8))


class StateTracker:
    """Tracks all signals, maintains CSV log, and auto-settles contracts."""

    def __init__(self, log_dir: str = "live_bot/logs"):
        self._log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._signals_today: List[dict] = []
        self._current_beijing_day = None
        self._csv_path = None
        self._csv_writer = None
        self._csv_file = None

        # Pending: (candle_idx, signal, settle_candles, payout)
        self._pending: dict[str, list[tuple[int, dict, int, float]]] = {}
        self._candle_idx: dict[str, int] = {}
        self._settled_today: List[dict] = []
        self._settle_csv_path = None
        self._settle_csv_file = None
        self._settle_csv_writer = None

    # ------------------------------------------------------------------
    # CSV writers
    # ------------------------------------------------------------------
    def _ensure_csv(self):
        """Create or rotate signal CSV per day."""
        today = datetime.now(BEIJING_TZ).strftime('%Y%m%d')
        path = os.path.join(self._log_dir, f"signals_{today}.csv")

        if path != self._csv_path:
            if self._csv_file:
                self._csv_file.close()

            is_new = not os.path.exists(path)
            self._csv_path = path
            self._csv_file = open(path, 'a', newline='', encoding='utf-8')
            self._csv_writer = csv.writer(self._csv_file)

            if is_new:
                self._csv_writer.writerow([
                    'time', 'symbol', 'direction', 'timeframe', 'score', 'regime',
                    'price', 'rsi7', 'mfi', 'stoch_k', 'adx', 'cci',
                    'atr_pct', 'reasons', 'settled', 'result', 'pnl'
                ])
                self._csv_file.flush()

    def _ensure_settle_csv(self):
        """Create or rotate settlement CSV per day."""
        today = datetime.now(BEIJING_TZ).strftime('%Y%m%d')
        path = os.path.join(self._log_dir, f"settlements_{today}.csv")

        if path != self._settle_csv_path:
            if self._settle_csv_file:
                self._settle_csv_file.close()

            is_new = not os.path.exists(path)
            self._settle_csv_path = path
            self._settle_csv_file = open(path, 'a', newline='', encoding='utf-8')
            self._settle_csv_writer = csv.writer(self._settle_csv_file)

            if is_new:
                self._settle_csv_writer.writerow([
                    'signal_time', 'settle_time', 'symbol', 'direction',
                    'entry_price', 'exit_price', 'score', 'result', 'pnl'
                ])
                self._settle_csv_file.flush()

    # ------------------------------------------------------------------
    # Signal recording
    # ------------------------------------------------------------------
    def record_signal(self, signal: dict):
        """Record a new trade signal."""
        ts = signal['timestamp'] / 1000
        dt = datetime.fromtimestamp(ts, tz=BEIJING_TZ)
        beijing_day = int(dt.strftime('%Y%m%d'))

        # Daily reset
        if beijing_day != self._current_beijing_day:
            self._current_beijing_day = beijing_day
            self._signals_today = []
            self._settled_today = []

        # Store in memory
        self._signals_today.append(signal)

        # Write to CSV (unsettled)
        self._ensure_csv()
        self._csv_writer.writerow([
            dt.strftime('%Y-%m-%d %H:%M:%S'),
            signal['symbol'],
            signal['direction'],
            signal.get('timeframe', '10m'),
            signal['score'],
            signal.get('regime', ''),
            signal['price'],
            signal['rsi7'],
            signal['mfi'],
            signal['stoch_k'],
            signal['adx'],
            signal['cci'],
            signal['atr_pct'],
            '|'.join(signal.get('reasons', [])),
            '', '', ''  # settled, result, pnl (filled on settlement)
        ])
        self._csv_file.flush()

        logger.info(
            "Signal logged: %s %s score=%.1f regime=%s price=%.2f",
            signal['symbol'], signal['direction'],
            signal['score'], signal.get('regime', '?'),
            signal['price']
        )

    # ------------------------------------------------------------------
    # Pending & settlement
    # ------------------------------------------------------------------
    def increment_candle(self, symbol: str):
        """Increment candle counter for settlement tracking."""
        idx = self._candle_idx.get(symbol, -1) + 1
        self._candle_idx[symbol] = idx

    def add_pending(self, symbol: str, signal: dict, settle_candles: int = 2,
                    payout: float = 0.80):
        """Queue a signal for future settlement."""
        idx = self._candle_idx.get(symbol, 0)
        if symbol not in self._pending:
            self._pending[symbol] = []
        self._pending[symbol].append((idx, signal, settle_candles, payout))

    def settle(self, symbol: str, current_price: float,
               settle_ts: float) -> List[dict]:
        """
        Check and settle any pending signals that are due.
        Returns list of newly settled results.
        """
        current_idx = self._candle_idx.get(symbol, 0)
        if symbol not in self._pending:
            return []

        settled = []
        still_pending = []

        for sig_idx, sig, settle_candles, payout in self._pending[symbol]:
            if current_idx - sig_idx >= settle_candles:
                result = self._evaluate(sig, current_price, payout)
                result['settle_time'] = settle_ts
                result['timeframe'] = sig.get('timeframe', '10m')
                settled.append(result)
                self._settled_today.append(result)
                self._write_settlement(result)
                logger.info(
                    "%s %s [%s] settled: %s | entry=%.2f exit=%.2f pnl=%+.1f",
                    symbol, sig['direction'].upper(), sig.get('timeframe', '10m'),
                    result['result'], sig['price'], current_price, result['pnl']
                )
            else:
                still_pending.append((sig_idx, sig, settle_candles, payout))

        self._pending[symbol] = still_pending
        return settled

    @staticmethod
    def _evaluate(signal: dict, exit_price: float, payout_rate: float = 0.80) -> dict:
        """Evaluate a signal against exit price."""
        direction = signal['direction']
        entry = signal['price']
        stake = 25.0
        payout = stake * payout_rate

        if direction == 'up':
            won = exit_price > entry
        else:
            won = exit_price < entry

        return {
            'signal_time': signal['timestamp'],
            'symbol': signal['symbol'],
            'direction': direction,
            'timeframe': signal.get('timeframe', '10m'),
            'entry_price': entry,
            'exit_price': round(exit_price, 2),
            'score': signal['score'],
            'result': 'WIN' if won else 'LOSS',
            'pnl': round(payout if won else -stake, 2),
        }

    def _write_settlement(self, result: dict):
        """Write a settled result to CSV."""
        self._ensure_settle_csv()
        dt_sig = datetime.fromtimestamp(result['signal_time'] / 1000, tz=BEIJING_TZ)
        dt_set = datetime.fromtimestamp(result['settle_time'] / 1000, tz=BEIJING_TZ)
        self._settle_csv_writer.writerow([
            dt_sig.strftime('%Y-%m-%d %H:%M:%S'),
            dt_set.strftime('%Y-%m-%d %H:%M:%S'),
            result['symbol'],
            result['direction'],
            result['entry_price'],
            result['exit_price'],
            result['score'],
            result['result'],
            result['pnl'],
        ])
        self._settle_csv_file.flush()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def get_signals_today(self) -> List[dict]:
        """Get all signals for the current Beijing day."""
        return self._signals_today

    def get_settled_today(self) -> List[dict]:
        """Get settled results for the current Beijing day."""
        return self._settled_today

    def get_pending_count(self) -> int:
        """Total unsettled signals across all symbols."""
        return sum(len(v) for v in self._pending.values())

    def close(self):
        """Clean shutdown."""
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
        if self._settle_csv_file:
            self._settle_csv_file.close()
            self._settle_csv_file = None
