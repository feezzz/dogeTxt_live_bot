"""
Risk manager: daily limits, cooldown, volatility filter, circuit breakers.
"""
import logging
from datetime import datetime
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# UTC offset for Beijing time
BEIJING_OFFSET = 8


class RiskManager:
    """Manages per-symbol and global risk limits."""

    def __init__(self, config: dict):
        self._cooldown_candles = config.get('cooldown_candles', 2)
        self._max_daily_trades = config.get('max_daily_trades', 30)
        self._daily_loss_limit = config.get('daily_loss_limit', 75)
        self._max_consec_loss = config.get('max_consecutive_loss', 3)
        self._low_vol_hours = set(config.get('low_vol_hours', []))

        # Per-symbol state
        self._last_signal_idx: dict[str, int] = {}  # symbol -> candle index
        self._symbol_idx: dict[str, int] = {}       # symbol -> current candle idx

        # Daily state
        self._day_trades = 0
        self._day_pnl = 0.0
        self._consec_losses = 0
        self._current_beijing_day: Optional[int] = None

        # Trading state (user updates these after each trade result)
        self._last_trade_direction: Optional[str] = None
        self._last_trade_symbol: Optional[str] = None

    def increment_candle(self, symbol: str):
        """Increment the candle counter for a symbol."""
        self._symbol_idx[symbol] = self._symbol_idx.get(symbol, -1) + 1

    def check_daily_reset(self, utc_ts: float):
        """Reset daily counters at Beijing midnight (UTC 16:00)."""
        dt = datetime.fromtimestamp(utc_ts / 1000)
        beijing_hour = (dt.hour + BEIJING_OFFSET) % 24
        beijing_day = (dt.day + (1 if dt.hour + BEIJING_OFFSET >= 24 else 0))

        if self._current_beijing_day != beijing_day:
            is_new_day = self._current_beijing_day is not None
            self._current_beijing_day = beijing_day
            self._day_trades = 0
            self._day_pnl = 0.0
            self._consec_losses = 0
            self._last_signal_idx.clear()
            if is_new_day:
                logger.info("New day! Daily reset: BJ day=%d", beijing_day)
            else:
                logger.info("Daily counters initialized: BJ day=%d", beijing_day)

    def can_signal(self, symbol: str, signal: dict,
                   candle_ts: float) -> Tuple[bool, str]:
        """
        Check if a signal is allowed.
        Returns (allowed, reason_if_blocked).
        """
        dt = datetime.fromtimestamp(candle_ts / 1000)

        # 1. Daily trade limit
        if self._day_trades >= self._max_daily_trades * 2:  # x2 for both symbols
            # Still allow if we haven't hit per-symbol limit
            pass
        if self._day_trades >= self._max_daily_trades * 3:
            return False, f"日交易上限({self._max_daily_trades * 3})"

        # 2. Cooldown between signals (same symbol)
        current_idx = self._symbol_idx.get(symbol, 0)
        last_idx = self._last_signal_idx.get(symbol, -999)
        if current_idx - last_idx < self._cooldown_candles:
            remaining = self._cooldown_candles - (current_idx - last_idx)
            return False, f"冷却中(还需{remaining * 5}分钟)"

        # 3. Consecutive loss pause
        if self._consec_losses >= self._max_consec_loss:
            return False, f"连亏{self._consec_losses}笔,暂停等待"

        # 4. Daily loss circuit breaker
        if self._day_pnl <= -self._daily_loss_limit:
            return False, f"日亏损熔断(-${self._daily_loss_limit})"

        # 5. Low-vol hours: allow at reduced rate
        if dt.hour in self._low_vol_hours and self._day_trades >= self._max_daily_trades:
            return False, f"低波动时段,已达上限"

        return True, "OK"

    def record_signal(self, symbol: str):
        """Record that a signal was generated."""
        self._last_signal_idx[symbol] = self._symbol_idx.get(symbol, 0)
        self._day_trades += 1
        self._last_trade_symbol = symbol

    def record_result(self, pnl: float):
        """Record the result of a trade (called by user when they know outcome)."""
        self._day_pnl += pnl
        if pnl > 0:
            self._consec_losses = 0
        else:
            self._consec_losses += 1

    @property
    def daily_stats(self) -> dict:
        return {
            'trades': self._day_trades,
            'pnl': self._day_pnl,
            'consec_losses': self._consec_losses,
            'beijing_day': self._current_beijing_day,
        }
