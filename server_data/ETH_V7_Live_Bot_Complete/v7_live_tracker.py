"""V7 signal logging and backtest-aligned next-open settlement tracking."""
from __future__ import annotations

import csv
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Sequence

logger = logging.getLogger(__name__)
BEIJING_TZ = timezone(timedelta(hours=8))
FIVE_MINUTES_MS = 5 * 60 * 1000


class V7LiveTracker:
    """Track signals using next 5m open as entry and the following 5m close as exit."""

    def __init__(self, log_dir: str, stake: float = 25.0, payout: float = 0.80):
        self.log_dir = log_dir
        self.stake = float(stake)
        self.payout = float(payout)
        os.makedirs(log_dir, exist_ok=True)
        self._candle_idx: dict[str, int] = {}
        self._pending: dict[str, list[dict]] = {}
        self._signals: list[dict] = []
        self._settled: list[dict] = []

    def current_index(self, symbol: str) -> int:
        return self._candle_idx.get(symbol, -1)

    def process_candle(self, symbol: str, candle: Sequence[float]) -> list[dict]:
        """Activate next-open entries and settle signals due on this closed candle."""
        current_idx = self._candle_idx.get(symbol, -1) + 1
        self._candle_idx[symbol] = current_idx
        current_open = float(candle[1])
        current_close = float(candle[4])
        current_open_ts = int(candle[0])
        current_close_ts = current_open_ts + FIVE_MINUTES_MS

        settled: list[dict] = []
        still_pending: list[dict] = []
        for item in self._pending.get(symbol, []):
            bars_since_signal = current_idx - item["signal_idx"]
            if item["entry_price"] is None and bars_since_signal >= 1:
                item["entry_price"] = current_open
                item["entry_time"] = current_open_ts
                logger.info(
                    "%s V7 entry activated: %s next-open=%.2f",
                    symbol,
                    item["signal"]["direction"].upper(),
                    current_open,
                )
            if item["entry_price"] is not None and bars_since_signal >= 2:
                result = self._evaluate(item, current_close, current_close_ts)
                settled.append(result)
                self._settled.append(result)
                self._write_settlement(result)
            else:
                still_pending.append(item)
        self._pending[symbol] = still_pending
        return settled

    def add_signal(self, signal: dict) -> None:
        symbol = signal["symbol"]
        item = {
            "signal_idx": self.current_index(symbol),
            "signal": dict(signal),
            "entry_price": None,
            "entry_time": None,
        }
        self._pending.setdefault(symbol, []).append(item)
        self._signals.append(dict(signal))
        self._write_signal(signal)

    def _evaluate(self, item: dict, exit_price: float, settle_time: int) -> dict:
        signal = item["signal"]
        entry = float(item["entry_price"])
        direction = signal["direction"]
        won = exit_price > entry if direction == "up" else exit_price < entry
        pnl = self.stake * self.payout if won else -self.stake
        result = {
            "signal_time": int(signal["timestamp"]),
            "entry_time": int(item["entry_time"]),
            "settle_time": int(settle_time),
            "symbol": signal["symbol"],
            "direction": direction,
            "entry_price": entry,
            "exit_price": float(exit_price),
            "reference_close": float(signal.get("reference_close", signal["price"])),
            "score": float(signal["score"]),
            "probability_up": float(signal.get("probability_up", 0.5)),
            "result": "WIN" if won else "LOSS",
            "pnl": round(pnl, 2),
            "model_version": signal.get("model_version", "V7"),
        }
        logger.info(
            "%s V7 %s settled: %s entry=%.2f exit=%.2f pnl=%+.2f",
            result["symbol"],
            direction.upper(),
            result["result"],
            entry,
            exit_price,
            pnl,
        )
        return result

    def _daily_path(self, prefix: str, timestamp_ms: int) -> str:
        day = datetime.fromtimestamp(timestamp_ms / 1000, tz=BEIJING_TZ).strftime("%Y%m%d")
        return os.path.join(self.log_dir, f"{prefix}_{day}.csv")

    @staticmethod
    def _append_csv(path: str, header: list[str], row: list) -> None:
        is_new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if is_new:
                writer.writerow(header)
            writer.writerow(row)

    def _write_signal(self, signal: dict) -> None:
        path = self._daily_path("v7_signals", int(signal["timestamp"]))
        timestamp = datetime.fromtimestamp(signal["timestamp"] / 1000, tz=BEIJING_TZ)
        self._append_csv(
            path,
            [
                "signal_time",
                "symbol",
                "direction",
                "probability_up",
                "confidence",
                "reference_close",
                "entry_rule",
                "score",
                "atr_pct",
                "model_version",
                "reasons",
            ],
            [
                timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                signal["symbol"],
                signal["direction"],
                signal.get("probability_up"),
                signal.get("confidence"),
                signal.get("reference_close", signal["price"]),
                "next_5m_open",
                signal["score"],
                signal["atr_pct"],
                signal.get("model_version", "V7"),
                "|".join(signal.get("reasons", [])),
            ],
        )

    def _write_settlement(self, result: dict) -> None:
        path = self._daily_path("v7_settlements", int(result["settle_time"]))
        signal_dt = datetime.fromtimestamp(result["signal_time"] / 1000, tz=BEIJING_TZ)
        entry_dt = datetime.fromtimestamp(result["entry_time"] / 1000, tz=BEIJING_TZ)
        settle_dt = datetime.fromtimestamp(result["settle_time"] / 1000, tz=BEIJING_TZ)
        self._append_csv(
            path,
            [
                "signal_time",
                "entry_time",
                "settle_time",
                "symbol",
                "direction",
                "reference_close",
                "entry_price",
                "exit_price",
                "probability_up",
                "result",
                "pnl",
                "model_version",
            ],
            [
                signal_dt.strftime("%Y-%m-%d %H:%M:%S"),
                entry_dt.strftime("%Y-%m-%d %H:%M:%S"),
                settle_dt.strftime("%Y-%m-%d %H:%M:%S"),
                result["symbol"],
                result["direction"],
                result["reference_close"],
                result["entry_price"],
                result["exit_price"],
                result["probability_up"],
                result["result"],
                result["pnl"],
                result["model_version"],
            ],
        )

    def get_signals(self) -> list[dict]:
        return list(self._signals)

    def get_settled(self) -> list[dict]:
        return list(self._settled)

    def get_pending_count(self) -> int:
        return sum(len(items) for items in self._pending.values())
