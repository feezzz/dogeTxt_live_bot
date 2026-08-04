from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from v7_feature_engine import build_latest_feature_row
from v7_live_tracker import FIVE_MINUTES_MS, V7LiveTracker
from v7_strategy_engine import V7StrategyEngine


def make_candles(count: int, interval_ms: int, end_close_ts: int, seed: int) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    first_open = end_close_ts - count * interval_ms
    price = 2000.0
    rows = []
    for index in range(count):
        open_time = first_open + index * interval_ms
        open_price = price
        move = rng.normal(0, 0.003)
        close = max(10.0, open_price * (1 + move))
        high = max(open_price, close) * (1 + abs(rng.normal(0, 0.001)))
        low = min(open_price, close) * (1 - abs(rng.normal(0, 0.001)))
        volume = float(100 + rng.lognormal(3, 0.4))
        rows.append([open_time, open_price, high, low, close, volume])
        price = close
    return rows


class V7FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.strategy = V7StrategyEngine(Path(__file__).resolve().parents[1] / "models" / "v7")
        cls.signal_close = 1_800_000_000_000
        cls.c5 = make_candles(600, 5 * 60 * 1000, cls.signal_close, 1)
        cls.c15 = make_candles(300, 15 * 60 * 1000, cls.signal_close, 2)
        cls.c1h = make_candles(300, 60 * 60 * 1000, cls.signal_close, 3)

    def test_model_and_config_feature_order_match(self):
        self.assertEqual(self.strategy.features, list(self.strategy.model.feature_name()))
        self.assertEqual(len(self.strategy.features), 100)
        self.assertAlmostEqual(self.strategy.threshold, 0.555)

    def test_unclosed_higher_timeframe_candles_are_ignored(self):
        baseline, _ = build_latest_feature_row(
            self.c5, self.c15, self.c1h, self.signal_close, self.strategy.features
        )
        unclosed_15m = [
            self.signal_close - 10 * 60 * 1000,
            1.0,
            999999.0,
            0.01,
            999999.0,
            999999999.0,
        ]
        unclosed_1h = [
            self.signal_close - 30 * 60 * 1000,
            1.0,
            999999.0,
            0.01,
            999999.0,
            999999999.0,
        ]
        guarded, _ = build_latest_feature_row(
            self.c5,
            [*self.c15, unclosed_15m],
            [*self.c1h, unclosed_1h],
            self.signal_close,
            self.strategy.features,
        )
        np.testing.assert_allclose(baseline.to_numpy(), guarded.to_numpy(), rtol=0, atol=0)

    def test_signal_payload_is_repository_compatible(self):
        signal = self.strategy.evaluate(
            self.c5, self.c15, self.c1h, "ETHUSDT", self.signal_close
        )
        # A synthetic row may fall inside the no-trade probability band.  Validate
        # the model prediction path even in that case.
        row, _ = build_latest_feature_row(
            self.c5, self.c15, self.c1h, self.signal_close, self.strategy.features
        )
        probability = float(self.strategy.model.predict(row)[0])
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)
        if signal is not None:
            for key in [
                "symbol",
                "direction",
                "score",
                "regime",
                "price",
                "timestamp",
                "rsi7",
                "mfi",
                "stoch_k",
                "adx",
                "cci",
                "atr_pct",
                "reasons",
            ]:
                self.assertIn(key, signal)


class V7TrackerTests(unittest.TestCase):
    def test_next_open_entry_and_following_close_settlement(self):
        with tempfile.TemporaryDirectory() as directory:
            tracker = V7LiveTracker(directory, stake=25, payout=0.8)
            t0 = 1_800_000_000_000
            candle_i = [t0, 2000, 2005, 1995, 2001, 100]
            tracker.process_candle("ETHUSDT", candle_i)
            tracker.add_signal(
                {
                    "timestamp": t0 + FIVE_MINUTES_MS,
                    "symbol": "ETHUSDT",
                    "direction": "up",
                    "score": 5.8,
                    "probability_up": 0.58,
                    "confidence": 0.58,
                    "price": 2001,
                    "reference_close": 2001,
                    "atr_pct": 0.1,
                    "model_version": "test",
                    "reasons": [],
                }
            )
            candle_i1 = [t0 + FIVE_MINUTES_MS, 2002, 2010, 1999, 2006, 110]
            self.assertEqual(tracker.process_candle("ETHUSDT", candle_i1), [])
            candle_i2 = [t0 + 2 * FIVE_MINUTES_MS, 2006, 2012, 2004, 2008, 120]
            settled = tracker.process_candle("ETHUSDT", candle_i2)
            self.assertEqual(len(settled), 1)
            self.assertEqual(settled[0]["entry_price"], 2002)
            self.assertEqual(settled[0]["exit_price"], 2008)
            self.assertEqual(settled[0]["result"], "WIN")
            self.assertEqual(settled[0]["pnl"], 20.0)


if __name__ == "__main__":
    unittest.main()
