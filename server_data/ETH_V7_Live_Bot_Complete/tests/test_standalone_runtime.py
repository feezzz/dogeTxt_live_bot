from __future__ import annotations

import unittest

from data_stream import DataStream
from notifier import Notifier


class DataStreamRuntimeTests(unittest.TestCase):
    def test_upsert_keeps_order_and_replaces_duplicate(self):
        stream = DataStream()
        stream._upsert("ETHUSDT", "5m", [1000, 1, 2, 0.5, 1.5, 10])
        stream._upsert("ETHUSDT", "5m", [2000, 1.5, 2.2, 1.2, 2, 11])
        stream._upsert("ETHUSDT", "5m", [2000, 1.5, 2.3, 1.1, 2.1, 12])
        rows = stream.get_candles("ETHUSDT", "5m")
        self.assertEqual([row[0] for row in rows], [1000, 2000])
        self.assertEqual(rows[-1][4], 2.1)


class NotifierRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_console_flags_disable_all_notifications(self):
        notifier = Notifier(
            {
                "signal_enabled": False,
                "summary_enabled": False,
                "loss_streak_enabled": True,
                "feishu_webhook_url": "https://invalid.example",
                "pushplus_token": "invalid",
            }
        )
        called = False

        async def fake_broadcast(*args, **kwargs):
            nonlocal called
            called = True

        notifier._broadcast = fake_broadcast
        await notifier.send_startup(["ETHUSDT"], {})
        await notifier.send_loss_streak_alert("ETHUSDT", 3, {})
        self.assertFalse(called)


if __name__ == "__main__":
    unittest.main()
