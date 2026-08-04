"""ETH V7.0 live signal bot.

This entry point reuses the repository's DataStream and Notifier while keeping
V3 untouched.  Start it with ``python v7_run.py``.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

import data_stream as data_stream_module
from data_stream import DataStream
from notifier import Notifier
from v7_live_tracker import V7LiveTracker
from v7_strategy_engine import V7StrategyEngine

logger = logging.getLogger("live_bot.v7")
BEIJING_TZ = timezone(timedelta(hours=8))


DEFAULT_CONFIG = {
    "symbols": ["ETHUSDT"],
    "proxy": {"enabled": False, "host": "127.0.0.1", "port": 7892},
    "notification": {
        "signal_enabled": True,
        "summary_enabled": True,
        "signal_cooldown_minutes": 0,
    },
    "alerts": {"loss_streak_enabled": True, "loss_streak_thresholds": [3, 5]},
    "pushplus_token": "",
    "feishu_webhook_url": "",
    "v7": {
        "model_dir": "models/v7",
        "stake": 25.0,
        "payout": 0.80,
        "shadow_mode": True,
    },
}


class V7SignalBot:
    def __init__(self, config: dict, console_only: bool = False):
        self.config = config
        requested_symbols = config.get("symbols", ["ETHUSDT"])
        self.symbols = [symbol for symbol in requested_symbols if symbol == "ETHUSDT"]
        if not self.symbols:
            self.symbols = ["ETHUSDT"]
        if requested_symbols != self.symbols:
            logger.warning("Frozen V7 model supports ETHUSDT only; other symbols were ignored")

        v7_cfg = config.get("v7", {})
        model_dir = v7_cfg.get("model_dir", "models/v7")
        if not os.path.isabs(model_dir):
            model_dir = os.path.join(os.path.dirname(__file__), model_dir)
        self.strategy = V7StrategyEngine(model_dir)
        self.stake = float(v7_cfg.get("stake", 25.0))
        self.payout = float(v7_cfg.get("payout", 0.80))
        self.shadow_mode = bool(v7_cfg.get("shadow_mode", True))

        proxy_cfg = config.get("proxy", {})
        proxy_url = ""
        if proxy_cfg.get("enabled", False):
            proxy_url = f"http://{proxy_cfg.get('host', '127.0.0.1')}:{proxy_cfg.get('port', 7892)}"
        # V7 uses recursive EMA/RMA features. Keep a longer warm-up window than V3
        # so live values match the full-history research calculation closely.
        data_stream_module.INITIAL_FETCH.update({"5m": 600, "15m": 300, "1h": 300})
        self.data = DataStream(proxy_url=proxy_url)

        notification = dict(config.get("notification", {}))
        if console_only:
            notification["signal_enabled"] = False
            notification["summary_enabled"] = False
        self.notifier = Notifier(
            {
                "pushplus_token": config.get("pushplus_token", ""),
                "feishu_webhook_url": config.get("feishu_webhook_url", ""),
                **notification,
                **config.get("alerts", {}),
            }
        )
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        self.tracker = V7LiveTracker(log_dir, stake=self.stake, payout=self.payout)

        self._running = False
        self._beijing_day: str | None = None
        self._daily_pnl = 0.0
        self._daily_signal_count = 0
        self._circuit_breaker = False
        self._loss_streak = 0
        self._loss_alerted: set[int] = set()
        self._last_signal_idx: dict[str, int] = {}

    def _reset_day_if_needed(self, timestamp_ms: int) -> None:
        day = datetime.fromtimestamp(timestamp_ms / 1000, tz=BEIJING_TZ).strftime("%Y%m%d")
        if day == self._beijing_day:
            return
        if self._beijing_day is not None:
            logger.info(
                "V7 day reset: previous pnl=%+.2f signals=%d",
                self._daily_pnl,
                self._daily_signal_count,
            )
        self._beijing_day = day
        self._daily_pnl = 0.0
        self._daily_signal_count = 0
        self._circuit_breaker = False
        self._loss_streak = 0
        self._loss_alerted.clear()

    def _risk_blocked(self) -> bool:
        if self._circuit_breaker:
            return True
        if self._daily_signal_count >= self.strategy.max_daily:
            self._circuit_breaker = True
            logger.warning("V7 daily signal limit reached: %d", self.strategy.max_daily)
            return True
        if self._daily_pnl <= self.strategy.max_daily_loss:
            self._circuit_breaker = True
            logger.warning(
                "V7 daily loss breaker: pnl=%+.2f threshold=%+.2f",
                self._daily_pnl,
                self.strategy.max_daily_loss,
            )
            return True
        return False

    async def _handle_settlements(self, results: list[dict]) -> None:
        alerts = self.config.get("alerts", {})
        thresholds = set(alerts.get("loss_streak_thresholds", [3, 5]))
        for result in results:
            self._daily_pnl += float(result["pnl"])
            if result["result"] == "WIN":
                self._loss_streak = 0
                self._loss_alerted.clear()
            else:
                self._loss_streak += 1
                if (
                    alerts.get("loss_streak_enabled", True)
                    and self._loss_streak in thresholds
                    and self._loss_streak not in self._loss_alerted
                ):
                    self._loss_alerted.add(self._loss_streak)
                    await self.notifier.send_loss_streak_alert(
                        result["symbol"], self._loss_streak, result
                    )

    async def run(self) -> None:
        self._running = True
        self.data.on_candle_close(self._on_candle_close)
        await self.data.start(self.symbols)
        startup_cfg = {
            "score_threshold": f"P≥{self.strategy.threshold:.3f} / P≤{1-self.strategy.threshold:.3f}",
            "timeframes": {"10m": {"payout": self.payout}},
            "loss_streak_thresholds": self.config.get("alerts", {}).get(
                "loss_streak_thresholds", [3, 5]
            ),
        }
        await self.notifier.send_startup(self.symbols, startup_cfg)
        logger.info(
            "V7 started: model=%s threshold=%.3f shadow=%s session_hours=%s entry=next-5m-open",
            self.strategy.version,
            self.strategy.threshold,
            self.shadow_mode,
            self.strategy.session_hours or "all",
        )
        print("\n[V7 bot is running. Press Ctrl+C to stop.]\n")
        try:
            while self._running:
                await asyncio.sleep(1)
        finally:
            await self.shutdown()

    async def _on_candle_close(self, symbol: str, candle) -> None:
        close_ts = int(candle[0] + 5 * 60 * 1000)
        self._reset_day_if_needed(close_ts)

        results = self.tracker.process_candle(symbol, candle)
        await self._handle_settlements(results)
        if self._risk_blocked():
            return
        if not self.strategy.in_session(close_ts):
            logger.debug("%s signal skipped: hour outside session_hours %s", symbol, self.strategy.session_hours)
            return

        candles_5m = self.data.get_candles(symbol, "5m")
        candles_15m = self.data.get_candles(symbol, "15m")
        candles_1h = self.data.get_candles(symbol, "1h")
        signal = self.strategy.evaluate(
            candles_5m, candles_15m, candles_1h, symbol, close_ts
        )
        if signal is None:
            return

        current_idx = self.tracker.current_index(symbol)
        last_idx = self._last_signal_idx.get(symbol, -10**9)
        if current_idx - last_idx < self.strategy.cooldown:
            return
        self._last_signal_idx[symbol] = current_idx

        signal["stake"] = self.stake
        signal["payout"] = self.payout
        signal["shadow_mode"] = self.shadow_mode
        self.tracker.add_signal(signal)
        self._daily_signal_count += 1

        await self.notifier.send_signal(
            signal,
            {"10m": {"settle_bars": 2, "payout": self.payout}},
        )
        direction_cn = "做多" if signal["direction"] == "up" else "做空"
        print(
            f"V7 {symbol} {direction_cn} | P(up)={signal['probability_up']:.4f} "
            f"| 参考收盘={signal['reference_close']:.2f} | 下一根5m开盘跟踪入场"
        )

    async def shutdown(self) -> None:
        if not self._running:
            return
        self._running = False
        logger.info(
            "V7 stopping: signals=%d settled=%d pending=%d pnl=%+.2f",
            len(self.tracker.get_signals()),
            len(self.tracker.get_settled()),
            self.tracker.get_pending_count(),
            self._daily_pnl,
        )
        await self.data.stop()


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        logger.warning("Config not found: %s; using safe defaults", path)
        return dict(DEFAULT_CONFIG)
    with open(path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    config = dict(DEFAULT_CONFIG)
    config.update(loaded)
    return config


def cli() -> None:
    parser = argparse.ArgumentParser(description="ETH V7 causal live signal bot")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    parser.add_argument("--console", action="store_true")
    args = parser.parse_args()

    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(log_dir, f"v7_bot_{datetime.now().strftime('%Y%m%d')}.log"),
                encoding="utf-8",
            ),
            logging.StreamHandler(sys.stdout),
        ],
    )
    bot = V7SignalBot(load_config(args.config), console_only=args.console)
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\n[V7 interrupted]")


if __name__ == "__main__":
    cli()
