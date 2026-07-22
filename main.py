"""
Live event contract signal bot — main entry point.
Monitors ETH/BTC 5m candles, runs V3 strategy, pushes signals to WeChat.

Usage:
    python -m live_bot.main              # Run with config.yaml defaults
    python -m live_bot.main --console     # Console-only mode (no PushPlus)
"""

import asyncio
import ctypes
import logging
import os
import signal as unix_signal
import sys
import threading
from datetime import datetime

import yaml

from data_stream import DataStream
from indicator_engine import IndicatorEngine
from strategy_engine import StrategyEngine
from risk_manager import RiskManager
from notifier import Notifier
from state_tracker import StateTracker

logger = logging.getLogger('live_bot')

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')


def load_config(path: str = CONFIG_PATH) -> dict:
    """Load YAML config file."""
    if not os.path.exists(path):
        logger.warning("No config.yaml found, using defaults. Copy config.yaml and set your PushPlus token.")
        return {
            'symbols': ['ETHUSDT'],
            'strategy': {'score_threshold': 5.0, 'cooldown_candles': 2, 'max_daily_trades': 30, 'min_atr_pct': 0.08},
            'risk': {'daily_loss_limit': 75, 'max_consecutive_loss': 3, 'low_vol_hours': [22, 23, 0, 1, 2, 3, 4, 5]},
            'notification': {'signal_enabled': True, 'summary_enabled': True, 'signal_cooldown_minutes': 5},
            'pushplus_token': '',
        }
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class SignalBot:
    """Orchestrates all modules for the live signal bot."""

    def __init__(self, config: dict, console_only: bool = False):
        self._config = config
        self._symbols = config.get('symbols', ['ETHUSDT'])
        self._strategy_cfg = config.get('strategy', {})
        self._risk_cfg = config.get('risk', {})
        self._notif_cfg = config.get('notification', {})

        if console_only:
            self._notif_cfg = dict(self._notif_cfg, signal_enabled=False, summary_enabled=False)

        # Timeframe configs
        self._timeframes = self._strategy_cfg.get('timeframes', {
            '10m': {'settle_bars': 2, 'payout': 0.80},
        })

        # Modules
        self._data = DataStream()
        self._indicators: dict[str, IndicatorEngine] = {}
        self._strategy = StrategyEngine({**self._strategy_cfg, 'low_vol_hours': self._risk_cfg.get('low_vol_hours', [])})
        self._risk = RiskManager(self._risk_cfg)
        self._notifier = Notifier({
            'pushplus_token': config.get('pushplus_token', ''),
            **self._notif_cfg,
        })
        self._tracker = StateTracker()

        # Initialize indicator engines per symbol
        for sym in self._symbols:
            self._indicators[sym] = IndicatorEngine()

        self._running = False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    async def run(self):
        """Start the bot."""
        self._running = True

        # Startup notification
        await self._notifier.send_startup(self._symbols, self._strategy_cfg)
        logger.info("Bot starting for symbols: %s", self._symbols)

        # Register candle close callback
        self._data.on_candle_close(self._on_candle_close)

        # Start data stream
        await self._data.start(self._symbols)

        logger.info("Bot running. Waiting for 5m candle closes...")
        print("\n[Bot is running. Press Ctrl+C to stop.]\n")

        # Keep alive
        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await self._shutdown()

    async def _on_candle_close(self, symbol: str, candle):
        """
        Called when a new 5m candle closes.
        This is the core signal pipeline.
        """
        ts = candle[0]
        dt = datetime.fromtimestamp(ts / 1000)
        logger.debug("%s 5m candle closed at %s", symbol, dt.strftime('%H:%M'))

        # Get all timeframe data
        candles_5m = self._data.get_candles(symbol, '5m')
        candles_15m = self._data.get_candles(symbol, '15m')
        candles_1h = self._data.get_candles(symbol, '1h')

        if not candles_5m or len(candles_5m) < 60:
            logger.debug("%s: not enough data yet (%d candles)", symbol, len(candles_5m) if candles_5m else 0)
            return

        # Update indicators
        engine = self._indicators[symbol]
        engine.update(candles_5m, candles_15m, candles_1h)

        # Update risk manager candle counter
        self._risk.increment_candle(symbol)
        self._risk.check_daily_reset(ts)
        self._tracker.increment_candle(symbol)

        # Settle pending signals
        self._tracker.settle(symbol, candle[4], ts)

        # Run strategy
        signal = self._strategy.evaluate(engine, symbol, ts)
        if signal is None:
            return

        # Preview signals: log only, no popup, no tracking, no push
        if signal.get('is_preview'):
            reasons_str = ' | '.join(signal['reasons'][:6])
            logger.info(
                "%s PREVIEW: %s score=%.1f regime=%s price=%.2f | %s",
                symbol, signal['direction'].upper(),
                signal['score'], signal['regime'],
                signal['price'], reasons_str
            )
            direction_cn = '做多 ▲' if signal['direction'] == 'up' else '做空 ▼'
            print(f"\n{'='*60}")
            print(f"🔍 预览 {symbol} {direction_cn}  |  得分: {signal['score']:+.1f}  |  {signal['regime']}行情")
            print(f"   价格: ${signal['price']:.2f}  |  RSI7: {signal['rsi7']:.0f}  MFI: {signal['mfi']:.0f}  CCI: {signal['cci']:.0f}")
            print(f"   打分原因: {reasons_str}")
            print(f"{'='*60}\n")
            return

        # Risk check (once per signal, before splitting into timeframes)
        allowed, reason = self._risk.can_signal(symbol, signal, ts)
        if not allowed:
            logger.info("%s signal blocked: %s (score=%.1f)", symbol, reason, signal['score'])
            return

        self._risk.record_signal(symbol)

        # Split into configured timeframes
        for tf_name, tf_cfg in sorted(self._timeframes.items()):
            settle_bars = tf_cfg['settle_bars']
            payout = tf_cfg['payout']

            tf_signal = {
                **signal,
                'timeframe': tf_name,
                'settle_bars': settle_bars,
                'payout': payout,
            }

            # Record and track per timeframe
            self._tracker.record_signal(tf_signal)
            self._tracker.add_pending(symbol, tf_signal, settle_candles=settle_bars, payout=payout)

        # Notify (combined message for all timeframes)
        await self._notifier.send_signal(signal, self._timeframes)
        _show_windows_popup(signal)
        logger.info(
            "%s SIGNAL: %s score=%.1f regime=%s price=%.2f rsi=%.0f",
            symbol, signal['direction'].upper(),
            signal['score'], signal['regime'],
            signal['price'], signal['rsi7']
        )

    async def _shutdown(self):
        """Clean shutdown."""
        logger.info("Shutting down...")
        self._running = False

        # Daily summary with stats
        signals = self._tracker.get_signals_today()
        settled = self._tracker.get_settled_today()
        if signals:
            await self._notifier.send_daily_summary(signals, settled)

        await self._data.stop()
        self._tracker.close()
        logger.info("Bot stopped.")


# ====================================================================
# Entry point
# ====================================================================

def _show_windows_popup(signal: dict):
    """Show a Windows MessageBox popup for a trading signal (non-blocking)."""
    if sys.platform != 'win32':
        return
    symbol = signal['symbol']
    direction = signal['direction']
    direction_cn = '做多 ▲' if direction == 'up' else '做空 ▼'
    score = signal['score']
    price = signal['price']
    regime = signal['regime']
    is_preview = signal.get('is_preview', False)

    ts = signal['timestamp'] / 1000
    dt = datetime.fromtimestamp(ts)
    time_str = dt.strftime('%Y-%m-%d %H:%M:%S')

    tag = '【预览】' if is_preview else '【正式】'
    tf_note = '' if is_preview else '  10m(80%) + 30m(85%)'
    msg = (
        f"{tag} {symbol}  {direction_cn}{tf_note}\n"
        f"────────────────────────\n"
        f"时间: {time_str}\n"
        f"得分: {score:+.1f}  |  行情: {regime}\n"
        f"价格: ${price:.2f}\n"
        f"RSI7: {signal['rsi7']:.0f}  MFI: {signal['mfi']:.0f}  CCI: {signal['cci']:.0f}\n"
        f"StochK: {signal['stoch_k']:.0f}  ADX: {signal['adx']:.0f}  ATR%: {signal['atr_pct']:.3f}"
    )

    reasons = signal.get('reasons', [])
    if reasons:
        msg += "\n────────────────────────\n"
        msg += "\n".join(f"  {r}" for r in reasons)

    title = f"{'🔍' if is_preview else '⚡'} {symbol} {direction_cn} 信号"

    def _popup():
        ctypes.windll.user32.MessageBoxW(0, msg, title, 0x40)

    threading.Thread(target=_popup, daemon=True).start()


def main():
    """CLI entry point."""
    console_only = '--console' in sys.argv

    # Setup logging
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"bot_{datetime.now().strftime('%Y%m%d')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ]
    )

    # Load config
    config = load_config()
    if console_only:
        config['pushplus_token'] = ''
        print("Running in CONSOLE-ONLY mode (no PushPlus notifications)")

    # Run
    bot = SignalBot(config, console_only=console_only)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    main_task = None
    try:
        main_task = loop.create_task(bot.run())
        loop.run_until_complete(main_task)
    except KeyboardInterrupt:
        print("\n[Interrupted by user]")
        if main_task:
            main_task.cancel()
            try:
                loop.run_until_complete(main_task)
            except (asyncio.CancelledError, KeyboardInterrupt):
                pass
    finally:
        # Run any pending tasks briefly to allow cleanup
        pending = asyncio.all_tasks(loop)
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()


if __name__ == '__main__':
    main()
