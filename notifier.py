"""
PushPlus WeChat push notification + console logging.
PushPlus: free WeChat push, one HTTP call. Register at https://www.pushplus.plus
"""
import logging
from datetime import datetime, timezone, timedelta

import httpx

logger = logging.getLogger(__name__)

PUSHPLUS_URL = "http://www.pushplus.plus/send"
BEIJING_TZ = timezone(timedelta(hours=8))


class Notifier:
    """Sends trade signals via PushPlus to WeChat."""

    def __init__(self, config: dict):
        self._token = config.get('pushplus_token', '')
        self._signal_enabled = config.get('signal_enabled', True)
        self._summary_enabled = config.get('summary_enabled', True)
        self._last_signal_time: dict[str, float] = {}
        self._signal_cooldown_s = config.get('signal_cooldown_minutes', 5) * 60

    async def send_signal(self, signal: dict, timeframes: dict = None):
        """Send a trade signal notification to WeChat."""
        symbol = signal['symbol']
        direction = signal['direction']
        direction_cn = '做多' if direction == 'up' else '做空'
        emoji = '🟢' if direction == 'up' else '🔴'

        # Cooldown check (don't spam same symbol)
        now = datetime.now().timestamp()
        if symbol in self._last_signal_time:
            if now - self._last_signal_time[symbol] < self._signal_cooldown_s:
                return

        ts = signal['timestamp'] / 1000
        dt = datetime.fromtimestamp(ts, tz=BEIJING_TZ)

        # Timeframe info
        tf_str = ''
        if timeframes:
            tf_parts = [f"{n}({c['payout']*100:.0f}%)" for n, c in timeframes.items()]
            tf_str = f" | 周期: {'+'.join(tf_parts)}"

        # Build message
        lines = [
            f"{emoji} **{symbol} {direction_cn}** | 得分 {signal['score']:.1f} | {signal['regime']}行情{tf_str}",
            f"价格: ${signal['price']:.2f} | 时间: {dt.strftime('%H:%M')}",
            f"RSI7={signal['rsi7']:.0f} | MFI={signal['mfi']:.0f} | StochRSI={signal['stoch_k']:.0f}",
            f"ADX={signal['adx']:.0f} | CCI={signal['cci']:.0f} | ATR%={signal['atr_pct']:.3f}",
        ]
        if signal.get('reasons'):
            lines.append(f"信号: {' | '.join(signal['reasons'][:4])}")

        content = "\n".join(lines)

        if self._signal_enabled and self._token and self._token != 'YOUR_PUSPLUS_TOKEN_HERE':
            await self._push(content, f"{symbol} {direction_cn} 信号")
        else:
            # Console-only mode
            print(f"\n{'='*60}")
            print(content)
            print(f"{'='*60}\n")
            logger.info("Signal: %s", content.replace('\n', ' | '))

    async def send_daily_summary(self, signals_today: list, settled: list = None):
        """Send daily summary at end of Beijing day."""
        if not signals_today:
            return

        total = len(signals_today)
        up_count = sum(1 for s in signals_today if s['direction'] == 'up')
        down_count = total - up_count
        avg_score = sum(abs(s['score']) for s in signals_today) / total if total else 0

        by_symbol = {}
        for s in signals_today:
            sym = s['symbol']
            by_symbol.setdefault(sym, 0)
            by_symbol[sym] += 1

        sym_parts = [f"{k}:{v}笔" for k, v in by_symbol.items()]

        lines = [
            f"📊 今日信号总结 (7:00-23:00)",
            f"总信号: {total}笔 | 做多: {up_count} | 做空: {down_count}",
            f"平均得分: {avg_score:.1f}",
            f"品种分布: {', '.join(sym_parts)}",
        ]

        # Settlement stats
        if settled:
            settled_7_23 = [
                s for s in settled
                if 7 <= datetime.fromtimestamp(s['signal_time'] / 1000,
                                               tz=BEIJING_TZ).hour < 23
            ]
            if settled_7_23:
                won = sum(1 for s in settled_7_23 if s['result'] == 'WIN')
                total_s = len(settled_7_23)
                wr = won / total_s * 100
                pnl = sum(s['pnl'] for s in settled_7_23)
                lines.append(f"已结算: {won}胜/{total_s - won}负 | 胜率: {wr:.0f}% | 盈亏: ${pnl:+.2f}")

                # Per timeframe
                for tf in ['10m', '30m']:
                    ss = [s for s in settled_7_23 if s.get('timeframe', '10m') == tf]
                    if ss:
                        w = sum(1 for s in ss if s['result'] == 'WIN')
                        lines.append(f"  {tf}: {w}/{len(ss)} {w/len(ss)*100:.0f}% ${sum(s['pnl'] for s in ss):+.2f}")

                # Per symbol
                for sym in ['ETHUSDT', 'BTCUSDT']:
                    ss = [s for s in settled_7_23 if s['symbol'] == sym]
                    if ss:
                        w = sum(1 for s in ss if s['result'] == 'WIN')
                        lines.append(f"  {sym}: {w}/{len(ss)} {w/len(ss)*100:.0f}%")
            else:
                lines.append("已结算: 0笔 (等待自动结算)")

        pending = total - (len(settled) if settled else 0)
        if pending > 0:
            lines.append(f"⏳ 等待结算: {pending}笔")

        content = "\n".join(lines)

        if self._summary_enabled and self._token and self._token != 'YOUR_PUSPLUS_TOKEN_HERE':
            await self._push(content, "每日交易总结")
        else:
            print(f"\n{content}\n")
            logger.info("Daily summary: %s", content.replace('\n', ' | '))

    async def send_startup(self, symbols: list, config: dict):
        """Notify that the bot has started."""
        timeframes = config.get('timeframes', {})
        tf_str = '+'.join(timeframes.keys()) if timeframes else '10m'
        content = (
            f"🚀 事件合约信号机器人已启动\n"
            f"监控: {', '.join(symbols)} | 周期: {tf_str}\n"
            f"分数阈值: {config.get('score_threshold', 5.0)}\n"
            f"日限额: {config.get('max_daily_trades', 50)}笔\n"
            f"时间: {datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')}"
        )
        if self._token and self._token != 'YOUR_PUSPLUS_TOKEN_HERE':
            await self._push(content, "机器人启动")
        print(f"\n{content}\n")

    async def _push(self, content: str, title: str = ""):
        """Send via PushPlus API."""
        if not self._token or self._token == 'YOUR_PUSPLUS_TOKEN_HERE':
            return
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(PUSHPLUS_URL, json={
                    'token': self._token,
                    'title': title,
                    'content': content,
                    'template': 'markdown',
                })
                if resp.status_code == 200:
                    result = resp.json()
                    if result.get('code') != 200:
                        logger.warning("PushPlus error: %s", result.get('msg'))
                else:
                    logger.warning("PushPlus HTTP %s", resp.status_code)
        except Exception as e:
            logger.warning("PushPlus push failed: %s", e)
