"""
PushPlus WeChat + Feishu Bot notification + console logging.
"""
import logging
from datetime import datetime, timezone, timedelta

import httpx

logger = logging.getLogger(__name__)

PUSHPLUS_URL = "https://www.pushplus.plus/send"
BEIJING_TZ = timezone(timedelta(hours=8))


class Notifier:
    """Sends trade signals via PushPlus and/or Feishu Bot."""

    def __init__(self, config: dict):
        self._token = config.get('pushplus_token', '')
        self._feishu_url = config.get('feishu_webhook_url', '')
        self._signal_enabled = config.get('signal_enabled', True)
        self._summary_enabled = config.get('summary_enabled', True)
        self._last_signal_time: dict[str, float] = {}
        self._signal_cooldown_s = config.get('signal_cooldown_minutes', 5) * 60

    async def send_signal(self, signal: dict, timeframes: dict = None, stake: float = 25.0):
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
            f"价格: ${signal['price']:.2f} | 仓位: ${stake:.0f} | 时间: {dt.strftime('%H:%M')}",
            f"RSI7={signal['rsi7']:.0f} | MFI={signal['mfi']:.0f} | StochRSI={signal['stoch_k']:.0f}",
            f"ADX={signal['adx']:.0f} | CCI={signal['cci']:.0f} | ATR%={signal['atr_pct']:.3f}",
        ]
        if signal.get('reasons'):
            lines.append(f"信号: {' | '.join(signal['reasons'][:4])}")

        content = "\n".join(lines)

        if self._signal_enabled and self._token and 'YOUR_PUS' not in self._token:
            await self._push(content, f"{symbol} {direction_cn} 信号")
        else:
            # Console-only mode
            print(f"\n{'='*60}")
            print(content)
            print(f"{'='*60}\n")
            logger.info("Signal: %s", content.replace('\n', ' | '))

        self._last_signal_time[symbol] = now

        # Feishu card
        if self._feishu_url:
            color = 'green' if direction == 'up' else 'red'
            arrow = '▲' if direction == 'up' else '▼'
            tf_s = ' | '.join(tf_parts) if timeframes else '10m(80%)'
            body_lines = [
                f"**价格**：${signal['price']:.2f}  |  **时间**：{dt.strftime('%H:%M')}",
                f"**得分**：{signal['score']:+.1f}  |  仓位：${stake:.0f}  |  {signal['regime']}行情  |  {tf_s}",
                f"RSI7：{signal['rsi7']:.0f}  |  MFI：{signal['mfi']:.0f}  |  CCI：{signal['cci']:.0f}",
                f"StochK：{signal['stoch_k']:.0f}  |  ADX：{signal['adx']:.0f}  |  ATR%：{signal['atr_pct']:.3f}",
            ]
            if signal.get('reasons'):
                body_lines.append(f"**原因**：{' | '.join(signal['reasons'][:4])}")
            await self._push_feishu(self._feishu_card(
                header=f"{emoji} {symbol} {direction_cn} {arrow}",
                template=color,
                body="\n".join(body_lines),
                note=f"{dt.strftime('%Y-%m-%d %H:%M')}  |  10m(80%)",
            ))

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
            won = sum(1 for s in settled if s['result'] == 'WIN')
            total_s = len(settled)
            wr = won / total_s * 100
            pnl = sum(s['pnl'] for s in settled)
            lines.append(f"已结算: {won}胜/{total_s - won}负 | 胜率: {wr:.0f}% | 盈亏: ${pnl:+.2f}")

            # Per timeframe
            for tf in ['10m']:
                ss = [s for s in settled if s.get('timeframe', '10m') == tf]
                if ss:
                    w = sum(1 for s in ss if s['result'] == 'WIN')
                    lines.append(f"  {tf}: {w}/{len(ss)} {w/len(ss)*100:.0f}% ${sum(s['pnl'] for s in ss):+.2f}")

            # Per symbol
            for sym in ['ETHUSDT', 'BTCUSDT']:
                ss = [s for s in settled if s['symbol'] == sym]
                if ss:
                    w = sum(1 for s in ss if s['result'] == 'WIN')
                    lines.append(f"  {sym}: {w}/{len(ss)} {w/len(ss)*100:.0f}%")

        pending = total - (len(settled) if settled else 0)
        if pending > 0:
            lines.append(f"⏳ 等待结算: {pending}笔")

        content = "\n".join(lines)

        if self._summary_enabled and self._token and 'YOUR_PUS' not in self._token:
            await self._push(content, "每日交易总结")
        else:
            print(f"\n{content}\n")
            logger.info("Daily summary: %s", content.replace('\n', ' | '))

        # Feishu
        if self._feishu_url:
            await self._push_feishu(self._feishu_card(
                header=f"📊 今日信号总结",
                template="blue",
                body=content.replace('\n', '\n\n'),
                note=datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M'),
            ))

    async def send_startup(self, symbols: list, config: dict):
        """Notify that the bot has started."""
        timeframes = config.get('timeframes', {})
        tf_str = '+'.join(timeframes.keys()) if timeframes else '10m'
        content = (
            f"🚀 事件合约信号机器人已启动\n"
            f"监控: {', '.join(symbols)} | 周期: {tf_str}\n"
            f"分数阈值: {config.get('score_threshold', 5.0)}\n"
            f"连亏提醒: {config.get('loss_streak_thresholds', [3, 5])}\n"
            f"时间: {datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')}"
        )
        if self._token and 'YOUR_PUS' not in self._token:
            await self._push(content, "机器人启动")
        print(f"\n{content}\n")

        # Feishu
        if self._feishu_url:
            await self._push_feishu(self._feishu_card(
                header="🚀 机器人已启动",
                template="blue",
                body=content,
                note="",
            ))

    async def send_loss_streak_alert(self, symbol: str, streak: int, result: dict):
        """Send a consecutive loss alert to WeChat and console."""
        level = '‼️ 严重提醒' if streak >= 5 else '⚠️ 注意提醒'
        content = (
            f"{level}\n"
            f"{symbol} 连续亏损: {streak} 笔\n"
            f"最近方向: {result['direction']}\n"
            f"入场价: ${result['entry_price']:.2f}\n"
            f"结算价: ${result['exit_price']:.2f}\n"
            f"得分: {result['score']:.1f}"
        )

        print(f"\n{'!' * 60}")
        print(content)
        print(f"{'!' * 60}\n")
        logger.warning("%s consecutive losses: %d", symbol, streak)

        if self._token and 'YOUR_PUS' not in self._token:
            await self._push(content, f"{symbol} 连亏{streak}笔提醒")

        # Feishu
        if self._feishu_url:
            await self._push_feishu(self._feishu_card(
                header=f"{level}",
                template="red",
                body=content.replace('\n', '\n\n'),
                note=datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S'),
            ))

    async def send_circuit_breaker(self, symbol: str, daily_pnl: float, max_loss: float):
        """Notify that daily loss circuit breaker has been triggered."""
        content = (
            f"🛑 **日内熔断触发**\n"
            f"当前累计盈亏: ${daily_pnl:+.2f}\n"
            f"熔断阈值: ${max_loss:+.0f}\n"
            f"今日剩余时间暂停所有信号\n"
            f"触发品种: {symbol}\n"
            f"时间: {datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')}"
        )

        print(f"\n{'#'*60}")
        print(content)
        print(f"{'#'*60}\n")
        logger.warning("Circuit breaker: daily PnL=$%.2f, max=$%.0f", daily_pnl, max_loss)

        if self._feishu_url:
            await self._push_feishu(self._feishu_card(
                header="🛑 日内熔断已触发",
                template="red",
                body=content.replace('\n', '\n\n'),
                note=datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S'),
            ))

    async def _push(self, content: str, title: str = ""):
        """Send via PushPlus API."""
        if not self._token or 'YOUR_PUS' in self._token:
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

    # ------------------------------------------------------------------
    # Feishu Bot (Lark) webhook
    # ------------------------------------------------------------------
    @staticmethod
    def _feishu_card(header: str, template: str, body: str, note: str = "") -> dict:
        """Build a Feishu interactive card message."""
        elements = [
            {"tag": "div", "text": {"tag": "lark_md", "content": body}},
        ]
        if note:
            elements.append({"tag": "note", "elements": [{"tag": "plain_text", "content": note}]})
        return {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {"tag": "plain_text", "content": header},
                    "template": template,
                },
                "elements": elements,
            },
        }

    async def _push_feishu(self, payload: dict):
        """Send via Feishu webhook."""
        if not self._feishu_url:
            return
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(self._feishu_url, json=payload)
                if resp.status_code != 200:
                    logger.warning("Feishu HTTP %s: %s", resp.status_code, resp.text)
                else:
                    result = resp.json()
                    if result.get('code') != 0:
                        logger.warning("Feishu error: %s", result.get('msg'))
        except Exception as e:
            logger.warning("Feishu push failed: %s", e)
