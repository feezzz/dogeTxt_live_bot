"""Feishu and PushPlus notifications for the standalone ETH V7 bot."""
from __future__ import annotations

import html
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)
BEIJING_TZ = timezone(timedelta(hours=8))
PUSHPLUS_URL = "https://www.pushplus.plus/send"


class Notifier:
    def __init__(self, config: dict[str, Any]):
        self.pushplus_token = str(config.get("pushplus_token", "") or "").strip()
        self.feishu_webhook_url = str(config.get("feishu_webhook_url", "") or "").strip()
        self.signal_enabled = bool(config.get("signal_enabled", True))
        self.summary_enabled = bool(config.get("summary_enabled", True))
        self.loss_streak_enabled = bool(config.get("loss_streak_enabled", True))
        self.notifications_enabled = self.signal_enabled or self.summary_enabled
        self.timeout = aiohttp.ClientTimeout(total=15, connect=8, sock_read=15)

    async def _post_json(self, url: str, payload: dict[str, Any]) -> bool:
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(url, json=payload) as response:
                    text = await response.text()
                    if response.status < 200 or response.status >= 300:
                        logger.warning("Notification HTTP %s: %s", response.status, text[:300])
                        return False
                    return True
        except Exception as exc:
            logger.warning("Notification request failed: %s", exc)
            return False

    async def _send_feishu(self, title: str, lines: list[str], color: str = "blue") -> bool:
        if not self.feishu_webhook_url:
            return False
        content = "\n".join(lines)
        payload = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {"tag": "plain_text", "content": title},
                    "template": color,
                },
                "elements": [
                    {"tag": "div", "text": {"tag": "lark_md", "content": content}}
                ],
            },
        }
        return await self._post_json(self.feishu_webhook_url, payload)

    async def _send_pushplus(self, title: str, lines: list[str]) -> bool:
        if not self.pushplus_token:
            return False
        body = "<br>".join(html.escape(str(line)) for line in lines)
        payload = {
            "token": self.pushplus_token,
            "title": title,
            "content": body,
            "template": "html",
        }
        return await self._post_json(PUSHPLUS_URL, payload)

    async def _broadcast(self, title: str, lines: list[str], color: str = "blue") -> None:
        sent_feishu = await self._send_feishu(title, lines, color=color)
        sent_pushplus = await self._send_pushplus(title, lines)
        if not sent_feishu and not sent_pushplus:
            logger.info("Notification not configured: %s | %s", title, " | ".join(lines))

    async def send_startup(self, symbols: list[str], strategy_config: dict[str, Any]) -> None:
        if not self.notifications_enabled:
            return
        lines = [
            f"**监控币种：** {', '.join(symbols)}",
            f"**模型：** ETH V7.0 causal ML",
            f"**阈值：** {strategy_config.get('score_threshold', '-')}",
            "**入场口径：** 信号后下一根 5m 开盘",
            "**结算口径：** 再下一根 5m 收盘",
            f"**启动时间：** {datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        await self._broadcast("ETH V7 机器人已启动", lines, color="green")

    async def send_signal(self, signal: dict[str, Any], timeframes: dict[str, Any]) -> None:
        if not self.signal_enabled:
            return
        direction = signal.get("direction")
        direction_cn = "做多" if direction == "up" else "做空"
        probability_up = float(signal.get("probability_up", 0.5))
        confidence = float(signal.get("confidence", max(probability_up, 1 - probability_up)))
        timestamp_ms = int(signal.get("timestamp", 0))
        signal_time = datetime.fromtimestamp(timestamp_ms / 1000, tz=BEIJING_TZ)
        reasons = signal.get("reasons", [])
        lines = [
            f"**方向：** {direction_cn}",
            f"**上涨概率：** {probability_up * 100:.2f}%",
            f"**方向置信度：** {confidence * 100:.2f}%",
            f"**参考收盘价：** {float(signal.get('reference_close', signal.get('price', 0))):.2f}",
            f"**ATR：** {float(signal.get('atr_pct', 0)):.3f}%",
            f"**模型：** {signal.get('model_version', 'V7')}",
            f"**信号时间：** {signal_time.strftime('%Y-%m-%d %H:%M:%S')}",
            "**跟踪规则：** 下一根 5m 开盘入场，随后一根 5m 收盘结算",
        ]
        if reasons:
            lines.append("**原因：** " + "；".join(str(x) for x in reasons[:5]))
        color = "green" if direction == "up" else "red"
        await self._broadcast(f"{signal.get('symbol', 'ETHUSDT')} V7 {direction_cn}信号", lines, color)

    async def send_loss_streak_alert(
        self, symbol: str, loss_streak: int, result: dict[str, Any]
    ) -> None:
        if not self.notifications_enabled or not self.loss_streak_enabled:
            return
        lines = [
            f"**币种：** {symbol}",
            f"**连续亏损：** {loss_streak} 笔",
            f"**最近方向：** {'做多' if result.get('direction') == 'up' else '做空'}",
            f"**最近盈亏：** {float(result.get('pnl', 0)):+.2f} USDT",
            "请检查网络延迟、平台报价偏差和当前市场状态。",
        ]
        await self._broadcast(f"V7 连亏提醒：{loss_streak} 笔", lines, color="orange")

    async def send_daily_summary(self, summary: dict[str, Any]) -> None:
        if not self.summary_enabled:
            return
        lines = [f"**{key}：** {value}" for key, value in summary.items()]
        await self._broadcast("ETH V7 每日总结", lines, color="blue")
