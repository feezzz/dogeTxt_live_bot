"""Binance spot candle stream used by the standalone ETH V7 bot.

Only fully closed candles are stored.  Before a 5-minute signal callback is
emitted at a 15-minute or 1-hour boundary, the corresponding higher timeframe
is refreshed over REST so the model sees the newest *closed* candle without
reading an unfinished candle.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from collections import defaultdict
from typing import Awaitable, Callable, Sequence

import aiohttp

logger = logging.getLogger(__name__)

INTERVAL_MS = {
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
}

# v7_main enlarges these values before DataStream is constructed.
INITIAL_FETCH = {"5m": 600, "15m": 300, "1h": 300}

REST_BASES = (
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
)
WS_BASES = (
    "wss://stream.binance.com:9443/stream",
    "wss://stream.binance.com:443/stream",
)

Candle = list[float]
CandleCallback = Callable[[str, Candle], Awaitable[None] | None]


class DataStream:
    """Maintain closed Binance spot candles and emit closed 5m callbacks."""

    def __init__(self, proxy_url: str = ""):
        self.proxy_url = proxy_url or None
        self._session: aiohttp.ClientSession | None = None
        self._candles: dict[str, dict[str, list[Candle]]] = defaultdict(
            lambda: {"5m": [], "15m": [], "1h": []}
        )
        self._callbacks: list[CandleCallback] = []
        self._tasks: list[asyncio.Task] = []
        self._running = False
        self._last_emitted_5m: dict[str, int] = {}
        self._symbols: list[str] = []

    def on_candle_close(self, callback: CandleCallback) -> None:
        self._callbacks.append(callback)

    def get_candles(self, symbol: str, interval: str) -> list[Candle]:
        return list(self._candles[symbol.upper()][interval])

    async def start(self, symbols: Sequence[str]) -> None:
        if self._running:
            return
        self._symbols = [s.upper() for s in symbols]
        timeout = aiohttp.ClientTimeout(total=30, connect=15, sock_read=30)
        self._session = aiohttp.ClientSession(timeout=timeout)
        self._running = True

        logger.info("Fetching initial closed Binance spot candles...")
        for symbol in self._symbols:
            for interval in ("5m", "15m", "1h"):
                rows = await self._fetch_klines(
                    symbol, interval, limit=INITIAL_FETCH[interval]
                )
                self._candles[symbol][interval] = rows
                logger.info("%s %s initialized: %d candles", symbol, interval, len(rows))

        for symbol in self._symbols:
            self._tasks.append(asyncio.create_task(self._ws_loop(symbol)))

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _fetch_klines(self, symbol: str, interval: str, limit: int) -> list[Candle]:
        if self._session is None:
            raise RuntimeError("DataStream session is not initialized")
        params = {"symbol": symbol, "interval": interval, "limit": int(limit)}
        last_error: Exception | None = None
        for base in REST_BASES:
            url = f"{base}/api/v3/klines"
            try:
                async with self._session.get(url, params=params, proxy=self.proxy_url) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        raise RuntimeError(f"HTTP {resp.status}: {text[:300]}")
                    payload = json.loads(text)
                now_ms = int(time.time() * 1000)
                rows: list[Candle] = []
                for item in payload:
                    # Binance REST index 6 is close time. Reject the currently forming bar.
                    if int(item[6]) > now_ms:
                        continue
                    rows.append(
                        [
                            int(item[0]),
                            float(item[1]),
                            float(item[2]),
                            float(item[3]),
                            float(item[4]),
                            float(item[5]),
                        ]
                    )
                if not rows:
                    raise RuntimeError("Binance returned no fully closed candles")
                return rows
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, RuntimeError) as exc:
                last_error = exc
                logger.warning("REST failed %s via %s: %s", interval, base, exc)
        raise RuntimeError(
            f"Unable to fetch {symbol} {interval} candles from Binance. "
            f"Check network/proxy. Last error: {last_error}"
        )

    async def _refresh_closed(self, symbol: str, interval: str) -> None:
        rows = await self._fetch_klines(symbol, interval, limit=3)
        for candle in rows:
            self._upsert(symbol, interval, candle)

    def _upsert(self, symbol: str, interval: str, candle: Candle) -> None:
        rows = self._candles[symbol][interval]
        open_time = int(candle[0])
        if rows and int(rows[-1][0]) == open_time:
            rows[-1] = candle
        elif not rows or int(rows[-1][0]) < open_time:
            rows.append(candle)
        else:
            # Rare out-of-order reconnect data.
            replaced = False
            for idx in range(len(rows) - 1, -1, -1):
                ts = int(rows[idx][0])
                if ts == open_time:
                    rows[idx] = candle
                    replaced = True
                    break
                if ts < open_time:
                    rows.insert(idx + 1, candle)
                    replaced = True
                    break
            if not replaced:
                rows.insert(0, candle)

        keep = max(INITIAL_FETCH.get(interval, 300), 120)
        if len(rows) > keep:
            del rows[:-keep]

    async def _ws_loop(self, symbol: str) -> None:
        streams = "/".join(
            f"{symbol.lower()}@kline_{interval}" for interval in ("5m", "15m", "1h")
        )
        backoff = 2
        base_index = 0
        while self._running:
            base = WS_BASES[base_index % len(WS_BASES)]
            base_index += 1
            url = f"{base}?streams={streams}"
            try:
                if self._session is None:
                    return
                logger.info("Connecting Binance WebSocket: %s", url)
                async with self._session.ws_connect(
                    url,
                    proxy=self.proxy_url,
                    heartbeat=25,
                    receive_timeout=90,
                    autoping=True,
                ) as ws:
                    logger.info("Binance WebSocket connected for %s", symbol)
                    backoff = 2
                    async for message in ws:
                        if not self._running:
                            return
                        if message.type == aiohttp.WSMsgType.TEXT:
                            await self._handle_ws_message(symbol, message.data)
                        elif message.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.CLOSE,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            break
            except asyncio.CancelledError:
                return
            except Exception as exc:  # reconnect loop must survive transient failures
                logger.exception("WebSocket error for %s: %s", symbol, exc)
            if self._running:
                logger.warning("Reconnecting %s in %ds", symbol, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def _handle_ws_message(self, expected_symbol: str, raw: str) -> None:
        try:
            payload = json.loads(raw)
            data = payload.get("data", payload)
            event = data.get("k")
            if not event or not bool(event.get("x")):
                return
            symbol = str(data.get("s", expected_symbol)).upper()
            interval = str(event["i"])
            if interval not in INTERVAL_MS:
                return
            candle: Candle = [
                int(event["t"]),
                float(event["o"]),
                float(event["h"]),
                float(event["l"]),
                float(event["c"]),
                float(event["v"]),
            ]
            self._upsert(symbol, interval, candle)

            if interval != "5m":
                return
            open_time = int(candle[0])
            if open_time <= self._last_emitted_5m.get(symbol, -1):
                return
            signal_close = open_time + INTERVAL_MS["5m"]

            # Make the latest newly closed HTF candle available before V7 inference.
            if signal_close % INTERVAL_MS["15m"] == 0:
                try:
                    await self._refresh_closed(symbol, "15m")
                except Exception as exc:
                    logger.warning("15m boundary refresh failed; using last closed candle: %s", exc)
            if signal_close % INTERVAL_MS["1h"] == 0:
                try:
                    await self._refresh_closed(symbol, "1h")
                except Exception as exc:
                    logger.warning("1h boundary refresh failed; using last closed candle: %s", exc)

            self._last_emitted_5m[symbol] = open_time
            for callback in list(self._callbacks):
                result = callback(symbol, candle)
                if inspect.isawaitable(result):
                    await result
        except Exception as exc:
            logger.exception("Failed to process Binance message: %s", exc)
