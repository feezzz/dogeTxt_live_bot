"""
Real-time K-line data stream via Binance WebSocket + REST fallback.
"""
import asyncio
import glob
import json
import logging
import os
from datetime import datetime
from typing import Callable, Dict, List, Optional

import aiohttp
import websockets

logger = logging.getLogger(__name__)

# Binance WebSocket stream
WS_URL = "wss://stream.binance.com/ws"
# Binance REST API
REST_URL = "https://api.binance.com/api/v3/klines"
# HTTP proxy for REST requests (same as event_backtest/data_fetcher.py)
HTTP_PROXY = "http://127.0.0.1:7892"
# Cache directory (reuse event_backtest cache)
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'event_backtest', 'cache')

# Timeframe intervals in minutes
INTERVAL_MAP = {
    '5m': 5,
    '15m': 15,
    '1h': 60,
}

# How many candles to fetch initially for each interval
INITIAL_FETCH = {
    '5m': 200,
    '15m': 200,
    '1h': 200,
}


def normalize_kline(k: dict) -> List[float]:
    """Convert Binance kline dict to internal format [ts_ms, open, high, low, close, volume]."""
    return [float(k['t']), float(k['o']), float(k['h']),
            float(k['l']), float(k['c']), float(k['v'])]


class DataStream:
    """Real-time multi-symbol, multi-timeframe K-line data stream."""

    def __init__(self):
        self._candles: Dict[str, Dict[str, List[List[float]]]] = {}  # symbol -> {tf: [kline,...]}
        self._callbacks: List[Callable] = []
        self._running = False
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._rest_task: Optional[asyncio.Task] = None
        self._ws_stream_ids: Dict[int, str] = {}  # stream id -> symbol
        self._last_close_ts: Dict[str, int] = {}  # symbol -> last processed kline close time ms

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    async def start(self, symbols: List[str]):
        """Fetch initial data then connect WebSocket."""
        self._running = True
        self._started_at = datetime.now().timestamp() * 1000  # ms, filter stale WS replays
        self._session = aiohttp.ClientSession(proxy=HTTP_PROXY)

        for sym in symbols:
            self._candles[sym] = {}

        # Fetch initial historical candles for all symbols and timeframes
        logger.info("Fetching initial historical candles...")
        await self._fetch_initial(symbols)

        # Connect WebSocket for 5m klines
        self._ws_task = asyncio.create_task(self._ws_loop(symbols))

        # Start REST fallback poller
        self._rest_task = asyncio.create_task(self._rest_fallback_loop(symbols))

        logger.info("DataStream started for %s", symbols)

    async def stop(self):
        """Clean shutdown."""
        self._running = False
        if self._rest_task:
            self._rest_task.cancel()
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()
        logger.info("DataStream stopped.")

    # ------------------------------------------------------------------
    # Historical fetch (REST)
    # ------------------------------------------------------------------
    async def _fetch_klines_rest(self, symbol: str, interval: str,
                                  limit: int = 200) -> List[List[float]]:
        """Fetch klines from REST API."""
        url = f"{REST_URL}?symbol={symbol}&interval={interval}&limit={limit}"
        try:
            async with self._session.get(url, timeout=15) as resp:
                if resp.status != 200:
                    logger.warning("REST fetch %s %s: HTTP %s", symbol, interval, resp.status)
                    return []
                data = await resp.json()
                result = []
                for k in data:
                    result.append([float(k[0]), float(k[1]), float(k[2]),
                                   float(k[3]), float(k[4]), float(k[5])])
                return result
        except Exception as e:
            logger.warning("REST fetch %s %s failed: %s", symbol, interval, e)
            return []

    @staticmethod
    def _load_from_cache(symbol: str, interval: str, count: int = 200) -> List[List[float]]:
        """Load recent candles from local cache files when REST is unavailable."""
        pattern = os.path.join(CACHE_DIR, f'{symbol}_{interval}_*.json')
        files = sorted(glob.glob(pattern), reverse=True)
        if not files:
            logger.warning("Cache miss: %s %s", symbol, interval)
            return []
        logger.info("Cache hit: %s %s <- %s", symbol, interval, os.path.basename(files[0]))
        try:
            with open(files[0], 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data[-count:] if len(data) > count else data
        except Exception as e:
            logger.warning("Cache load failed: %s", e)
            return []

    async def _fetch_initial(self, symbols: List[str]):
        """Fetch initial candles for all symbols and timeframes."""
        tasks = []
        for sym in symbols:
            for tf, limit in INITIAL_FETCH.items():
                tasks.append((sym, tf, self._fetch_klines_rest(sym, tf, limit)))

        results = await asyncio.gather(
            *(t[2] for t in tasks), return_exceptions=True
        )

        for (sym, tf, _), result in zip(tasks, results):
            from_cache = False
            if isinstance(result, Exception):
                logger.error("Initial fetch %s %s error: %s", sym, tf, result)
                result = []
            # REST failed (likely geo-blocked), try local cache
            if not result:
                result = self._load_from_cache(sym, tf, INITIAL_FETCH[tf])
                from_cache = True
            self._candles[sym][tf] = result
            logger.info("Initial %s %s: %d candles%s",
                        sym, tf, len(result),
                        ' (from cache)' if from_cache else '')

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------
    async def _ws_loop(self, symbols: List[str]):
        """WebSocket connection loop with auto-reconnect."""
        while self._running:
            try:
                await self._connect_ws(symbols)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("WS disconnected: %s, reconnecting in 5s...", e)
                await asyncio.sleep(5)

    async def _connect_ws(self, symbols: List[str]):
        """Connect and handle WebSocket stream."""
        streams = [f"{sym.lower()}@kline_5m" for sym in symbols]
        url = f"{WS_URL}/{'/'.join(streams)}"

        async with websockets.connect(url, proxy=HTTP_PROXY) as ws:
            self._ws = ws
            logger.info("WebSocket connected: %s", streams)

            while self._running:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=30)
                    await self._handle_ws_message(json.loads(msg))
                except asyncio.TimeoutError:
                    # Health check
                    try:
                        await ws.ping()
                    except Exception:
                        break
                except websockets.ConnectionClosed:
                    break

    async def _handle_ws_message(self, msg: dict):
        """Parse WebSocket kline message."""
        data = msg.get('data', {})
        kline = data.get('k', {})
        if not kline.get('x'):  # x = is this kline closed?
            return  # Only process closed candles

        k_open_ms = kline.get('t', 0)
        k_close_ms = k_open_ms + 5 * 60 * 1000  # 5m candle close time

        # Skip stale candles replayed on WS connect/reconnect
        if k_open_ms and k_close_ms < self._started_at:
            return

        # Skip candles whose close time is in the future (Binance edge case)
        now_ms = datetime.now().timestamp() * 1000
        if k_close_ms > now_ms + 60 * 1000:
            return

        symbol = data.get('s', '')
        if symbol not in self._candles:
            return

        # Skip if this candle close time is not newer than last processed
        last_close = self._last_close_ts.get(symbol, 0)
        if k_close_ms <= last_close:
            return

        k = normalize_kline(kline)
        self._add_candle(symbol, '5m', k)
        self._last_close_ts[symbol] = k_close_ms
        await self._on_candle_close(symbol, k)

    # ------------------------------------------------------------------
    # REST fallback
    # ------------------------------------------------------------------
    async def _rest_fallback_loop(self, symbols: List[str]):
        """Periodically fetch latest klines as WebSocket fallback."""
        await asyncio.sleep(60)  # Wait 60s before first check
        while self._running:
            try:
                for sym in symbols:
                    candles_5m = self._candles.get(sym, {}).get('5m', [])
                    if not candles_5m:
                        continue
                    last_ts = candles_5m[-1][0]
                    now = int(datetime.now().timestamp() * 1000)
                    # If last candle is more than 10 minutes old, fetch fresh
                    if now - last_ts > 10 * 60 * 1000:
                        fresh = await self._fetch_klines_rest(sym, '5m', 5)
                        for c in fresh:
                            if c[0] > last_ts:
                                self._add_candle(sym, '5m', c)
                                await self._on_candle_close(sym, c)
            except Exception as e:
                logger.warning("REST fallback error: %s", e)
            await asyncio.sleep(60)

    # ------------------------------------------------------------------
    # Candle management
    # ------------------------------------------------------------------
    def _add_candle(self, symbol: str, tf: str, candle: List[float]):
        """Add a candle to in-memory cache, maintain max size."""
        if tf not in self._candles[symbol]:
            self._candles[symbol][tf] = []
        candles = self._candles[symbol][tf]

        ts = candle[0]
        # Dedup: replace if same timestamp
        if candles and candles[-1][0] == ts:
            candles[-1] = candle
        else:
            candles.append(candle)

        # Trim
        max_size = INITIAL_FETCH.get(tf, 200)
        if len(candles) > max_size:
            self._candles[symbol][tf] = candles[-max_size:]

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def on_candle_close(self, callback: Callable):
        """Register callback: async callback(symbol, candle)."""
        self._callbacks.append(callback)

    async def _on_candle_close(self, symbol: str, candle: List[float]):
        """Notify all callbacks when a new 5m candle closes."""
        # Also update 15m and 1h candles by pulling fresh data
        await self._refresh_higher_tf(symbol)

        for cb in self._callbacks:
            try:
                await cb(symbol, candle)
            except Exception as e:
                logger.error("Callback error: %s", e)

    async def _refresh_higher_tf(self, symbol: str):
        """Refresh 15m and 1h candles after 5m update."""
        for tf in ['15m', '1h']:
            try:
                fresh = await self._fetch_klines_rest(symbol, tf, 200)
                if fresh:
                    self._candles[symbol][tf] = fresh
                elif not self._candles[symbol].get(tf):
                    # REST failed and we have nothing — try cache once
                    self._candles[symbol][tf] = self._load_from_cache(symbol, tf, 200)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public access
    # ------------------------------------------------------------------
    def get_candles(self, symbol: str, tf: str) -> List[List[float]]:
        """Get current candle list for a symbol/timeframe."""
        return self._candles.get(symbol, {}).get(tf, [])

    def get_timestamps(self, symbol: str, tf: str) -> List[float]:
        """Get all timestamps for a symbol/timeframe."""
        return [c[0] for c in self.get_candles(symbol, tf)]

    @property
    def is_running(self) -> bool:
        return self._running
