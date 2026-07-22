"""
Incremental indicator engine — reuses event_backtest/indicators.py functions.
"""
import sys
import os
import logging
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from event_backtest.indicators import (
    sma, ema, rsi, kdj, kdj_golden_cross, kdj_death_cross,
    bollinger_bands, adx, atr, atr_pct, bb_width, volume_spike,
    cci, williams_r, donchian,
    stochastic_rsi, aroon, aroon_osc, mfi, parabolic_sar,
    detect_candle_patterns,
)

logger = logging.getLogger(__name__)


class IndicatorEngine:
    """Computes and caches all indicators for a single symbol."""

    def __init__(self):
        self._cache: Dict[str, list] = {}
        self._patterns_cache: Dict[str, list] = {}

    def update(self, candles_5m: List[List[float]],
               candles_15m: List[List[float]],
               candles_1h: List[List[float]]):
        """Compute all indicators from current candle data. Called every 5m candle close."""
        # Extract OHLCV arrays
        closes = [c[4] for c in candles_5m]
        opens = [c[1] for c in candles_5m]
        highs = [c[2] for c in candles_5m]
        lows = [c[3] for c in candles_5m]
        volumes = [c[5] for c in candles_5m]

        c15 = [c[4] for c in candles_15m]
        c1h = [c[4] for c in candles_1h]
        h1h = [c[2] for c in candles_1h]
        l1h = [c[3] for c in candles_1h]

        n = len(closes)
        if n < 60:
            return  # Not enough data

        # Core indicators
        self._cache['rsi7'] = rsi(closes, 7)
        self._cache['rsi14'] = rsi(closes, 14)
        self._cache['rsi15m'] = rsi(c15, 7)  # 15m RSI for TF confluence check
        self._cache['ma5'] = sma(closes, 5)
        self._cache['ma10'] = sma(closes, 10)
        self._cache['ma20'] = sma(closes, 20)
        self._cache['ma45'] = sma(closes, 45)
        k, d, j = kdj(highs, lows, closes, period=6, k_period=3, d_period=3)
        self._cache['kdj_k'] = k
        self._cache['kdj_d'] = d
        self._cache['kdj_j'] = j
        self._cache['kg'] = kdj_golden_cross(k, d)
        self._cache['kd'] = kdj_death_cross(k, d)
        bb_mid, bb_up, bb_low = bollinger_bands(closes, period=20, std_mult=2.0)
        self._cache['bb_mid'] = bb_mid
        self._cache['bb_up'] = bb_up
        self._cache['bb_low'] = bb_low
        self._cache['bbw'] = bb_width(bb_up, bb_low, bb_mid)
        self._cache['vol_spike'] = volume_spike(volumes, period=20, threshold=1.5)
        atr_vals = atr(highs, lows, closes, 14)
        self._cache['atr_pct'] = atr_pct(atr_vals, closes)
        self._cache['ema9'] = ema(closes, 9)
        self._cache['ema21'] = ema(closes, 21)

        # 1h ADX
        adx_1h, pdi_1h, mdi_1h = adx(h1h, l1h, c1h, period=14)
        self._cache['adx_1h'] = adx_1h
        self._cache['pdi_1h'] = pdi_1h
        self._cache['mdi_1h'] = mdi_1h

        # New indicators
        self._cache['cci14'] = cci(highs, lows, closes, period=14)
        self._cache['wr14'] = williams_r(highs, lows, closes, period=14)
        stoch_k, stoch_d = stochastic_rsi(closes, period=14, stoch_period=14)
        self._cache['stoch_k'] = stoch_k
        self._cache['stoch_d'] = stoch_d
        aroon_up, aroon_down = aroon(highs, lows, period=14)
        self._cache['aroon_up'] = aroon_up
        self._cache['aroon_down'] = aroon_down
        self._cache['aroon_osc'] = aroon_osc(aroon_up, aroon_down)
        self._cache['mfi'] = mfi(highs, lows, closes, volumes, period=14)
        self._cache['sar'] = parabolic_sar(highs, lows)

        # Candle patterns
        patterns = detect_candle_patterns(opens, highs, lows, closes)
        self._patterns_cache = patterns

        # Store raw data for tf_idx lookup
        self._cache['_closes'] = closes
        self._cache['_opens'] = opens
        self._cache['_highs'] = highs
        self._cache['_lows'] = lows
        self._cache['_volumes'] = volumes
        self._cache['_t5'] = [c[0] for c in candles_5m]
        self._cache['_t15'] = [c[0] for c in candles_15m]
        self._cache['_t1h'] = [c[0] for c in candles_1h]

    def get(self, name: str, idx: int = -1):
        """Get indicator value at index. Default: latest value."""
        arr = self._cache.get(name, [])
        if not arr:
            return 0.0
        if idx < 0:
            idx = len(arr) + idx
        if 0 <= idx < len(arr):
            return arr[idx]
        return 0.0

    def get_pattern(self, name: str, idx: int = -1) -> bool:
        """Get candle pattern at index."""
        arr = self._patterns_cache.get(name, [])
        if not arr:
            return False
        if idx < 0:
            idx = len(arr) + idx
        if 0 <= idx < len(arr):
            return arr[idx]
        return False

    def tf_idx(self, t1h: List[float], target_ts: float) -> int:
        """Find index in 1h timestamps for a given 5m timestamp."""
        for i in range(len(t1h) - 1, -1, -1):
            if t1h[i] <= target_ts:
                return i
        return -1

    def get_15m_idx(self, target_ts: float) -> int:
        """Get the 15m candle index corresponding to a 5m timestamp."""
        t15 = self._cache.get('_t15', [])
        return self.tf_idx(t15, target_ts)

    def get_1h_idx(self, target_ts: float) -> int:
        """Get the 1h candle index corresponding to a 5m timestamp."""
        t1h = self._cache.get('_t1h', [])
        return self.tf_idx(t1h, target_ts)
