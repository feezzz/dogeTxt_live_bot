"""
V3 Ensemble Strategy — real-time signal generation.
Directly ported from event_backtest/ensemble_v3.py backtest_ensemble_v3().
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from indicator_engine import IndicatorEngine

logger = logging.getLogger(__name__)


class StrategyEngine:
    """V3 multi-indicator scoring strategy for 10-min event contracts."""

    def __init__(self, config: dict):
        self._score_threshold = config.get('score_threshold', 5.0)
        self._preview_threshold = config.get('preview_threshold', 3.0)
        self._min_atr_pct = config.get('min_atr_pct', 0.08)
        self._low_vol_hours = set(config.get('low_vol_hours', []))

    def evaluate(self, indicator: IndicatorEngine,
                 symbol: str, candle_ts: float,
                 idx_5m: int = -1) -> Optional[dict]:
        """
        Run V3 scoring. Returns signal dict or None.
        Called on each 5m candle close.
        Pass idx_5m to evaluate a specific historical candle (for backtesting).
        """
        price = indicator.get('_closes', idx_5m)
        ts_5m = indicator.get('_t5', idx_5m)

        if not isinstance(ts_5m, (int, float)) or ts_5m == 0:
            return None

        # ATR filter
        atr_pct = indicator.get('atr_pct', idx_5m)
        if atr_pct < self._min_atr_pct:
            return None

        # Get 1h index
        idx_1h = indicator.get_1h_idx(ts_5m)
        if idx_1h < 0:
            return None

        # ADX 1h
        adx_val = indicator.get('adx_1h', idx_1h)
        pdi = indicator.get('pdi_1h', idx_1h)
        mdi = indicator.get('mdi_1h', idx_1h)
        di_diff = pdi - mdi

        # ================================================================
        # Regime detection
        # ================================================================
        aroon_osc_val = indicator.get('aroon_osc', idx_5m)
        bbw = indicator.get('bbw', idx_5m)

        if adx_val > 25 or abs(aroon_osc_val) > 50:
            regime = 'trending'
        elif adx_val < 18 or (bbw < 1.5 and abs(aroon_osc_val) < 25):
            regime = 'ranging'
        else:
            regime = 'neutral'

        # ================================================================
        # V3 Scoring (14 components)
        # ================================================================
        score = 0.0
        reasons = []

        # --- RSI(7) ---
        rsi7 = indicator.get('rsi7', idx_5m)
        if rsi7 < 20:
            score += 2.0; reasons.append(f'RSI7={rsi7:.0f}(超卖)')
        elif rsi7 < 30:
            score += 1.2; reasons.append(f'RSI7={rsi7:.0f}(低)')
        elif rsi7 < 40:
            score += 0.3
        elif rsi7 > 80:
            score -= 2.0; reasons.append(f'RSI7={rsi7:.0f}(超买)')
        elif rsi7 > 70:
            score -= 1.2; reasons.append(f'RSI7={rsi7:.0f}(高)')
        elif rsi7 > 60:
            score -= 0.3

        # --- Stochastic RSI ---
        stoch_k = indicator.get('stoch_k', idx_5m)
        stoch_d = indicator.get('stoch_d', idx_5m)
        if stoch_k < 10 and stoch_d < 15:
            score += 1.5; reasons.append(f'StochRSI={stoch_k:.0f}(极低)')
        elif stoch_k < 20:
            score += 0.5
        elif stoch_k > 90 and stoch_d > 85:
            score -= 1.5; reasons.append(f'StochRSI={stoch_k:.0f}(极高)')
        elif stoch_k > 80:
            score -= 0.5

        # Stoch RSI cross
        stoch_k_prev = indicator.get('stoch_k', -2)
        stoch_d_prev = indicator.get('stoch_d', -2)
        if stoch_k_prev <= stoch_d_prev and stoch_k > stoch_d:
            score += 0.8
        elif stoch_k_prev >= stoch_d_prev and stoch_k < stoch_d:
            score -= 0.8

        # --- MFI ---
        mfi = indicator.get('mfi', idx_5m)
        if mfi < 15:
            score += 1.5; reasons.append(f'MFI={mfi:.0f}(超卖)')
        elif mfi < 25:
            score += 0.5
        elif mfi > 85:
            score -= 1.5; reasons.append(f'MFI={mfi:.0f}(超买)')
        elif mfi > 75:
            score -= 0.5

        # --- CCI ---
        cci = indicator.get('cci14', idx_5m)
        if cci < -200:
            score += 1.5
        elif cci < -100:
            score += 0.7
        elif cci > 200:
            score -= 1.5
        elif cci > 100:
            score -= 0.7

        # --- Williams %R ---
        wr = indicator.get('wr14', idx_5m)
        if wr < -90:
            score += 1.2
        elif wr < -80:
            score += 0.5
        elif wr > -10:
            score -= 1.2
        elif wr > -20:
            score -= 0.5

        # --- Parabolic SAR ---
        sar = indicator.get('sar', idx_5m)
        if price > sar:
            score += 0.5
        else:
            score -= 0.5

        # --- Aroon ---
        aroon_up = indicator.get('aroon_up', idx_5m)
        aroon_down = indicator.get('aroon_down', idx_5m)
        if aroon_up > 70 and aroon_down < 30:
            score += 0.5
        elif aroon_down > 70 and aroon_up < 30:
            score -= 0.5

        # --- MA trend ---
        ma5 = indicator.get('ma5', idx_5m)
        ma10 = indicator.get('ma10', idx_5m)
        ma20 = indicator.get('ma20', idx_5m)
        if price > ma20 and ma5 > ma10:
            score += 0.8
        elif price < ma20 and ma5 < ma10:
            score -= 0.8

        # --- EMA 9/21 ---
        ema9 = indicator.get('ema9', idx_5m)
        ema21 = indicator.get('ema21', idx_5m)
        if ema9 > ema21:
            score += 0.3
        else:
            score -= 0.3

        # --- KDJ ---
        kg = indicator.get('kg', idx_5m)
        kd = indicator.get('kd', idx_5m)
        j = indicator.get('kdj_j', idx_5m)
        if kg:
            score += 0.8; reasons.append('KDJ金叉')
        elif kd:
            score -= 0.8; reasons.append('KDJ死叉')
        if j < 0:
            score += 0.5
        elif j > 100:
            score -= 0.5

        # --- BB position ---
        bb_up = indicator.get('bb_up', idx_5m)
        bb_low = indicator.get('bb_low', idx_5m)
        if bb_up > bb_low:
            bb_pos = (price - bb_low) / (bb_up - bb_low)
            if bb_pos < 0.08:
                score += 1.2; reasons.append(f'BB底(位置={bb_pos:.2f})')
            elif bb_pos < 0.2:
                score += 0.5
            elif bb_pos > 0.92:
                score -= 1.2; reasons.append(f'BB顶(位置={bb_pos:.2f})')
            elif bb_pos > 0.8:
                score -= 0.5

        # --- Volume spike ---
        vol_spike = indicator.get('vol_spike', idx_5m)
        opens_val = indicator.get('_opens', idx_5m)
        closes_val = indicator.get('_closes', idx_5m)
        if vol_spike:
            if closes_val > opens_val:
                score += 0.5
            else:
                score -= 0.5

        # --- Candle patterns ---
        if indicator.get_pattern('hammer', idx_5m):
            score += 1.0; reasons.append('锤子线')
        elif indicator.get_pattern('shooting_star', idx_5m):
            score -= 1.0; reasons.append('射击之星')
        if indicator.get_pattern('bullish_engulfing', idx_5m):
            score += 1.5; reasons.append('多头吞没')
        elif indicator.get_pattern('bearish_engulfing', idx_5m):
            score -= 1.5; reasons.append('空头吞没')

        # --- Regime adjustment ---
        if regime == 'trending':
            if di_diff > 5:
                score += 0.8
            elif di_diff < -5:
                score -= 0.8
            if di_diff > 3 and rsi7 < 50:
                score += 0.3
            elif di_diff < -3 and rsi7 > 50:
                score -= 0.3
        elif regime == 'ranging':
            bb_pos_val = (price - bb_low) / (bb_up - bb_low) if bb_up > bb_low else 0.5
            if bb_pos_val < 0.15:
                score += 0.5
            elif bb_pos_val > 0.85:
                score -= 0.5

        # ================================================================
        # Decision
        # ================================================================
        direction = None
        is_preview = False

        if score >= self._score_threshold:
            direction = 'up'
        elif score <= -self._score_threshold:
            direction = 'down'
        elif score >= self._preview_threshold:
            direction = 'up'
            is_preview = True
        elif score <= -self._preview_threshold:
            direction = 'down'
            is_preview = True

        if direction is None:
            return None

        # Indicator agreement check (skip for preview signals)
        if not is_preview:
            agree = 0
            if direction == 'up':
                if stoch_k < 30: agree += 1
                if mfi < 40: agree += 1
                if aroon_osc_val > -30: agree += 1
                if price > sar: agree += 1
            else:
                if stoch_k > 70: agree += 1
                if mfi > 60: agree += 1
                if aroon_osc_val < 30: agree += 1
                if price < sar: agree += 1

            if agree < 1:
                return None

        # Build signal
        return {
            'symbol': symbol,
            'direction': direction,
            'score': round(score, 1),
            'regime': regime,
            'price': price,
            'rsi7': round(rsi7, 1),
            'mfi': round(mfi, 1),
            'stoch_k': round(stoch_k, 1),
            'adx': round(adx_val, 1),
            'cci': round(cci, 1),
            'atr_pct': round(atr_pct, 3),
            'reasons': reasons if is_preview else reasons[:5],
            'is_preview': is_preview,
            'timestamp': candle_ts,
        }
