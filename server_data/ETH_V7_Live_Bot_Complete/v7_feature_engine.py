"""Strictly causal feature builder for the ETH V7 live model.

The implementation mirrors ``research/eth_auto_optimizer.py``.  It only uses
candles that were fully closed at ``signal_close_ts`` and returns the feature
columns in the exact order stored in the model configuration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

FIVE_MINUTES_MS = 5 * 60 * 1000
FIFTEEN_MINUTES_MS = 15 * 60 * 1000
ONE_HOUR_MS = 60 * 60 * 1000


class FeatureBuildError(RuntimeError):
    """Raised when live candles cannot produce a complete V7 feature row."""


@dataclass(frozen=True)
class FeatureDiagnostics:
    base_open_time: int
    signal_close_time: int
    latest_15m_open_time: int
    latest_1h_open_time: int
    base_count: int
    count_15m: int
    count_1h: int


def _candles_to_frame(candles: Sequence[Sequence[float]]) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(
        candles,
        columns=["open_time", "open", "high", "low", "close", "volume"],
    )
    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    frame["open_time"] = pd.to_numeric(frame["open_time"], errors="coerce").astype("int64")
    return (
        frame.dropna()
        .sort_values("open_time")
        .drop_duplicates("open_time", keep="last")
        .reset_index(drop=True)
    )


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rma(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = rma(gain, period)
    avg_loss = rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return (
        out.mask((avg_loss == 0) & (avg_gain > 0), 100)
        .mask((avg_gain == 0) & (avg_loss > 0), 0)
        .mask((avg_gain == 0) & (avg_loss == 0), 50)
    )


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return rma(true_range, period)


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    atr_values = atr(high, low, close, period)
    plus_di = 100 * rma(plus_dm, period) / atr_values.replace(0, np.nan)
    minus_di = 100 * rma(minus_dm, period) / atr_values.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return rma(dx, period), plus_di, minus_di


def stoch_rsi(close: pd.Series, period: int = 14) -> tuple[pd.Series, pd.Series]:
    rsi_values = rsi(close, period)
    rolling_low = rsi_values.rolling(period, min_periods=period).min()
    rolling_high = rsi_values.rolling(period, min_periods=period).max()
    raw = 100 * (rsi_values - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)
    k = raw.rolling(3, min_periods=1).mean()
    d = k.rolling(3, min_periods=1).mean()
    return k, d


def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    typical = (high + low + close) / 3
    flow = typical * volume
    direction = typical.diff()
    positive = pd.Series(np.where(direction > 0, flow, 0.0), index=typical.index).rolling(
        period, min_periods=period
    ).sum()
    negative = pd.Series(np.where(direction < 0, flow, 0.0), index=typical.index).rolling(
        period, min_periods=period
    ).sum()
    ratio = positive / negative.replace(0, np.nan)
    out = 100 - 100 / (1 + ratio)
    return out.mask((negative == 0) & (positive > 0), 100).mask(
        (positive == 0) & (negative > 0), 0
    )


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    typical = (high + low + close) / 3
    average = typical.rolling(period, min_periods=period).mean()
    mean_deviation = typical.rolling(period, min_periods=period).apply(
        lambda values: np.mean(np.abs(values - values.mean())), raw=True
    )
    return (typical - average) / (0.015 * mean_deviation.replace(0, np.nan))


def williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    highest = high.rolling(period, min_periods=period).max()
    lowest = low.rolling(period, min_periods=period).min()
    return -100 * (highest - close) / (highest - lowest).replace(0, np.nan)


def _add_timeframe_features(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    result = frame.copy()
    close = result["close"]
    high = result["high"]
    low = result["low"]
    volume = result["volume"]
    for period in [1, 2, 3, 6, 12]:
        result[f"{prefix}_ret{period}"] = close.pct_change(period)
    for period in [7, 14, 28]:
        result[f"{prefix}_rsi{period}"] = rsi(close, period) / 100
    atr_values = atr(high, low, close, 14)
    result[f"{prefix}_atrp"] = atr_values / close
    adx_values, plus_di, minus_di = adx(high, low, close, 14)
    result[f"{prefix}_adx"] = adx_values / 100
    result[f"{prefix}_di"] = (plus_di - minus_di) / 100
    for period in [9, 21, 50]:
        result[f"{prefix}_ema{period}_gap"] = close / ema(close, period) - 1
    log_volume = np.log1p(volume)
    result[f"{prefix}_volz20"] = (
        log_volume - log_volume.rolling(20, min_periods=20).mean()
    ) / log_volume.rolling(20, min_periods=20).std(ddof=0)
    return result


def _build_base_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    open_ = result["open"]
    high = result["high"]
    low = result["low"]
    close = result["close"]
    volume = result["volume"]

    atr_values = atr(high, low, close, 14)
    result["atrp"] = atr_values / close
    for period in [1, 2, 3, 4, 6, 9, 12, 18, 24, 36, 48, 72]:
        result[f"ret{period}"] = close.pct_change(period)
    result["gap1"] = open_ / close.shift(1) - 1
    result["body"] = (close - open_) / atr_values
    result["range"] = (high - low) / atr_values
    result["upper_wick"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / atr_values
    result["lower_wick"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / atr_values
    result["close_pos"] = (close - low) / (high - low).replace(0, np.nan)

    for period in [5, 10, 20, 40, 80]:
        average = close.rolling(period, min_periods=period).mean()
        stddev = close.rolling(period, min_periods=period).std(ddof=0)
        result[f"z{period}"] = (close - average) / stddev.replace(0, np.nan)
        result[f"ma{period}_gap"] = close / average - 1
    for period in [5, 9, 12, 21, 34, 55]:
        result[f"ema{period}_gap"] = close / ema(close, period) - 1
    for period in [5, 7, 9, 14, 21, 28]:
        result[f"rsi{period}"] = rsi(close, period) / 100

    stoch_k, stoch_d = stoch_rsi(close, 14)
    result["stoch_k"] = stoch_k / 100
    result["stoch_d"] = stoch_d / 100
    result["stoch_diff"] = (stoch_k - stoch_d) / 100
    result["mfi14"] = mfi(high, low, close, volume, 14) / 100
    result["cci20"] = np.tanh(cci(high, low, close, 20) / 200)
    result["willr14"] = williams_r(high, low, close, 14) / 100

    adx_values, plus_di, minus_di = adx(high, low, close, 14)
    result["adx5"] = adx_values / 100
    result["di5"] = (plus_di - minus_di) / 100

    middle = close.rolling(20, min_periods=20).mean()
    stddev = close.rolling(20, min_periods=20).std(ddof=0)
    upper = middle + 2 * stddev
    lower = middle - 2 * stddev
    result["bb_pos"] = (close - lower) / (upper - lower).replace(0, np.nan)
    result["bb_width"] = (upper - lower) / middle

    log_volume = np.log1p(volume)
    for period in [10, 20, 50, 100]:
        result[f"volz{period}"] = (
            log_volume - log_volume.rolling(period, min_periods=period).mean()
        ) / log_volume.rolling(period, min_periods=period).std(ddof=0)
    result["buy_pressure"] = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)

    log_return = np.log(close).diff()
    for period in [3, 6, 12, 24, 48]:
        result[f"vol{period}"] = log_return.rolling(period, min_periods=period).std(ddof=0)
        result[f"upfrac{period}"] = (log_return > 0).rolling(period, min_periods=period).mean()

    dt = pd.to_datetime(result["open_time"], unit="ms", utc=True)
    hour = dt.dt.hour + dt.dt.minute / 60
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    day_of_week = dt.dt.dayofweek
    result["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    result["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    return result.replace([np.inf, -np.inf], np.nan)


def _filter_fully_closed(
    frame: pd.DataFrame,
    signal_close_ts: int,
    duration_ms: int,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.loc[frame["open_time"] + duration_ms <= signal_close_ts].reset_index(drop=True)


def build_latest_feature_row(
    candles_5m: Sequence[Sequence[float]],
    candles_15m: Sequence[Sequence[float]],
    candles_1h: Sequence[Sequence[float]],
    signal_close_ts: int,
    feature_names: Iterable[str],
) -> tuple[pd.DataFrame, FeatureDiagnostics]:
    """Build one model-ready feature row from fully closed live candles."""
    feature_names = list(feature_names)
    base = _filter_fully_closed(_candles_to_frame(candles_5m), signal_close_ts, FIVE_MINUTES_MS)
    frame_15m = _filter_fully_closed(
        _candles_to_frame(candles_15m), signal_close_ts, FIFTEEN_MINUTES_MS
    )
    frame_1h = _filter_fully_closed(_candles_to_frame(candles_1h), signal_close_ts, ONE_HOUR_MS)

    if len(base) < 120:
        raise FeatureBuildError(f"5m candles insufficient: {len(base)} < 120")
    if len(frame_15m) < 60:
        raise FeatureBuildError(f"15m candles insufficient: {len(frame_15m)} < 60")
    if len(frame_1h) < 60:
        raise FeatureBuildError(f"1h candles insufficient: {len(frame_1h)} < 60")

    base_features = _build_base_features(base)
    latest = base_features.iloc[-1].copy()
    tf15_features = _add_timeframe_features(frame_15m, "m15").iloc[-1]
    tf1h_features = _add_timeframe_features(frame_1h, "h1").iloc[-1]
    for name, value in tf15_features.items():
        if name.startswith("m15_"):
            latest[name] = value
    for name, value in tf1h_features.items():
        if name.startswith("h1_"):
            latest[name] = value

    missing = [name for name in feature_names if name not in latest.index]
    if missing:
        raise FeatureBuildError(f"missing feature columns: {missing}")
    row = pd.DataFrame([[latest[name] for name in feature_names]], columns=feature_names)
    invalid = [name for name in feature_names if not np.isfinite(float(row.iloc[0][name]))]
    if invalid:
        raise FeatureBuildError(f"non-finite feature values: {invalid[:12]}")

    diagnostics = FeatureDiagnostics(
        base_open_time=int(base.iloc[-1]["open_time"]),
        signal_close_time=int(signal_close_ts),
        latest_15m_open_time=int(frame_15m.iloc[-1]["open_time"]),
        latest_1h_open_time=int(frame_1h.iloc[-1]["open_time"]),
        base_count=len(base),
        count_15m=len(frame_15m),
        count_1h=len(frame_1h),
    )
    return row.astype("float64"), diagnostics
