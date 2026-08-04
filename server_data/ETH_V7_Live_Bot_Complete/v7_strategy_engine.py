"""Live inference adapter for the frozen ETH V7.0 causal model."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import lightgbm as lgb

from v7_feature_engine import FeatureBuildError, build_latest_feature_row

logger = logging.getLogger(__name__)
BEIJING_TZ = timezone(timedelta(hours=8))


def beijing_hour(ts_ms: int) -> int:
    return datetime.fromtimestamp(ts_ms / 1000, tz=BEIJING_TZ).hour


class V7StrategyEngine:
    """Generate ETH event-contract signals from the frozen V7 LightGBM model."""

    def __init__(self, model_dir: str | Path | None = None):
        root = Path(__file__).resolve().parent
        self.model_dir = Path(model_dir) if model_dir else root / "models" / "v7"
        config_path = self.model_dir / "eth_v7_balanced_config.json"
        model_path = self.model_dir / "eth_v7_balanced_model.txt"
        if not config_path.exists():
            raise FileNotFoundError(f"V7 config not found: {config_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"V7 model not found: {model_path}")

        self.metadata = json.loads(config_path.read_text(encoding="utf-8"))
        self.version = str(self.metadata.get("version", "V7.0-causal-ml-balanced"))
        self.config = dict(self.metadata["config"])
        self.features = list(self.metadata["features"])
        self.threshold = float(self.config["prob_threshold"])
        self.min_atrp = float(self.config["min_atrp"])
        self.cooldown = int(self.config.get("cooldown", 1))
        self.max_daily = int(self.config.get("max_daily", 50))
        self.max_daily_loss = float(self.config.get("max_daily_loss", -75.0))
        self.session_hours = sorted(int(h) for h in self.config.get("session_hours", []))
        try:
            self.model = lgb.Booster(model_file=str(model_path))
        except lgb.basic.LightGBMError:
            # Windows 下 LightGBM C API 无法打开含非 ASCII 字符的路径（中文目录），
            # 回退为从字符串加载（模型为 ASCII 文本）。
            self.model = lgb.Booster(model_str=model_path.read_text(encoding="utf-8"))
        model_features = list(self.model.feature_name())
        if model_features != self.features:
            raise ValueError("V7 model feature order does not match config")

    def in_session(self, ts_ms: int) -> bool:
        """信号层时段过滤（北京时间小时）。空集合 = 不过滤。

        被过滤的信号不占冷却位、不计数、不影响熔断——语义与回测 simulate 一致。
        """
        return not self.session_hours or beijing_hour(ts_ms) in self.session_hours

    def evaluate(
        self,
        candles_5m: Sequence[Sequence[float]],
        candles_15m: Sequence[Sequence[float]],
        candles_1h: Sequence[Sequence[float]],
        symbol: str,
        signal_close_ts: int,
    ) -> dict | None:
        if symbol != "ETHUSDT":
            logger.debug("V7 frozen model only supports ETHUSDT, skipping %s", symbol)
            return None
        try:
            row, diagnostics = build_latest_feature_row(
                candles_5m,
                candles_15m,
                candles_1h,
                signal_close_ts,
                self.features,
            )
        except FeatureBuildError as exc:
            logger.warning("%s V7 feature build skipped: %s", symbol, exc)
            return None

        probability_up = float(self.model.predict(row, num_iteration=self.model.num_trees())[0])
        confidence = max(probability_up, 1.0 - probability_up)
        atrp = float(row.iloc[0]["atrp"])
        if atrp < self.min_atrp:
            return None
        if probability_up >= self.threshold:
            direction = "up"
        elif probability_up <= 1.0 - self.threshold:
            direction = "down"
        else:
            return None

        latest = candles_5m[-1]
        rsi7 = float(row.iloc[0]["rsi7"]) * 100
        mfi14 = float(row.iloc[0]["mfi14"]) * 100
        stoch_k = float(row.iloc[0]["stoch_k"]) * 100
        cci_scaled = float(row.iloc[0]["cci20"])
        adx_1h = float(row.iloc[0]["h1_adx"]) * 100
        regime = "趋势" if adx_1h >= 25 else "震荡"
        direction_cn = "上涨" if direction == "up" else "下跌"
        probability_for_direction = probability_up if direction == "up" else 1.0 - probability_up
        reasons = [
            f"V7预测{direction_cn}概率 {probability_for_direction * 100:.2f}%",
            f"冻结阈值 {self.threshold * 100:.1f}%",
            f"ATR {atrp * 100:.3f}%",
            f"严格收盘对齐 15m={diagnostics.latest_15m_open_time} 1h={diagnostics.latest_1h_open_time}",
        ]
        return {
            "symbol": symbol,
            "direction": direction,
            # Keep compatibility with existing notifier/tracker.  Score 5.55 means 55.5% confidence.
            "score": round(confidence * 10, 4),
            "probability_up": probability_up,
            "confidence": confidence,
            "model_version": self.version,
            "regime": regime,
            "price": float(latest[4]),  # reference close; actual tracked entry is next 5m open
            "reference_close": float(latest[4]),
            "timestamp": int(signal_close_ts),
            "rsi7": rsi7,
            "mfi": mfi14,
            "stoch_k": stoch_k,
            "adx": adx_1h,
            "cci": cci_scaled * 200,
            "atr_pct": atrp * 100,
            "reasons": reasons,
            "is_preview": False,
        }
