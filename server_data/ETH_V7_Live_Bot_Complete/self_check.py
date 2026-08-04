"""Offline package integrity check. Does not connect to Binance."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    required = [
        ROOT / "config.yaml",
        ROOT / "models/v7/eth_v7_balanced_model.txt",
        ROOT / "models/v7/eth_v7_balanced_config.json",
        ROOT / "v7_run.py",
        ROOT / "data_stream.py",
        ROOT / "notifier.py",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        print("[FAIL] Missing files:", ", ".join(missing))
        return 1

    for module_name in ("aiohttp", "yaml", "numpy", "pandas", "lightgbm"):
        importlib.import_module(module_name)
        print(f"[OK] import {module_name}")

    config_path = ROOT / "models/v7/eth_v7_balanced_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    features = config.get("features", [])
    if len(features) != 100:
        print(f"[FAIL] Expected 100 model features, got {len(features)}")
        return 1

    from v7_strategy_engine import V7StrategyEngine

    strategy = V7StrategyEngine(ROOT / "models/v7")
    print(f"[OK] model loaded: {strategy.version}")
    print(f"[OK] threshold: {strategy.threshold}")
    print("[OK] package integrity check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
