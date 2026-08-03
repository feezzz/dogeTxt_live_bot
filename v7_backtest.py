"""V7 frozen-model historical backtest (2024-2026 ETHUSDT)."""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
V7_DIR = ROOT / "server_data" / "ETH_V7_Live_Bot_Complete"
sys.path.insert(0, str(V7_DIR))
import v7_feature_engine as fe  # noqa: E402

CACHE_DIR = ROOT / "server_data" / "cache"
BEIJING_TZ = timezone(timedelta(hours=8))
FIVE_MINUTES_MS = fe.FIVE_MINUTES_MS
INTERVALS = {"5m": 5 * 60 * 1000, "15m": 15 * 60 * 1000, "1h": 60 * 60 * 1000}


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int,
                 proxy_url: str | None = None) -> list[list[float]]:
    rows: list[list[float]] = []
    proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None
    cursor = start_ms
    while cursor < end_ms:
        data = None
        for attempt in range(3):
            try:
                resp = requests.get(
                    "https://api.binance.com/api/v3/klines",
                    params={"symbol": symbol, "interval": interval,
                            "startTime": cursor, "endTime": end_ms, "limit": 1000},
                    proxies=proxies, timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        if not data:
            break
        rows.extend([float(v) for v in r[:6]] for r in data)
        cursor = int(data[-1][0]) + 1
        time.sleep(0.15)
    return rows


def load_or_fetch(symbol: str, interval: str, start_ms: int, end_ms: int,
                  proxy_url: str | None = None) -> list[list[float]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{symbol}_{interval}_{start_ms}_{end_ms}.csv"
    if path.exists():
        return pd.read_csv(path).values.tolist()
    rows = fetch_klines(symbol, interval, start_ms, end_ms, proxy_url)
    pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"]
                 ).to_csv(path, index=False)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="V7 历史回测数据获取与缓存")
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--start", help="起始时间, 如 2024-01-01 或 unix ms")
    parser.add_argument("--end", help="结束时间, 如 2026-08-01 或 unix ms")
    parser.add_argument("--no-proxy", action="store_true", help="禁用代理直连 Binance")
    args = parser.parse_args()

    def to_ms(s: str | None, default_dt: datetime) -> int:
        if not s:
            return int(default_dt.timestamp() * 1000)
        if s.isdigit():
            return int(s)
        return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=BEIJING_TZ).timestamp() * 1000)

    start_ms = to_ms(args.start, datetime(2024, 1, 1, tzinfo=BEIJING_TZ))
    end_ms = to_ms(args.end, datetime(2026, 8, 1, tzinfo=BEIJING_TZ))
    proxy_url = None if args.no_proxy else "http://127.0.0.1:7892"
    rows = load_or_fetch(args.symbol, args.interval, start_ms, end_ms, proxy_url)
    print(f"candles: {len(rows)}")
    if rows:
        print("first:", rows[0])
        print("last:", rows[-1])


if __name__ == "__main__":
    main()
