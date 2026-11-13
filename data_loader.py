"""
data_loader.py
Coletor de dados históricos OHLCV (candlesticks) da API pública de mercado da Binance
(data-api.binance.vision), sem necessidade de chave de API e com cache local em CSV.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

BINANCE_DATA_URL = "https://data-api.binance.vision/api/v3/klines"

INTERVAL_MS = {
    "1h": 3_600_000,
    "4h": 4 * 3_600_000,
    "1d": 24 * 3_600_000,
}


def _to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_klines(
    symbol: str,
    interval: str = "1d",
    start_date: str = "2021-01-01",
    end_date: Optional[str] = None,
    cache_dir: str = "data",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Baixa candles históricos completos entre start_date e end_date (UTC).
    Usa cache CSV local e faz paginação automática (1000 candles/request).
    """
    if interval not in INTERVAL_MS:
        raise ValueError(f"Intervalo não suportado: {interval}. Use {list(INTERVAL_MS.keys())}")

    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{symbol}_{interval}.csv")

    start_ms = _to_ms(start_date)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    end_ms = _to_ms(end_date) if end_date else now_ms

    df_cached: Optional[pd.DataFrame] = None
    fetch_from_ms = start_ms

    if os.path.exists(cache_file) and not force_refresh:
        df_cached = pd.read_csv(cache_file, parse_dates=["open_time", "close_time"])
        if not df_cached.empty:
            cached_min_ms = int(df_cached["open_time_ms"].min())
            cached_max_ms = int(df_cached["open_time_ms"].max())
            cached_max_close_ms = int(df_cached["close_time_ms"].max())
            # Se o cache cobre o início e já possui o último candle completamente fechado
            step = INTERVAL_MS[interval]
            if cached_min_ms <= start_ms + step:
                if cached_max_close_ms + step >= end_ms:
                    filtered = df_cached[
                        (df_cached["open_time_ms"] >= start_ms)
                        & (df_cached["open_time_ms"] <= end_ms)
                    ].copy()
                    return filtered.reset_index(drop=True)
                # Atualiza apenas do último candle em diante
                fetch_from_ms = cached_max_ms + step

    all_rows = []
    cursor_ms = fetch_from_ms
    step = INTERVAL_MS[interval]

    while cursor_ms < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor_ms,
            "endTime": end_ms,
            "limit": 1000,
        }
        resp = None
        for attempt in range(4):
            try:
                resp = requests.get(BINANCE_DATA_URL, params=params, timeout=15)
                if resp.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(1.5 * (attempt + 1))

        assert resp is not None
        batch = resp.json()
        if not batch:
            break

        all_rows.extend(batch)
        last_open_ms = int(batch[-1][0])
        if last_open_ms <= cursor_ms:
            cursor_ms += step
        else:
            cursor_ms = last_open_ms + step

        if len(batch) < 1000:
            break
        time.sleep(0.08)

    if all_rows:
        df_new = pd.DataFrame(
            all_rows,
            columns=[
                "open_time_ms",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time_ms",
                "quote_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )
        numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume"]
        for col in numeric_cols:
            df_new[col] = df_new[col].astype(float)
        df_new["open_time_ms"] = df_new["open_time_ms"].astype("int64")
        df_new["close_time_ms"] = df_new["close_time_ms"].astype("int64")
        df_new["open_time"] = pd.to_datetime(df_new["open_time_ms"], unit="ms", utc=True)
        df_new["close_time"] = pd.to_datetime(df_new["close_time_ms"], unit="ms", utc=True)
        df_new = df_new[
            [
                "open_time_ms",
                "open_time",
                "close_time_ms",
                "close_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume",
            ]
        ]

        if df_cached is not None and not df_cached.empty and not force_refresh:
            df_merged = pd.concat([df_cached, df_new], ignore_index=True)
        else:
            df_merged = df_new

        df_merged = (
            df_merged.drop_duplicates(subset=["open_time_ms"])
            .sort_values("open_time_ms")
            .reset_index(drop=True)
        )
        # Remove candle ainda aberto (incompleto) para evitar look-ahead / distorção no último ponto
        df_merged = df_merged[df_merged["close_time_ms"] <= now_ms].reset_index(drop=True)
        df_merged.to_csv(cache_file, index=False)
    elif df_cached is not None:
        df_merged = df_cached
    else:
        raise RuntimeError(f"Nenhum dado retornado para {symbol} ({interval})")

    filtered = df_merged[
        (df_merged["open_time_ms"] >= start_ms) & (df_merged["open_time_ms"] <= end_ms)
    ].copy()
    return filtered.reset_index(drop=True)
