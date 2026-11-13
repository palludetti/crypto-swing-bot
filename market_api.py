"""
market_api.py
Consultas públicas em tempo real na Binance (sem chave privada ou API key)
e atualização incremental dos candles diários fechados.
"""

from __future__ import annotations

import json
from typing import Iterable

import pandas as pd
import requests

from config import DATA_DIR, UNIVERSE
from data_loader import fetch_klines

BINANCE_TICKER_PRICE_URL = "https://data-api.binance.vision/api/v3/ticker/price"


def fetch_live_prices(symbols: Iterable[str] = UNIVERSE, include_brl: bool = True) -> dict[str, float]:
    """
    Busca cotações em tempo real em uma única chamada em lote na Binance.
    Retorna dicionário {symbol: price_float}.
    """
    sym_list = list(symbols)
    if include_brl and "USDTBRL" not in sym_list:
        sym_list.append("USDTBRL")

    symbols_json = json.dumps(sym_list, separators=(",", ":"))
    resp = requests.get(
        BINANCE_TICKER_PRICE_URL,
        params={"symbols": symbols_json},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return {item["symbol"]: float(item["price"]) for item in data}


def fetch_universe_daily_candles(
    symbols: Iterable[str] = UNIVERSE,
    start_date: str = "2025-01-01",
    cache_dir: str = DATA_DIR,
) -> dict[str, pd.DataFrame]:
    """
    Carrega do cache local e atualiza até o último candle diário fechado (1d)
    para todos os ativos do universo.
    """
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        out[sym] = fetch_klines(sym, interval="1d", start_date=start_date, cache_dir=cache_dir)
    return out
