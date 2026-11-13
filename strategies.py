"""
strategies.py
Implementação das estratégias clássicas de Swing Trade para validação em backtest.
Todas as regras respeitam estritamente o fechamento dos candles (sem look-ahead bias).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from indicators import adx, atr, donchian_high, donchian_low, ema, rsi, sma, supertrend


def attach_btc_regime(
    df: pd.DataFrame,
    btc_daily_df: Optional[pd.DataFrame],
    sma_period: int = 200,
) -> pd.Series:
    """
    Mapeia para cada candle de `df` se o BTC estava em regime de alta (close > SMA200)
    com base apenas no ÚLTIMO candle diário de BTC completamente fechado até a data do candle.
    """
    if btc_daily_df is None or btc_daily_df.empty:
        return pd.Series(True, index=df.index)

    btc = btc_daily_df[["open_time", "close_time", "close"]].copy().sort_values("close_time")
    btc["btc_sma"] = sma(btc["close"], sma_period)
    btc["btc_bull"] = (btc["close"] > btc["btc_sma"]).fillna(False)

    left = df[["open_time", "close_time"]].copy().sort_values("close_time")
    merged = pd.merge_asof(
        left,
        btc[["close_time", "btc_bull"]],
        on="close_time",
        direction="backward",
    )
    # Alinha de volta ao índice original de df
    merged.index = left.index
    return merged.sort_index()["btc_bull"].fillna(False)


def compute_momentum_score(
    df: pd.DataFrame,
    atr_series: pd.Series,
    lookback: int = 30,
) -> pd.Series:
    """
    Momentum ajustado ao risco (estilo Clenow): retorno acumulado dividido pela
    volatilidade percentual (ATR/Close). Prioriza tendências fortes e limpas.
    """
    roc = (df["close"] / df["close"].shift(lookback)) - 1.0
    atr_pct = (atr_series / df["close"]).replace(0.0, np.nan)
    return (roc / atr_pct).fillna(0.0)


def attach_rs_vs_btc(
    df: pd.DataFrame,
    btc_daily_df: Optional[pd.DataFrame],
    sma_period: int = 50,
    is_btc: bool = False,
) -> pd.Series:
    """
    Filtro de Força Relativa contra o Bitcoin (ALT/BTC > SMA(ALT/BTC, sma_period)).
    Garante que só alocamos em altcoins que estão superando ou acompanhando o BTC.
    Para o próprio BTC (`is_btc=True`), retorna True.
    """
    if is_btc or btc_daily_df is None or btc_daily_df.empty:
        return pd.Series(True, index=df.index)

    btc = btc_daily_df[["close_time", "close"]].rename(columns={"close": "btc_close"}).sort_values("close_time")
    left = df[["close_time", "close"]].copy().sort_values("close_time")
    merged = pd.merge_asof(
        left,
        btc,
        on="close_time",
        direction="backward",
    )
    merged.index = left.index
    merged = merged.sort_index()

    ratio = merged["close"] / merged["btc_close"].replace(0.0, np.nan)
    ratio_sma = sma(ratio, sma_period)
    return (ratio > ratio_sma).fillna(False)


def strategy_donchian_breakout(
    df: pd.DataFrame,
    btc_daily_df: Optional[pd.DataFrame] = None,
    entry_period: int = 20,
    exit_period: int = 10,
    use_btc_filter: bool = True,
    atr_stop_mult: float = 2.5,
    rs_btc_period: Optional[int] = None,
    is_btc: bool = False,
    momentum_lookback: int = 30,
    own_sma_period: Optional[int] = None,
    max_stop_pct: Optional[float] = None,
) -> pd.DataFrame:
    """
    Estratégia 1: Donchian Breakout (Turtle Trend Following).
    - Compra quando fecha acima da máxima dos últimos `entry_period` candles (excluindo o atual),
      desde que preço > EMA(50), (opcional) BTC > SMA(200) diário,
      (opcional) preço > SMA(own_sma_period) do próprio ativo,
      (opcional) Força Relativa > BTC e (opcional) distância até o stop <= max_stop_pct.
    - Stop inicial em `atr_stop_mult * ATR(14)`.
    - Sinal de saída quando fecha abaixo da mínima dos últimos `exit_period` candles.
    """
    out = df.copy()
    out["atr"] = atr(out, 14)
    out["ema50"] = ema(out["close"], 50)
    out["don_hi"] = donchian_high(out, entry_period)
    out["don_lo"] = donchian_low(out, exit_period)
    out["btc_bull"] = attach_btc_regime(out, btc_daily_df) if use_btc_filter else True
    out["rs_btc_ok"] = (
        attach_rs_vs_btc(out, btc_daily_df, sma_period=rs_btc_period, is_btc=is_btc)
        if rs_btc_period is not None
        else True
    )
    out["own_sma_ok"] = (
        (out["close"] > sma(out["close"], own_sma_period)).fillna(False)
        if own_sma_period is not None
        else True
    )
    out["momentum_score"] = compute_momentum_score(out, out["atr"], lookback=momentum_lookback)

    stop_price = out["close"] - atr_stop_mult * out["atr"]
    stop_dist_pct = (out["close"] - stop_price) / out["close"].replace(0.0, np.nan)
    dist_ok = (stop_dist_pct <= max_stop_pct).fillna(False) if max_stop_pct is not None else True

    raw_buy = (
        (out["close"] > out["don_hi"])
        & (out["close"] > out["ema50"])
        & out["btc_bull"]
        & out["rs_btc_ok"]
        & out["own_sma_ok"]
        & dist_ok
    ).fillna(False)

    out["signal_buy"] = raw_buy & ~raw_buy.shift(1, fill_value=False)
    out["signal_sell"] = (out["close"] < out["don_lo"]).fillna(False)
    out["stop_price_init"] = stop_price
    out["tp_price_init"] = np.nan
    return out


def strategy_supertrend_adx(
    df: pd.DataFrame,
    btc_daily_df: Optional[pd.DataFrame] = None,
    st_period: int = 10,
    st_mult: float = 3.0,
    min_adx: float = 18.0,
    use_btc_filter: bool = True,
    atr_stop_mult: float = 3.0,
) -> pd.DataFrame:
    """
    Estratégia 2: Supertrend + Filtro ADX + Tendência EMA200.
    - Compra quando Supertrend está em alta (dir == 1), ADX > min_adx (há tendência),
      close > EMA(200) e (opcional) BTC > SMA(200).
    - Sai quando Supertrend vira para baixa (dir == -1) ou via stop ATR.
    """
    out = df.copy()
    out["atr"] = atr(out, 14)
    out["ema200"] = ema(out["close"], 200)
    out["adx"] = adx(out, 14)
    st_line, st_dir = supertrend(out, period=st_period, multiplier=st_mult)
    out["st_line"] = st_line
    out["st_dir"] = st_dir
    out["btc_bull"] = attach_btc_regime(out, btc_daily_df) if use_btc_filter else True

    raw_buy = (
        (out["st_dir"] == 1)
        & (out["adx"] >= min_adx)
        & (out["close"] > out["ema200"])
        & out["btc_bull"]
    ).fillna(False)

    out["signal_buy"] = raw_buy & ~raw_buy.shift(1, fill_value=False)
    out["signal_sell"] = (out["st_dir"] == -1).fillna(False)
    # Stop inicial no menor valor entre linha do Supertrend e close - mult*ATR
    atr_stop = out["close"] - atr_stop_mult * out["atr"]
    out["stop_price_init"] = np.minimum(out["st_line"], atr_stop)
    out["tp_price_init"] = np.nan
    return out


def strategy_ema_crossover(
    df: pd.DataFrame,
    btc_daily_df: Optional[pd.DataFrame] = None,
    fast_period: int = 20,
    slow_period: int = 50,
    trend_period: int = 200,
    use_btc_filter: bool = True,
    atr_stop_mult: float = 2.5,
) -> pd.DataFrame:
    """
    Estratégia 3: Cruzamento de Médias (EMA20 x EMA50) com filtro EMA200 e BTC.
    - Compra quando EMA rápida > EMA lenta, preço > EMA200 e BTC em tendência de alta.
    - Sai quando EMA rápida cruza abaixo da EMA lenta.
    """
    out = df.copy()
    out["atr"] = atr(out, 14)
    out["ema_fast"] = ema(out["close"], fast_period)
    out["ema_slow"] = ema(out["close"], slow_period)
    out["ema_trend"] = ema(out["close"], trend_period)
    out["btc_bull"] = attach_btc_regime(out, btc_daily_df) if use_btc_filter else True

    raw_buy = (
        (out["ema_fast"] > out["ema_slow"])
        & (out["close"] > out["ema_trend"])
        & out["btc_bull"]
    ).fillna(False)

    out["signal_buy"] = raw_buy & ~raw_buy.shift(1, fill_value=False)
    out["signal_sell"] = (out["ema_fast"] < out["ema_slow"]).fillna(False)
    out["stop_price_init"] = out["close"] - atr_stop_mult * out["atr"]
    out["tp_price_init"] = np.nan
    return out


def strategy_trend_pullback_rsi(
    df: pd.DataFrame,
    btc_daily_df: Optional[pd.DataFrame] = None,
    rsi_entry_max: float = 42.0,
    use_btc_filter: bool = True,
    atr_stop_mult: float = 2.2,
    atr_tp_mult: float = 4.4,
) -> pd.DataFrame:
    """
    Estratégia 4: Pullback em Tendência de Alta (Mean Reversion a favor da tendência).
    - Macro: close > EMA(200), EMA(50) > EMA(200) e BTC > SMA(200).
    - Entrada: RSI(14) caiu abaixo de `rsi_entry_max` nos últimos 3 candles e agora
      está subindo (rsi > rsi_prev) com candle comprador (close > open).
    - Saída: Stop em 2.2*ATR ou Alvo fixo em 4.4*ATR (Risco/Retorno 1:2).
    """
    out = df.copy()
    out["atr"] = atr(out, 14)
    out["ema50"] = ema(out["close"], 50)
    out["ema200"] = ema(out["close"], 200)
    out["rsi"] = rsi(out["close"], 14)
    out["btc_bull"] = attach_btc_regime(out, btc_daily_df) if use_btc_filter else True

    recent_oversold = out["rsi"].rolling(window=3, min_periods=1).min() <= rsi_entry_max
    turning_up = (out["rsi"] > out["rsi"].shift(1)) & (out["close"] > out["open"])
    uptrend = (out["close"] > out["ema200"]) & (out["ema50"] > out["ema200"])

    out["signal_buy"] = (
        uptrend & recent_oversold & turning_up & out["btc_bull"]
    ).fillna(False)

    out["signal_sell"] = False  # Sai por Stop ou Take Profit (ou perda clara da tendência)
    out["stop_price_init"] = out["close"] - atr_stop_mult * out["atr"]
    out["tp_price_init"] = out["close"] + atr_tp_mult * out["atr"]
    return out
