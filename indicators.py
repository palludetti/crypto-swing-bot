"""
indicators.py
Indicadores técnicos implementados em Pandas/NumPy seguindo convenções padrão
(Wilder smoothing para RSI, ATR e ADX) e sem look-ahead bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    result = 100.0 - (100.0 / (1.0 + rs))
    return result.fillna(100.0).where(avg_loss.notna(), np.nan)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def donchian_high(df: pd.DataFrame, period: int) -> pd.Series:
    """Máxima dos últimos `period` candles FECHADOS ANTERIORES (shift 1 evita look-ahead)."""
    return df["high"].rolling(window=period, min_periods=period).max().shift(1)


def donchian_low(df: pd.DataFrame, period: int) -> pd.Series:
    """Mínima dos últimos `period` candles FECHADOS ANTERIORES (shift 1 evita look-ahead)."""
    return df["low"].rolling(window=period, min_periods=period).min().shift(1)


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_s = atr(df, period=period)
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean() / tr_s.replace(0.0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean() / tr_s.replace(0.0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def supertrend(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> tuple[pd.Series, pd.Series]:
    """
    Calcula o Supertrend clássico.
    Retorna (supertrend_line, direction) onde direction = 1 (alta) ou -1 (baixa).
    """
    hl2 = (df["high"] + df["low"]) / 2.0
    atr_val = atr(df, period=period)

    upper_basic = hl2 + multiplier * atr_val
    lower_basic = hl2 - multiplier * atr_val

    close = df["close"].to_numpy()
    ub = upper_basic.to_numpy().copy()
    lb = lower_basic.to_numpy().copy()
    n = len(df)

    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)
    st_line = np.full(n, np.nan)

    first_valid = atr_val.first_valid_index()
    if first_valid is None:
        return pd.Series(st_line, index=df.index), pd.Series(direction, index=df.index)

    start_idx = int(df.index.get_loc(first_valid))
    final_upper[start_idx] = ub[start_idx]
    final_lower[start_idx] = lb[start_idx]
    direction[start_idx] = 1
    st_line[start_idx] = final_lower[start_idx]

    for i in range(start_idx + 1, n):
        # Final upper band
        if np.isnan(final_upper[i - 1]):
            final_upper[i] = ub[i]
        elif ub[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = ub[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Final lower band
        if np.isnan(final_lower[i - 1]):
            final_lower[i] = lb[i]
        elif lb[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = lb[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Direction
        prev_dir = direction[i - 1]
        if prev_dir == 1:
            if close[i] < final_lower[i]:
                direction[i] = -1
                st_line[i] = final_upper[i]
            else:
                direction[i] = 1
                st_line[i] = final_lower[i]
        else:
            if close[i] > final_upper[i]:
                direction[i] = 1
                st_line[i] = final_lower[i]
            else:
                direction[i] = -1
                st_line[i] = final_upper[i]

    return (
        pd.Series(st_line, index=df.index),
        pd.Series(direction, index=df.index),
    )
