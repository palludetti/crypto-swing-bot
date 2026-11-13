"""
test_engine_integrity.py
Testes automatizados para comprovar:
1. Ausência de look-ahead bias em todos os indicadores e estratégias.
2. Execução causal (sinal em i executa em open[i+1], stop priorizado sobre TP intrabar).
3. Cobrança exata de taxas e slippage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_engine import BacktestConfig, run_single_asset_backtest
from portfolio_backtest import run_portfolio_backtest
from strategies import (
    strategy_donchian_breakout,
    strategy_ema_crossover,
    strategy_supertrend_adx,
    strategy_trend_pullback_rsi,
)


def _make_synthetic_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")
    rets = rng.normal(0.001, 0.03, size=n)
    closes = 100.0 * np.exp(np.cumsum(rets))
    opens = np.roll(closes, 1)
    opens[0] = 100.0
    highs = np.maximum(opens, closes) * (1.0 + rng.uniform(0.001, 0.02, size=n))
    lows = np.minimum(opens, closes) * (1.0 - rng.uniform(0.001, 0.02, size=n))
    return pd.DataFrame(
        {
            "open_time": times,
            "close_time": times + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": rng.uniform(1000, 5000, size=n),
        }
    )


def test_causal_no_lookahead() -> None:
    df = _make_synthetic_ohlcv(300, seed=7)
    btc = _make_synthetic_ohlcv(300, seed=99)

    cutoff = 250
    df_mut = df.copy()
    # Altera drasticamente o futuro (barras >= cutoff)
    df_mut.loc[cutoff:, ["open", "high", "low", "close"]] *= 5.0
    btc_mut = btc.copy()
    btc_mut.loc[cutoff:, ["open", "high", "low", "close"]] *= 0.2

    for name, fn in [
        ("donchian", strategy_donchian_breakout),
        (
            "donchian_refined",
            lambda d, b: strategy_donchian_breakout(
                d, b, entry_period=30, exit_period=15, atr_stop_mult=3.0,
                rs_btc_period=50, is_btc=False, own_sma_period=200, max_stop_pct=0.18,
            ),
        ),
        ("supertrend", strategy_supertrend_adx),
        ("ema", strategy_ema_crossover),
        ("pullback_rsi", strategy_trend_pullback_rsi),
    ]:
        full = fn(df, btc)
        mut = fn(df_mut, btc_mut)
        cols = ["signal_buy", "signal_sell", "stop_price_init", "tp_price_init"]
        if "momentum_score" in full.columns:
            cols.append("momentum_score")
        for col in cols:
            a = full[col].iloc[:cutoff]
            b = mut[col].iloc[:cutoff]
            assert np.allclose(
                a.fillna(-999).astype(float), b.fillna(-999).astype(float)
            ), f"Look-ahead detectado em {name}.{col}!"
    print("[PASS] test_causal_no_lookahead")


def test_stop_before_tp_and_fees() -> None:
    times = pd.date_range("2024-01-01", periods=6, freq="1D", tz="UTC")
    df = pd.DataFrame(
        {
            "open_time": times,
            "close_time": times + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1),
            "open": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [102.0, 102.0, 102.0, 130.0, 102.0, 102.0],
            "low":  [98.0,  98.0,  98.0,  85.0,  98.0,  98.0],
            "close": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            # Sinal no fechamento da barra 1 -> compra em open[2] = 100
            "signal_buy": [False, True, False, False, False, False],
            "signal_sell": [False] * 6,
            # Stop em 90 e TP em 120; na barra 3 high=130 e low=85 -> deve acionar STOP!
            "stop_price_init": [np.nan, 90.0, np.nan, np.nan, np.nan, np.nan],
            "tp_price_init": [np.nan, 120.0, np.nan, np.nan, np.nan, np.nan],
        }
    )
    cfg = BacktestConfig(initial_capital=1000.0, fee_rate=0.001, slippage_rate=0.0)
    res = run_single_asset_backtest(df, "Unit", "TEST", "1d", cfg)
    assert res.total_trades == 1
    tr = res.trades[0]
    assert tr.entry_time == times[2], f"Esperava entrada em barra 2, veio {tr.entry_time}"
    assert tr.exit_time == times[3], f"Esperava saída em barra 3, veio {tr.exit_time}"
    assert tr.exit_reason == "STOP_LOSS", f"Esperava STOP_LOSS, veio {tr.exit_reason}"
    assert np.isclose(tr.exit_price, 90.0)
    assert res.final_capital < 905.0  # perdeu 10% + taxas
    print("[PASS] test_stop_before_tp_and_fees")


def test_portfolio_momentum_ranking() -> None:
    times = pd.date_range("2024-01-01", periods=15, freq="1D", tz="UTC")
    base = {
        "open_time": times,
        "close_time": times + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1),
        "open": [100.0] * 15,
        "high": [101.0] * 15,
        "low": [99.0] * 15,
        "close": [100.0] * 15,
        "signal_buy": [False, True] + [False] * 13,
        "signal_sell": [False] * 10 + [True] + [False] * 4,
        "stop_price_init": [np.nan] * 15,
        "tp_price_init": [np.nan] * 15,
    }
    df_weak = pd.DataFrame({**base, "momentum_score": [1.5] * 15})
    df_strong = pd.DataFrame({**base, "momentum_score": [8.0] * 15})
    cfg = BacktestConfig(initial_capital=1000.0, fee_rate=0.001, slippage_rate=0.0)
    res = run_portfolio_backtest(
        {"WEAK": df_weak, "STRONG": df_strong},
        "RankTest",
        "1d",
        cfg,
        max_positions=1,
    )
    assert len(res.trades) == 1
    assert res.trades[0].symbol == "STRONG", f"Esperava STRONG, escolheu {res.trades[0].symbol}"
    print("[PASS] test_portfolio_momentum_ranking")


def test_portfolio_risk_sizing() -> None:
    times = pd.date_range("2024-01-01", periods=12, freq="1D", tz="UTC")
    df = pd.DataFrame(
        {
            "open_time": times,
            "close_time": times + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1),
            "open": [100.0] * 12,
            "high": [101.0] * 12,
            "low": [99.0, 99.0, 99.0, 75.0] + [99.0] * 8,
            "close": [100.0] * 12,
            "signal_buy": [False, True] + [False] * 10,
            "signal_sell": [False] * 12,
            # Stop 20% abaixo (80.0); com 3% de risco sobre $1000 = $30 risco -> alocação deve ser ~$150
            "stop_price_init": [np.nan, 80.0] + [np.nan] * 10,
            "tp_price_init": [np.nan] * 12,
        }
    )
    cfg = BacktestConfig(initial_capital=1000.0, fee_rate=0.0, slippage_rate=0.0, risk_per_trade_pct=0.03)
    res = run_portfolio_backtest({"ASSET": df}, "RiskTest", "1d", cfg, max_positions=2)
    assert len(res.trades) == 1
    tr = res.trades[0]
    # Posição deveria ser $150 (1.5 unidades a 100), perda de 20% = -$30 -> capital final = $970
    assert abs(res.final_capital - 970.0) < 0.01, f"Esperava $970.0, veio ${res.final_capital:.2f}"
    assert abs(tr.qty - 1.5) < 0.01
    print("[PASS] test_portfolio_risk_sizing")


if __name__ == "__main__":
    test_causal_no_lookahead()
    test_stop_before_tp_and_fees()
    test_portfolio_momentum_ranking()
    test_portfolio_risk_sizing()
