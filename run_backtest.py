"""
run_backtest.py
Executa o laboratório comparativo de estratégias de Swing Trade em dados reais (2021-2026)
para portfólio multi-ativo e análise ano a ano.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from backtest_engine import BacktestConfig, run_single_asset_backtest
from data_loader import fetch_klines
from portfolio_backtest import PortfolioResult, run_portfolio_backtest
from strategies import (
    strategy_donchian_breakout,
    strategy_ema_crossover,
    strategy_supertrend_adx,
    strategy_trend_pullback_rsi,
)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "LINKUSDT", "AVAXUSDT"]
START_DATE = "2021-01-01"


def yearly_returns(eq_df: pd.DataFrame) -> dict[int, float]:
    df = eq_df.copy()
    df["year"] = pd.to_datetime(df["time"]).dt.year
    out = {}
    years = sorted(df["year"].unique())
    prev_end_eq = float(df["equity"].iloc[0])
    for yr in years:
        sub = df[df["year"] == yr]
        end_eq = float(sub["equity"].iloc[-1])
        ret = ((end_eq / prev_end_eq) - 1.0) * 100.0
        out[int(yr)] = ret
        prev_end_eq = end_eq
    return out


def print_portfolio_summary(results: list[tuple[PortfolioResult, dict[int, float]]]) -> None:
    print("\n" + "=" * 112)
    print("COMPARATIVO DE PORTFÓLIO MULTI-ATIVO (Banca inicial: $1,000 | Máx 4 posições simultâneas | Taxa 0.10% + Slippage 0.05%)")
    print("=" * 112)
    header = (
        f"{'Estratégia':<34} {'TF':<4} {'Final($)':>9} {'Ret%':>8} {'B&H%':>8} "
        f"{'CAGR%':>7} {'MaxDD%':>8} {'BH_DD%':>8} {'Sharpe':>7} {'PF':>6} {'Win%':>6} {'Trades':>6}"
    )
    print(header)
    print("-" * 112)
    for res, _ in results:
        pf_str = f"{res.profit_factor:.2f}" if res.profit_factor < 100 else "INF"
        print(
            f"{res.strategy_name:<34} {res.interval:<4} "
            f"{res.final_capital:>9.2f} {res.total_return_pct:>7.1f}% {res.equal_weight_bh_return_pct:>7.1f}% "
            f"{res.cagr_pct:>6.1f}% {res.max_drawdown_pct:>7.1f}% {res.bh_max_drawdown_pct:>7.1f}% "
            f"{res.sharpe_ratio:>7.2f} {pf_str:>6} {res.win_rate_pct:>5.1f}% {res.total_trades:>6}"
        )

    print("\n" + "=" * 112)
    print("CONSISTÊNCIA ANO A ANO (% de retorno do Portfólio em cada ano)")
    print("=" * 112)
    all_years = sorted({yr for _, yrs in results for yr in yrs.keys()})
    yr_hdr = f"{'Estratégia':<34} {'TF':<4} " + " ".join(f"{yr:>9}" for yr in all_years)
    print(yr_hdr)
    print("-" * 112)
    for res, yrs in results:
        vals = " ".join(f"{yrs.get(yr, 0.0):>8.1f}%" for yr in all_years)
        print(f"{res.strategy_name:<34} {res.interval:<4} {vals}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    print("Baixando / carregando dados históricos da Binance (data-api.binance.vision)...")
    raw_1d: dict[str, pd.DataFrame] = {}
    raw_4h: dict[str, pd.DataFrame] = {}

    for sym in SYMBOLS:
        print(f"  -> {sym} (1d & 4h)...", end="", flush=True)
        raw_1d[sym] = fetch_klines(sym, "1d", start_date=START_DATE, cache_dir=data_dir)
        raw_4h[sym] = fetch_klines(sym, "4h", start_date=START_DATE, cache_dir=data_dir)
        print(f" OK ({len(raw_1d[sym])} dias | {len(raw_4h[sym])} velas 4h)")

    btc_1d = raw_1d["BTCUSDT"]

    cfg_1d = BacktestConfig(initial_capital=1000.0, fee_rate=0.001, slippage_rate=0.0005, periods_per_year=365.0)
    cfg_4h = BacktestConfig(initial_capital=1000.0, fee_rate=0.001, slippage_rate=0.0005, periods_per_year=365.0 * 6)

    experiments = [
        ("1. Donchian 30/15 + BTC (1D)", "1d", raw_1d, cfg_1d, lambda df: strategy_donchian_breakout(df, btc_1d, 30, 15, True, 3.0), None),
        ("2. Donchian 40/20 + BTC (1D)", "1d", raw_1d, cfg_1d, lambda df: strategy_donchian_breakout(df, btc_1d, 40, 20, True, 3.0), None),
        ("3. Donchian 20/10 + BTC + Trail", "1d", raw_1d, cfg_1d, lambda df: strategy_donchian_breakout(df, btc_1d, 20, 10, True, 2.5), 3.0),
        ("4. Donchian 20/10 (Sem Filtro)", "1d", raw_1d, cfg_1d, lambda df: strategy_donchian_breakout(df, btc_1d, 20, 10, False, 2.5), 3.0),
        ("5. Supertrend + ADX + BTC", "1d", raw_1d, cfg_1d, lambda df: strategy_supertrend_adx(df, btc_1d, 10, 3.0, 18.0, True, 3.0), None),
        ("6. EMA 20/50 + EMA200 + BTC", "1d", raw_1d, cfg_1d, lambda df: strategy_ema_crossover(df, btc_1d, 20, 50, 200, True, 2.5), 3.0),
        ("7. Supertrend + ADX + BTC", "4h", raw_4h, cfg_4h, lambda df: strategy_supertrend_adx(df, btc_1d, 10, 3.0, 20.0, True, 3.0), None),
        ("8. Pullback RSI + EMA200 + BTC", "4h", raw_4h, cfg_4h, lambda df: strategy_trend_pullback_rsi(df, btc_1d, 42.0, True, 2.2, 4.4), None),
    ]

    portfolio_results: list[tuple[PortfolioResult, dict[int, float]]] = []
    for name, tf, raw_map, cfg, strat_fn, trail_mult in experiments:
        annotated = {sym: strat_fn(df) for sym, df in raw_map.items()}
        res = run_portfolio_backtest(
            annotated,
            strategy_name=name,
            interval=tf,
            config=cfg,
            max_positions=4,
            trailing_atr_mult=trail_mult,
        )
        portfolio_results.append((res, yearly_returns(res.equity_curve)))

    print_portfolio_summary(portfolio_results)

    # Detalhamento da melhor estratégia (por Sharpe ou Calmar com pelo menos 30 trades)
    valid = [r for r, _ in portfolio_results if r.total_trades >= 25]
    best = max(valid or [r for r, _ in portfolio_results], key=lambda r: r.calmar_ratio)

    print(f"\n" + "=" * 112)
    print(f"DETALHAMENTO POR ATIVO DA ESTRATÉGIA DESTAQUE: {best.strategy_name} ({best.interval})")
    print("=" * 112)

    best_exp = next(exp for exp in experiments if exp[0] == best.strategy_name and exp[1] == best.interval)
    _, tf, raw_map, cfg, strat_fn, trail_mult = best_exp

    print(f"{'Ativo':<10} {'Final($)':>9} {'Ret%':>8} {'B&H%':>8} {'MaxDD%':>8} {'BH_DD%':>8} {'Win%':>6} {'PF':>6} {'Trades':>6} {'Taxas($)':>9}")
    print("-" * 112)
    for sym in SYMBOLS:
        df_sig = strat_fn(raw_map[sym])
        s_res = run_single_asset_backtest(df_sig, best.strategy_name, sym, tf, cfg, trailing_atr_mult=trail_mult)
        pf_s = f"{s_res.profit_factor:.2f}" if s_res.profit_factor < 100 else "INF"
        print(
            f"{sym:<10} {s_res.final_capital:>9.2f} {s_res.total_return_pct:>7.1f}% "
            f"{s_res.buy_hold_return_pct:>7.1f}% {s_res.max_drawdown_pct:>7.1f}% "
            f"{s_res.buy_hold_max_dd_pct:>7.1f}% {s_res.win_rate_pct:>5.1f}% "
            f"{pf_s:>6} {s_res.total_trades:>6} {s_res.total_fees_usd:>9.2f}"
        )


if __name__ == "__main__":
    main()
