"""
validate_refinements.py
Estudo completo de refinamento e validação anti-overfitting em universo expandido (16 Top Caps):
1. Progressão de refinamentos (Baseline vs OwnSMA200 vs Risk Sizing vs Controle de Volatilidade).
2. Validação In-Sample (2021-2023) vs Out-of-Sample puro (2024-2026).
3. Grade de sensibilidade de parâmetros (platô de MaxStop e janela Donchian).
4. Stress test com custos dobrados (0.60% ida e volta).
"""

from __future__ import annotations

import os
import sys

import pandas as pd

from backtest_engine import BacktestConfig
from data_loader import fetch_klines
from portfolio_backtest import PortfolioResult, run_portfolio_backtest
from strategies import strategy_donchian_breakout

UNIVERSE_16 = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
    "LINKUSDT", "LTCUSDT", "NEARUSDT", "AAVEUSDT",
    "UNIUSDT", "DOTUSDT", "ATOMUSDT", "TRXUSDT",
]


def yearly_returns(eq_df: pd.DataFrame) -> dict[int, float]:
    df = eq_df.copy()
    df["year"] = pd.to_datetime(df["time"]).dt.year
    out = {}
    prev_end = float(df["equity"].iloc[0])
    for yr, sub in df.groupby("year"):
        end_eq = float(sub["equity"].iloc[-1])
        out[int(yr)] = ((end_eq / prev_end) - 1.0) * 100.0
        prev_end = end_eq
    return out


def print_row(label: str, res: PortfolioResult) -> None:
    yrs = yearly_returns(res.equity_curve)
    ystr = " ".join(f"{y}:{yrs.get(y, 0.0):>5.1f}%" for y in sorted(yrs))
    pf = f"{res.profit_factor:.2f}" if res.profit_factor < 100 else "INF"
    print(
        f"{label:<45} | ${res.final_capital:>7.2f} | Ret={res.total_return_pct:>6.1f}% | "
        f"DD={res.max_drawdown_pct:>5.1f}% | Sh={res.sharpe_ratio:.2f} | PF={pf:>4} | "
        f"Win={res.win_rate_pct:>4.1f}% | Tr={res.total_trades:>3} | {ystr}"
    )


def build_signals(
    raw_1d: dict[str, pd.DataFrame],
    entry_p: int = 30,
    exit_p: int = 15,
    rs_btc_period: int | None = None,
    use_momentum_rank: bool = True,
    own_sma_period: int | None = None,
    max_stop_pct: float | None = None,
    atr_stop_mult: float = 3.0,
) -> dict[str, pd.DataFrame]:
    btc_1d = raw_1d["BTCUSDT"]
    out = {}
    for sym, df in raw_1d.items():
        sig = strategy_donchian_breakout(
            df,
            btc_daily_df=btc_1d,
            entry_period=entry_p,
            exit_period=exit_p,
            use_btc_filter=True,
            atr_stop_mult=atr_stop_mult,
            rs_btc_period=rs_btc_period,
            is_btc=(sym == "BTCUSDT"),
            momentum_lookback=30,
            own_sma_period=own_sma_period,
            max_stop_pct=max_stop_pct,
        )
        if not use_momentum_rank:
            sig["momentum_score"] = 0.0
        out[sym] = sig
    return out


def slice_after_warmup(
    sig_map: dict[str, pd.DataFrame],
    start_ts: str,
    end_ts: str | None = None,
) -> dict[str, pd.DataFrame]:
    t0 = pd.Timestamp(start_ts, tz="UTC")
    t1 = pd.Timestamp(end_ts, tz="UTC") if end_ts else None
    sliced = {}
    for sym, df in sig_map.items():
        mask = df["open_time"] >= t0
        if t1 is not None:
            mask &= df["open_time"] <= t1
        sliced[sym] = df[mask].reset_index(drop=True)
    return sliced


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    raw_1d = {sym: fetch_klines(sym, "1d", "2021-01-01", cache_dir=data_dir) for sym in UNIVERSE_16}
    cfg_std = BacktestConfig(initial_capital=1000.0, fee_rate=0.001, slippage_rate=0.0005, periods_per_year=365.0)
    cfg_risk3 = BacktestConfig(
        initial_capital=1000.0, fee_rate=0.001, slippage_rate=0.0005,
        risk_per_trade_pct=0.03, periods_per_year=365.0,
    )

    print("\n=== 1. PROGRESSÃO DOS REFINAMENTOS NO UNIVERSO DE 16 ATIVOS (2021-2026) ===")
    exp1 = [
        ("A. Baseline Donchian 30/15 + BTC", build_signals(raw_1d, 30, 15, None, False, None, None), cfg_std),
        ("B. + Ranking + Filtro RS(50) vs BTC", build_signals(raw_1d, 30, 15, 50, True, None, None), cfg_std),
        ("C. + Risk Sizing 3% (sem OwnSMA200)", build_signals(raw_1d, 30, 15, None, True, None, None), cfg_risk3),
        ("D. + Tendência Própria (OwnSMA200) + Risk3%", build_signals(raw_1d, 30, 15, None, True, 200, None), cfg_risk3),
        ("E. + Controle Volatilidade (MaxStop 20%)", build_signals(raw_1d, 30, 15, None, True, 200, 0.20), cfg_risk3),
        ("F. + Controle Volatilidade (MaxStop 18%)", build_signals(raw_1d, 30, 15, None, True, 200, 0.18), cfg_risk3),
    ]
    bh_shown = False
    for name, sigs, cfg in exp1:
        res = run_portfolio_backtest(sigs, name, "1d", cfg, max_positions=4)
        if not bh_shown:
            print(f"Benchmark Buy&Hold 16 ativos (2021-2026): Ret={res.equal_weight_bh_return_pct:.1f}% | MaxDD={res.bh_max_drawdown_pct:.1f}%\n")
            bh_shown = True
        print_row(name, res)

    print("\n=== 2. VALIDAÇÃO OUT-OF-SAMPLE PURO (2024-2026) ===")
    for label, sigs, cfg in [
        ("A. Baseline Donchian 30/15 + BTC", build_signals(raw_1d, 30, 15, None, False, None, None), cfg_std),
        ("D. + OwnSMA200 + Risk Sizing 3%", build_signals(raw_1d, 30, 15, None, True, 200, None), cfg_risk3),
        ("E. + OwnSMA200 + Risk 3% + MaxStop 20%", build_signals(raw_1d, 30, 15, None, True, 200, 0.20), cfg_risk3),
        ("F. + OwnSMA200 + Risk 3% + MaxStop 18%", build_signals(raw_1d, 30, 15, None, True, 200, 0.18), cfg_risk3),
    ]:
        r_out = run_portfolio_backtest(slice_after_warmup(sigs, "2024-01-01"), label, "1d", cfg, max_positions=4)
        print_row(f"[OUT 24-26] {label}", r_out)
    print(f"    -> B&H Out-of-Sample (2024-2026): Ret={r_out.equal_weight_bh_return_pct:.1f}% | MaxDD={r_out.bh_max_drawdown_pct:.1f}%")

    print("\n=== 3. PLATÔ DE SENSIBILIDADE (JANELA DONCHIAN COM MaxStop=20%, OwnSMA200, Risk 3%) ===")
    for ep, ex in [(20, 10), (25, 12), (30, 15), (35, 17), (40, 20)]:
        sigs = build_signals(raw_1d, ep, ex, None, True, 200, 0.20)
        r_all = run_portfolio_backtest(sigs, f"Don {ep}/{ex}", "1d", cfg_risk3, max_positions=4)
        r_out = run_portfolio_backtest(slice_after_warmup(sigs, "2024-01-01"), f"Don {ep}/{ex}", "1d", cfg_risk3, max_positions=4)
        print_row(f"[ALL] Don {ep}/{ex} + OwnSMA200 + MaxStop20%", r_all)
        print_row(f"[OUT] Don {ep}/{ex} + OwnSMA200 + MaxStop20%", r_out)

    print("\n=== 4. STRESS TEST DE TAXAS DOBRADAS (0.60% IDA E VOLTA) ===")
    cfg_stress = BacktestConfig(
        initial_capital=1000.0, fee_rate=0.002, slippage_rate=0.001,
        risk_per_trade_pct=0.03, periods_per_year=365.0,
    )
    sigs_final = build_signals(raw_1d, 30, 15, None, True, 200, 0.18)
    print_row("[ALL] Normal 0.30% RT (MaxStop 18%)", run_portfolio_backtest(sigs_final, "Normal", "1d", cfg_risk3, max_positions=4))
    print_row("[ALL] Stress 0.60% RT (MaxStop 18%)", run_portfolio_backtest(sigs_final, "Stress", "1d", cfg_stress, max_positions=4))
    print_row("[OUT] Stress 0.60% RT (MaxStop 18%)", run_portfolio_backtest(slice_after_warmup(sigs_final, "2024-01-01"), "Stress OUT", "1d", cfg_stress, max_positions=4))


if __name__ == "__main__":
    main()
