"""
portfolio_backtest.py
Simulador multi-ativo com banca compartilhada em USDT.
Permite até `max_positions` trades abertos simultaneamente (ex: 4 slots de 25% cada),
executando sinais em ordem cronológica estrita sem look-ahead bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from backtest_engine import BacktestConfig, Trade, _max_drawdown_pct


@dataclass
class OpenPos:
    symbol: str
    qty: float
    entry_price: float
    entry_time: pd.Timestamp
    entry_idx: int
    entry_fee: float
    active_stop: Optional[float]
    active_tp: Optional[float]
    highest_high: float


@dataclass
class PortfolioResult:
    strategy_name: str
    interval: str
    symbols: list[str]
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    equal_weight_bh_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    bh_max_drawdown_pct: float
    sharpe_ratio: float
    calmar_ratio: float
    total_trades: int
    trades_per_month: float
    win_rate_pct: float
    profit_factor: float
    total_fees_usd: float
    trades: list[Trade]
    equity_curve: pd.DataFrame


def run_portfolio_backtest(
    asset_dfs: Dict[str, pd.DataFrame],
    strategy_name: str,
    interval: str,
    config: BacktestConfig,
    max_positions: int = 4,
    trailing_atr_mult: Optional[float] = None,
) -> PortfolioResult:
    symbols = list(asset_dfs.keys())
    # Alinha timeline comum (interseção das datas disponíveis de todos os ativos, ou união ordenada)
    # Usar a interseção garante que o Buy & Hold comparativo comece e termine no mesmo dia para todos.
    common_times = None
    indexed: Dict[str, pd.DataFrame] = {}
    for sym, df in asset_dfs.items():
        d = df.copy().sort_values("open_time").drop_duplicates("open_time").set_index("open_time")
        indexed[sym] = d
        common_times = d.index if common_times is None else common_times.intersection(d.index)

    assert common_times is not None and len(common_times) > 10
    timeline = sorted(common_times)
    n = len(timeline)

    for sym in symbols:
        indexed[sym] = indexed[sym].loc[timeline]

    cash = config.initial_capital
    positions: Dict[str, OpenPos] = {}
    pending_buy: Dict[str, tuple[bool, Optional[float], Optional[float], float]] = {
        s: (False, None, None, 0.0) for s in symbols
    }
    pending_sell: Dict[str, bool] = {s: False for s in symbols}

    trades: list[Trade] = []
    equity_rows = []
    total_fees = 0.0

    # Buy & Hold Equal-Weight inicial (1/M da banca em cada ativo)
    bh_shares = {}
    bh_alloc = config.initial_capital / len(symbols)
    for sym in symbols:
        p0 = float(indexed[sym]["close"].iloc[0])
        bh_shares[sym] = (bh_alloc * (1.0 - config.fee_rate)) / p0

    for i, t in enumerate(timeline):
        t_ts = pd.Timestamp(t)

        # 1) Processa saídas em cada ativo já posicionado antes de abrir novas compras
        for sym in list(positions.keys()):
            row = indexed[sym].iloc[i]
            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
            pos = positions[sym]

            exit_reason: Optional[str] = None
            raw_exit = 0.0
            if pending_sell[sym]:
                exit_reason = "SIGNAL_EXIT"
                raw_exit = o * (1.0 - config.slippage_rate)
            elif pos.active_stop is not None and l <= pos.active_stop:
                exit_reason = "TRAILING_STOP" if trailing_atr_mult else "STOP_LOSS"
                raw_exit = min(o, pos.active_stop) * (1.0 - config.slippage_rate)
            elif pos.active_tp is not None and h >= pos.active_tp:
                exit_reason = "TAKE_PROFIT"
                raw_exit = max(o, pos.active_tp) * (1.0 - config.slippage_rate * 0.5)

            if exit_reason is not None:
                gross_proceeds = pos.qty * raw_exit
                exit_fee = gross_proceeds * config.fee_rate
                net_proceeds = gross_proceeds - exit_fee
                cash += net_proceeds
                total_fees += exit_fee

                cost_basis = pos.qty * pos.entry_price
                gross_pnl = gross_proceeds - cost_basis
                trade_fees = pos.entry_fee + exit_fee
                net_pnl = gross_pnl - trade_fees
                ret_pct = (net_pnl / (cost_basis + pos.entry_fee)) * 100.0

                trades.append(
                    Trade(
                        symbol=sym,
                        entry_time=pos.entry_time,
                        exit_time=t_ts,
                        entry_price=pos.entry_price,
                        exit_price=raw_exit,
                        qty=pos.qty,
                        gross_pnl_usd=gross_pnl,
                        fees_usd=trade_fees,
                        net_pnl_usd=net_pnl,
                        return_pct=ret_pct,
                        exit_reason=exit_reason,
                        bars_held=i - pos.entry_idx,
                    )
                )
                del positions[sym]

        # 2) Calcula patrimônio na abertura para dimensionar novos slots igualmente
        open_equity = cash
        for sym, pos in positions.items():
            open_equity += pos.qty * float(indexed[sym]["open"].iloc[i])

        slot_target_usd = open_equity / max_positions

        # 3) Processa entradas pendentes respeitando max_positions, ordenadas por momentum_score
        candidates = []
        for sym in symbols:
            if sym in positions:
                continue
            do_buy, p_stop, p_tp, p_score = pending_buy[sym]
            if do_buy:
                candidates.append((p_score, sym, p_stop, p_tp))

        candidates.sort(key=lambda item: item[0], reverse=True)

        for p_score, sym, p_stop, p_tp in candidates:
            if len(positions) >= max_positions:
                break

            o = float(indexed[sym]["open"].iloc[i])
            h = float(indexed[sym]["high"].iloc[i])
            exec_price = o * (1.0 + config.slippage_rate)

            alloc_usd = min(slot_target_usd, cash)
            if (
                config.risk_per_trade_pct is not None
                and p_stop is not None
                and exec_price > p_stop
            ):
                stop_dist_pct = (exec_price - p_stop) / exec_price
                if stop_dist_pct > 0.005:
                    risk_usd = open_equity * config.risk_per_trade_pct
                    pos_by_risk = risk_usd / stop_dist_pct
                    alloc_usd = min(alloc_usd, pos_by_risk)

            if alloc_usd < 15.0:
                continue

            spend_usd = alloc_usd / (1.0 + config.fee_rate)
            buy_fee = spend_usd * config.fee_rate
            qty = spend_usd / exec_price
            cash -= (spend_usd + buy_fee)
            total_fees += buy_fee

            positions[sym] = OpenPos(
                symbol=sym,
                qty=qty,
                entry_price=exec_price,
                entry_time=t_ts,
                entry_idx=i,
                entry_fee=buy_fee,
                active_stop=p_stop,
                active_tp=p_tp,
                highest_high=h,
            )

        # 4) Ao final da barra: atualiza trailing stop, mark-to-market e agenda sinais para i+1
        close_equity = cash
        bh_equity = 0.0
        for sym in symbols:
            row = indexed[sym].iloc[i]
            c = float(row["close"])
            h = float(row["high"])
            bh_equity += bh_shares[sym] * c

            if sym in positions:
                pos = positions[sym]
                close_equity += pos.qty * c
                pos.highest_high = max(pos.highest_high, h)
                atr_val = float(row["atr"]) if "atr" in row and not pd.isna(row["atr"]) else np.nan
                if trailing_atr_mult is not None and not np.isnan(atr_val):
                    cand = pos.highest_high - trailing_atr_mult * atr_val
                    if pos.active_stop is None or cand > pos.active_stop:
                        pos.active_stop = cand

            sig_b = bool(row["signal_buy"]) if not pd.isna(row["signal_buy"]) else False
            sig_s = bool(row["signal_sell"]) if not pd.isna(row["signal_sell"]) else False
            s_init = float(row["stop_price_init"]) if "stop_price_init" in row and not pd.isna(row["stop_price_init"]) else None
            t_init = float(row["tp_price_init"]) if "tp_price_init" in row and not pd.isna(row["tp_price_init"]) else None
            m_score = float(row["momentum_score"]) if "momentum_score" in row and not pd.isna(row["momentum_score"]) else 0.0
            pending_buy[sym] = (sig_b, s_init, t_init, m_score)
            pending_sell[sym] = sig_s

        equity_rows.append((t_ts, close_equity, bh_equity))

    eq_df = pd.DataFrame(equity_rows, columns=["time", "equity", "bh_equity"])
    eq_s = eq_df["equity"]
    bh_s = eq_df["bh_equity"]

    final_cap = float(eq_s.iloc[-1])
    total_ret = ((final_cap / config.initial_capital) - 1.0) * 100.0
    bh_ret = ((float(bh_s.iloc[-1]) / config.initial_capital) - 1.0) * 100.0

    total_days = max((pd.Timestamp(timeline[-1]) - pd.Timestamp(timeline[0])).total_seconds() / 86400.0, 1.0)
    years = total_days / 365.25
    cagr = (((final_cap / config.initial_capital) ** (1.0 / years)) - 1.0) * 100.0 if final_cap > 0 else -100.0

    max_dd = _max_drawdown_pct(eq_s)
    bh_dd = _max_drawdown_pct(bh_s)

    rets = eq_s.pct_change().dropna()
    sharpe = float((rets.mean() / rets.std()) * np.sqrt(config.periods_per_year)) if len(rets) > 2 and rets.std() > 0 else 0.0
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0

    total_trades = len(trades)
    months = max(total_days / 30.4375, 1.0)
    wins = [tr for tr in trades if tr.net_pnl_usd > 0]
    losses = [tr for tr in trades if tr.net_pnl_usd <= 0]
    win_rate = (len(wins) / total_trades) * 100.0 if total_trades > 0 else 0.0
    gross_w = sum(tr.net_pnl_usd for tr in wins)
    gross_l = abs(sum(tr.net_pnl_usd for tr in losses))
    pf = (gross_w / gross_l) if gross_l > 0 else (float("inf") if gross_w > 0 else 0.0)

    return PortfolioResult(
        strategy_name=strategy_name,
        interval=interval,
        symbols=symbols,
        start_date=str(pd.Timestamp(timeline[0]).date()),
        end_date=str(pd.Timestamp(timeline[-1]).date()),
        initial_capital=config.initial_capital,
        final_capital=final_cap,
        total_return_pct=total_ret,
        equal_weight_bh_return_pct=bh_ret,
        cagr_pct=cagr,
        max_drawdown_pct=max_dd,
        bh_max_drawdown_pct=bh_dd,
        sharpe_ratio=sharpe,
        calmar_ratio=calmar,
        total_trades=total_trades,
        trades_per_month=total_trades / months,
        win_rate_pct=win_rate,
        profit_factor=pf,
        total_fees_usd=total_fees,
        trades=trades,
        equity_curve=eq_df,
    )
