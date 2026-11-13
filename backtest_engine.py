"""
backtest_engine.py
Motor de backtesting causal (sem look-ahead bias) para Spot CEX:
- Sinais gerados no fechamento do candle i executam no open do candle i+1.
- Stops e Take Profits são verificados intrabar (high/low) usando níveis conhecidos
  antes da barra abrir, sempre testando Stop Loss antes de Take Profit (conservador).
- Taxas e slippage cobrados em ambas as pontas.
- Calcula equity curve mark-to-market barra a barra para Max Drawdown e Sharpe reais.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd


@dataclass
class Trade:
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: float
    gross_pnl_usd: float
    fees_usd: float
    net_pnl_usd: float
    return_pct: float
    exit_reason: str
    bars_held: int


@dataclass
class BacktestConfig:
    initial_capital: float = 1000.0
    fee_rate: float = 0.001       # 0.10% Binance spot padrão sem desconto BNB
    slippage_rate: float = 0.0005  # 0.05% conservador por execução a mercado
    risk_per_trade_pct: Optional[float] = None  # Se definido (ex: 0.02), dimensiona por risco do stop
    max_position_pct: float = 1.0  # Máximo da banca por ativo (1.0 = 100% em ativo único)
    periods_per_year: float = 365.0  # 365 para 1d, 365*6 para 4h


@dataclass
class BacktestResult:
    strategy_name: str
    symbol: str
    interval: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    buy_hold_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    buy_hold_max_dd_pct: float
    sharpe_ratio: float
    calmar_ratio: float
    total_trades: int
    trades_per_month: float
    win_rate_pct: float
    profit_factor: float
    avg_win_pct: float
    avg_loss_pct: float
    exposure_pct: float
    total_fees_usd: float
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)


def _max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    cum_max = equity.cummax()
    dd = (equity / cum_max) - 1.0
    return float(dd.min() * 100.0)


def run_single_asset_backtest(
    df: pd.DataFrame,
    strategy_name: str,
    symbol: str,
    interval: str,
    config: BacktestConfig,
    trailing_atr_mult: Optional[float] = None,
) -> BacktestResult:
    """
    Espera que `df` contenha colunas:
      open_time, open, high, low, close
      signal_buy (bool), signal_sell (bool)
      stop_price_init (float/NaN), tp_price_init (float/NaN)
      atr (opcional, se trailing_atr_mult != None)
    """
    data = df.reset_index(drop=True)
    n = len(data)
    if n < 5:
        raise ValueError(f"Poucos candles para {symbol}: {n}")

    cash = config.initial_capital
    qty = 0.0
    in_pos = False
    entry_price = 0.0
    entry_time: Optional[pd.Timestamp] = None
    entry_idx = 0
    entry_fee = 0.0
    active_stop: Optional[float] = None
    active_tp: Optional[float] = None
    highest_high = 0.0

    pending_buy = False
    pending_sell = False
    pending_stop: Optional[float] = None
    pending_tp: Optional[float] = None

    trades: list[Trade] = []
    equity_records = []
    bars_in_pos = 0
    total_fees = 0.0

    opens = data["open"].to_numpy()
    highs = data["high"].to_numpy()
    lows = data["low"].to_numpy()
    closes = data["close"].to_numpy()
    times = data["open_time"].to_numpy()
    sig_buys = data["signal_buy"].fillna(False).to_numpy(dtype=bool)
    sig_sells = data["signal_sell"].fillna(False).to_numpy(dtype=bool)
    stop_inits = (
        data["stop_price_init"].to_numpy()
        if "stop_price_init" in data.columns
        else np.full(n, np.nan)
    )
    tp_inits = (
        data["tp_price_init"].to_numpy()
        if "tp_price_init" in data.columns
        else np.full(n, np.nan)
    )
    atrs = (
        data["atr"].to_numpy()
        if "atr" in data.columns
        else np.full(n, np.nan)
    )

    for i in range(n):
        o = float(opens[i])
        h = float(highs[i])
        l = float(lows[i])
        c = float(closes[i])
        t = pd.Timestamp(times[i])

        # 1) Processar saídas na barra i usando níveis definidos ANTES da abertura de i
        if in_pos:
            exit_reason: Optional[str] = None
            raw_exit_price = 0.0

            if pending_sell:
                exit_reason = "SIGNAL_EXIT"
                raw_exit_price = o * (1.0 - config.slippage_rate)
            elif active_stop is not None and l <= active_stop:
                exit_reason = "TRAILING_STOP" if trailing_atr_mult else "STOP_LOSS"
                base = min(o, active_stop)
                raw_exit_price = base * (1.0 - config.slippage_rate)
            elif active_tp is not None and h >= active_tp:
                exit_reason = "TAKE_PROFIT"
                base = max(o, active_tp)
                raw_exit_price = base * (1.0 - config.slippage_rate * 0.5)

            if exit_reason is not None:
                gross_proceeds = qty * raw_exit_price
                exit_fee = gross_proceeds * config.fee_rate
                net_proceeds = gross_proceeds - exit_fee
                cash += net_proceeds
                total_fees += exit_fee

                cost_basis = qty * entry_price
                gross_pnl = gross_proceeds - cost_basis
                trade_fees = entry_fee + exit_fee
                net_pnl = gross_pnl - trade_fees
                ret_pct = (net_pnl / (cost_basis + entry_fee)) * 100.0

                trades.append(
                    Trade(
                        symbol=symbol,
                        entry_time=entry_time,
                        exit_time=t,
                        entry_price=entry_price,
                        exit_price=raw_exit_price,
                        qty=qty,
                        gross_pnl_usd=gross_pnl,
                        fees_usd=trade_fees,
                        net_pnl_usd=net_pnl,
                        return_pct=ret_pct,
                        exit_reason=exit_reason,
                        bars_held=i - entry_idx,
                    )
                )
                in_pos = False
                qty = 0.0
                active_stop = None
                active_tp = None
                entry_fee = 0.0

        # 2) Processar entrada pendente na abertura da barra i (gerada no fechamento de i-1)
        if not in_pos and pending_buy:
            exec_price = o * (1.0 + config.slippage_rate)
            max_alloc_usd = cash * config.max_position_pct

            alloc_usd = max_alloc_usd
            if (
                config.risk_per_trade_pct is not None
                and pending_stop is not None
                and not np.isnan(pending_stop)
                and exec_price > pending_stop
            ):
                stop_dist_pct = (exec_price - pending_stop) / exec_price
                if stop_dist_pct > 0.005:
                    risk_usd = cash * config.risk_per_trade_pct
                    pos_by_risk = risk_usd / stop_dist_pct
                    alloc_usd = min(max_alloc_usd, pos_by_risk)

            # Reserva margem para a taxa de compra
            spend_usd = alloc_usd / (1.0 + config.fee_rate)
            buy_fee = spend_usd * config.fee_rate
            if spend_usd > 10.0 and (spend_usd + buy_fee) <= cash + 1e-9:
                qty = spend_usd / exec_price
                cash -= (spend_usd + buy_fee)
                total_fees += buy_fee
                in_pos = True
                entry_price = exec_price
                entry_time = t
                entry_idx = i
                entry_fee = buy_fee
                highest_high = h
                active_stop = (
                    float(pending_stop)
                    if pending_stop is not None and not np.isnan(pending_stop)
                    else None
                )
                active_tp = (
                    float(pending_tp)
                    if pending_tp is not None and not np.isnan(pending_tp)
                    else None
                )

        if in_pos:
            bars_in_pos += 1
            # Atualiza trailing stop apenas no fechamento da barra i (para valer na barra i+1)
            highest_high = max(highest_high, h)
            if trailing_atr_mult is not None and not np.isnan(atrs[i]):
                candidate_stop = highest_high - trailing_atr_mult * float(atrs[i])
                if active_stop is None or candidate_stop > active_stop:
                    active_stop = candidate_stop

        # Mark-to-market no fechamento de i
        equity_mtm = cash + (qty * c if in_pos else 0.0)
        equity_records.append((t, equity_mtm, c))

        # Sinais do fechamento de i serão executados no open de i+1
        pending_buy = bool(sig_buys[i])
        pending_sell = bool(sig_sells[i])
        pending_stop = float(stop_inits[i]) if not np.isnan(stop_inits[i]) else None
        pending_tp = float(tp_inits[i]) if not np.isnan(tp_inits[i]) else None

    eq_df = pd.DataFrame(equity_records, columns=["time", "equity", "close"])
    eq_series = eq_df["equity"]
    bh_series = config.initial_capital * (eq_df["close"] / eq_df["close"].iloc[0])

    final_capital = float(eq_series.iloc[-1])
    total_ret = ((final_capital / config.initial_capital) - 1.0) * 100.0
    bh_ret = ((float(bh_series.iloc[-1]) / config.initial_capital) - 1.0) * 100.0

    # CAGR
    total_days = max(
        (pd.Timestamp(times[-1]) - pd.Timestamp(times[0])).total_seconds() / 86400.0,
        1.0,
    )
    years = total_days / 365.25
    cagr = (((final_capital / config.initial_capital) ** (1.0 / years)) - 1.0) * 100.0 if final_capital > 0 else -100.0

    max_dd = _max_drawdown_pct(eq_series)
    bh_max_dd = _max_drawdown_pct(bh_series)

    # Sharpe anualizado sobre retornos por barra
    rets = eq_series.pct_change().dropna()
    if len(rets) > 2 and rets.std() > 0:
        sharpe = float((rets.mean() / rets.std()) * np.sqrt(config.periods_per_year))
    else:
        sharpe = 0.0

    calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0

    # Estatísticas de Trades fechados
    total_trades = len(trades)
    months = max(total_days / 30.4375, 1.0)
    trades_per_month = total_trades / months

    if total_trades > 0:
        wins = [tr for tr in trades if tr.net_pnl_usd > 0]
        losses = [tr for tr in trades if tr.net_pnl_usd <= 0]
        win_rate = (len(wins) / total_trades) * 100.0
        gross_win = sum(tr.net_pnl_usd for tr in wins)
        gross_loss = abs(sum(tr.net_pnl_usd for tr in losses))
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
        avg_win = float(np.mean([tr.return_pct for tr in wins])) if wins else 0.0
        avg_loss = float(np.mean([tr.return_pct for tr in losses])) if losses else 0.0
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_win = 0.0
        avg_loss = 0.0

    exposure = (bars_in_pos / n) * 100.0

    return BacktestResult(
        strategy_name=strategy_name,
        symbol=symbol,
        interval=interval,
        start_date=str(pd.Timestamp(times[0]).date()),
        end_date=str(pd.Timestamp(times[-1]).date()),
        initial_capital=config.initial_capital,
        final_capital=final_capital,
        total_return_pct=total_ret,
        buy_hold_return_pct=bh_ret,
        cagr_pct=cagr,
        max_drawdown_pct=max_dd,
        buy_hold_max_dd_pct=bh_max_dd,
        sharpe_ratio=sharpe,
        calmar_ratio=calmar,
        total_trades=total_trades,
        trades_per_month=trades_per_month,
        win_rate_pct=win_rate,
        profit_factor=profit_factor,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        exposure_pct=exposure,
        total_fees_usd=total_fees,
        trades=trades,
        equity_curve=eq_df,
    )
