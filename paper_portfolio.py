"""
paper_portfolio.py
Contabilidade única (Single Source of Truth) e persistência atômica para o Paper Trader.
Garante que: equity - initial_capital == realized_pnl + unrealized_pnl em qualquer instante.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from config import PAPER_CONFIG, STATE_FILE, STRATEGY_CONFIG
from indicators import atr, donchian_high, donchian_low, ema, sma
from strategies import compute_momentum_score

_STATE_LOCK = threading.RLock()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_state(initial_capital: float | None = None) -> dict[str, Any]:
    cap = float(initial_capital if initial_capital is not None else PAPER_CONFIG["initial_capital_usd"])
    return {
        "mode": "PAPER",
        "paused": False,
        "started_at": utc_now_iso(),
        "initial_capital_usd": cap,
        "cash_usd": cap,
        "total_fees_usd": 0.0,
        "positions": {},
        "closed_trades": [],
        "equity_history": [],
        "last_daily_candle_date": None,
        "last_radar": {
            "updated_at": None,
            "candle_date": None,
            "btc_bull": False,
            "btc_price": 0.0,
            "btc_sma200": 0.0,
            "assets": [],
        },
        "last_telegram_update_id": None,
    }


def load_state(path: str = STATE_FILE) -> dict[str, Any]:
    with _STATE_LOCK:
        if not os.path.exists(path):
            st = default_state()
            save_state(st, path=path)
            return st
        with open(path, "r", encoding="utf-8") as f:
            st = json.load(f)
        # Garante chaves mínimas
        base = default_state(st.get("initial_capital_usd"))
        for k, v in base.items():
            st.setdefault(k, v)
        return st


def save_state(state: dict[str, Any], path: str = STATE_FILE) -> None:
    with _STATE_LOCK:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=os.path.dirname(os.path.abspath(path)),
            prefix="paper_state_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


def compute_position_size_usd(
    open_equity_usd: float,
    cash_usd: float,
    entry_price: float,
    stop_price: float,
    max_positions: int = PAPER_CONFIG["max_positions"],
    risk_per_trade_pct: float = PAPER_CONFIG["risk_per_trade_pct"],
    min_order_usd: float = PAPER_CONFIG["min_order_usd"],
) -> float:
    slot_cap = open_equity_usd / max_positions
    alloc = min(slot_cap, cash_usd)
    if entry_price > stop_price > 0:
        stop_dist_pct = (entry_price - stop_price) / entry_price
        if stop_dist_pct > 0.005:
            risk_usd = open_equity_usd * risk_per_trade_pct
            pos_by_risk = risk_usd / stop_dist_pct
            alloc = min(alloc, pos_by_risk)
    if alloc < min_order_usd:
        return 0.0
    return round(float(alloc), 4)


def open_paper_position(
    state: dict[str, Any],
    symbol: str,
    ref_price: float,
    alloc_usd: float,
    stop_price: float,
    don_lo_exit: float,
    atr_val: float,
    momentum_score: float,
    fee_rate: float = PAPER_CONFIG["fee_rate"],
    slippage_rate: float = PAPER_CONFIG["slippage_rate"],
    entry_time: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    if symbol in state["positions"]:
        return None
    if alloc_usd <= 0 or alloc_usd > state["cash_usd"] + 1e-6:
        return None

    alloc_usd = min(alloc_usd, state["cash_usd"])
    exec_price = ref_price * (1.0 + slippage_rate)
    spend_usd = alloc_usd / (1.0 + fee_rate)
    buy_fee = spend_usd * fee_rate
    qty = spend_usd / exec_price

    state["cash_usd"] = round(float(state["cash_usd"] - (spend_usd + buy_fee)), 8)
    state["total_fees_usd"] = round(float(state["total_fees_usd"] + buy_fee), 8)

    pos = {
        "symbol": symbol,
        "qty": qty,
        "entry_price": exec_price,
        "entry_time": entry_time or utc_now_iso(),
        "alloc_usd": spend_usd + buy_fee,
        "cost_basis_usd": spend_usd,
        "entry_fee_usd": buy_fee,
        "stop_price": float(stop_price),
        "don_lo_exit": float(don_lo_exit),
        "atr_at_entry": float(atr_val),
        "momentum_score": float(momentum_score),
        "highest_price": exec_price,
    }
    state["positions"][symbol] = pos
    return pos


def close_paper_position(
    state: dict[str, Any],
    symbol: str,
    ref_price: float,
    reason: str,
    fee_rate: float = PAPER_CONFIG["fee_rate"],
    slippage_rate: float = PAPER_CONFIG["slippage_rate"],
    exit_time: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    pos = state["positions"].get(symbol)
    if not pos:
        return None

    slip = slippage_rate * 0.5 if reason == "MANUAL_SELL" else slippage_rate
    exec_price = ref_price * (1.0 - slip)
    qty = float(pos["qty"])
    gross_proceeds = qty * exec_price
    exit_fee = gross_proceeds * fee_rate
    net_proceeds = gross_proceeds - exit_fee

    state["cash_usd"] = round(float(state["cash_usd"] + net_proceeds), 8)
    state["total_fees_usd"] = round(float(state["total_fees_usd"] + exit_fee), 8)

    total_invested = float(pos["cost_basis_usd"]) + float(pos["entry_fee_usd"])
    total_fees = float(pos["entry_fee_usd"]) + exit_fee
    net_pnl = net_proceeds - total_invested
    ret_pct = (net_pnl / total_invested) * 100.0 if total_invested > 0 else 0.0

    trade_record = {
        "symbol": symbol,
        "entry_time": pos["entry_time"],
        "exit_time": exit_time or utc_now_iso(),
        "entry_price": float(pos["entry_price"]),
        "exit_price": exec_price,
        "qty": qty,
        "invested_usd": total_invested,
        "proceeds_usd": net_proceeds,
        "fees_usd": total_fees,
        "net_pnl_usd": net_pnl,
        "return_pct": ret_pct,
        "exit_reason": reason,
    }
    state["closed_trades"].append(trade_record)
    del state["positions"][symbol]
    return trade_record


def compute_market_radar(
    raw_1d: dict[str, pd.DataFrame],
    live_prices: dict[str, float],
) -> dict[str, Any]:
    """
    Calcula o estado atual (último candle diário fechado + preço ao vivo) de todo o universo.
    Usado tanto para gerar alertas quanto para alimentar a aba Radar do Dashboard.
    """
    ep = STRATEGY_CONFIG["entry_period"]
    ex = STRATEGY_CONFIG["exit_period"]
    own_sma_p = STRATEGY_CONFIG["own_sma_period"]
    btc_sma_p = STRATEGY_CONFIG["btc_sma_period"]
    atr_p = STRATEGY_CONFIG["atr_period"]
    atr_mult = STRATEGY_CONFIG["atr_stop_mult"]
    max_stop = STRATEGY_CONFIG["max_stop_pct"]

    btc_df = raw_1d["BTCUSDT"]
    btc_sma200 = float(sma(btc_df["close"], btc_sma_p).iloc[-1])
    btc_closed = float(btc_df["close"].iloc[-1])
    btc_live = float(live_prices.get("BTCUSDT", btc_closed))
    btc_bull = btc_closed > btc_sma200
    last_candle_date = str(pd.Timestamp(btc_df["open_time"].iloc[-1]).date())

    assets = []
    for sym, df in raw_1d.items():
        if len(df) < own_sma_p + 5:
            continue
        atr_s = atr(df, atr_p)
        ema50_s = ema(df["close"], 50)
        sma200_s = sma(df["close"], own_sma_p)
        don_hi_s = donchian_high(df, ep)
        don_lo_s = donchian_low(df, ex)
        mom_s = compute_momentum_score(df, atr_s, STRATEGY_CONFIG["momentum_lookback"])

        prev_close = float(df["close"].iloc[-1])
        live_px = float(live_prices.get(sym, prev_close))
        cur_sma200 = float(sma200_s.iloc[-1])
        cur_ema50 = float(ema50_s.iloc[-1])
        # Donchian high/low incluindo o último candle fechado para monitorar rompimento de hoje
        cur_don_hi = float(df["high"].tail(ep).max())
        cur_don_lo = float(df["low"].tail(ex).min())
        cur_atr = float(atr_s.iloc[-1])
        stop_dist_pct = (atr_mult * cur_atr / prev_close) if prev_close > 0 else 1.0

        dist_to_breakout_pct = ((cur_don_hi / live_px) - 1.0) * 100.0 if live_px > 0 else 0.0
        dist_sma200_pct = ((live_px / cur_sma200) - 1.0) * 100.0 if cur_sma200 > 0 else 0.0
        change_24h_pct = ((live_px / prev_close) - 1.0) * 100.0 if prev_close > 0 else 0.0

        above_sma200 = live_px > cur_sma200
        above_ema50 = live_px > cur_ema50
        vol_ok = stop_dist_pct <= max_stop

        if not btc_bull:
            status_tag = "MACRO BEAR"
        elif not above_sma200:
            status_tag = "ABAIXO SMA200"
        elif not vol_ok:
            status_tag = "VOL ALTA"
        elif live_px > cur_don_hi and above_ema50:
            status_tag = "ROMPENDO AGORA"
        elif dist_to_breakout_pct <= 5.0 and above_ema50:
            status_tag = "PERTO DE ROMPER"
        else:
            status_tag = "EM TENDÊNCIA"

        assets.append({
            "symbol": sym,
            "ticker": sym.replace("USDT", ""),
            "price": live_px,
            "prev_close": prev_close,
            "change_today_pct": round(change_24h_pct, 2),
            "don_hi_30": round(cur_don_hi, 6),
            "don_lo_15": round(cur_don_lo, 6),
            "sma200": round(cur_sma200, 6),
            "dist_sma200_pct": round(dist_sma200_pct, 2),
            "dist_breakout_pct": round(dist_to_breakout_pct, 2),
            "stop_dist_pct": round(stop_dist_pct * 100.0, 2),
            "vol_ok": bool(vol_ok),
            "above_sma200": bool(above_sma200),
            "momentum_score": round(float(mom_s.iloc[-1]), 2),
            "status": status_tag,
        })

    # Ordena: quem está rompendo ou mais perto de romper com filtros OK aparece no topo
    assets.sort(key=lambda a: (not a["above_sma200"], not a["vol_ok"], a["dist_breakout_pct"]))

    return {
        "updated_at": utc_now_iso(),
        "candle_date": last_candle_date,
        "btc_bull": bool(btc_bull),
        "btc_price": btc_live,
        "btc_closed": btc_closed,
        "btc_sma200": round(btc_sma200, 2),
        "btc_dist_sma200_pct": round(((btc_live / btc_sma200) - 1.0) * 100.0, 2) if btc_sma200 > 0 else 0.0,
        "assets": assets,
    }


def build_dashboard_snapshot(
    state: dict[str, Any],
    live_prices: Optional[dict[str, float]] = None,
    usdt_brl: float = 5.75,
) -> dict[str, Any]:
    """
    Calcula o retrato financeiro único (single source of truth) consumido pelo Dashboard e pelo Telegram.
    """
    live_prices = live_prices or {}
    initial_cap = float(state["initial_capital_usd"])
    cash_usd = float(state["cash_usd"])

    positions_out = []
    positions_value_usd = 0.0
    unrealized_pnl_usd = 0.0

    for sym, pos in state.get("positions", {}).items():
        entry_px = float(pos["entry_price"])
        qty = float(pos["qty"])
        invested = float(pos["cost_basis_usd"]) + float(pos["entry_fee_usd"])
        curr_px = float(live_prices.get(sym, entry_px))
        curr_val = qty * curr_px
        pnl_u = curr_val - invested
        pnl_pct = (pnl_u / invested) * 100.0 if invested > 0 else 0.0
        stop_px = float(pos["stop_price"])
        don_exit = float(pos["don_lo_exit"])
        dist_stop_pct = ((curr_px / stop_px) - 1.0) * 100.0 if stop_px > 0 else 0.0

        positions_value_usd += curr_val
        unrealized_pnl_usd += pnl_u

        positions_out.append({
            "symbol": sym,
            "ticker": sym.replace("USDT", ""),
            "qty": qty,
            "entry_price": entry_px,
            "current_price": curr_px,
            "invested_usd": round(invested, 2),
            "current_value_usd": round(curr_val, 2),
            "pnl_usd": round(pnl_u, 2),
            "pnl_pct": round(pnl_pct, 2),
            "stop_price": stop_px,
            "don_lo_exit": don_exit,
            "dist_stop_pct": round(dist_stop_pct, 2),
            "entry_time": pos["entry_time"],
            "momentum_score": float(pos.get("momentum_score", 0.0)),
        })

    total_equity_usd = cash_usd + positions_value_usd
    closed_trades = state.get("closed_trades", [])
    realized_pnl_usd = sum(float(tr["net_pnl_usd"]) for tr in closed_trades)
    total_pnl_usd = total_equity_usd - initial_cap
    total_return_pct = (total_pnl_usd / initial_cap) * 100.0 if initial_cap > 0 else 0.0
    exposure_pct = (positions_value_usd / total_equity_usd) * 100.0 if total_equity_usd > 0 else 0.0

    wins = [tr for tr in closed_trades if float(tr["net_pnl_usd"]) > 0]
    losses = [tr for tr in closed_trades if float(tr["net_pnl_usd"]) <= 0]
    win_rate_pct = (len(wins) / len(closed_trades) * 100.0) if closed_trades else 0.0
    gross_w = sum(float(tr["net_pnl_usd"]) for tr in wins)
    gross_l = abs(sum(float(tr["net_pnl_usd"]) for tr in losses))
    profit_factor = (gross_w / gross_l) if gross_l > 0 else (99.0 if gross_w > 0 else 0.0)

    return {
        "ok": True,
        "mode": state.get("mode", "PAPER"),
        "paused": bool(state.get("paused", False)),
        "strategy_name": STRATEGY_CONFIG["name"],
        "usdt_brl": round(float(usdt_brl), 4),
        "summary": {
            "initial_capital_usd": round(initial_cap, 2),
            "total_equity_usd": round(total_equity_usd, 2),
            "total_equity_brl": round(total_equity_usd * usdt_brl, 2),
            "cash_usd": round(cash_usd, 2),
            "positions_value_usd": round(positions_value_usd, 2),
            "exposure_pct": round(exposure_pct, 1),
            "realized_pnl_usd": round(realized_pnl_usd, 2),
            "unrealized_pnl_usd": round(unrealized_pnl_usd, 2),
            "total_pnl_usd": round(total_pnl_usd, 2),
            "total_pnl_brl": round(total_pnl_usd * usdt_brl, 2),
            "total_return_pct": round(total_return_pct, 2),
            "total_fees_usd": round(float(state.get("total_fees_usd", 0.0)), 2),
            "open_count": len(positions_out),
            "max_positions": PAPER_CONFIG["max_positions"],
            "total_trades": len(closed_trades),
            "win_rate_pct": round(win_rate_pct, 1),
            "profit_factor": round(profit_factor, 2),
            "last_daily_candle_date": state.get("last_daily_candle_date"),
        },
        "positions": positions_out,
        "closed_trades": list(reversed(closed_trades[-50:])),
        "equity_history": state.get("equity_history", [])[-180:],
        "radar": state.get("last_radar", {}),
    }
