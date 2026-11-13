"""
paper_trader.py
Motor operacional de Paper Trading para Swing Trade diário (1d).
- Avalia fechamentos diários (00:00 UTC / 21:00 BRT) usando a estratégia validada
  Donchian 30/15 + OwnSMA200 + Filtro Macro BTC + Risk Sizing 3% + MaxStop 18%.
- Monitora Stop Loss intradiário em tempo real a cada 60s.
- Responde comandos no Telegram (/status, /radar, /trades, /pausar, /retomar, /ajuda).
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from config import DATA_DIR, PAPER_CONFIG, STATE_FILE, STRATEGY_CONFIG, UNIVERSE
from market_api import fetch_live_prices, fetch_universe_daily_candles
from paper_portfolio import (
    build_dashboard_snapshot,
    close_paper_position,
    compute_market_radar,
    compute_position_size_usd,
    load_state,
    open_paper_position,
    save_state,
    utc_now_iso,
)
from strategies import strategy_donchian_breakout
from telegram_client import TelegramClient

logger = logging.getLogger("swingbot.paper")


def format_status_message(snapshot: dict[str, Any]) -> str:
    s = snapshot["summary"]
    radar = snapshot.get("radar") or {}
    btc_bull = radar.get("btc_bull", False)
    btc_px = float(radar.get("btc_price", 0.0))
    btc_sma = float(radar.get("btc_sma200", 0.0))
    regime_txt = "🟢 ALTA (BTC &gt; SMA200)" if btc_bull else "🔴 DEFENSIVO (BTC &lt; SMA200)"
    paused_txt = " ⏸️ <b>[PAUSADO]</b>" if snapshot.get("paused") else ""

    pnl_sign = "+" if s["total_pnl_usd"] >= 0 else ""
    ret_sign = "+" if s["total_return_pct"] >= 0 else ""

    lines = [
        f"📊 <b>SwingBot Paper Status</b>{paused_txt}",
        "",
        f"💰 <b>Patrimônio</b>: <b>US$ {s['total_equity_usd']:.2f}</b> (~R$ {s['total_equity_brl']:.2f})",
        f"📈 <b>Resultado</b>: <b>{pnl_sign}US$ {s['total_pnl_usd']:.2f} ({ret_sign}{s['total_return_pct']:.2f}%)</b>",
        f"💵 <b>Caixa Livre</b>: US$ {s['cash_usd']:.2f} | <b>Alocado</b>: {s['exposure_pct']:.1f}%",
        f"🎯 <b>Trades</b>: {s['total_trades']} fechados (Win: {s['win_rate_pct']:.1f}% | PF: {s['profit_factor']:.2f})",
        f"🧭 <b>Regime BTC</b>: {regime_txt}",
        f"   BTC: <code>US$ {btc_px:,.2f}</code> vs SMA200: <code>US$ {btc_sma:,.2f}</code>",
    ]

    positions = snapshot.get("positions", [])
    if positions:
        lines.append("")
        lines.append(f"📌 <b>Posições Abertas ({len(positions)}/{s['max_positions']})</b>:")
        for p in positions:
            ps = "+" if p["pnl_usd"] >= 0 else ""
            lines.append(
                f"• <b>{p['ticker']}</b>: US$ {p['current_value_usd']:.2f} "
                f"(<b>{ps}{p['pnl_pct']:.2f}%</b>) | Stop: <code>${p['stop_price']:.4f}</code> ({p['dist_stop_pct']:.1f}%)"
            )
    else:
        lines.append("")
        lines.append("📌 <b>Posições Abertas</b>: Nenhuma (100% em USDT simulado).")

    return "\n".join(lines)


def format_radar_message(snapshot: dict[str, Any]) -> str:
    radar = snapshot.get("radar") or {}
    assets = radar.get("assets", [])
    btc_bull = radar.get("btc_bull", False)
    header_regime = "🟢 BTC em Alta (&gt; SMA200)" if btc_bull else "🔴 BTC Defensivo (&lt; SMA200 — compras bloqueadas)"

    lines = [
        "🔭 <b>Radar de Mercado (Donchian 30/15 | 1D)</b>",
        f"Regime: {header_regime}",
        f"Último candle fechado: <code>{radar.get('candle_date', '-')}</code>",
        "",
    ]
    for a in assets[:8]:
        if a["above_sma200"] and a["vol_ok"]:
            icon = "🔥" if a["dist_breakout_pct"] <= 3.0 else "🟢"
            info = f"falta <b>{a['dist_breakout_pct']:.1f}%</b> p/ romper"
        elif not a["above_sma200"]:
            icon = "⚪"
            info = f"{a['dist_sma200_pct']:.1f}% vs SMA200"
        else:
            icon = "🟡"
            info = f"vol alta (stop {a['stop_dist_pct']:.1f}%)"
        lines.append(f"{icon} <b>{a['ticker']}</b>: ${a['price']:.4g} — {info}")

    return "\n".join(lines)


def format_trades_message(snapshot: dict[str, Any]) -> str:
    closed = snapshot.get("closed_trades", [])
    if not closed:
        return "📒 <b>Histórico de Trades</b>\n\nNenhum trade encerrado ainda na simulação."
    lines = ["📒 <b>Últimos Trades Encerrados (Paper)</b>", ""]
    for tr in closed[:10]:
        sign = "+" if tr["net_pnl_usd"] >= 0 else ""
        emoji = "✅" if tr["net_pnl_usd"] > 0 else "❌"
        sym = tr["symbol"].replace("USDT", "")
        lines.append(
            f"{emoji} <b>{sym}</b>: {sign}US$ {tr['net_pnl_usd']:.2f} ({sign}{tr['return_pct']:.2f}%) "
            f"[{tr['exit_reason']}]"
        )
    return "\n".join(lines)


def _compute_signals_universe(raw_1d: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    btc_df = raw_1d["BTCUSDT"]
    out = {}
    for sym, df in raw_1d.items():
        out[sym] = strategy_donchian_breakout(
            df,
            btc_daily_df=btc_df,
            entry_period=STRATEGY_CONFIG["entry_period"],
            exit_period=STRATEGY_CONFIG["exit_period"],
            use_btc_filter=True,
            atr_stop_mult=STRATEGY_CONFIG["atr_stop_mult"],
            rs_btc_period=None,
            is_btc=(sym == "BTCUSDT"),
            momentum_lookback=STRATEGY_CONFIG["momentum_lookback"],
            own_sma_period=STRATEGY_CONFIG["own_sma_period"],
            max_stop_pct=STRATEGY_CONFIG["max_stop_pct"],
        )
    return out


class PaperTraderEngine:
    def __init__(
        self,
        state_path: str = STATE_FILE,
        data_dir: str = DATA_DIR,
        tg_client: Optional[TelegramClient] = None,
    ) -> None:
        self.state_path = state_path
        self.data_dir = data_dir
        self.tg = tg_client or TelegramClient()
        self.live_prices: dict[str, float] = {}
        self.usdt_brl: float = 5.75
        self.last_price_sync: float = 0.0

    def refresh_live_prices(self) -> dict[str, float]:
        prices = fetch_live_prices(UNIVERSE, include_brl=True)
        if "USDTBRL" in prices:
            self.usdt_brl = prices.pop("USDTBRL")
        self.live_prices = prices
        self.last_price_sync = time.time()
        return self.live_prices

    def get_snapshot(self, force_price_refresh: bool = False) -> dict[str, Any]:
        if force_price_refresh or not self.live_prices or (time.time() - self.last_price_sync > 25):
            try:
                self.refresh_live_prices()
            except Exception as exc:
                logger.warning("Falha ao buscar preços ao vivo: %s", exc)
        state = load_state(self.state_path)
        return build_dashboard_snapshot(state, self.live_prices, self.usdt_brl)

    def check_intraday_stops(
        self,
        state: dict[str, Any],
        live_prices: dict[str, float],
    ) -> list[dict[str, Any]]:
        """
        Verifica se alguma posição aberta atingiu o stop_price intradiário.
        """
        closed_events = []
        for sym in list(state.get("positions", {}).keys()):
            pos = state["positions"][sym]
            curr_px = live_prices.get(sym)
            if curr_px is None:
                continue
            stop_px = float(pos["stop_price"])
            if curr_px <= stop_px:
                tr = close_paper_position(state, sym, curr_px, reason="STOP_LOSS")
                if tr:
                    closed_events.append(tr)
        return closed_events

    def run_daily_cycle(
        self,
        raw_1d: dict[str, pd.DataFrame],
        live_prices: dict[str, float],
        notify: bool = True,
    ) -> dict[str, Any]:
        """
        Executa inicialização ou avaliação do fechamento diário mais recente (idempotente).
        """
        state = load_state(self.state_path)
        btc_df = raw_1d["BTCUSDT"]
        latest_date = str(pd.Timestamp(btc_df["open_time"].iloc[-1]).date())
        radar = compute_market_radar(raw_1d, live_prices)
        state["last_radar"] = radar

        prev_processed = state.get("last_daily_candle_date")

        # Primeira execução: sincroniza baseline e não compra com atraso no meio do dia
        if prev_processed is None:
            state["last_daily_candle_date"] = latest_date
            snap = build_dashboard_snapshot(state, live_prices, self.usdt_brl)
            state["equity_history"].append({
                "date": latest_date,
                "equity_usd": snap["summary"]["total_equity_usd"],
                "cash_usd": snap["summary"]["cash_usd"],
                "btc_bull": radar["btc_bull"],
            })
            save_state(state, self.state_path)
            if notify and self.tg.enabled:
                self.tg.send_message(
                    "🤖 <b>Paper Trader Inicializado!</b>\n\n"
                    + format_status_message(snap)
                    + "\n\n⏰ <i>Próxima avaliação oficial no fechamento diário (21:00 BRT / 00:00 UTC).</i>"
                )
            return {"action": "initialized", "date": latest_date, "buys": [], "sells": []}

        # Mesmo dia já processado: apenas atualiza o radar
        if latest_date <= prev_processed:
            save_state(state, self.state_path)
            return {"action": "radar_refreshed", "date": latest_date, "buys": [], "sells": []}

        # Novo fechamento diário confirmado -> processa sinais!
        sigs = _compute_signals_universe(raw_1d)
        sells_done = []
        buys_done = []

        # 1) Processa saídas por sinal Donchian Low (15) e atualiza referência das mantidas
        for sym in list(state["positions"].keys()):
            df_sig = sigs.get(sym)
            if df_sig is None or df_sig.empty:
                continue
            last_row = df_sig.iloc[-1]
            exec_px = float(live_prices.get(sym, last_row["close"]))
            if bool(last_row["signal_sell"]):
                tr = close_paper_position(state, sym, exec_px, reason="SIGNAL_EXIT")
                if tr:
                    sells_done.append(tr)
            else:
                # Atualiza Donchian Low informativo na posição aberta
                state["positions"][sym]["don_lo_exit"] = float(last_row["don_lo"])

        # 2) Calcula patrimônio para dimensionar possíveis entradas
        open_equity = float(state["cash_usd"])
        for sym, pos in state["positions"].items():
            open_equity += float(pos["qty"]) * float(live_prices.get(sym, pos["entry_price"]))

        # 3) Coleta e ranqueia sinais de compra se não estiver pausado
        if not state.get("paused", False):
            candidates = []
            for sym, df_sig in sigs.items():
                if sym in state["positions"] or df_sig.empty:
                    continue
                last_row = df_sig.iloc[-1]
                if bool(last_row["signal_buy"]):
                    score = float(last_row.get("momentum_score", 0.0))
                    candidates.append((score, sym, last_row))

            candidates.sort(key=lambda item: item[0], reverse=True)
            for score, sym, row in candidates:
                if len(state["positions"]) >= PAPER_CONFIG["max_positions"]:
                    break
                ref_px = float(live_prices.get(sym, row["close"]))
                stop_px = float(row["stop_price_init"])
                alloc = compute_position_size_usd(
                    open_equity_usd=open_equity,
                    cash_usd=float(state["cash_usd"]),
                    entry_price=ref_px,
                    stop_price=stop_px,
                )
                if alloc > 0:
                    pos = open_paper_position(
                        state=state,
                        symbol=sym,
                        ref_price=ref_px,
                        alloc_usd=alloc,
                        stop_price=stop_px,
                        don_lo_exit=float(row["don_lo"]),
                        atr_val=float(row["atr"]),
                        momentum_score=score,
                    )
                    if pos:
                        buys_done.append(pos)

        state["last_daily_candle_date"] = latest_date
        snap = build_dashboard_snapshot(state, live_prices, self.usdt_brl)
        state["equity_history"].append({
            "date": latest_date,
            "equity_usd": snap["summary"]["total_equity_usd"],
            "cash_usd": snap["summary"]["cash_usd"],
            "btc_bull": radar["btc_bull"],
        })
        save_state(state, self.state_path)

        if notify and self.tg.enabled:
            self._send_daily_notifications(snap, buys_done, sells_done)

        return {"action": "evaluated", "date": latest_date, "buys": buys_done, "sells": sells_done}

    def _send_daily_notifications(
        self,
        snap: dict[str, Any],
        buys: list[dict[str, Any]],
        sells: list[dict[str, Any]],
    ) -> None:
        for tr in sells:
            sign = "+" if tr["net_pnl_usd"] >= 0 else ""
            emoji = "✅" if tr["net_pnl_usd"] > 0 else "🛑"
            self.tg.send_message(
                f"{emoji} <b>SAÍDA EXECUTADA (PAPER) — {tr['symbol']}</b>\n\n"
                f"• Motivo: <b>{tr['exit_reason']}</b>\n"
                f"• Entrada: <code>${tr['entry_price']:.4f}</code> → Saída: <code>${tr['exit_price']:.4f}</code>\n"
                f"• Resultado líquido: <b>{sign}US$ {tr['net_pnl_usd']:.2f} ({sign}{tr['return_pct']:.2f}%)</b>\n"
                f"• Taxa simulada total: US$ {tr['fees_usd']:.2f}"
            )

        for pos in buys:
            dist_pct = ((pos["entry_price"] - pos["stop_price"]) / pos["entry_price"]) * 100.0
            self.tg.send_message(
                f"🚀 <b>COMPRA EXECUTADA (PAPER) — {pos['symbol']}</b>\n\n"
                f"• Preço entrada: <code>${pos['entry_price']:.4f}</code>\n"
                f"• Valor alocado: <b>US$ {pos['alloc_usd']:.2f}</b>\n"
                f"• Stop Loss (3×ATR): <code>${pos['stop_price']:.4f}</code> (-{dist_pct:.1f}%)\n"
                f"• Saída tendência (Don15): <code>${pos['don_lo_exit']:.4f}</code>"
            )

        self.tg.send_message(
            "🌙 <b>Fechamento Diário (21h BRT)</b>\n\n" + format_status_message(snap)
        )

    def handle_telegram_commands(self) -> None:
        if not self.tg.enabled:
            return
        state = load_state(self.state_path)
        last_uid = state.get("last_telegram_update_id")
        offset = (last_uid + 1) if last_uid is not None else None
        updates = self.tg.get_updates(offset=offset, timeout=1)
        if not updates:
            return

        max_uid = last_uid
        for upd in updates:
            uid = upd.get("update_id")
            if max_uid is None or (uid is not None and uid > max_uid):
                max_uid = uid
            msg = upd.get("message") or {}
            chat_id = str((msg.get("chat") or {}).get("id", ""))
            if chat_id != str(self.tg.chat_id):
                continue
            text = (msg.get("text") or "").strip().lower()
            if not text.startswith("/"):
                continue

            cmd = text.split()[0]
            if cmd in ("/start", "/status"):
                snap = self.get_snapshot(force_price_refresh=True)
                self.tg.send_message(format_status_message(snap))
            elif cmd == "/radar":
                snap = self.get_snapshot(force_price_refresh=True)
                self.tg.send_message(format_radar_message(snap))
            elif cmd == "/trades":
                snap = self.get_snapshot(force_price_refresh=False)
                self.tg.send_message(format_trades_message(snap))
            elif cmd == "/pausar":
                st = load_state(self.state_path)
                st["paused"] = True
                save_state(st, self.state_path)
                self.tg.send_message("⏸️ Novas compras em Paper Trading foram <b>pausadas</b>. Stops continuam monitorados.")
            elif cmd == "/retomar":
                st = load_state(self.state_path)
                st["paused"] = False
                save_state(st, self.state_path)
                self.tg.send_message("▶️ Operação em Paper Trading <b>retomada</b>.")
            elif cmd in ("/ajuda", "/help"):
                self.tg.send_message(
                    "🤖 <b>Comandos do SwingBot Paper</b>\n\n"
                    "/status — Saldo, regime BTC e posições abertas\n"
                    "/radar — Proximidade de rompimento nas 16 moedas\n"
                    "/trades — Últimos trades encerrados\n"
                    "/pausar — Pausa novas compras automáticas\n"
                    "/retomar — Retoma compras automáticas"
                )

        if max_uid is not None and max_uid != last_uid:
            st = load_state(self.state_path)
            st["last_telegram_update_id"] = max_uid
            save_state(st, self.state_path)

    def run_once(self, notify: bool = True) -> dict[str, Any]:
        prices = self.refresh_live_prices()
        raw_1d = fetch_universe_daily_candles(UNIVERSE, start_date="2025-01-01", cache_dir=self.data_dir)
        state = load_state(self.state_path)

        stops = self.check_intraday_stops(state, prices)
        if stops:
            save_state(state, self.state_path)
            if notify and self.tg.enabled:
                for tr in stops:
                    sign = "+" if tr["net_pnl_usd"] >= 0 else ""
                    self.tg.send_message(
                        f"🛑 <b>STOP LOSS ACIONADO (PAPER) — {tr['symbol']}</b>\n\n"
                        f"• Preço saída: <code>${tr['exit_price']:.4f}</code>\n"
                        f"• Resultado líquido: <b>{sign}US$ {tr['net_pnl_usd']:.2f} ({sign}{tr['return_pct']:.2f}%)</b>"
                    )

        res = self.run_daily_cycle(raw_1d, prices, notify=notify)
        return res

    def run_forever(self, stop_event: Any = None) -> None:
        logger.info("Iniciando loop contínuo do Paper Trader (intervalo %ss)...", PAPER_CONFIG["stop_check_interval_sec"])
        last_candle_check = 0.0
        # Primeira passada imediata
        try:
            self.run_once(notify=True)
            last_candle_check = time.time()
        except Exception as exc:
            logger.exception("Erro na inicialização do Paper Trader: %s", exc)

        while stop_event is None or not stop_event.is_set():
            try:
                self.handle_telegram_commands()
            except Exception as exc:
                logger.warning("Erro ao ler comandos Telegram: %s", exc)

            now = time.time()
            # A cada 60s checa preços ao vivo e stops intradiários
            if now - self.last_price_sync >= PAPER_CONFIG["stop_check_interval_sec"]:
                try:
                    prices = self.refresh_live_prices()
                    state = load_state(self.state_path)
                    if state.get("positions"):
                        stops = self.check_intraday_stops(state, prices)
                        if stops:
                            save_state(state, self.state_path)
                            for tr in stops:
                                sign = "+" if tr["net_pnl_usd"] >= 0 else ""
                                self.tg.send_message(
                                    f"🛑 <b>STOP LOSS ACIONADO (PAPER) — {tr['symbol']}</b>\n\n"
                                    f"• Preço saída: <code>${tr['exit_price']:.4f}</code>\n"
                                    f"• Resultado líquido: <b>{sign}US$ {tr['net_pnl_usd']:.2f} ({sign}{tr['return_pct']:.2f}%)</b>"
                                )
                except Exception as exc:
                    logger.warning("Erro ao checar stops: %s", exc)

            # A cada 120s (ou logo após 00:00 UTC) checa se fechou novo candle diário 1d
            utc_now = datetime.now(timezone.utc)
            check_interval = 30 if (utc_now.hour == 0 and utc_now.minute < 10) else 120
            if now - last_candle_check >= check_interval:
                try:
                    raw_1d = fetch_universe_daily_candles(UNIVERSE, start_date="2025-01-01", cache_dir=self.data_dir)
                    self.run_daily_cycle(raw_1d, self.live_prices, notify=True)
                    last_candle_check = time.time()
                except Exception as exc:
                    logger.warning("Erro no ciclo diário: %s", exc)

            time.sleep(3)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    engine = PaperTraderEngine()
    engine.run_forever()
