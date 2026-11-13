"""
cloud_runner.py
Executor serverless para GitHub Actions:
1. Busca cotações ao vivo e candles diários fechados (1d) na Binance.
2. Verifica stops intradiários, fechamento diário (21h BRT / 00:00 UTC) e comandos do Telegram.
3. Exporta `twa/status.json` para o Dashboard no GitHub Pages.
4. Sincroniza automaticamente a URL fixa do GitHub Pages no botão '📊 Dashboard' do Telegram.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from config import BASE_DIR, DATA_DIR, STATE_FILE, TWA_DIR, UNIVERSE
from market_api import fetch_universe_daily_candles
from paper_portfolio import (
    close_paper_position,
    compute_market_radar,
    default_state,
    load_state,
    save_state,
)
from paper_trader import PaperTraderEngine

logger = logging.getLogger("swingbot.cloud")


def resolve_github_pages_url() -> str | None:
    """
    Descobre a URL pública do Dashboard no GitHub Pages:
    1) Variável explícita WEBAPP_PUBLIC_URL
    2) Derivada automaticamente de GITHUB_REPOSITORY (owner/repo -> https://owner.github.io/repo/)
    """
    explicit = os.getenv("WEBAPP_PUBLIC_URL", "").strip()
    if explicit.startswith("https://"):
        return explicit

    gh_repo = os.getenv("GITHUB_REPOSITORY", "").strip()
    if "/" in gh_repo:
        owner, repo = gh_repo.split("/", 1)
        return f"https://{owner.lower()}.github.io/{repo}/"
    return None


def apply_manual_action(engine: PaperTraderEngine, action: str) -> None:
    action = action.strip().lower()
    if not action or action == "sync":
        return

    state = load_state(engine.state_path)
    if action == "pause":
        state["paused"] = True
        save_state(state, engine.state_path)
        if engine.tg.enabled:
            engine.tg.send_message("⏸️ Paper Trading <b>pausado</b> via GitHub Actions.")
    elif action == "resume":
        state["paused"] = False
        save_state(state, engine.state_path)
        if engine.tg.enabled:
            engine.tg.send_message("▶️ Paper Trading <b>retomado</b> via GitHub Actions.")
    elif action == "sell_all":
        prices = engine.refresh_live_prices()
        closed = []
        for sym in list(state.get("positions", {}).keys()):
            pos = state["positions"][sym]
            px = float(prices.get(sym, pos["entry_price"]))
            tr = close_paper_position(state, sym, px, reason="MANUAL_SELL")
            if tr:
                closed.append(tr)
        save_state(state, engine.state_path)
        if closed and engine.tg.enabled:
            engine.tg.send_message(
                f"⚡ <b>VENDA TOTAL MANUAL (PAPER)</b>: {len(closed)} posição(ões) encerrada(s)."
            )
    elif action == "reset":
        fresh = default_state(state.get("initial_capital_usd", 300.0))
        fresh["last_daily_candle_date"] = state.get("last_daily_candle_date")
        fresh["last_radar"] = state.get("last_radar")
        fresh["last_telegram_update_id"] = state.get("last_telegram_update_id")
        save_state(fresh, engine.state_path)
        if engine.tg.enabled:
            engine.tg.send_message("🔄 Simulação Paper Trading <b>reiniciada em US$ 300,00</b>.")


def export_static_dashboard_data(engine: PaperTraderEngine) -> dict:
    os.makedirs(TWA_DIR, exist_ok=True)
    snap = engine.get_snapshot(force_price_refresh=False)
    status_path = os.path.join(TWA_DIR, "status.json")
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2, ensure_ascii=False)
    return snap


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Executa 1 ciclo completo do SwingBot na nuvem.")
    parser.add_argument(
        "--action",
        default="sync",
        choices=["sync", "pause", "resume", "sell_all", "reset"],
        help="Ação manual opcional disparada via workflow_dispatch.",
    )
    parser.add_argument(
        "--notify-menu",
        action="store_true",
        help="Envia mensagem no Telegram confirmando o link do Dashboard.",
    )
    args = parser.parse_args()

    engine = PaperTraderEngine(state_path=STATE_FILE, data_dir=DATA_DIR)

    # 1) Aplica ação manual se solicitada
    apply_manual_action(engine, args.action)

    # 2) Executa ciclo de preços, stops intradiários e fechamento diário (1d)
    result = engine.run_once(notify=True)
    logger.info("Resultado do ciclo diário: %s", result["action"])

    # 3) Processa comandos enviados pelo usuário no chat do Telegram (/status, /radar, etc.)
    try:
        engine.handle_telegram_commands()
    except Exception as exc:
        logger.warning("Aviso ao processar comandos Telegram: %s", exc)

    # 4) Exporta snapshot atualizado para twa/status.json (usado pelo GitHub Pages)
    snap = export_static_dashboard_data(engine)
    logger.info(
        "Snapshot exportado: Patrimônio US$ %.2f | Posições: %s/%s | Candle: %s",
        snap["summary"]["total_equity_usd"],
        snap["summary"]["open_count"],
        snap["summary"]["max_positions"],
        snap["summary"]["last_daily_candle_date"],
    )

    # 5) Sincroniza o botão '📊 Dashboard' no Telegram caso tenhamos URL do GitHub Pages
    pages_url = resolve_github_pages_url()
    if pages_url and engine.tg.enabled:
        engine.tg.set_menu_button(pages_url, button_text="📊 Dashboard")
        logger.info("Menu do Telegram sincronizado para GitHub Pages: %s", pages_url)
        if args.notify_menu:
            engine.tg.notify_dashboard_online(pages_url)


if __name__ == "__main__":
    main()
