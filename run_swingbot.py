"""
run_swingbot.py
Launcher unificado do SwingBot Cripto (Paper Trading US$ 300 + Telegram Bot + WebApp Dashboard).
Inicia o motor de simulação em thread daemon, o servidor HTTP local e o túnel Cloudflare.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading

from config import BASE_DIR, WEBAPP_PORT
from paper_trader import PaperTraderEngine
from telegram_webapp import run_cloudflared_watchdog, start_http_server

PID_FILE = os.path.join(BASE_DIR, "swingbot.pid")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(BASE_DIR, "swingbot_paper.log"), encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("swingbot.main")

    with open(PID_FILE, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))

    stop_event = threading.Event()

    def _handle_sig(signum: int, _frame: object) -> None:
        logger.info("Sinal %s recebido. Encerrando SwingBot Paper...", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    logger.info("=" * 60)
    logger.info("SWINGBOT CRIPTO — PAPER TRADING (BANCA INICIAL US$ 300)")
    logger.info("=" * 60)

    engine = PaperTraderEngine()
    server = start_http_server(engine, port=WEBAPP_PORT)

    trader_thread = threading.Thread(
        target=engine.run_forever,
        args=(stop_event,),
        daemon=True,
        name="paper-trader",
    )
    trader_thread.start()

    try:
        run_cloudflared_watchdog(engine, port=WEBAPP_PORT, stop_event=stop_event)
    finally:
        stop_event.set()
        server.shutdown()
        if os.path.exists(PID_FILE):
            try:
                os.remove(PID_FILE)
            except OSError:
                pass
        logger.info("SwingBot Paper encerrado com segurança.")


if __name__ == "__main__":
    main()
