"""
telegram_webapp.py
Servidor Web & API do SwingBot Paper Trader + Túnel HTTPS Cloudflare com Auto-Healing.
Sincroniza o botão permanente '📊 Dashboard' no Telegram Web App.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

import requests

from config import CLOUDFLARED_PATH, TWA_DIR, TWA_URL_FILE, WEBAPP_PORT
from market_api import fetch_universe_daily_candles
from paper_portfolio import (
    close_paper_position,
    compute_market_radar,
    default_state,
    load_state,
    save_state,
)
from paper_trader import PaperTraderEngine
from config import DATA_DIR, UNIVERSE

logger = logging.getLogger("swingbot.webapp")


def make_handler(engine: PaperTraderEngine):
    class SwingWebHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=TWA_DIR, **kwargs)

        def _send_json(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            return json.loads(raw) if raw.strip() else {}

        def do_GET(self) -> None:
            if self.path.startswith("/api/health"):
                self._send_json(200, {"ok": True, "status": "online", "mode": "PAPER"})
                return
            if self.path.startswith("/api/status"):
                try:
                    force = "refresh=1" in self.path
                    snap = engine.get_snapshot(force_price_refresh=force)
                    self._send_json(200, snap)
                except Exception as exc:
                    self._send_json(500, {"ok": False, "error": str(exc)})
                return
            super().do_GET()

        def do_POST(self) -> None:
            try:
                if self.path.startswith("/api/sell"):
                    body = self._read_json_body()
                    symbol = str(body.get("symbol", "")).upper().strip()
                    if not symbol:
                        self._send_json(400, {"ok": False, "error": "Parâmetro 'symbol' obrigatório."})
                        return
                    engine.refresh_live_prices()
                    state = load_state(engine.state_path)
                    closed = []
                    targets = list(state["positions"].keys()) if symbol == "ALL" else [symbol]
                    for sym in targets:
                        pos = state["positions"].get(sym)
                        if not pos:
                            continue
                        px = float(engine.live_prices.get(sym, pos["entry_price"]))
                        tr = close_paper_position(state, sym, px, reason="MANUAL_SELL")
                        if tr:
                            closed.append(tr)
                    if not closed:
                        self._send_json(404, {"ok": False, "error": "Nenhuma posição encontrada para encerrar."})
                        return
                    save_state(state, engine.state_path)
                    if engine.tg.enabled:
                        for tr in closed:
                            sign = "+" if tr["net_pnl_usd"] >= 0 else ""
                            engine.tg.send_message(
                                f"⚡ <b>VENDA MANUAL (PAPER) — {tr['symbol']}</b>\n\n"
                                f"• Preço saída: <code>${tr['exit_price']:.4f}</code>\n"
                                f"• Resultado líquido: <b>{sign}US$ {tr['net_pnl_usd']:.2f} ({sign}{tr['return_pct']:.2f}%)</b>"
                            )
                    self._send_json(200, {"ok": True, "closed": closed})
                    return

                if self.path.startswith("/api/pause"):
                    st = load_state(engine.state_path)
                    st["paused"] = True
                    save_state(st, engine.state_path)
                    self._send_json(200, {"ok": True, "paused": True})
                    return

                if self.path.startswith("/api/resume"):
                    st = load_state(engine.state_path)
                    st["paused"] = False
                    save_state(st, engine.state_path)
                    self._send_json(200, {"ok": True, "paused": False})
                    return

                if self.path.startswith("/api/refresh"):
                    prices = engine.refresh_live_prices()
                    raw_1d = fetch_universe_daily_candles(UNIVERSE, start_date="2025-01-01", cache_dir=DATA_DIR)
                    st = load_state(engine.state_path)
                    st["last_radar"] = compute_market_radar(raw_1d, prices)
                    save_state(st, engine.state_path)
                    snap = engine.get_snapshot(force_price_refresh=False)
                    self._send_json(200, snap)
                    return

                if self.path.startswith("/api/reset"):
                    body = self._read_json_body()
                    if not body.get("confirm"):
                        self._send_json(400, {"ok": False, "error": "Confirmação obrigatória."})
                        return
                    old = load_state(engine.state_path)
                    fresh = default_state(old.get("initial_capital_usd", 300.0))
                    fresh["last_daily_candle_date"] = old.get("last_daily_candle_date")
                    fresh["last_radar"] = old.get("last_radar")
                    fresh["last_telegram_update_id"] = old.get("last_telegram_update_id")
                    save_state(fresh, engine.state_path)
                    self._send_json(200, {"ok": True})
                    return

                self._send_json(404, {"ok": False, "error": "Rota não encontrada."})
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": str(exc)})

        def log_message(self, format: str, *args: Any) -> None:
            pass

    return SwingWebHandler


def start_http_server(engine: PaperTraderEngine, port: int = WEBAPP_PORT) -> ThreadingHTTPServer:
    handler = make_handler(engine)
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("Servidor Web & API ativo em http://127.0.0.1:%s", port)
    return server


def run_cloudflared_watchdog(
    engine: PaperTraderEngine,
    port: int = WEBAPP_PORT,
    stop_event: Optional[threading.Event] = None,
) -> None:
    if not os.path.exists(CLOUDFLARED_PATH):
        logger.error("cloudflared.exe não encontrado em %s", CLOUDFLARED_PATH)
        return

    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    is_first_launch = True

    while stop_event is None or not stop_event.is_set():
        logger.info("Conectando túnel HTTPS Cloudflare na porta %s...", port)
        proc = subprocess.Popen(
            [CLOUDFLARED_PATH, "tunnel", "--url", f"http://127.0.0.1:{port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        tunnel_url = None
        assert proc.stdout is not None
        for line in proc.stdout:
            match = url_pattern.search(line)
            if match:
                tunnel_url = match.group(0)
                break

        local_stop = threading.Event()
        if tunnel_url:
            logger.info("Túnel HTTPS ativo: %s", tunnel_url)
            with open(TWA_URL_FILE, "w", encoding="utf-8") as f:
                f.write(tunnel_url)

            if engine.tg.enabled:
                engine.tg.set_menu_button(tunnel_url, button_text="📊 Dashboard")
                if is_first_launch:
                    engine.tg.notify_dashboard_online(tunnel_url)
                    is_first_launch = False

            def _drain(pipe: Any) -> None:
                try:
                    for _ in pipe:
                        pass
                except Exception:
                    pass

            def _health_watchdog(url: str, target_proc: subprocess.Popen[str]) -> None:
                fails = 0
                while not local_stop.is_set() and (stop_event is None or not stop_event.is_set()):
                    time.sleep(60)
                    if local_stop.is_set():
                        break
                    try:
                        r = requests.get(f"{url}/api/health", timeout=10)
                        fails = 0 if r.status_code == 200 else fails + 1
                    except Exception:
                        fails += 1
                    if fails >= 5:
                        logger.warning("Túnel %s sem resposta (5 falhas). Reiniciando...", url)
                        try:
                            target_proc.terminate()
                        except Exception:
                            pass
                        break

            threading.Thread(target=_drain, args=(proc.stdout,), daemon=True).start()
            threading.Thread(target=_health_watchdog, args=(tunnel_url, proc), daemon=True).start()
        else:
            logger.warning("Não foi possível capturar URL do Cloudflare. Tentando novamente...")

        while proc.poll() is None:
            if stop_event is not None and stop_event.is_set():
                local_stop.set()
                proc.terminate()
                return
            time.sleep(1)

        local_stop.set()
        if stop_event is not None and stop_event.is_set():
            return
        logger.warning("Túnel desconectado. Reconectando em 3s...")
        time.sleep(3)
