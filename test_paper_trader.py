"""
test_paper_trader.py
Testes automatizados para contabilidade única, idempotência diária,
stop loss intradiário e API do Telegram WebApp (usando arquivos temporários).
"""

from __future__ import annotations

import json
import os
import tempfile
import urllib.request

import numpy as np
import pandas as pd

from paper_portfolio import (
    build_dashboard_snapshot,
    close_paper_position,
    compute_position_size_usd,
    default_state,
    load_state,
    open_paper_position,
    save_state,
)
from paper_trader import PaperTraderEngine
from telegram_client import TelegramClient
from telegram_webapp import make_handler
from http.server import ThreadingHTTPServer
import threading


def _fake_daily_df(n: int = 230, breakout_last: bool = False, base_price: float = 100.0) -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=n, freq="1D", tz="UTC")
    closes = np.linspace(base_price * 0.85, base_price, n)
    if breakout_last:
        closes[-1] = base_price * 1.15  # rompe claramente os últimos 30 dias
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    highs = np.maximum(opens, closes) * 1.01
    lows = np.minimum(opens, closes) * 0.99
    return pd.DataFrame({
        "open_time": times,
        "close_time": times + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": [1000.0] * n,
    })


def test_accounting_invariant() -> None:
    st = default_state(300.0)
    alloc = compute_position_size_usd(300.0, st["cash_usd"], entry_price=100.0, stop_price=85.0)
    assert alloc > 0
    pos = open_paper_position(st, "SOLUSDT", 100.0, alloc, stop_price=85.0, don_lo_exit=90.0, atr_val=5.0, momentum_score=2.5)
    assert pos is not None

    snap1 = build_dashboard_snapshot(st, {"SOLUSDT": 112.0}, usdt_brl=5.5)
    s1 = snap1["summary"]
    diff1 = abs((s1["total_equity_usd"] - 300.0) - (s1["realized_pnl_usd"] + s1["unrealized_pnl_usd"]))
    assert diff1 < 0.03, f"Invariante quebrada aberto: {diff1}"

    tr = close_paper_position(st, "SOLUSDT", 112.0, reason="MANUAL_SELL")
    assert tr is not None and tr["net_pnl_usd"] > 0
    snap2 = build_dashboard_snapshot(st, {"SOLUSDT": 112.0}, usdt_brl=5.5)
    s2 = snap2["summary"]
    diff2 = abs((s2["total_equity_usd"] - 300.0) - (s2["realized_pnl_usd"] + s2["unrealized_pnl_usd"]))
    assert diff2 < 0.03, f"Invariante quebrada fechado: {diff2}"
    print("[PASS] test_accounting_invariant")


def test_daily_idempotency_and_intraday_stop() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        state_path = os.path.join(tmp, "state.json")
        tg_dummy = TelegramClient(token="", chat_id="")
        eng = PaperTraderEngine(state_path=state_path, data_dir=tmp, tg_client=tg_dummy)

        df_btc = _fake_daily_df(230, breakout_last=False, base_price=90000.0)
        df_sol = _fake_daily_df(230, breakout_last=True, base_price=150.0)
        raw_day1 = {"BTCUSDT": df_btc.iloc[:-1].copy(), "SOLUSDT": df_sol.iloc[:-1].copy()}
        prices_day1 = {"BTCUSDT": float(raw_day1["BTCUSDT"]["close"].iloc[-1]), "SOLUSDT": float(raw_day1["SOLUSDT"]["close"].iloc[-1])}

        # 1) Inicialização no dia anterior -> não abre compras atrasadas
        r_init = eng.run_daily_cycle(raw_day1, prices_day1, notify=False)
        assert r_init["action"] == "initialized"
        assert len(load_state(state_path)["positions"]) == 0

        # 2) Novo candle fechado com breakout em SOL -> abre compra
        raw_day2 = {"BTCUSDT": df_btc.copy(), "SOLUSDT": df_sol.copy()}
        prices_day2 = {"BTCUSDT": float(df_btc["close"].iloc[-1]), "SOLUSDT": float(df_sol["close"].iloc[-1])}
        r_eval = eng.run_daily_cycle(raw_day2, prices_day2, notify=False)
        assert r_eval["action"] == "evaluated"
        assert len(r_eval["buys"]) == 1 and r_eval["buys"][0]["symbol"] == "SOLUSDT"

        # 3) Rodar de novo para o mesmo dia -> idempotente (não recompra)
        r_dup = eng.run_daily_cycle(raw_day2, prices_day2, notify=False)
        assert r_dup["action"] == "radar_refreshed"
        assert len(load_state(state_path)["positions"]) == 1

        # 4) Queda até abaixo do stop -> stop loss intradiário fecha
        st = load_state(state_path)
        stop_px = st["positions"]["SOLUSDT"]["stop_price"]
        stops = eng.check_intraday_stops(st, {"SOLUSDT": stop_px * 0.99})
        save_state(st, state_path)
        assert len(stops) == 1 and stops[0]["exit_reason"] == "STOP_LOSS"
        assert len(load_state(state_path)["positions"]) == 0
    print("[PASS] test_daily_idempotency_and_intraday_stop")


def test_webapp_api_status() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        state_path = os.path.join(tmp, "state.json")
        tg_dummy = TelegramClient(token="", chat_id="")
        eng = PaperTraderEngine(state_path=state_path, data_dir=tmp, tg_client=tg_dummy)
        eng.live_prices = {"BTCUSDT": 90000.0}
        eng.last_price_sync = 9e12

        handler = make_handler(eng)
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=5) as resp:
                assert resp.status == 200
                data = json.loads(resp.read().decode("utf-8"))
                assert data["ok"] is True
                assert data["summary"]["initial_capital_usd"] == 300.0
        finally:
            server.shutdown()
    print("[PASS] test_webapp_api_status")


def test_cloud_runner_helpers() -> None:
    from cloud_runner import resolve_github_pages_url

    old_repo = os.environ.get("GITHUB_REPOSITORY")
    old_url = os.environ.get("WEBAPP_PUBLIC_URL")
    try:
        os.environ["WEBAPP_PUBLIC_URL"] = ""
        os.environ["GITHUB_REPOSITORY"] = "CaioUser/crypto-swing-bot"
        assert resolve_github_pages_url() == "https://caiouser.github.io/crypto-swing-bot/"
    finally:
        if old_repo is None:
            os.environ.pop("GITHUB_REPOSITORY", None)
        else:
            os.environ["GITHUB_REPOSITORY"] = old_repo
        if old_url is None:
            os.environ.pop("WEBAPP_PUBLIC_URL", None)
        else:
            os.environ["WEBAPP_PUBLIC_URL"] = old_url
    print("[PASS] test_cloud_runner_helpers")


if __name__ == "__main__":
    test_accounting_invariant()
    test_daily_idempotency_and_intraday_stop()
    test_webapp_api_status()
    test_cloud_runner_helpers()
