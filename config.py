"""
config.py
Configuração central do SwingBot Paper Trader e integração com Telegram Web App.
"""

from __future__ import annotations

import os
from dotenv import dotenv_values

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATE_FILE = os.path.join(BASE_DIR, "paper_state.json")
TWA_DIR = os.path.join(BASE_DIR, "twa")
TWA_URL_FILE = os.path.join(BASE_DIR, "twa_url.txt")
CLOUDFLARED_PATH = os.path.join(BASE_DIR, "tools", "cloudflared.exe")

# Universo validado no backtest (16 Top Caps líquidos na Binance Spot)
UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
    "LINKUSDT", "LTCUSDT", "NEARUSDT", "AAVEUSDT",
    "UNIUSDT", "DOTUSDT", "ATOMUSDT", "TRXUSDT",
]

# Parâmetros validados da estratégia Donchian 30/15 + Filtro Macro + Anti-Esticamento
STRATEGY_CONFIG = {
    "name": "Donchian 30/15 + OwnSMA200 + BTC Filter",
    "interval": "1d",
    "entry_period": 30,
    "exit_period": 15,
    "btc_sma_period": 200,
    "own_sma_period": 200,
    "atr_period": 14,
    "atr_stop_mult": 3.0,
    "max_stop_pct": 0.18,
    "momentum_lookback": 30,
}

# Gestão de Risco e Custos Simulados (Paper Trading)
PAPER_CONFIG = {
    "initial_capital_usd": 300.0,
    "max_positions": 4,
    "risk_per_trade_pct": 0.03,   # 3% da banca total até o Stop
    "fee_rate": 0.001,            # 0.10% Binance Spot padrão
    "slippage_rate": 0.0005,      # 0.05% slippage simulado
    "min_order_usd": 10.0,        # Mínimo nominal realista da Binance
    "stop_check_interval_sec": 60,
    "daily_check_interval_sec": 60,
}

WEBAPP_PORT = 8090


def get_telegram_credentials() -> tuple[str | None, str | None]:
    """
    Busca apenas TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID de:
    1) Variáveis de ambiente atuais
    2) .env.local na pasta do projeto
    3) C:\\nanobot_secrets\\.env.local (reaproveitando o esquema existente)
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        return token.strip(), chat_id.strip()

    for candidate in (
        os.path.join(BASE_DIR, ".env.local"),
        os.path.join(BASE_DIR, ".env"),
        r"C:\nanobot_secrets\.env.local",
    ):
        if os.path.exists(candidate):
            vals = dotenv_values(candidate)
            token = token or vals.get("TELEGRAM_BOT_TOKEN")
            chat_id = chat_id or vals.get("TELEGRAM_CHAT_ID")
            if token and chat_id:
                return token.strip(), chat_id.strip()

    return (token.strip() if token else None), (chat_id.strip() if chat_id else None)
