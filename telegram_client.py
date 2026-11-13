"""
telegram_client.py
Cliente Telegram leve para envio de alertas HTML, atualização do botão WebApp
e leitura opcional de comandos do usuário autorizado (/status, /radar, /trades).
"""

from __future__ import annotations

from typing import Any, Optional
import requests

from config import get_telegram_credentials


class TelegramClient:
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None) -> None:
        t, c = get_telegram_credentials()
        self.token = token or t
        self.chat_id = str(chat_id or c) if (chat_id or c) else None

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def _post(self, method: str, payload: dict[str, Any], timeout: int = 10) -> dict[str, Any]:
        if not self.token:
            return {"ok": False, "error": "Token ausente"}
        url = f"https://api.telegram.org/bot{self.token}/{method}"
        resp = requests.post(url, json=payload, timeout=timeout)
        return resp.json()

    def send_message(
        self,
        text: str,
        reply_markup: Optional[dict[str, Any]] = None,
        disable_preview: bool = True,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"ok": False, "error": "Telegram não configurado"}
        payload: dict[str, Any] = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": disable_preview,
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup
        try:
            return self._post("sendMessage", payload, timeout=10)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def set_menu_button(self, webapp_url: str, button_text: str = "📊 Dashboard") -> dict[str, Any]:
        if not self.token:
            return {"ok": False, "error": "Token ausente"}
        menu = {"type": "web_app", "text": button_text, "web_app": {"url": webapp_url}}
        res = {}
        try:
            res = self._post("setChatMenuButton", {"menu_button": menu}, timeout=8)
            if self.chat_id:
                res = self._post(
                    "setChatMenuButton",
                    {"chat_id": self.chat_id, "menu_button": menu},
                    timeout=8,
                )
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return res

    def notify_dashboard_online(self, webapp_url: str) -> dict[str, Any]:
        msg = (
            "📈 <b>SwingBot Cripto (Paper Trading) — Online!</b>\n\n"
            "Modo: <b>Simulação (US$ 300 virtuais)</b>\n"
            "Estratégia: <b>Donchian 30/15 + SMA200 + Filtro BTC</b>\n\n"
            "O link do painel foi sincronizado. Use o botão abaixo ou o menu do bot para acompanhar."
        )
        markup = {
            "inline_keyboard": [
                [{"text": "📊 Abrir Dashboard Swing", "web_app": {"url": webapp_url}}]
            ]
        }
        return self.send_message(msg, reply_markup=markup)

    def get_updates(self, offset: Optional[int] = None, timeout: int = 2) -> list[dict[str, Any]]:
        if not self.token:
            return []
        payload: dict[str, Any] = {"timeout": timeout, "allowed_updates": ["message"]}
        if offset is not None:
            payload["offset"] = offset
        try:
            data = self._post("getUpdates", payload, timeout=timeout + 5)
            if data.get("ok"):
                return data.get("result", [])
        except Exception:
            pass
        return []
