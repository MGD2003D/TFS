"""
Gemini client через Caila API.

Использует тот же токен CAILA_TOKEN, но другой endpoint и поле model в payload.
По умолчанию: gemini-2.5-flash.
"""

import aiohttp
import os
import json
from .caila_client import CailaClient
from typing import List, Dict, Optional


GEMINI_API_URL = "https://caila.io/api/mlpgate/account/just-ai/model/gemini/predict"

AVAILABLE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]


class GeminiClient(CailaClient):
    """
    Gemini через Caila API.

    Наследует всю логику CailaClient (retry, extract_aspects, extract_triplets),
    переопределяет только URL и payload (_generate добавляет поле model).
    """

    def __init__(self, model: str = "gemini-2.5-flash", token: str = None):
        super().__init__(api_url=GEMINI_API_URL, token=token)
        self.model = model

    async def initialize(self):
        print(f"Инициализация Gemini клиента ({self.model})")
        print(f"API URL: {self.api_url}")
        self.session = aiohttp.ClientSession()
        print(f"Gemini клиент готов ({self.model})")

    async def _generate(self, messages: List[Dict[str, str]]) -> str:
        """Отправка запроса к Gemini через Caila API."""
        if not self.session:
            raise RuntimeError("Клиент не инициализирован. Вызовите initialize() сначала.")

        payload = {
            "model": self.model,
            "messages": messages,
        }
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        try:
            async with self.session.post(self.api_url, json=payload, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()

                if "response" in data:
                    return data["response"]
                elif "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0].get("message", {}).get("content", "")
                elif "text" in data:
                    return data["text"]
                else:
                    return str(data)

        except aiohttp.ClientError as e:
            print(f"[GEMINI] Ошибка API: {e}")
            raise
        except Exception as e:
            print(f"[GEMINI] Неожиданная ошибка: {e}")
            raise
