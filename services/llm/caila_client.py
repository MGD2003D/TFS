import aiohttp
import os
from .base import BaseLLMClient
from typing import List, Dict
from app_state import system_prompt


class CailaClient(BaseLLMClient):

    def __init__(self, api_url: str = "https://caila.io/api/mlpgateway/account/just-ai/model/Qwen3-30B-A3B/predict", token: str = None):
        self.api_url = api_url
        self.token = token or os.getenv("CAILA_TOKEN")
        self.session = None

        if not self.token:
            raise ValueError("CAILA_TOKEN не найден в переменных окружения")

    async def initialize(self):
        """Инициализация HTTP сессии"""
        print(f"Инициализация Caila API клиента")
        print(f"API URL: {self.api_url}")
        self.session = aiohttp.ClientSession()
        print("Caila клиент готов к работе")

    async def simple_query(self, prompt: str) -> str:
        """Простой запрос к модели"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        return await self._generate(messages)

    async def chat_query(self, messages: List[Dict[str, str]]) -> str:
        """Запрос с историей сообщений"""
        full_messages = [
            {"role": "system", "content": system_prompt}
        ] + messages

        print(f"В generate ушло:\n{full_messages}")

        return await self._generate(full_messages)

    async def cleanup(self):
        """Закрытие HTTP сессии"""
        if self.session:
            await self.session.close()
            print("Caila клиент закрыт")

    async def _generate(self, messages: List[Dict[str, str]]) -> str:
        """Отправка запроса к Caila API"""
        if not self.session:
            raise RuntimeError("Клиент не инициализирован. Вызовите initialize() сначала.")

        payload = {"messages": messages}
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
            print(f"Ошибка при обращении к Caila API: {e}")
            raise
        except Exception as e:
            print(f"Неожиданная ошибка: {e}")
            raise

