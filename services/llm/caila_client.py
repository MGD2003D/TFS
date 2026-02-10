import aiohttp
import os
import json
from .base import BaseLLMClient
from typing import List, Dict, Optional
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

    async def extract_aspects(self, query: str) -> Optional[Dict[str, str]]:
        """
        Извлекает аспекты из запроса для query decomposition.

        Args:
            query: Пользовательский запрос

        Returns:
            Dict[aspect_name, search_query] или None при ошибке
            - Простой запрос: {"original": "query"}
            - Сложный запрос: {"original": "query", "aspect1": "...", ...}
        """
        from prompts_config import build_aspect_extraction_prompt

        try:
            prompt = build_aspect_extraction_prompt(query)
            response = await self.simple_query(prompt)

            response_clean = response.strip()

            if "<think>" in response_clean and "</think>" in response_clean:
                response_clean = response_clean.split("</think>", 1)[1].strip()

            if response_clean.startswith("```json"):
                response_clean = response_clean[7:]
            if response_clean.startswith("```"):
                response_clean = response_clean[3:]
            if response_clean.endswith("```"):
                response_clean = response_clean[:-3]
            response_clean = response_clean.strip()

            aspects = json.loads(response_clean)

            if not isinstance(aspects, dict):
                print(f"[ASPECT EXTRACTION] Invalid format (not dict): {aspects}")
                return None

            if "original" not in aspects:
                print(f"[ASPECT EXTRACTION] Missing 'original' key, adding it")
                aspects["original"] = query

            if "aspects" in aspects and isinstance(aspects["aspects"], dict):
                nested_aspects = aspects.pop("aspects")
                aspects.update(nested_aspects)
                print(f"[ASPECT EXTRACTION] Unpacked nested 'aspects' dict")

            invalid_keys = [k for k, v in aspects.items() if not isinstance(v, str)]
            if invalid_keys:
                print(f"[ASPECT EXTRACTION] Invalid non-string values for keys: {invalid_keys}")
                return None

            print(f"[ASPECT EXTRACTION] Extracted {len(aspects)} aspects")
            for aspect_name, aspect_query in aspects.items():
                print(f"  - {aspect_name}: {aspect_query}")

            return aspects

        except json.JSONDecodeError as e:
            print(f"[ASPECT EXTRACTION] JSON parse error: {e}")
            print(f"[ASPECT EXTRACTION] Response was: {response[:200]}...")
            return None
        except Exception as e:
            print(f"[ASPECT EXTRACTION] Unexpected error: {e}")
            return None
