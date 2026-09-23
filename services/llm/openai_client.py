import aiohttp
import os
import json
from .base import BaseLLMClient
from typing import List, Dict, Optional
from app_state import system_prompt


class OpenAICompatibleClient(BaseLLMClient):
    """
    Клиент для любого OpenAI-совместимого сервера (llama.cpp, vLLM, Ollama, LM Studio и т.п.).

    Отправляет POST {api_url} с телом {"model", "messages", ...} и Bearer-токеном.
    Для thinking-моделей (Qwen3 и др.) по умолчанию отключает режим размышлений —
    иначе ответ уходит в reasoning_content, а content приходит пустым.
    """

    def __init__(
        self,
        api_url: str,
        model: str,
        token: Optional[str] = None,
        enable_thinking: bool = False,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: int = 300,
    ):
        self.api_url = api_url
        self.model = model
        self.token = token
        self.enable_thinking = enable_thinking
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.session = None

    async def initialize(self):
        """Инициализация HTTP сессии"""
        print(f"Инициализация LLM клиента (OpenAI-compatible)")
        print(f"API URL: {self.api_url}")
        print(f"Модель: {self.model}")
        print(f"Thinking: {'включён' if self.enable_thinking else 'выключен'}")
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        print("LLM клиент готов к работе")

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
            print("LLM клиент закрыт")

    async def _generate(self, messages: List[Dict[str, str]]) -> str:
        """Отправка запроса к OpenAI-совместимому API"""
        if not self.session:
            raise RuntimeError("Клиент не инициализирован. Вызовите initialize() сначала.")

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            # llama.cpp / vLLM: управление thinking-режимом через chat template (Qwen3 и др.)
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        try:
            async with self.session.post(self.api_url, json=payload, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()

                if "choices" in data and len(data["choices"]) > 0:
                    message = data["choices"][0].get("message", {})
                    content = message.get("content") or ""
                    # Некоторые серверы кладут размышления прямо в текст ответа
                    if "<think>" in content and "</think>" in content:
                        content = content.split("</think>", 1)[1].strip()
                    if not content and message.get("reasoning_content"):
                        # Модель потратила весь лимит на размышления и не дошла до ответа
                        print("[LLM] content пустой, ответ обрезан в reasoning_content "
                              f"(finish_reason={data['choices'][0].get('finish_reason')})")
                    return content
                elif "response" in data:
                    return data["response"]
                elif "text" in data:
                    return data["text"]
                else:
                    return str(data)

        except aiohttp.ClientError as e:
            print(f"Ошибка при обращении к LLM API: {e}")
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
