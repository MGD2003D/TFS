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

    async def extract_triplets(self, query: str, context: str) -> List[Dict]:
        """
        Извлекает триплеты из документов с LLM confidence.

        Args:
            query: исходный запрос
            context: текст документов

        Returns:
            [{subject, predicate, object, confidence}, ...] или [] при ошибке
        """
        from prompts_config import build_triplet_extraction_prompt

        try:
            prompt = build_triplet_extraction_prompt(query, context)
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

            result = json.loads(response_clean)

            triplets = result.get("triplets", [])
            if not isinstance(triplets, list):
                return []

            # Валидация и фильтрация
            valid = []
            for t in triplets:
                if not isinstance(t, dict):
                    continue
                if not all(k in t for k in ("subject", "predicate", "object")):
                    continue
                conf = float(t.get("confidence", 0.8))
                if conf < 0.7:
                    continue
                valid.append({
                    "subject": str(t["subject"]),
                    "predicate": str(t["predicate"]),
                    "object": str(t["object"]),
                    "confidence": conf,
                })

            if valid:
                print(f"[TRIPLET EXTRACTION] Extracted {len(valid)} triplets")
                for t in valid:
                    print(f"  [{t['confidence']:.2f}] ({t['subject']}) --[{t['predicate']}]--> ({t['object']})")

            return valid

        except json.JSONDecodeError as e:
            print(f"[TRIPLET EXTRACTION] JSON parse error: {e}")
            return []
        except Exception as e:
            print(f"[TRIPLET EXTRACTION] Error: {e}")
            return []

    async def extract_aspects(self, query: str) -> Optional[Dict]:
        """
        Анализирует сложность запроса и извлекает аспекты/хопы.

        Returns:
            - Simple: {"type": "simple", "original": "query"}
            - Parallel: {"type": "parallel", "original": "query", "aspects": {...}}
            - Sequential: {"type": "sequential", "original": "query", "hops": [...]}
            - None при ошибке
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

            result = json.loads(response_clean)

            if not isinstance(result, dict):
                print(f"[ASPECT EXTRACTION] Invalid format (not dict): {result}")
                return None

            if "original" not in result:
                result["original"] = query

            if "type" not in result:
                result["type"] = "simple"

            query_type = result["type"]

            if query_type == "parallel":
                aspects = result.get("aspects", {})
                if not isinstance(aspects, dict):
                    print(f"[ASPECT EXTRACTION] Invalid aspects format: {aspects}")
                    result["type"] = "simple"
                else:
                    invalid = [k for k, v in aspects.items() if not isinstance(v, str)]
                    if invalid:
                        print(f"[ASPECT EXTRACTION] Non-string aspect values: {invalid}")
                        result["type"] = "simple"

            elif query_type == "sequential":
                hops = result.get("hops", [])
                if not isinstance(hops, list) or len(hops) < 2:
                    print(f"[ASPECT EXTRACTION] Invalid hops (need >=2): {hops}")
                    result["type"] = "simple"
                else:
                    for hop in hops:
                        if not isinstance(hop, dict) or "query" not in hop:
                            print(f"[ASPECT EXTRACTION] Invalid hop format: {hop}")
                            result["type"] = "simple"
                            break

            print(f"[ASPECT EXTRACTION] Type: {result['type']}")
            if result["type"] == "parallel":
                for name, q in result.get("aspects", {}).items():
                    print(f"  aspect '{name}': {q}")
            elif result["type"] == "sequential":
                for i, hop in enumerate(result.get("hops", []), 1):
                    print(f"  hop {i}: {hop.get('query', '?')} → extract: {hop.get('extract', '?')}")

            return result

        except json.JSONDecodeError as e:
            print(f"[ASPECT EXTRACTION] JSON parse error: {e}")
            print(f"[ASPECT EXTRACTION] Response was: {response[:200]}...")
            return None
        except Exception as e:
            print(f"[ASPECT EXTRACTION] Unexpected error: {e}")
            return None
