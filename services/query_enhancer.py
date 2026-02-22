import app_state
import prompts_config
from typing import Dict, List, Optional
import json
import re
import time


class QueryEnhancerService:
    """
    Сервис для улучшения поисковых запросов через LLM.

    Функции:
    - Извлечение ключевых сущностей и терминов
    - Переформулирование запроса в поисковый вид
    - Генерация альтернативных вариантов запроса
    """

    def __init__(self):
        pass

    async def enhance_query(self, original_query: str) -> Dict:
        """
        Улучшает поисковый запрос через LLM.

        Returns:
            {
                "original_query": str,
                "intent": str,  # factual, definition, comparison, process, general
                "rewritten_query": str,
                "alternative_queries": List[str],
                "entities": {
                    "key_terms": List[str],
                    "named_entities": List[str],
                    "temporal": List[str],
                    "numerical": List[str]
                }
            }
        """
        start_time = time.perf_counter()

        prompt = self._build_enhancement_prompt(original_query)

        llm_start = time.perf_counter()
        response = await app_state.llm_client.simple_query(prompt)
        llm_time = time.perf_counter() - llm_start

        result = self._parse_llm_response(response, original_query)

        total_time = time.perf_counter() - start_time
        print(f"[Query Enhancement] Completed in {total_time:.3f}s (LLM: {llm_time:.3f}s)")

        return result

    def _build_enhancement_prompt(self, query: str) -> str:
        """Строит промпт для LLM для улучшения запроса используя конфигурацию"""
        return prompts_config.build_query_enhancement_prompt(query)

    def _parse_llm_response(self, response: str, original_query: str) -> Dict:
        """Парсит ответ LLM и возвращает структурированный результат"""
        try:
            response = response.strip()

            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)

                return {
                    "original_query": original_query,
                    "intent": parsed.get("intent", "general"),
                    "rewritten_query": parsed.get("rewritten_query", original_query),
                    "alternative_queries": parsed.get("alternative_queries", []),
                    "entities": parsed.get("entities", {})
                }
            else:
                print(f"[Query Enhancement] WARNING: Failed to find JSON in LLM response")
                return self._fallback_result(original_query)

        except json.JSONDecodeError as e:
            print(f"[Query Enhancement] ERROR: Failed to parse LLM response as JSON: {e}")
            print(f"[Query Enhancement] Response was: {response[:200]}")
            return self._fallback_result(original_query)
        except Exception as e:
            print(f"[Query Enhancement] ERROR: Unexpected error: {e}")
            return self._fallback_result(original_query)

    def _fallback_result(self, original_query: str) -> Dict:
        """Возвращает базовый результат если LLM не смог обработать запрос"""
        return {
            "original_query": original_query,
            "intent": "general",
            "rewritten_query": original_query,
            "alternative_queries": [],
            "entities": {
                "key_terms": [],
                "named_entities": [],
                "temporal": [],
                "numerical": []
            }
        }

    async def decompose_query(self, query: str) -> Optional[Dict[str, str]]:
        """
        Декомпозиция запроса на независимые аспекты для параллельного поиска.

        Returns:
            {"original": query} — простой запрос, использовать baseline.
            {"original": query, "aspect1": q1, ...} — сложный запрос, использовать decomposition.
            None — ошибка парсинга, использовать baseline.
        """
        prompt = prompts_config.build_aspect_extraction_prompt(query)

        try:
            response = await app_state.llm_client.simple_query(prompt)
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
                print(f"[DECOMPOSITION] Invalid format (not dict): {type(aspects)}")
                return None

            if "original" not in aspects:
                aspects["original"] = query

            if "aspects" in aspects and isinstance(aspects["aspects"], dict):
                nested = aspects.pop("aspects")
                aspects.update(nested)

            if any(not isinstance(v, str) for v in aspects.values()):
                print(f"[DECOMPOSITION] Non-string values in aspects, skipping")
                return None

            print(f"[DECOMPOSITION] Extracted {len(aspects)} aspects for: {query[:60]}")
            for name, q in aspects.items():
                print(f"  - {name}: {q}")

            return aspects

        except json.JSONDecodeError as e:
            print(f"[DECOMPOSITION] JSON parse error: {e}")
            return None
        except Exception as e:
            print(f"[DECOMPOSITION] Unexpected error: {e}")
            return None

    def build_search_queries(self, enhanced: Dict) -> List[str]:
        """
        Строит список запросов для поиска на основе результата enhancement.

        Returns:
            List[str] - список запросов для поиска (основной + альтернативные)
        """
        queries = [enhanced["rewritten_query"]]
        queries.extend(enhanced.get("alternative_queries", []))

        entities = enhanced.get("entities", {})
        key_terms = entities.get("key_terms", [])
        if key_terms and len(key_terms) >= 2:
            queries.append(" ".join(key_terms[:3]))

        return queries[:5]
