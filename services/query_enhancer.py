import app_state
import prompts_config
from typing import Dict, List, Optional
import json
import time


class QueryEnhancerService:
    """
    Сервис для улучшения поисковых запросов через LLM.

    Функции:
    - Извлечение сущностей (страны, города, типы туров, даты, бюджет)
    - Переформулирование запроса в поисковый вид
    - Добавление доменных терминов туризма
    - Генерация альтернативных вариантов запроса
    - Анализ сложности запроса (simple / parallel / sequential)
    - Адаптивный выбор стратегии поиска
    """

    # Стратегии поиска
    STRATEGY_BASELINE = "baseline"           # Простой запрос → прямой поиск
    STRATEGY_REFORMULATION = "reformulation" # Переформулирование + альтернативы
    STRATEGY_DECOMPOSITION = "decomposition" # Параллельные аспекты + weighted RRF
    STRATEGY_MULTIHOP = "multihop"           # Последовательные хопы с контекстом

    def __init__(self):
        self.tourism_domains = [
            "тур", "путешествие", "отдых", "поездка", "экскурсия",
            "пляж", "море", "горы", "страна", "город", "отель",
            "виза", "страховка", "трансфер", "перелет", "багаж",
            "питание", "проживание", "маршрут", "гид", "группа"
        ]

    async def enhance_query(self, original_query: str) -> Dict:
        """
        Улучшает поисковый запрос через LLM.

        Returns:
            {
                "original_query": str,
                "intent": str,  # list_tours, tour_info, general_question
                "rewritten_query": str,  # основной переформулированный запрос
                "alternative_queries": List[str],  # альтернативные варианты
                "entities": {
                    "destinations": List[str],  # страны/города
                    "tour_types": List[str],  # типы туров
                    "dates": List[str],  # даты/периоды
                    "budget": Optional[str],  # бюджет
                    "duration": Optional[str],  # длительность
                    "other": Dict  # прочие сущности
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
                    "intent": parsed.get("intent", "tour_info"),
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

    def build_search_queries(self, enhanced: Dict) -> List[str]:
        """
        Строит список запросов для поиска на основе результата enhancement.

        Returns:
            List[str] - список запросов для поиска (основной + альтернативные)
        """
        queries = [enhanced["rewritten_query"]]
        queries.extend(enhanced.get("alternative_queries", []))

        entities = enhanced.get("entities", {})

        destinations = entities.get("destinations", [])
        tour_types = entities.get("tour_types", [])

        if destinations and tour_types:
            for dest in destinations[:2]:
                for tour_type in tour_types[:2]:
                    queries.append(f"{tour_type} тур {dest}")

        key_terms = entities.get("key_terms", [])
        if key_terms and len(key_terms) >= 2:
            queries.append(" ".join(key_terms[:3]))

        return queries[:5]

    async def analyze_complexity(self, query: str) -> Optional[Dict]:
        """
        Анализирует сложность запроса через LLM (aspect extraction).

        Returns:
            - {"type": "simple", "original": "query"}
            - {"type": "parallel", "original": "query", "aspects": {...}}
            - {"type": "sequential", "original": "query", "hops": [...]}
            - None при ошибке (fallback to baseline)
        """
        start = time.perf_counter()
        result = await app_state.llm_client.extract_aspects(query)
        elapsed = time.perf_counter() - start
        print(f"[COMPLEXITY ANALYSIS] Completed in {elapsed:.3f}s")
        return result

    def detect_strategy(self, enhanced: Dict, complexity: Optional[Dict] = None) -> str:
        """
        Определяет оптимальную стратегию поиска на основе результатов enhance и complexity.

        Args:
            enhanced: Результат enhance_query()
            complexity: Результат analyze_complexity() (опционально)

        Returns:
            Одна из стратегий: baseline, reformulation, decomposition, multihop
        """
        intent = enhanced.get("intent", "")

        # Специальные интенты не требуют сложных стратегий
        if intent in ("small_talk", "inappropriate", "off_topic", "list_tours", "filtered_list"):
            return self.STRATEGY_BASELINE

        # Если есть результат анализа сложности — используем его
        if complexity and isinstance(complexity, dict):
            query_type = complexity.get("type", "simple")

            if query_type == "sequential":
                hops = complexity.get("hops", [])
                if len(hops) >= 2:
                    print(f"[STRATEGY] → multihop ({len(hops)} hops)")
                    return self.STRATEGY_MULTIHOP

            if query_type == "parallel":
                aspects = complexity.get("aspects", {})
                if len(aspects) >= 2:
                    print(f"[STRATEGY] → decomposition ({len(aspects)} aspects)")
                    return self.STRATEGY_DECOMPOSITION

        # Если есть альтернативные запросы — reformulation
        alternatives = enhanced.get("alternative_queries", [])
        if alternatives:
            print(f"[STRATEGY] → reformulation ({len(alternatives)} alternatives)")
            return self.STRATEGY_REFORMULATION

        print(f"[STRATEGY] → baseline")
        return self.STRATEGY_BASELINE
