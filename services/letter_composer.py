import time
import app_state
import prompts_config
from typing import Dict, List


class LetterComposer:

    async def compose(self, incoming_letter: str, user_id: str) -> Dict:
        total_start = time.perf_counter()

        topic_start = time.perf_counter()
        topic = await self._extract_topic(incoming_letter)
        topic_time = time.perf_counter() - topic_start
        print(f"[COMPOSER] Topic: {topic[:120]}")

        search_start = time.perf_counter()
        import asyncio
        template_results, law_results = await asyncio.gather(
            self._search_templates(topic, top_k=3),
            self._search_laws(topic, top_k=5),
        )
        search_time = time.perf_counter() - search_start
        print(f"[COMPOSER] Templates found: {len(template_results)}, Laws found: {len(law_results)}")

        llm_start = time.perf_counter()
        examples_block = self._build_examples_block(template_results)
        legal_block = self._build_legal_block(law_results)

        prompt = prompts_config.build_letter_composition_prompt(
            incoming=incoming_letter,
            examples=examples_block,
            legal_context=legal_block,
        )
        draft = await app_state.llm_client.simple_query(prompt)
        llm_time = time.perf_counter() - llm_start

        total_time = time.perf_counter() - total_start
        print(
            f"[timing] compose: topic={topic_time:.2f}s search={search_time:.2f}s "
            f"llm={llm_time:.2f}s total={total_time:.2f}s"
        )

        return {
            "draft": draft,
            "template_sources": [r.get("metadata", {}).get("source", "") for r in template_results],
            "law_sources": [r.get("metadata", {}).get("source", "") for r in law_results],
        }

    async def _extract_topic(self, letter: str) -> str:
        prompt = prompts_config.build_topic_extraction_prompt(letter)
        topic = await app_state.llm_client.simple_query(prompt)
        return topic.strip()

    async def _search_templates(self, topic: str, top_k: int = 3) -> List[Dict]:
        if app_state.vector_store_templates is None:
            return []
        try:
            return await app_state.vector_store_templates.search(topic, top_k=top_k)
        except Exception as e:
            print(f"[COMPOSER] Ошибка поиска шаблонов: {e}")
            return []

    async def _search_laws(self, topic: str, top_k: int = 5) -> List[Dict]:
        if app_state.vector_store is None:
            return []
        try:
            return await app_state.vector_store.search(topic, top_k=top_k)
        except Exception as e:
            print(f"[COMPOSER] Ошибка поиска нормативки: {e}")
            return []

    def _build_examples_block(self, results: List[Dict]) -> str:
        if not results:
            return "Похожих примеров не найдено."
        parts = []
        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            request_text = r.get("text", "")[:500]
            response_text = meta.get("response", "")[:800]
            parts.append(
                f"[Пример {i}]\n"
                f"Обращение: {request_text}...\n"
                f"Ответ: {response_text}..."
            )
        return "\n\n".join(parts)

    def _build_legal_block(self, results: List[Dict]) -> str:
        if not results:
            return "Нормативная база не найдена."
        parts = []
        for i, r in enumerate(results, 1):
            source = r.get("metadata", {}).get("source", "неизвестно")
            text = r.get("text", "")[:600]
            parts.append(f"[{i}. {source}]\n{text}")
        return "\n\n".join(parts)
