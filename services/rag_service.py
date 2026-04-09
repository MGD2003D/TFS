import app_state
import prompts_config
from typing import List, Dict, Tuple, Optional
from fastapi import UploadFile
import tempfile
import os
import io
import time
import asyncio
from services.document_types import is_supported_document, temp_suffix_for
from services.fusion import weighted_rrf_fusion, aspect_fusion, multihop_merge
from services.query_enhancer import QueryEnhancerService
from tg_bot import custom_emoji as ce


class RAGService:

    def __init__(self, min_relevance: float = 0.35, default_top_k: int = 8, enable_query_enhancement: bool = True):
        self.min_relevance = min_relevance
        self.default_top_k = default_top_k
        self.enable_query_enhancement = enable_query_enhancement

    async def upload_and_index_document(self, file: UploadFile) -> Dict:
        if not is_supported_document(file.filename or ""):
            raise ValueError(f"Unsupported file type: {file.filename}. Only .pdf and .docx are supported.")

        content = await file.read()
        file_obj = io.BytesIO(content)

        minio_result = await app_state.minio_storage.upload_document(file_obj, file.filename)
        document_id = minio_result["document_id"]

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=temp_suffix_for(file.filename or ""))
        temp_file.write(content)
        temp_file.close()

        try:
            chunks, metadata = await app_state.document_indexer.process_document(temp_file.name, document_id=document_id)
            await app_state.vector_store.add_documents(chunks, metadata)

            return {
                "document_id": document_id,
                "filename": file.filename,
                "chunks_indexed": len(chunks),
                "size": minio_result["size"],
                "uploaded_at": minio_result["uploaded_at"]
            }

        finally:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    async def index_documents(self, files: List[UploadFile]) -> Tuple[int, int]:
        """
        УСТАРЕВШИЙ МЕТОД: Индексация PDF документов только в векторное хранилище
        Есть upload_and_index_document для новых загрузок
        """
        temp_files = []

        try:
            for file in files:
                if not is_supported_document(file.filename or ""):
                    raise ValueError(f"Unsupported file type: {file.filename}. Only .pdf and .docx are supported.")

                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=temp_suffix_for(file.filename or ""))
                content = await file.read()
                temp_file.write(content)
                temp_file.close()
                temp_files.append(temp_file.name)

            chunks, metadata = await app_state.document_indexer.process_multiple_documents(temp_files)
            await app_state.vector_store.add_documents(chunks, metadata)

            return len(files), len(chunks)

        finally:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

    async def query(self, query: str, top_k: int = None) -> Dict:
        total_start = time.perf_counter()
        if top_k is None:
            top_k = self.default_top_k

        search_start = time.perf_counter()
        search_results = await app_state.vector_store.search(query, top_k=top_k)
        search_time = time.perf_counter() - search_start

        print(f"\n{'='*80}")
        print(f"[RAG QUERY] Query: {query}")
        print(f"[RAG QUERY] Found {len(search_results)} results")
        if search_results:
            for i, doc in enumerate(search_results, 1):
                print(f"\n--- Result {i} ---")
                print(f"  Score: {doc.get('score', 0):.4f}")
                print(f"  Source: {doc.get('metadata', {}).get('source', 'unknown')}")
                print(f"  Chunk ID: {doc.get('metadata', {}).get('chunk_id', 'N/A')}")
                print(f"  Text preview: {doc.get('text', '')[:150]}...")
        print(f"{'='*80}\n")

        if not search_results:
            total_time = time.perf_counter() - total_start
            print(f"[timing] rag_query: search={search_time:.3f}s total={total_time:.3f}s (no_results)")
            return {
                "answer": "К сожалению, в базе знаний нет информации, которая могла бы помочь ответить на ваш вопрос.",
                "sources": []
            }

        filter_start = time.perf_counter()
        relevant_results = [doc for doc in search_results if doc['score'] >= self.min_relevance]
        filter_time = time.perf_counter() - filter_start

        print(f"[RAG QUERY] Filtered: {len(relevant_results)}/{len(search_results)} results above threshold {self.min_relevance}")

        if not relevant_results:
            total_time = time.perf_counter() - total_start
            print(
                "[timing] rag_query: "
                f"search={search_time:.3f}s "
                f"filter={filter_time:.3f}s "
                f"total={total_time:.3f}s (no_relevant)"
            )
            return {
                "answer": "К сожалению, в базе знаний нет информации, которая могла бы помочь ответить на ваш вопрос.",
                "sources": []
            }

        context_start = time.perf_counter()
        context = self._build_context(relevant_results)
        rag_prompt = self._build_rag_prompt(context, query)
        context_time = time.perf_counter() - context_start

        print(f"\n[RAG QUERY] Context sent to LLM:")
        print(f"{'-'*80}")
        print(context[:500] + "..." if len(context) > 500 else context)
        print(f"{'-'*80}\n")

        llm_start = time.perf_counter()
        answer = await app_state.llm_client.simple_query(rag_prompt)
        llm_time = time.perf_counter() - llm_start

        sources = [{
            "text": doc["text"][:200] + "...",
            "score": doc["score"],
            "metadata": doc["metadata"]
        } for doc in relevant_results]

        total_time = time.perf_counter() - total_start
        print(
            "[timing] rag_query: "
            f"search={search_time:.3f}s "
            f"filter={filter_time:.3f}s "
            f"context={context_time:.3f}s "
            f"llm={llm_time:.3f}s "
            f"total={total_time:.3f}s"
        )
        return {
            "answer": answer,
            "sources": sources
        }

    async def chat_query(self, user_id: str, query: str, top_k: int = None) -> Dict:
        total_start = time.perf_counter()
        if top_k is None:
            top_k = self.default_top_k

        enhancement_time = 0.0
        intent = "tour_info"

        if self.enable_query_enhancement:
            enhancement_start = time.perf_counter()
            enhanced = await app_state.query_enhancer.enhance_query(query)
            enhancement_time = time.perf_counter() - enhancement_start
            intent = enhanced.get("intent", "tour_info")

            print(f"\n{'='*80}")
            print(f"[QUERY ENHANCEMENT]")
            print(f"  Intent: {intent}")
            print(f"  Original: {enhanced['original_query']}")
            print(f"  Rewritten: {enhanced['rewritten_query']}")
            if enhanced.get('alternative_queries'):
                print(f"  Alternatives: {enhanced['alternative_queries']}")
            if enhanced.get('entities'):
                entities = enhanced['entities']
                for key, value in entities.items():
                    if value:
                        print(f"  {key.replace('_', ' ').title()}: {value}")
            print(f"{'='*80}\n")

            if intent == "small_talk":
                return await self._handle_small_talk(user_id, query, enhancement_time, total_start)

            if intent == "inappropriate":
                return await self._handle_inappropriate(user_id, query, enhancement_time, total_start)

            if intent == "off_topic":
                return await self._handle_off_topic(user_id, query, enhancement_time, total_start)

            if intent == "list_tours":
                return await self._handle_list_tours_intent(
                    user_id=user_id,
                    query=query,
                    enhanced=enhanced,
                    enhancement_time=enhancement_time,
                    total_start=total_start
                )

            if intent == "filtered_list":
                entities = enhanced.get('entities', {})
                destinations = entities.get('destinations', [])
                tour_types = entities.get('tour_types', [])

                if destinations or tour_types:
                    tours = app_state.tour_catalog.filter_tours(
                        destinations=destinations,
                        tour_types=tour_types
                    )

                    if len(tours) > 6:
                        print(f"[FILTERED LIST] Found {len(tours)} tours - showing compact list")
                        return await self._handle_list_tours_intent(
                            user_id=user_id,
                            query=query,
                            enhanced=enhanced,
                            enhancement_time=enhancement_time,
                            total_start=total_start
                        )
                    else:
                        print(f"[FILTERED LIST] Found {len(tours)} tours - using RAG with full details")
                else:
                    print(f"[FILTERED LIST] No filters found - using RAG")


            # Adaptive search: анализ сложности + выбор стратегии
            search_start = time.perf_counter()
            search_results, strategy, analysis_time = await self._search_adaptive(
                query=query,
                enhanced=enhanced,
                top_k=top_k,
            )
            search_time = time.perf_counter() - search_start
        else:
            search_start = time.perf_counter()
            search_results = await app_state.vector_store.search(query, top_k=top_k)
            search_time = time.perf_counter() - search_start
            strategy = "baseline"
            analysis_time = 0.0

        print(f"\n{'='*80}")
        print(f"[RAG CHAT] User ID: {user_id}")
        print(f"[RAG CHAT] Original Query: {query}")
        print(f"[RAG CHAT] Strategy: {strategy}")
        print(f"[RAG CHAT] Found {len(search_results)} results")

        if search_results and 'search_type' in search_results[0]:
            search_type_stats = {}
            for doc in search_results:
                stype = doc.get('search_type', 'unknown')
                search_type_stats[stype] = search_type_stats.get(stype, 0) + 1
            print(f"[RAG CHAT] Search types: {search_type_stats}")

        if search_results:
            for i, doc in enumerate(search_results[:5], 1):
                print(f"\n--- Result {i} ---")
                print(f"  Score: {doc.get('score', 0):.4f}")

                if 'dense_score' in doc and 'sparse_score' in doc:
                    print(f"    Dense: {doc.get('dense_score', 0):.4f} | Sparse: {doc.get('sparse_score', 0):.4f} | Combined: {doc.get('combined_score', 0):.4f}")
                    print(f"  Search type: {doc.get('search_type', 'N/A')}")

                print(f"  Source: {doc.get('metadata', {}).get('source', 'unknown')}")
                print(f"  Chunk ID: {doc.get('metadata', {}).get('chunk_id', 'N/A')}")
                if doc.get('covered_aspects'):
                    print(f"  Covered aspects: {doc['covered_aspects']}")
                if doc.get('hop_index') is not None:
                    print(f"  Hop index: {doc['hop_index']}")
                print(f"  Text preview: {doc.get('text', '')[:150]}...")
        print(f"{'='*80}\n")

        if not search_results:
            app_state.add_role_message(user_id, query, role="user")
            history = app_state.get_user_messages(user_id)
            llm_start = time.perf_counter()
            answer = await app_state.llm_client.chat_query(history)
            llm_time = time.perf_counter() - llm_start
            app_state.add_role_message(user_id, answer, role="assistant")
            total_time = time.perf_counter() - total_start
            print(
                "[timing] rag_chat: "
                f"enhancement={enhancement_time:.3f}s "
                f"analysis={analysis_time:.3f}s "
                f"search={search_time:.3f}s "
                f"llm={llm_time:.3f}s "
                f"total={total_time:.3f}s "
                f"strategy={strategy} (no_results)"
            )
            return {"answer": answer, "sources": []}

        filter_start = time.perf_counter()
        # Для RRF-based стратегий порог не применяем (scores несравнимы с cosine)
        if strategy in ("decomposition", "multihop", "reformulation"):
            relevant_results = search_results
        else:
            relevant_results = [doc for doc in search_results if doc['score'] >= self.min_relevance]
        filter_time = time.perf_counter() - filter_start

        print(f"[RAG CHAT] Filtered: {len(relevant_results)}/{len(search_results)} results (strategy={strategy})")

        context_time = 0.0
        if relevant_results:
            context_start = time.perf_counter()
            context = self._build_context(relevant_results)
            user_message = f"""=== РЕЛЕВАНТНАЯ ИНФОРМАЦИЯ ИЗ БД ===
{context}

=== ВОПРОС ===
{query}"""
            context_time = time.perf_counter() - context_start

            print(f"\n[RAG CHAT] Context sent to LLM:")
            print(f"{'-'*80}")
            print(context[:500] + "..." if len(context) > 500 else context)
            print(f"{'-'*80}\n")
        else:
            user_message = query

        app_state.add_role_message(user_id, user_message, role="user")
        history = app_state.get_user_messages(user_id)

        llm_start = time.perf_counter()
        answer = await app_state.llm_client.chat_query(history)
        llm_time = time.perf_counter() - llm_start
        app_state.add_role_message(user_id, answer, role="assistant")

        sources = [{
            "text": doc["text"][:200] + "...",
            "score": doc["score"],
            "metadata": doc["metadata"]
        } for doc in relevant_results] if relevant_results else []

        total_time = time.perf_counter() - total_start
        print(
            "[timing] rag_chat: "
            f"enhancement={enhancement_time:.3f}s "
            f"analysis={analysis_time:.3f}s "
            f"search={search_time:.3f}s "
            f"filter={filter_time:.3f}s "
            f"context={context_time:.3f}s "
            f"llm={llm_time:.3f}s "
            f"total={total_time:.3f}s "
            f"strategy={strategy}"
        )
        return {
            "answer": answer,
            "sources": sources,
            "strategy": strategy
        }

    async def delete_document(self, document_id: str, filename: str) -> None:
        await app_state.vector_store.delete_by_document_id(document_id)

        await app_state.minio_storage.delete_document(document_id, filename)

        print(f"Документ {filename} (ID: {document_id}) полностью удален")

    async def replace_document(self, document_id: str, old_filename: str, new_file: UploadFile) -> Dict:
        await self.delete_document(document_id, old_filename)

        result = await self.upload_and_index_document(new_file)

        print(f"Документ {old_filename} заменен на {new_file.filename}")
        return result

    async def list_documents(self) -> List[Dict]:
        minio_docs = await app_state.minio_storage.list_documents()
        qdrant_docs = await app_state.vector_store.get_documents_list()

        docs_map = {}

        for doc in minio_docs:
            doc_id = doc["document_id"]
            docs_map[doc_id] = {
                "document_id": doc_id,
                "filename": doc["filename"],
                "size": doc["size"],
                "uploaded_at": doc["last_modified"],
                "indexed": False,
                "chunks_count": 0
            }

        for doc in qdrant_docs:
            doc_id = doc["document_id"]
            if doc_id in docs_map:
                docs_map[doc_id]["indexed"] = True
                docs_map[doc_id]["chunks_count"] = doc.get("total_chunks", 0)
            else:
                docs_map[doc_id] = {
                    "document_id": doc_id,
                    "filename": doc.get("source", "unknown"),
                    "size": 0,
                    "uploaded_at": None,
                    "indexed": True,
                    "chunks_count": doc.get("total_chunks", 0),
                    "warning": "Документ проиндексирован, но отсутствует в хранилище"
                }

        return list(docs_map.values())

    async def delete_collection(self) -> None:
        await app_state.vector_store.delete_collection()

    # =========================================================================
    # ADAPTIVE SEARCH STRATEGIES
    # =========================================================================

    async def _extract_and_save_to_graph(
        self,
        query: str,
        docs: List[Dict],
        source_strategy: str = "",
    ) -> None:
        """
        Фоновая задача: извлекает триплеты из документов и сохраняет в граф.

        Вызывается через asyncio.create_task() — fire and forget.
        Если упадёт — не страшно, граф просто не пополнится этим запросом.
        """
        graph = app_state.knowledge_graph
        if not graph or not graph._initialized:
            return
        if not docs:
            return

        try:
            # Берём текст топ-5 документов (не перегружаем контекст LLM)
            context_parts = [doc["text"][:300] for doc in docs[:5]]
            context = "\n---\n".join(context_parts)

            triplets = await app_state.llm_client.extract_triplets(query, context)
            if not triplets:
                return

            for t in triplets:
                t["source_query"] = query
                t["source_strategy"] = source_strategy

            await graph.save_triplets_batch(triplets)

        except Exception as e:
            print(f"[GRAPH BACKGROUND] Error: {e}")

    async def _search_adaptive(
        self,
        query: str,
        enhanced: Dict,
        top_k: int,
    ) -> Tuple[List[Dict], str, float]:
        """
        Адаптивный поиск: анализирует сложность запроса и выбирает стратегию.

        Returns:
            (search_results, strategy_name, analysis_time)
        """
        analysis_start = time.perf_counter()

        # Анализ сложности через LLM
        complexity = await app_state.query_enhancer.analyze_complexity(query)
        analysis_time = time.perf_counter() - analysis_start

        # Определение стратегии
        strategy = app_state.query_enhancer.detect_strategy(enhanced, complexity)

        print(f"\n[ADAPTIVE] Strategy: {strategy} (analysis: {analysis_time:.3f}s)")

        if strategy == QueryEnhancerService.STRATEGY_MULTIHOP:
            results = await self._search_multihop(query, complexity, top_k)
        elif strategy == QueryEnhancerService.STRATEGY_DECOMPOSITION:
            results = await self._search_decomposition(query, complexity, top_k)
        elif strategy == QueryEnhancerService.STRATEGY_REFORMULATION:
            search_queries = app_state.query_enhancer.build_search_queries(enhanced)
            results = await self._search_reformulation(search_queries, top_k, query=query)
        else:
            results = await self._search_baseline(query, top_k)

        return results, strategy, analysis_time

    async def _search_baseline(self, query: str, top_k: int) -> List[Dict]:
        """
        Baseline стратегия: простой поиск + graph lookup для enrichment.
        """
        # Graph lookup перед поиском
        graph_docs = []
        graph = app_state.knowledge_graph
        if graph and graph._initialized:
            graph_docs = await graph.lookup(query, min_score=0.75)

        results = await app_state.vector_store.search(query, top_k=top_k)

        # Фоновое сохранение в граф
        if results:
            asyncio.create_task(
                self._extract_and_save_to_graph(query, results, source_strategy="baseline")
            )

        # Graph hits добавляем в начало — они уже верифицированы
        return graph_docs + results

    async def _search_reformulation(
        self,
        search_queries: List[str],
        top_k: int,
        query: str = "",
    ) -> List[Dict]:
        """
        Reformulation стратегия: поиск по нескольким вариантам запроса + weighted RRF.
        """
        # Graph lookup по оригинальному запросу
        graph_docs = []
        graph = app_state.knowledge_graph
        if graph and graph._initialized and query:
            graph_docs = await graph.lookup(query, min_score=0.75)

        variant_results = []

        for idx, search_query in enumerate(search_queries, 1):
            if len(search_queries) > 1:
                print(f"[REFORMULATION] Query {idx}/{len(search_queries)}: {search_query}")
            results = await app_state.vector_store.search(search_query, top_k=top_k)
            variant_results.append(results)

        if len(variant_results) <= 1:
            fused = variant_results[0] if variant_results else []
        else:
            fused = weighted_rrf_fusion(
                variant_results,
                top_k=top_k,
                original_weight=3.0,
                variant_weight=1.0,
            )

        # Фоновое сохранение
        if fused:
            asyncio.create_task(
                self._extract_and_save_to_graph(query or search_queries[0], fused, source_strategy="reformulation")
            )

        return graph_docs + fused

    async def _search_decomposition(
        self,
        query: str,
        complexity: Dict,
        top_k: int,
    ) -> List[Dict]:
        """
        Decomposition стратегия: параллельный поиск по аспектам + aspect_fusion.
        """
        aspects = complexity.get("aspects", {})
        if not aspects:
            return await self._search_baseline(query, top_k)

        # Graph lookup перед поиском
        graph_docs = []
        graph = app_state.knowledge_graph
        if graph and graph._initialized:
            graph_docs = await graph.lookup(query, min_score=0.75)

        print(f"[DECOMPOSITION] Searching {len(aspects)} aspects in parallel")

        async def search_aspect(name: str, aspect_query: str) -> Tuple[str, List[Dict]]:
            print(f"  → aspect '{name}': {aspect_query}")
            results = await app_state.vector_store.search(aspect_query, top_k=top_k)
            return name, results

        tasks = [search_aspect(name, aq) for name, aq in aspects.items()]
        original_task = app_state.vector_store.search(query, top_k=top_k)

        original_results, *aspect_pairs = await asyncio.gather(
            original_task, *tasks
        )

        aspect_results = {name: results for name, results in aspect_pairs}

        print(f"[DECOMPOSITION] Original: {len(original_results)} results")
        for name, results in aspect_results.items():
            print(f"[DECOMPOSITION] Aspect '{name}': {len(results)} results")

        fused = aspect_fusion(
            original_results=original_results,
            aspect_results=aspect_results,
            top_k=top_k,
            original_weight=3.0,
            aspect_weight=1.0,
            coverage_bonus=0.5,
        )

        # Фоновое сохранение
        if fused:
            asyncio.create_task(
                self._extract_and_save_to_graph(query, fused, source_strategy="decomposition")
            )

        return graph_docs + fused

    async def _search_multihop(
        self,
        query: str,
        complexity: Dict,
        top_k: int,
    ) -> List[Dict]:
        """
        Multi-hop стратегия: последовательные хопы с graph cache.

        1. Перед каждым хопом — проверяем граф (cache hit = skip LLM)
        2. Если cache miss — search + LLM extraction
        3. После extraction — сохраняем в граф для будущих запросов
        """
        hops = complexity.get("hops", [])
        if not hops:
            return await app_state.vector_store.search(query, top_k=top_k)

        graph = app_state.knowledge_graph
        print(f"[MULTIHOP] Executing {len(hops)} sequential hops (graph={'ON' if graph and graph._initialized else 'OFF'})")

        hop_results = []
        prev_context = ""
        resolved_hops = []  # Для сохранения в граф

        for hop_idx, hop in enumerate(hops, 1):
            hop_query_template = hop.get("query", "")
            extract_hint = hop.get("extract", "")

            # Подставляем контекст из предыдущего хопа
            hop_query = hop_query_template
            if prev_context and "{prev}" in hop_query:
                hop_query = hop_query.replace("{prev}", prev_context)

            print(f"  → Hop {hop_idx}: {hop_query}")

            # === GRAPH LOOKUP: пробуем resolve через кэш ===
            graph_hit = None
            if graph and graph._initialized and hop_idx < len(hops):
                graph_hit = await graph.resolve_hop(
                    hop_query=hop_query,
                    extract_hint=extract_hint,
                    min_score=0.75,
                )

            results = await app_state.vector_store.search(hop_query, top_k=top_k)
            hop_results.append(results)
            print(f"    found: {len(results)} results")

            # Извлекаем контекст для следующего хопа
            if hop_idx < len(hops):
                if graph_hit:
                    # Cache hit — используем значение из графа
                    prev_context = graph_hit
                    print(f"    [GRAPH HIT] Using cached: {prev_context[:100]}")
                elif results:
                    # Cache miss — извлекаем через LLM
                    prev_context = await self._extract_hop_context(
                        results=results[:3],
                        extract_hint=extract_hint,
                        hop_query=hop_query,
                    )
                    print(f"    [LLM EXTRACT] {prev_context[:100]}...")

                    # Сохраняем для графа
                    if prev_context and len(prev_context) < 200:
                        resolved_hops.append({
                            "subject": hop_query,
                            "predicate": extract_hint or "resolves_to",
                            "object": prev_context,
                            "source_query": query,
                        })

        # === GRAPH SAVE: сохраняем resolved hops (синхронно, т.к. уже в конце) ===
        if resolved_hops and graph and graph._initialized:
            saved = await graph.save_triplets_batch(resolved_hops)
            if saved:
                print(f"[MULTIHOP] Saved {saved} hop triplets to graph")

        merged = multihop_merge(
            hop_results=hop_results,
            top_k=top_k,
            later_hop_boost=1.5,
        )

        # Фоновое извлечение дополнительных триплетов из всех документов
        all_docs = [doc for hop in hop_results for doc in hop]
        if all_docs and graph and graph._initialized:
            asyncio.create_task(
                self._extract_and_save_to_graph(query, all_docs, source_strategy="multihop")
            )

        return merged

    async def _extract_hop_context(
        self,
        results: List[Dict],
        extract_hint: str,
        hop_query: str,
    ) -> str:
        """
        Извлекает ключевую информацию из результатов хопа для следующего хопа.

        Использует LLM для точного извлечения, с fallback на top-1 текст.
        """
        if not results:
            return ""

        # Собираем текст из top результатов
        texts = [doc['text'][:300] for doc in results[:3]]
        context_text = "\n---\n".join(texts)

        # Просим LLM извлечь нужную информацию
        extraction_prompt = f"""Extract the specific information from the search results below.

SEARCH RESULTS:
{context_text}

WHAT TO EXTRACT: {extract_hint}
ORIGINAL QUESTION: {hop_query}

Return ONLY the extracted value (name, term, fact). Keep it short (1-2 sentences max). No explanations."""

        try:
            extracted = await app_state.llm_client.simple_query(extraction_prompt)
            extracted = extracted.strip()

            # Очистка от think-тегов (Qwen3)
            if "<think>" in extracted and "</think>" in extracted:
                extracted = extracted.split("</think>", 1)[1].strip()

            if extracted and len(extracted) < 200:
                return extracted
        except Exception as e:
            print(f"[MULTIHOP] Extraction failed: {e}")

        # Fallback: берём первые N символов из top-1 результата
        return results[0]['text'][:150]

    def _build_context(self, relevant_results: List[Dict]) -> str:
        context_parts = []
        seen_sources = {}

        for i, doc in enumerate(relevant_results):
            source_info = doc['metadata'].get('source', 'неизвестно')
            chunk_id = doc['metadata'].get('chunk_id', 0)
            score = doc.get('score', 0)

            if source_info not in seen_sources:
                seen_sources[source_info] = []
            seen_sources[source_info].append({
                'text': doc['text'],
                'chunk_id': chunk_id,
                'score': score
            })

        for idx, (source, chunks) in enumerate(seen_sources.items(), 1):
            context_parts.append(f"[Документ {idx}: {source}]")
            for chunk in sorted(chunks, key=lambda x: -x['score']):
                context_parts.append(f"{chunk['text']}\n")

        return "\n".join(context_parts)

    def _build_rag_prompt(self, context: str, query: str) -> str:
        """Строит промпт для RAG запроса используя конфигурацию"""
        return prompts_config.build_rag_prompt(context, query)

    async def _handle_small_talk(self, user_id: str, query: str, enhancement_time: float, total_start: float) -> Dict:
        """
        Обрабатывает small talk: приветствия, благодарности, "как дела" и т.д.
        """
        print(f"[SMALL TALK] User {user_id}: {query}")

        query_lower = query.lower()

        # Приветствия
        if any(word in query_lower for word in ["привет", "здравствуй", "добрый день", "добрый вечер", "доброе утро", "хай", "хэй"]):
            answer = f"{ce.wave()} Привет! Я помогу вам подобрать тур {ce.plane()}. Спросите меня о наших направлениях или конкретных турах!"

        # Как дела
        elif any(phrase in query_lower for phrase in ["как дела", "как ты", "как поживаешь", "что нового"]):
            answer = f"Всё отлично, спасибо! {ce.sparkles()} Готов помочь вам с подбором тура. Какое направление вас интересует?"

        # Благодарности
        elif any(word in query_lower for word in ["спасибо", "благодарю", "thanks", "пасиб"]):
            answer = f"{ce.check()} Рад помочь! Обращайтесь, если будут ещё вопросы по турам."

        # Прощания
        elif any(word in query_lower for word in ["пока", "до свидания", "бай", "досвидос"]):
            answer = f"{ce.wave()} До встречи! Обращайтесь, если понадобится помощь с выбором тура!"

        # Кто ты
        elif any(phrase in query_lower for phrase in ["кто ты", "что ты", "ты кто", "представься"]):
            answer = f"Я ИИ-ассистент туристического агентства TFS {ce.world()}. Помогаю подбирать туры и отвечаю на вопросы о наших направлениях. Чем могу помочь?"

        # Общий small talk
        else:
            answer = f"Я здесь, чтобы помочь вам с выбором тура {ce.plane()}. Спросите меня о направлениях, ценах или конкретных турах!"

        app_state.add_role_message(user_id, query, role="user")
        app_state.add_role_message(user_id, answer, role="assistant")

        total_time = time.perf_counter() - total_start
        print(f"[timing] small_talk: enhancement={enhancement_time:.3f}s total={total_time:.3f}s")

        return {"answer": answer, "sources": []}

    async def _handle_inappropriate(self, user_id: str, query: str, enhancement_time: float, total_start: float) -> Dict:
        """
        Обрабатывает неуместные запросы: грубость, мат, оскорбления.
        """
        print(f"[INAPPROPRIATE] User {user_id} sent inappropriate message")

        answer = "Пожалуйста, общайтесь вежливо. Я здесь, чтобы помочь вам с подбором тура. Чем могу помочь?"

        app_state.add_role_message(user_id, query, role="user")
        app_state.add_role_message(user_id, answer, role="assistant")

        total_time = time.perf_counter() - total_start
        print(f"[timing] inappropriate: enhancement={enhancement_time:.3f}s total={total_time:.3f}s")

        return {"answer": answer, "sources": []}

    async def _handle_off_topic(self, user_id: str, query: str, enhancement_time: float, total_start: float) -> Dict:
        """
        Обрабатывает вопросы не по теме туризма.
        """
        print(f"[OFF TOPIC] User {user_id}: {query}")

        answer = "Извините, я специализируюсь только на туристических вопросах. Могу рассказать про туры, направления, цены и условия. Чем могу помочь?"

        app_state.add_role_message(user_id, query, role="user")
        app_state.add_role_message(user_id, answer, role="assistant")

        total_time = time.perf_counter() - total_start
        print(f"[timing] off_topic: enhancement={enhancement_time:.3f}s total={total_time:.3f}s")

        return {"answer": answer, "sources": []}

    async def _handle_list_tours_intent(
        self,
        user_id: str,
        query: str,
        enhanced: Dict = None,
        enhancement_time: float = 0.0,
        total_start: float = None
    ) -> Dict:

        if total_start is None:
            total_start = time.perf_counter()

        destinations = []
        tour_types = []
        is_filtered = False

        if enhanced and enhanced.get('entities'):
            entities = enhanced['entities']
            destinations = entities.get('destinations', [])
            tour_types = entities.get('tour_types', [])
            is_filtered = bool(destinations or tour_types)

        intent = enhanced.get('intent', 'list_tours') if enhanced else 'list_tours'

        if is_filtered:
            print(f"[FILTERED LIST TOURS] User {user_id} requested filtered catalog")
            print(f"  Destinations: {destinations}")
            print(f"  Tour Types: {tour_types}")
        else:
            print(f"[LIST TOURS] User {user_id} requested full tours catalog")

        if not app_state.tour_catalog or not app_state.tour_catalog.initialized:
            answer = "⏳ Каталог туров еще загружается, пожалуйста подождите немного..."
            app_state.add_role_message(user_id, query, role="user")
            app_state.add_role_message(user_id, answer, role="assistant")

            total_time = time.perf_counter() - total_start
            print(f"[timing] list_tours: total={total_time:.3f}s (catalog_not_ready)")
            return {"answer": answer, "sources": []}

        if is_filtered:
            tours = app_state.tour_catalog.filter_tours(
                destinations=destinations,
                tour_types=tour_types
            )
        else:
            tours = app_state.tour_catalog.get_all_tours()

        if not tours:
            if is_filtered:
                filter_desc = []
                if destinations:
                    filter_desc.append(f"направлениям: {', '.join(destinations)}")
                if tour_types:
                    filter_desc.append(f"типам: {', '.join(tour_types)}")
                filters_str = " и ".join(filter_desc)
                answer = f"К сожалению, я не нашел туров по {filters_str}. Попробуйте изменить критерии поиска или спросите про все доступные туры."
            else:
                answer = "К сожалению, в данный момент в каталоге нет доступных туров."

            app_state.add_role_message(user_id, query, role="user")
            app_state.add_role_message(user_id, answer, role="assistant")

            total_time = time.perf_counter() - total_start
            print(f"[timing] list_tours: enhancement={enhancement_time:.3f}s total={total_time:.3f}s (no_tours)")
            return {"answer": answer, "sources": []}

        has_descriptions = any(tour.get('description') for tour in tours)

        tours_list_lines = []
        for idx, tour in enumerate(tours, 1):
            tour_name = tour.get('tour_name', 'Неизвестный тур')
            description = tour.get('description')

            if has_descriptions and description:
                tours_list_lines.append(f"{idx}. **{tour_name}**")
                tours_list_lines.append(f"   {description}")
                tours_list_lines.append("")
            else:
                tours_list_lines.append(f"{idx}. {tour_name}")

        tours_list = "\n".join(tours_list_lines)

        if is_filtered:
            filter_desc = []
            if destinations:
                filter_desc.append(', '.join(destinations))
            if tour_types:
                filter_desc.append(', '.join(tour_types))
            filters_str = ' - '.join(filter_desc)

            if len(tours) > 6:
                answer = f"""{ce.memo()} Найдено {len(tours)} туров по запросу "{filters_str}":

{tours_list}

Я не могу рассказать обо всех турах в одном сообщении, но вы можете спросить меня о любом из них подробнее! {ce.sparkles()}"""
            else:
                answer = f"""{ce.memo()} Найдено {len(tours)} {'тур' if len(tours) == 1 else 'туров'} по запросу "{filters_str}":

{tours_list}

{ce.sparkles()} Спросите меня о любом из туров, и я расскажу подробности!"""
        else:
            answer = f"""{ce.memo()} У нас есть {len(tours)} туров в каталоге:

{tours_list}

{ce.sparkles()} Спросите меня о любом из туров, и я расскажу подробности!"""

        app_state.add_role_message(user_id, query, role="user")
        app_state.add_role_message(user_id, answer, role="assistant")

        total_time = time.perf_counter() - total_start
        print(f"[timing] list_tours: enhancement={enhancement_time:.3f}s total={total_time:.3f}s (found {len(tours)} tours)")

        return {"answer": answer, "sources": []}
