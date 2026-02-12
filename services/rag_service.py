import app_state
import prompts_config
from typing import List, Dict, Tuple
from fastapi import UploadFile
import tempfile
import os
import io
import time
from services.document_types import is_supported_document, temp_suffix_for
from tg_bot import custom_emoji as ce


class RAGService:

    def __init__(self, min_relevance: float = 0.35, default_top_k: int = 8, enable_query_enhancement: bool = True):
        self.min_relevance = min_relevance
        self.default_top_k = default_top_k
        self.enable_query_enhancement = enable_query_enhancement

    def _resolve_namespace(self, user_id: str, is_corporate: bool) -> str:

        return "corporate" if is_corporate else f"user_{user_id}"

    def _resolve_namespaces(self, user_id: str, scope: str) -> List[str]:

        if scope == "personal":
            return [f"user_{user_id}"]
        elif scope == "corporate":
            return ["corporate"]
        elif scope == "personal_corporate":
            return [f"user_{user_id}", "corporate"]
        else:
            return [f"user_{user_id}", "corporate"]

    async def upload_and_index_document(
        self,
        file: UploadFile,
        user_id: str = None,
        is_corporate: bool = False
    ) -> Dict:

        if not is_supported_document(file.filename or ""):
            raise ValueError(f"Unsupported file type: {file.filename}. Only .pdf and .docx are supported.")

        content = await file.read()
        file_obj = io.BytesIO(content)

        namespace = self._resolve_namespace(user_id, is_corporate)

        minio_metadata = {
            "user_id": user_id,
            "is_corporate": str(is_corporate),
            "uploaded_by": user_id
        }

        minio_result = await app_state.minio_storage.upload_document(
            file_obj,
            file.filename,
            namespace=namespace,
            metadata=minio_metadata
        )
        document_id = minio_result["document_id"]

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=temp_suffix_for(file.filename or ""))
        temp_file.write(content)
        temp_file.close()

        try:
            chunks, metadata = await app_state.document_indexer.process_document(temp_file.name, document_id=document_id)

            namespace = self._resolve_namespace(user_id, is_corporate)

            from datetime import datetime
            for meta in metadata:
                meta["user_id"] = user_id
                meta["is_corporate"] = is_corporate
                meta["namespace"] = namespace
                meta["uploaded_by"] = user_id
                if "uploaded_at" not in meta:
                    meta["uploaded_at"] = datetime.now().isoformat()

            await app_state.vector_store.add_documents(
                chunks,
                metadata,
                namespace=namespace
            )

            return {
                "document_id": document_id,
                "filename": file.filename,
                "chunks_indexed": len(chunks),
                "size": minio_result["size"],
                "bucket_name": minio_result["bucket_name"],
                "namespace": minio_result["namespace"],
                "is_corporate": is_corporate,
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

    async def query(self, query: str, top_k: int = None, namespaces: List[str] = None) -> Dict:

        total_start = time.perf_counter()
        if top_k is None:
            top_k = self.default_top_k

        search_start = time.perf_counter()
        search_results = await app_state.vector_store.search(query, top_k=top_k, namespaces=namespaces)
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

    async def chat_query(
        self,
        user_id: str,
        query: str,
        top_k: int = None,
        scope: str = "personal_corporate"
    ) -> Dict:

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


            search_queries = app_state.query_enhancer.build_search_queries(enhanced)
        else:
            search_queries = [query]

        namespaces = self._resolve_namespaces(user_id, scope)

        search_start = time.perf_counter()
        all_results = []
        seen_texts = set()

        for idx, search_query in enumerate(search_queries, 1):
            if len(search_queries) > 1:
                print(f"[MULTI-QUERY SEARCH] Query {idx}/{len(search_queries)}: {search_query}")
            results = await app_state.vector_store.search(
                search_query,
                top_k=top_k,
                namespaces=namespaces
            )

            for result in results:
                text_key = result['text'][:100]
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    result['search_query'] = search_query
                    all_results.append(result)

        search_results = sorted(all_results, key=lambda x: x['score'], reverse=True)[:top_k]
        search_time = time.perf_counter() - search_start

        print(f"\n{'='*80}")
        print(f"[RAG CHAT] User ID: {user_id}")
        print(f"[RAG CHAT] Original Query: {query}")
        print(f"[RAG CHAT] Query Enhancement: {'ENABLED' if self.enable_query_enhancement else 'DISABLED'}")
        if self.enable_query_enhancement:
            print(f"[RAG CHAT] Searched {len(search_queries)} query variations")
        print(f"[RAG CHAT] Found {len(search_results)} unique results (after deduplication)")

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
                if self.enable_query_enhancement:
                    print(f"  Found by query: {doc.get('search_query', 'N/A')}")
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
                f"search={search_time:.3f}s "
                f"llm={llm_time:.3f}s "
                f"total={total_time:.3f}s (no_results)"
            )
            return {"answer": answer, "sources": []}

        filter_start = time.perf_counter()
        relevant_results = [doc for doc in search_results if doc['score'] >= self.min_relevance]
        filter_time = time.perf_counter() - filter_start

        print(f"[RAG CHAT] Filtered: {len(relevant_results)}/{len(search_results)} results above threshold {self.min_relevance}")

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

    async def delete_document(self, document_id: str, filename: str, user_id: str) -> None:

        from qdrant_client import models

        scroll_result = app_state.vector_store.client.scroll(
            collection_name=app_state.vector_store.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id)
                    )
                ]
            ),
            limit=1,
            with_payload=True
        )

        if not scroll_result[0]:
            raise ValueError(f"Document {document_id} not found")

        doc_metadata = scroll_result[0][0].payload

        if doc_metadata.get("is_corporate"):
            pass
        elif doc_metadata.get("user_id") != user_id:
            raise PermissionError("Cannot delete other users' personal documents")

        await app_state.vector_store.delete_by_document_id(document_id)

        bucket_name = (
            "documents-corporate" if doc_metadata.get("is_corporate")
            else f"documents-user-{doc_metadata.get('user_id')}"
        )

        await app_state.minio_storage.delete_document(document_id, filename, bucket_name)

        print(f"Документ {filename} (ID: {document_id}) полностью удален [bucket: {bucket_name}]")

    async def replace_document(self, document_id: str, old_filename: str, new_file: UploadFile) -> Dict:
        await self.delete_document(document_id, old_filename)

        result = await self.upload_and_index_document(new_file)

        print(f"Документ {old_filename} заменен на {new_file.filename}")
        return result

    async def list_documents(
        self,
        user_id: str = None,
        scope: str = "personal_corporate"
    ) -> List[Dict]:

        namespaces = self._resolve_namespaces(user_id, scope) if user_id else None

        minio_docs = await app_state.minio_storage.list_documents(namespaces=namespaces)
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
            for chunk in sorted(chunks, key=lambda x: x['chunk_id']):
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

        if any(word in query_lower for word in ["привет", "здравствуй", "добрый день", "добрый вечер", "доброе утро", "хай", "хэй"]):
            answer = f"{ce.wave()} Привет! Я помогу вам подобрать тур {ce.plane()}. Спросите меня о наших направлениях или конкретных турах!"

        elif any(phrase in query_lower for phrase in ["как дела", "как ты", "как поживаешь", "что нового"]):
            answer = f"Всё отлично, спасибо! {ce.sparkles()} Готов помочь вам с подбором тура. Какое направление вас интересует?"

        elif any(word in query_lower for word in ["спасибо", "благодарю", "thanks", "пасиб"]):
            answer = f"{ce.check()} Рад помочь! Обращайтесь, если будут ещё вопросы по турам."

        elif any(word in query_lower for word in ["пока", "до свидания", "бай", "досвидос"]):
            answer = f"{ce.wave()} До встречи! Обращайтесь, если понадобится помощь с выбором тура!"

        elif any(phrase in query_lower for phrase in ["кто ты", "что ты", "ты кто", "представься"]):
            answer = f"Я ИИ-ассистент туристического агентства TFS {ce.world()}. Помогаю подбирать туры и отвечаю на вопросы о наших направлениях. Чем могу помочь?"

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
