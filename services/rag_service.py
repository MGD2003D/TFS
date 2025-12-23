import app_state
from typing import List, Dict, Tuple
from fastapi import UploadFile
import tempfile
import os
import io
import time
from services.document_types import is_supported_document, temp_suffix_for


class RAGService:

    def __init__(self, min_relevance: float = 0.35, default_top_k: int = 8):
        self.min_relevance = min_relevance
        self.default_top_k = default_top_k

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

        search_start = time.perf_counter()
        search_results = await app_state.vector_store.search(query, top_k=top_k)
        search_time = time.perf_counter() - search_start

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
                f"search={search_time:.3f}s "
                f"llm={llm_time:.3f}s "
                f"total={total_time:.3f}s (no_results)"
            )
            return {"answer": answer, "sources": []}

        filter_start = time.perf_counter()
        relevant_results = [doc for doc in search_results if doc['score'] >= self.min_relevance]
        filter_time = time.perf_counter() - filter_start

        context_time = 0.0
        if relevant_results:
            context_start = time.perf_counter()
            context = self._build_context(relevant_results)
            user_message = f"""=== РЕЛЕВАНТНАЯ ИНФОРМАЦИЯ ИЗ БД ===
{context}

=== ВОПРОС ===
{query}"""
            context_time = time.perf_counter() - context_start
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
        return f"""Ты консультант туристического агентства. Используй ТОЛЬКО информацию из документов ниже для ответа.

=== ДОКУМЕНТЫ ===
{context}

=== ВОПРОС КЛИЕНТА ===
{query}

=== ИНСТРУКЦИИ ===
1. Ответь на вопрос, опираясь СТРОГО на информацию из документов выше
2. Структурируй ответ: используй абзацы, списки если нужно
3. Упоминай конкретные детали: цены, даты, места (если есть в документах)
4. Если в документах нет полного ответа, укажи какая информация есть, а какой не хватает
5. Будь дружелюбным и профессиональным
6. НЕ придумывай информацию, которой нет в документах

Твой ответ:"""
