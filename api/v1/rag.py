from fastapi import APIRouter, HTTPException, UploadFile, File
import app_state
from pydantic import BaseModel
from typing import List
import tempfile
import os


class RAGQueryRequest(BaseModel):
    query: str
    top_k: int = 5


class RAGQueryResponse(BaseModel):
    answer: str
    sources: List[dict]


class IndexResponse(BaseModel):
    status: str
    message: str
    chunks_indexed: int


router = APIRouter()


@router.post("/index", response_model=IndexResponse)
async def index_documents(files: List[UploadFile] = File(...)):
    try:
        temp_files = []

        for file in files:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"Файл {file.filename} не является PDF")

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            content = await file.read()
            temp_file.write(content)
            temp_file.close()
            temp_files.append(temp_file.name)

        chunks, metadata = await app_state.document_indexer.process_multiple_pdfs(temp_files)
        await app_state.vector_store.add_documents(chunks, metadata)

        for temp_file in temp_files:
            os.unlink(temp_file)

        return IndexResponse(
            status="success",
            message=f"Проиндексировано {len(files)} файлов",
            chunks_indexed=len(chunks)
        )

    except Exception as e:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=RAGQueryResponse)
async def rag_query(req: RAGQueryRequest):
    try:
        search_results = await app_state.vector_store.search(req.query, top_k=req.top_k)

        if not search_results:
            raise HTTPException(status_code=404, detail="Релевантные документы не найдены")

        MIN_RELEVANCE = 0.25
        relevant_results = [doc for doc in search_results if doc['score'] >= MIN_RELEVANCE]

        if not relevant_results:
            return RAGQueryResponse(
                answer="К сожалению, в базе знаний нет информации, которая могла бы помочь ответить на ваш вопрос.",
                sources=[]
            )

        context_parts = []
        for i, doc in enumerate(relevant_results):
            source_info = doc['metadata'].get('source', 'неизвестно')
            context_parts.append(f"[Источник {i+1} - {source_info}]:\n{doc['text']}")

        context = "\n\n".join(context_parts)

        rag_prompt = f"""Ты консультант туристического агентства. Используй ТОЛЬКО информацию из документов ниже для ответа.

=== ДОКУМЕНТЫ ===
{context}

=== ВОПРОС КЛИЕНТА ===
{req.query}

=== ИНСТРУКЦИИ ===
1. Ответь на вопрос, опираясь СТРОГО на информацию из документов выше
2. Структурируй ответ: используй абзацы, списки если нужно
3. Упоминай конкретные детали: цены, даты, места (если есть в документах)
4. Если в документах нет полного ответа, укажи какая информация есть, а какой не хватает
5. Будь дружелюбным и профессиональным
6. НЕ придумывай информацию, которой нет в документах

Твой ответ:"""

        answer = await app_state.llm_client.simple_query(rag_prompt)

        return RAGQueryResponse(
            answer=answer,
            sources=[{
                "text": doc["text"][:200] + "...",
                "score": doc["score"],
                "metadata": doc["metadata"]
            } for doc in relevant_results]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/{user_id}", response_model=RAGQueryResponse)
async def rag_chat(user_id: str, req: RAGQueryRequest):
    try:
        search_results = await app_state.vector_store.search(req.query, top_k=req.top_k)

        if not search_results:
            app_state.add_role_message(user_id, req.query, role="user")
            history = app_state.get_user_messages(user_id)
            answer = await app_state.llm_client.chat_query(history)
            app_state.add_role_message(user_id, answer, role="assistant")
            return RAGQueryResponse(answer=answer, sources=[])

        MIN_RELEVANCE = 0.25
        relevant_results = [doc for doc in search_results if doc['score'] >= MIN_RELEVANCE]

        if relevant_results:
            context_parts = []
            for i, doc in enumerate(relevant_results):
                source_info = doc['metadata'].get('source', 'неизвестно')
                context_parts.append(f"[Источник {i+1} - {source_info}]:\n{doc['text']}")
            context = "\n\n".join(context_parts)

            user_message = f"""=== РЕЛЕВАНТНАЯ ИНФОРМАЦИЯ ИЗ БД ===
{context}

=== ВОПРОС ===
{req.query}"""
        else:
            user_message = req.query

        app_state.add_role_message(user_id, user_message, role="user")
        history = app_state.get_user_messages(user_id)

        answer = await app_state.llm_client.chat_query(history)
        app_state.add_role_message(user_id, answer, role="assistant")

        return RAGQueryResponse(
            answer=answer,
            sources=[{
                "text": doc["text"][:200] + "...",
                "score": doc["score"],
                "metadata": doc["metadata"]
            } for doc in relevant_results] if relevant_results else []
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/collection")
async def delete_collection():
    try:
        await app_state.vector_store.delete_collection()
        return {"status": "success", "message": "Коллекция удалена"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
