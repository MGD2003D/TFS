from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
from services.rag_service import RAGService


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
rag_service = RAGService(min_relevance=0.25, default_top_k=5)


@router.post("/index", response_model=IndexResponse)
async def index_documents(files: List[UploadFile] = File(...)):
    try:
        files_count, chunks_count = await rag_service.index_documents(files)

        return IndexResponse(
            status="success",
            message=f"Проиндексировано {files_count} файлов",
            chunks_indexed=chunks_count
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=RAGQueryResponse)
async def rag_query(req: RAGQueryRequest):
    try:
        result = await rag_service.query(req.query, top_k=req.top_k)
        return RAGQueryResponse(
            answer=result["answer"],
            sources=result["sources"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/{user_id}", response_model=RAGQueryResponse)
async def rag_chat(user_id: str, req: RAGQueryRequest):
    try:
        result = await rag_service.chat_query(user_id, req.query, top_k=req.top_k)
        return RAGQueryResponse(
            answer=result["answer"],
            sources=result["sources"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/collection")
async def delete_collection():
    try:
        await rag_service.delete_collection()
        return {"status": "success", "message": "Коллекция удалена"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
