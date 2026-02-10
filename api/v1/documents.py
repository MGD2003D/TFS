from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
from services.rag_service import RAGService
import os


class DocumentUploadResponse(BaseModel):
    status: str
    document_id: str
    filename: str
    chunks_indexed: int
    size: int
    uploaded_at: str


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    size: int
    uploaded_at: str | None
    indexed: bool
    chunks_count: int
    warning: str | None = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int


router = APIRouter()

# Флаг для включения/выключения query enhancement
enable_query_enhancement = os.getenv('ENABLE_QUERY_ENHANCEMENT', 'true').lower() == 'true'

rag_service = RAGService(
    min_relevance=0.25,
    default_top_k=5,
    enable_query_enhancement=enable_query_enhancement
)


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    try:
        result = await rag_service.upload_and_index_document(file)

        return DocumentUploadResponse(
            status="success",
            document_id=result["document_id"],
            filename=result["filename"],
            chunks_indexed=result["chunks_indexed"],
            size=result["size"],
            uploaded_at=result["uploaded_at"]
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=DocumentListResponse)
async def list_documents():
    try:
        documents = await rag_service.list_documents()

        return DocumentListResponse(
            documents=documents,
            total=len(documents)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{document_id}")
async def delete_document(document_id: str, filename: str):
    try:
        await rag_service.delete_document(document_id, filename)

        return {
            "status": "success",
            "message": f"Документ {filename} (ID: {document_id}) удален"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{document_id}/replace")
async def replace_document(
    document_id: str,
    old_filename: str,
    new_file: UploadFile = File(...)
):

    try:
        result = await rag_service.replace_document(document_id, old_filename, new_file)

        return {
            "status": "success",
            "message": f"Документ {old_filename} заменен на {new_file.filename}",
            "new_document": result
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
