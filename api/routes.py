from fastapi import APIRouter
from api.v1 import query, rag

api_router = APIRouter()

api_router.include_router(query.router, prefix="/query", tags=["query"])
api_router.include_router(rag.router, prefix="/rag", tags=["rag"])