from fastapi import APIRouter
from api.v1 import query

api_router = APIRouter()

api_router.include_router(query.router, prefix="/query", tags=["query"])