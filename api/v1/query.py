from fastapi import APIRouter, HTTPException
import app_state
from pydantic import BaseModel

class QueryRequest(BaseModel):
    prompt: str

class QueryResponse(BaseModel):
    response: str

router = APIRouter()

@router.post("/direct")
async def query_direct(req: QueryRequest):
    try:
        response = await app_state.llm_client.simple_query(req.prompt)
        return QueryResponse(response=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))