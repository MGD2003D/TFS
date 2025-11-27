from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.chat_service import ChatService

class QueryRequest(BaseModel):
    prompt: str

class QueryResponse(BaseModel):
    response: str

router = APIRouter()
chat_service = ChatService()

@router.post("/direct")
async def query_direct(req: QueryRequest):
    try:
        response = await chat_service.simple_query(req.prompt)
        return QueryResponse(response=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/{user_id}")
async def chat_query(user_id: str, req: QueryRequest):
    try:
        response = await chat_service.chat_query(user_id, req.prompt)
        return QueryResponse(response=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/chat/{user_id}")
async def chat_clear(user_id: str):
    try:
        chat_service.clear_chat_history(user_id)
        return {"status": "success", "message": f"История чата пользователя {user_id} очищена"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def chats_get():
    return chat_service.get_all_chat_histories()