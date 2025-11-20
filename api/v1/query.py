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
    
@router.post("/chat/{user_id}")
async def chat_query(user_id: str, req: QueryRequest):
    try:

        app_state.add_role_message(user_id, req.prompt, role="user")
        history = app_state.get_user_messages(user_id)
        print(history)
        response = await app_state.llm_client.chat_query(history)
        app_state.add_role_message(user_id, response, role="assistant")
        return QueryResponse(response=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.delete("/chat/{user_id}")
async def chat_clear(user_id):
    try:
        app_state.delete_user_history(user_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/history")
async def chats_get():

    return app_state.chat_histories