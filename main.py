from fastapi import FastAPI
from api.routes import api_router
from contextlib import asynccontextmanager
from services.llm.qwen_client import QwenClient
import app_state

app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):

    global llm_client
    
    llm_client = QwenClient()
    app_state.llm_client = llm_client
    await llm_client.initialize()
    
    yield
    
    await llm_client.cleanup()


app = FastAPI(title="TFS AI API", lifespan=lifespan)

app.include_router(api_router, prefix="/api/v1")