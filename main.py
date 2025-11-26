from fastapi import FastAPI
from api.routes import api_router
from contextlib import asynccontextmanager
from services.llm.qwen_client import QwenClient
from services.vectorstore.qdrant_client import QdrantVectorStore
from services.document_indexer import DocumentIndexer
import app_state

app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_client, vector_store, document_indexer

    llm_client = QwenClient()
    app_state.llm_client = llm_client
    await llm_client.initialize()

    vector_store = QdrantVectorStore(
        embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    app_state.vector_store = vector_store
    await vector_store.initialize()

    document_indexer = DocumentIndexer(chunk_size=800, chunk_overlap=100)
    app_state.document_indexer = document_indexer

    yield

    await llm_client.cleanup()
    await vector_store.cleanup()


app = FastAPI(title="TFS AI API", lifespan=lifespan)

app.include_router(api_router, prefix="/api/v1")