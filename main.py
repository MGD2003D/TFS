from fastapi import FastAPI
from api.routes import api_router
from contextlib import asynccontextmanager
from services.llm.qwen_client import QwenClient
from services.vectorstore.qdrant_client import QdrantVectorStore
from services.document_indexer import DocumentIndexer
from services.minio_storage import MinioStorageService
from services.minio_event_listener import MinioEventListener
import app_state
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_client, vector_store, document_indexer, minio_storage, minio_event_listener, bot_task

    llm_client = QwenClient()
    app_state.llm_client = llm_client
    await llm_client.initialize()

    vector_store = QdrantVectorStore(
        embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    app_state.vector_store = vector_store
    await vector_store.initialize()

    minio_storage = MinioStorageService(
        endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.getenv("MINIO_ROOT_USER", "admin"),
        secret_key=os.getenv("MINIO_ROOT_PASSWORD", "admin12345678"),
        bucket_name=os.getenv("MINIO_BUCKET_NAME", "documents"),
        secure=os.getenv("MINIO_SECURE", "false").lower() == "true"
    )
    app_state.minio_storage = minio_storage
    await minio_storage.initialize()

    document_indexer = DocumentIndexer(chunk_size=800, chunk_overlap=100)
    app_state.document_indexer = document_indexer

    minio_event_listener = MinioEventListener(minio_storage)
    await minio_event_listener.start()

    bot_token = os.getenv('BOT_TOKEN')
    from tg_bot.bot import dp, bot
    bot_task = asyncio.create_task(dp.start_polling(bot))
    print("Telegram бот запущен")

    yield

    await minio_event_listener.stop()

    if bot_task:
        bot_task.cancel()
        try:
            await bot_task
        except asyncio.CancelledError:
            pass

    await llm_client.cleanup()
    await vector_store.cleanup()
    await minio_storage.cleanup()


app = FastAPI(title="TFS AI API", lifespan=lifespan)

app.include_router(api_router, prefix="/api/v1")