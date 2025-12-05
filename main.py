from fastapi import FastAPI
from api.routes import api_router
from contextlib import asynccontextmanager
from services.llm.caila_client import CailaClient
from services.vectorstore.qdrant_client import QdrantVectorStore
from services.document_indexer import DocumentIndexer
from services.minio_storage import MinioStorageService
from services.minio_event_listener import MinioEventListener
from services.startup_sync import sync_on_startup
import app_state
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_client, vector_store, document_indexer, minio_storage, minio_event_listener, bot_task

    print("\n=== ИНИЦИАЛИЗАЦИЯ СЕРВИСОВ ===")

    print("1/7 Инициализация LLM клиента...")
    llm_client = CailaClient()
    app_state.llm_client = llm_client
    await llm_client.initialize()
    print("LLM клиент готов")

    print("\n2/7 Инициализация Vector Store...")
    vector_store = QdrantVectorStore(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
        embedding_model="intfloat/multilingual-e5-base"
    )
    app_state.vector_store = vector_store
    await vector_store.initialize()
    print("Vector Store готов")

    print("\n3/7 Инициализация MinIO хранилища...")
    minio_storage = MinioStorageService(
        endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.getenv("MINIO_ROOT_USER", "admin"),
        secret_key=os.getenv("MINIO_ROOT_PASSWORD", "admin12345678"),
        bucket_name=os.getenv("MINIO_BUCKET_NAME", "documents"),
        secure=os.getenv("MINIO_SECURE", "false").lower() == "true"
    )
    app_state.minio_storage = minio_storage
    await minio_storage.initialize()
    print("MinIO хранилище готово")

    print("\n4/7 Инициализация Document Indexer...")
    document_indexer = DocumentIndexer(chunk_size=1000, chunk_overlap=200)
    app_state.document_indexer = document_indexer
    print("Document Indexer готов")

    print("\n5/7 Синхронизация MinIO -> Qdrant...")
    await sync_on_startup(minio_storage, vector_store, document_indexer)
    print("Синхронизация завершена")

    print("\n6/7 Запуск MinIO Event Listener...")
    minio_event_listener = MinioEventListener(minio_storage)
    await minio_event_listener.start()
    print("Event Listener запущен")

    print("\n7/7 Запуск Telegram бота...")
    bot_token = os.getenv('BOT_TOKEN')
    from tg_bot.bot import dp, bot
    bot_task = asyncio.create_task(dp.start_polling(bot))
    print("✓ Telegram бот запущен")

    print("\n=== ВСЕ СЕРВИСЫ ГОТОВЫ ===\n")

    yield

    print("\n=== ОСТАНОВКА СЕРВИСОВ ===")
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
    print("=== СЕРВИСЫ ОСТАНОВЛЕНЫ ===\n")


app = FastAPI(title="TFS AI API", lifespan=lifespan)

app.include_router(api_router, prefix="/api/v1")