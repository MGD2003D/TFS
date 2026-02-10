from fastapi import FastAPI
from api.routes import api_router
from contextlib import asynccontextmanager
from services.llm.caila_client import CailaClient
from services.vectorstore.qdrant_client import QdrantVectorStore
from services.document_indexer import DocumentIndexer
from services.minio_storage import MinioStorageService
from services.minio_event_listener import MinioEventListener
from services.startup_sync import sync_on_startup
from services.chat_cleanup import ChatCleanupWorker
from services.query_enhancer import QueryEnhancerService
from services.tour_catalog import TourCatalogService
import app_state
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_client, vector_store, document_indexer, minio_storage, minio_event_listener, bot_task, chat_cleanup_worker, query_enhancer

    print("\n=== ИНИЦИАЛИЗАЦИЯ СЕРВИСОВ ===")

    print("1/8 Инициализация LLM клиента...")
    llm_client = CailaClient()
    app_state.llm_client = llm_client
    await llm_client.initialize()
    print("LLM клиент готов")

    print("\n2/8 Инициализация Vector Store...")
    enable_hybrid_search = os.getenv('ENABLE_HYBRID_SEARCH', 'true').lower() == 'true'
    vector_store = QdrantVectorStore(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
        embedding_model="intfloat/multilingual-e5-base",
        enable_hybrid_search=enable_hybrid_search
    )
    app_state.vector_store = vector_store
    await vector_store.initialize()
    if enable_hybrid_search:
        print("Vector Store готов (Hybrid Search: Dense + Sparse)")
    else:
        print("Vector Store готов (только Dense vectors)")

    print("\n3/8 Инициализация MinIO хранилища...")
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

    print("\n4/8 Инициализация Document Indexer...")
    document_indexer = DocumentIndexer(chunk_size=1000, chunk_overlap=200)
    app_state.document_indexer = document_indexer
    print("Document Indexer готов")

    print("\n4.5/8 Инициализация Query Enhancer...")
    query_enhancer = QueryEnhancerService()
    app_state.query_enhancer = query_enhancer
    print("Query Enhancer готов")

    print("\n5/8 Starting chat cleanup worker...")
    chat_cleanup_worker = ChatCleanupWorker(
        ttl_seconds=int(os.getenv("CHAT_TTL_SECONDS", "3600")),
        interval_seconds=int(os.getenv("CHAT_CLEANUP_INTERVAL_SECONDS", "60"))
    )
    await chat_cleanup_worker.start()
    print("Chat cleanup worker started")

    print("\n6/8 Синхронизация MinIO -> Qdrant...")
    await sync_on_startup(minio_storage, vector_store, document_indexer)
    print("Синхронизация завершена")

    print("\n6.5/8 Построение каталога туров...")
    generate_tour_descriptions = os.getenv('GENERATE_TOUR_DESCRIPTIONS', 'false').lower() == 'true'
    tour_catalog = TourCatalogService(generate_descriptions=generate_tour_descriptions)
    app_state.tour_catalog = tour_catalog
    await tour_catalog.build_catalog(minio_storage, vector_store)
    if generate_tour_descriptions:
        print("Каталог туров готов (с описаниями через LLM)")
    else:
        print("Каталог туров готов (без описаний)")

    print("\n7/8 Запуск MinIO Event Listener...")
    minio_event_listener = MinioEventListener(minio_storage)
    await minio_event_listener.start()
    print("Event Listener запущен")

    print("\n8/8 Запуск Telegram бота...")
    bot_token = os.getenv('BOT_TOKEN')

    from tg_bot.bot import dp, bot, initialize_services

    initialize_services()

    app_state.services_ready = True
    print("Все сервисы инициализированы и готовы к работе")

    bot_task = asyncio.create_task(dp.start_polling(bot))
    print("Telegram бот запущен")

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

    if chat_cleanup_worker:
        await chat_cleanup_worker.stop()

    await llm_client.cleanup()
    await vector_store.cleanup()
    await minio_storage.cleanup()
    print("=== СЕРВИСЫ ОСТАНОВЛЕНЫ ===\n")


app = FastAPI(title="TFS AI API", lifespan=lifespan)

app.include_router(api_router, prefix="/api/v1")
