import asyncio
import os
import tempfile
from dotenv import load_dotenv
from services.minio_storage import MinioStorageService
from services.vectorstore.qdrant_client import QdrantVectorStore
from services.document_indexer import DocumentIndexer

load_dotenv()


async def sync_minio_to_qdrant():
    """Синхронизация всех документов из MinIO в Qdrant"""

    print("=" * 60)
    print("MinIO Qdrant Синхронизация")
    print("=" * 60)

    print("\n[1/4] Инициализация сервисов...")

    minio_storage = MinioStorageService(
        endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.getenv("MINIO_ROOT_USER", "admin"),
        secret_key=os.getenv("MINIO_ROOT_PASSWORD", "admin12345678"),
        bucket_name=os.getenv("MINIO_BUCKET_NAME", "documents"),
        secure=os.getenv("MINIO_SECURE", "false").lower() == "true"
    )
    await minio_storage.initialize()

    vector_store = QdrantVectorStore(
        embedding_model="intfloat/multilingual-e5-base"
    )
    await vector_store.initialize()

    document_indexer = DocumentIndexer(chunk_size=1000, chunk_overlap=200)

    print("Сервисы инициализированы")

    print("\n[2/4] Получение списков документов...")

    minio_docs = await minio_storage.list_documents()
    qdrant_docs = await vector_store.get_documents_list()

    print(f"В MinIO: {len(minio_docs)} документов")
    print(f"В Qdrant: {len(qdrant_docs)} документов")

    print("\n[3/4] Анализ синхронизации...")

    qdrant_doc_ids = {doc['document_id'] for doc in qdrant_docs}
    minio_doc_ids = {doc['document_id'] for doc in minio_docs}

    missing_in_qdrant = []
    for doc in minio_docs:
        if doc['document_id'] not in qdrant_doc_ids:
            missing_in_qdrant.append(doc)

    missing_in_minio = qdrant_doc_ids - minio_doc_ids

    print(f"\nСтатус синхронизации:")
    print(f"Синхронизировано: {len(qdrant_doc_ids & minio_doc_ids)} документов")
    print(f"Требуется индексация: {len(missing_in_qdrant)} документов")
    print(f"Только в Qdrant (аномалия): {len(missing_in_minio)} документов")

    if not missing_in_qdrant and not missing_in_minio:
        print("\nВсе документы синхронизированы!")
        await vector_store.cleanup()
        await minio_storage.cleanup()
        return

    if missing_in_qdrant:
        print(f"\n[4/4] Индексация {len(missing_in_qdrant)} документов...")

        for idx, doc in enumerate(missing_in_qdrant, 1):
            document_id = doc['document_id']
            filename = doc['filename']

            print(f"\n[{idx}/{len(missing_in_qdrant)}] {filename} (ID: {document_id})")

            try:
                content = await minio_storage.download_document(document_id, filename)

                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                temp_file.write(content)
                temp_file.close()

                try:
                    chunks, metadata = await document_indexer.process_pdf(
                        temp_file.name,
                        document_id=document_id
                    )
                    await vector_store.add_documents(chunks, metadata)

                    print(f"Проиндексировано {len(chunks)} чанков")

                finally:
                    if os.path.exists(temp_file.name):
                        os.unlink(temp_file.name)

            except Exception as e:
                print(f"Ошибка: {e}")
                import traceback
                traceback.print_exc()

    if missing_in_minio:
        print(f"\nВнимание! Найдены документы только в Qdrant (не в MinIO):")
        for doc_id in list(missing_in_minio)[:10]:
            print(f"  - {doc_id}")
        if len(missing_in_minio) > 10:
            print(f"  ... и еще {len(missing_in_minio) - 10}")

        print("\nРекомендация: удалите эти документы из Qdrant или восстановите в MinIO")

        # response = input("\nУдалить эти документы из Qdrant? (yes/no): ")
        # if response.lower() == 'yes':
        #     for doc_id in missing_in_minio:
        #         await vector_store.delete_by_document_id(doc_id)
        #     print("Документы удалены из Qdrant")

    await vector_store.cleanup()
    await minio_storage.cleanup()

    print("\n" + "=" * 60)
    print("Синхронизация завершена!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(sync_minio_to_qdrant())
