import tempfile
import os
import hashlib


async def sync_on_startup(minio_storage, vector_store, document_indexer):
    print("\n" + "=" * 60)
    print("Проверка синхронизации MinIO <-> Qdrant")
    print("=" * 60)

    try:
        minio_docs = await minio_storage.list_documents()
        qdrant_docs = await vector_store.get_documents_list()

        print(f"MinIO: {len(minio_docs)} документов")
        print(f"Qdrant: {len(qdrant_docs)} документов")

        minio_doc_hashes = {}
        for doc in minio_docs:
            try:
                content = await minio_storage.download_document(
                    doc['document_id'],
                    doc['filename']
                )
                content_hash = hashlib.sha256(content).hexdigest()[:16]
                minio_doc_hashes[content_hash] = {
                    'filename': doc['filename'],
                    'content': content,
                    'document_id': doc['document_id']
                }
            except Exception as e:
                print(f"⚠ Ошибка при загрузке {doc['filename']}: {e}")

        qdrant_doc_ids = {doc['document_id'] for doc in qdrant_docs}

        missing_in_qdrant = []
        for content_hash, doc_info in minio_doc_hashes.items():
            if content_hash not in qdrant_doc_ids:
                missing_in_qdrant.append((content_hash, doc_info))

        extra_in_qdrant = qdrant_doc_ids - set(minio_doc_hashes.keys())

        print(f"\nСтатус синхронизации:")
        print(f"Синхронизировано: {len(qdrant_doc_ids & set(minio_doc_hashes.keys()))} документов")
        print(f"Требуется индексация: {len(missing_in_qdrant)} документов")
        print(f"Лишние в Qdrant: {len(extra_in_qdrant)} документов")

        if missing_in_qdrant:
            print(f"\nИндексация недостающих документов...")
            for idx, (content_hash, doc_info) in enumerate(missing_in_qdrant, 1):
                filename = doc_info['filename']
                content = doc_info['content']

                print(f"  [{idx}/{len(missing_in_qdrant)}] {filename} (hash: {content_hash})")

                try:
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                    temp_file.write(content)
                    temp_file.close()

                    try:
                        chunks, metadata = await document_indexer.process_pdf(
                            temp_file.name,
                            document_id=content_hash
                        )
                        await vector_store.add_documents(chunks, metadata)
                        print(f"\
                              Проиндексировано {len(chunks)} чанков")

                    finally:
                        if os.path.exists(temp_file.name):
                            os.unlink(temp_file.name)

                except Exception as e:
                    print(f"    ✗ Ошибка: {e}")

        if extra_in_qdrant:
            print(f"\n🗑 Удаление лишних документов из Qdrant...")
            for doc_id in extra_in_qdrant:
                try:
                    await vector_store.delete_by_document_id(doc_id)
                    print(f"Удален документ с ID: {doc_id}")
                except Exception as e:
                    print(f"Ошибка при удалении {doc_id}: {e}")

        if not missing_in_qdrant and not extra_in_qdrant:
            print("\nВсе документы синхронизированы!")
        else:
            print("\nСинхронизация завершена!")

    except Exception as e:
        print(f"\nОшибка при синхронизации: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 60 + "\n")
