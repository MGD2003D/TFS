import tempfile
import os
from services.document_types import temp_suffix_for


async def sync_on_startup(minio_storage, vector_store, document_indexer):
    print("\n" + "=" * 60)
    print("Проверка синхронизации MinIO <-> Qdrant")
    print("=" * 60)

    try:
        minio_docs = await minio_storage.list_documents()
        qdrant_docs = await vector_store.get_documents_list()

        print(f"MinIO: {len(minio_docs)} документов")
        print(f"Qdrant: {len(qdrant_docs)} документов")

        minio_map = {doc['document_id']: doc for doc in minio_docs}
        qdrant_map = {doc['document_id']: doc for doc in qdrant_docs}

        # Detect incomplete indexes (indexed_chunks < total_chunks)
        incomplete_ids = {
            doc_id
            for doc_id, doc in qdrant_map.items()
            if doc.get('total_chunks', 0) > 0
            and doc.get('indexed_chunks', 0) != doc.get('total_chunks', 0)
        }
        if incomplete_ids:
            print(f"\n⚠ Обнаружены неполные индексы ({len(incomplete_ids)} документов):")
            for doc_id in incomplete_ids:
                doc = qdrant_map[doc_id]
                print(f"  - {doc.get('source', doc_id)}: "
                      f"{doc.get('indexed_chunks', 0)}/{doc.get('total_chunks', 0)} чанков")
            print("  → Удаляем из Qdrant и переиндексируем...")
            for doc_id in incomplete_ids:
                try:
                    await vector_store.delete_by_document_id(doc_id)
                except Exception as e:
                    print(f"  Не удалось удалить {doc_id}: {e}")
            qdrant_map = {k: v for k, v in qdrant_map.items() if k not in incomplete_ids}

        missing = [doc for doc in minio_docs if doc['document_id'] not in qdrant_map]
        # Re-add incomplete docs that exist in MinIO
        for doc_id in incomplete_ids:
            if doc_id in minio_map:
                missing.append(minio_map[doc_id])
        extra = [doc_id for doc_id in qdrant_map if doc_id not in minio_map]

        print(f"\nСтатус синхронизации:")
        print(f"Синхронизировано: {len(minio_map) - len(missing)} документов")
        print(f"Требуется индексация: {len(missing)} документов")
        print(f"Лишние в Qdrant: {len(extra)} документов")

        if missing:
            print(f"\nИндексация недостающих документов...")
            for idx, doc in enumerate(missing, 1):
                document_id = doc['document_id']
                filename = doc['filename']
                print(f"  [{idx}/{len(missing)}] {filename} (id: {document_id})")

                try:
                    content = await minio_storage.download_document(document_id, filename)

                    tmp = tempfile.NamedTemporaryFile(
                        delete=False, suffix=temp_suffix_for(filename)
                    )
                    tmp.write(content)
                    tmp.close()

                    try:
                        chunks, metadata = await document_indexer.process_document(
                            tmp.name,
                            document_id=document_id,
                            original_filename=filename
                        )
                        await vector_store.add_documents(chunks, metadata)
                        print(f"    ✓ Проиндексировано {len(chunks)} чанков")

                    except Exception as idx_err:
                        import traceback
                        print(f"    ✗ Ошибка индексации: {idx_err}")
                        traceback.print_exc()
                        try:
                            await vector_store.delete_by_document_id(document_id)
                        except Exception:
                            pass

                    finally:
                        if os.path.exists(tmp.name):
                            os.unlink(tmp.name)

                except Exception as e:
                    print(f"    ✗ Ошибка при подготовке документа: {e}")

        if extra:
            print(f"\nУдаление лишних документов из Qdrant...")
            for doc_id in extra:
                try:
                    await vector_store.delete_by_document_id(doc_id)
                    print(f"  Удален: {doc_id}")
                except Exception as e:
                    print(f"  Ошибка при удалении {doc_id}: {e}")

        if not missing and not extra:
            print("\nВсе документы синхронизированы!")
        else:
            print("\nСинхронизация завершена!")

    except Exception as e:
        print(f"\nОшибка при синхронизации: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 60 + "\n")
