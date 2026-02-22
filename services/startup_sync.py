import tempfile
import os
import hashlib
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

        qdrant_doc_ids = {doc['document_id'] for doc in qdrant_docs}

        # Check indexed docs for chunk completeness
        incomplete_in_qdrant = {
            doc['document_id']
            for doc in qdrant_docs
            if doc.get('total_chunks', 0) > 0
            and doc.get('indexed_chunks', 0) != doc.get('total_chunks', 0)
        }
        if incomplete_in_qdrant:
            print(f"\n⚠ Обнаружены неполные индексы ({len(incomplete_in_qdrant)} документов):")
            for doc in qdrant_docs:
                if doc['document_id'] in incomplete_in_qdrant:
                    print(f"  - {doc['source']}: {doc.get('indexed_chunks', 0)}/{doc.get('total_chunks', 0)} чанков")
            print("  → Удаляем из Qdrant и переиндексируем...")
            for doc_id in incomplete_in_qdrant:
                try:
                    await vector_store.delete_by_document_id(doc_id)
                except Exception as e:
                    print(f"  Не удалось удалить {doc_id} из Qdrant: {e}")
            # Treat as missing so they get re-indexed below
            qdrant_doc_ids -= incomplete_in_qdrant

        # Resolve document_id for each MinIO file.
        # Files uploaded via API already have document_id in metadata.
        # Files uploaded manually (via MinIO Console) have document_id=None —
        # we download them, compute hash, and write it back to metadata.
        resolved_docs = []
        for doc in minio_docs:
            doc_id = doc['document_id']
            content = None

            if doc_id is None:
                # Manually uploaded — need content to compute hash
                try:
                    content = await minio_storage.download_document(
                        None, doc['filename'], doc['bucket_name']
                    )
                    doc_id = hashlib.sha256(content).hexdigest()[:16]
                    print(f"  [MANUAL] {doc['filename']} → generated document_id={doc_id}")
                    # Write document_id back to MinIO metadata for future startups
                    await minio_storage.write_document_id_metadata(
                        doc['filename'], doc_id, doc['bucket_name']
                    )
                except Exception as e:
                    print(f"  ⚠ Не удалось обработать {doc['filename']}: {e}")
                    continue

            resolved_docs.append({
                **doc,
                'document_id': doc_id,
                '_content': content,  # cached if already downloaded
            })

        missing_in_qdrant = [d for d in resolved_docs if d['document_id'] not in qdrant_doc_ids]
        extra_in_qdrant = qdrant_doc_ids - {d['document_id'] for d in resolved_docs}

        print(f"\nСтатус синхронизации:")
        print(f"Синхронизировано: {len(resolved_docs) - len(missing_in_qdrant)} документов")
        print(f"Требуется индексация: {len(missing_in_qdrant)} документов")
        print(f"Лишние в Qdrant: {len(extra_in_qdrant)} документов")

        if missing_in_qdrant:
            print(f"\nИндексация недостающих документов...")
            for idx, doc in enumerate(missing_in_qdrant, 1):
                filename = doc['filename']
                doc_id = doc['document_id']
                print(f"  [{idx}/{len(missing_in_qdrant)}] {filename} (id: {doc_id})")

                try:
                    # Use cached content if already downloaded, otherwise fetch now
                    content = doc['_content']
                    if content is None:
                        content = await minio_storage.download_document(
                            doc_id, filename, doc['bucket_name']
                        )

                    temp_file = tempfile.NamedTemporaryFile(
                        delete=False, suffix=temp_suffix_for(filename)
                    )
                    temp_file.write(content)
                    temp_file.close()

                    try:
                        namespace = doc.get('namespace', 'corporate')
                        chunks, metadata = await document_indexer.process_document(
                            temp_file.name,
                            document_id=doc_id,
                            original_filename=filename
                        )
                        for meta in metadata:
                            meta['namespace'] = namespace
                        await vector_store.add_documents(chunks, metadata, namespace=namespace)
                        print(f"    ✓ Проиндексировано {len(chunks)} чанков")

                    except Exception as idx_err:
                        import traceback
                        print(f"    ✗ Ошибка индексации: {idx_err}")
                        traceback.print_exc()
                        print(f"    [ROLLBACK] Удаляем '{filename}' из MinIO...")
                        try:
                            await minio_storage.delete_document(doc_id, filename, doc['bucket_name'])
                            print(f"    [ROLLBACK] '{filename}' удалён из MinIO")
                        except Exception as del_e:
                            print(f"    [ROLLBACK] Не удалось удалить '{filename}' из MinIO: {del_e}")

                    finally:
                        if os.path.exists(temp_file.name):
                            os.unlink(temp_file.name)

                except Exception as e:
                    print(f"    ✗ Ошибка при подготовке документа: {e}")

        if extra_in_qdrant:
            print(f"\nУдаление лишних документов из Qdrant...")
            for doc_id in extra_in_qdrant:
                try:
                    await vector_store.delete_by_document_id(doc_id)
                    print(f"  Удален: {doc_id}")
                except Exception as e:
                    print(f"  Ошибка при удалении {doc_id}: {e}")

        if not missing_in_qdrant and not extra_in_qdrant:
            print("\nВсе документы синхронизированы!")
        else:
            print("\nСинхронизация завершена!")

    except Exception as e:
        print(f"\nОшибка при синхронизации: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 60 + "\n")
