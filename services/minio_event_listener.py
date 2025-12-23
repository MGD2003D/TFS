import asyncio
import json
from typing import Callable, Awaitable
import tempfile
import os
import hashlib
import app_state
import threading
import queue
from services.document_types import is_supported_document, temp_suffix_for


class MinioEventListener:
    def __init__(self, minio_storage):
        self.minio_storage = minio_storage
        self.listener_task = None
        self.listener_thread = None
        self.should_stop = False
        self.event_queue = queue.Queue()
        self.main_loop = None

    async def start(self):
        if self.listener_task is not None or self.listener_thread is not None:
            print("MinIO Event Listener уже запущен")
            return

        self.should_stop = False
        self.main_loop = asyncio.get_event_loop()

        self.listener_thread = threading.Thread(
            target=self._blocking_listen,
            daemon=True
        )
        self.listener_thread.start()

        self.listener_task = asyncio.create_task(self._process_events())
        print("MinIO Event Listener запущен")

    async def stop(self):
        self.should_stop = True

        if self.listener_task is not None:
            self.listener_task.cancel()
            try:
                await self.listener_task
            except asyncio.CancelledError:
                pass
            self.listener_task = None

        if self.listener_thread is not None:
            self.listener_thread.join(timeout=5)
            self.listener_thread = None

        print("MinIO Event Listener остановлен")

    def _blocking_listen(self):
        print(f"Начинаю прослушивание событий bucket '{self.minio_storage.bucket_name}'...")

        while not self.should_stop:
            try:
                events = self.minio_storage.client.listen_bucket_notification(
                    bucket_name=self.minio_storage.bucket_name,
                    prefix="",
                    suffix="",
                    events=[
                        "s3:ObjectCreated:*",
                        "s3:ObjectRemoved:*"
                    ]
                )

                for event in events:
                    if self.should_stop:
                        break

                    self.event_queue.put(event)

            except Exception as e:
                if not self.should_stop:
                    print(f"Ошибка при прослушивании событий MinIO: {e}")
                    print("Переподключение через 5 секунд...")
                    import time
                    time.sleep(5)

    async def _process_events(self):
        try:
            while not self.should_stop:
                await asyncio.sleep(0.1)

                while not self.event_queue.empty():
                    try:
                        event = self.event_queue.get_nowait()
                        await self._handle_event(event)
                    except queue.Empty:
                        break
                    except Exception as e:
                        print(f"Ошибка при обработке события из очереди: {e}")

        except asyncio.CancelledError:
            print("Event processor получил сигнал остановки")
            raise

    async def _handle_event(self, event_data: dict):
        try:
            if not event_data or 'Records' not in event_data:
                return

            for record in event_data['Records']:
                event_name = record.get('eventName', '')
                s3_info = record.get('s3', {})
                bucket_name = s3_info.get('bucket', {}).get('name', '')
                object_info = s3_info.get('object', {})
                object_name = object_info.get('key', '')

                if bucket_name != self.minio_storage.bucket_name:
                    continue

                parts = object_name.split('/', 1)

                if len(parts) == 2:
                    document_id, filename = parts
                elif len(parts) == 1:
                    filename = parts[0]
                    import hashlib
                    document_id = hashlib.sha256(filename.encode()).hexdigest()[:16]
                else:
                    print(f"Некорректный формат object_name: {object_name}")
                    continue

                if not is_supported_document(filename):
                    continue

                if event_name.startswith('s3:ObjectCreated'):
                    await self._handle_object_created(document_id, filename, object_name)
                elif event_name.startswith('s3:ObjectRemoved'):
                    await self._handle_object_removed(document_id, filename)

        except Exception as e:
            print(f"Ошибка при обработке события MinIO: {e}")
            import traceback
            traceback.print_exc()

    async def _handle_object_created(self, document_id: str, filename: str, object_name: str):
        print(f"Обнаружен новый документ: {filename} (ID: {document_id})")

        try:
            try:
                response = self.minio_storage.client.get_object(
                    self.minio_storage.bucket_name,
                    object_name
                )
                content = response.read()
                response.close()
                response.release_conn()
            except Exception as e:
                print(f"Ошибка при скачивании {object_name}: {e}")
                return

            content_hash = hashlib.sha256(content).hexdigest()[:16]

            existing_docs = await app_state.vector_store.get_documents_list()
            if any(doc['document_id'] == content_hash for doc in existing_docs):
                print(f"Документ {filename} уже проиндексирован (content hash: {content_hash}), пропускаем")
                return

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=temp_suffix_for(filename))
            temp_file.write(content)
            temp_file.close()

            try:
                chunks, metadata = await app_state.document_indexer.process_document(
                    temp_file.name,
                    document_id=content_hash
                )
                await app_state.vector_store.add_documents(chunks, metadata)

                print(f"Документ {filename} успешно проиндексирован (content hash: {content_hash}, {len(chunks)} чанков)")

            finally:
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)

        except Exception as e:
            print(f"Ошибка при индексации документа {filename}: {e}")
            import traceback
            traceback.print_exc()

    async def _handle_object_removed(self, document_id: str, filename: str):
        print(f"Обнаружено удаление документа: {filename} (ID: {document_id})")

        try:
            await app_state.vector_store.delete_by_document_id(document_id)
            print(f"Документ {filename} удален из индекса")

        except Exception as e:
            print(f"Ошибка при удалении документа {filename} из индекса: {e}")
            import traceback
            traceback.print_exc()
