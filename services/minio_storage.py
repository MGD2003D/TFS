from minio import Minio
from minio.error import S3Error
from typing import List, Dict, BinaryIO
import io
import os
import hashlib
from datetime import datetime
from services.document_types import get_content_type


class MinioStorageService:
    """
    Хранилище документов в MinIO.

    Структура хранения (плоская):
        bucket/filename.pdf
    document_id хранится в метаданных объекта (x-amz-meta-document-id).
    Это позволяет загружать файлы напрямую через MinIO Console —
    система обнаружит их при синхронизации и сгенерирует document_id.
    """

    def __init__(
        self,
        endpoint: str = "localhost:9000",
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin123",
        bucket_name: str = "documents",
        secure: bool = False
    ):
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket_name = bucket_name
        self.corporate_bucket = os.getenv("MINIO_CORPORATE_BUCKET", "documents-corporate")
        self.secure = secure
        self.client = None

    async def initialize(self):
        try:
            self.client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure
            )
            print(f"Подключение к MinIO: {self.endpoint}")

            if not self.client.bucket_exists(self.corporate_bucket):
                self.client.make_bucket(self.corporate_bucket)
                print(f"Создан корпоративный bucket MinIO: {self.corporate_bucket}")
            else:
                print(f"Корпоративный bucket MinIO '{self.corporate_bucket}' уже существует")

        except S3Error as e:
            print(f"Ошибка MinIO S3: {e}")
            raise
        except Exception as e:
            print(f"Ошибка подключения к MinIO ({self.endpoint}): {e}")
            raise

    def generate_document_id(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()[:16]

    def _namespace_to_bucket(self, namespace: str) -> str:
        if namespace == "corporate":
            return self.corporate_bucket
        else:
            return f"documents-{namespace.replace('_', '-')}"

    async def upload_document(
        self,
        file_data: BinaryIO,
        filename: str,
        namespace: str = "corporate",
        metadata: Dict = None
    ) -> Dict:
        content = file_data.read()
        file_data.seek(0)

        document_id = self.generate_document_id(content)
        bucket_name = self._namespace_to_bucket(namespace)

        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)
            print(f"Создан bucket: {bucket_name}")

        # Flat storage: just filename, document_id goes into metadata
        object_name = filename

        minio_metadata = metadata.copy() if metadata else {}
        minio_metadata["namespace"] = namespace
        minio_metadata["document-id"] = document_id
        if "uploaded_at" not in minio_metadata:
            minio_metadata["uploaded_at"] = datetime.now().isoformat()

        self.client.put_object(
            bucket_name=bucket_name,
            object_name=object_name,
            data=io.BytesIO(content),
            length=len(content),
            content_type=get_content_type(filename),
            metadata=minio_metadata
        )

        print(f"Документ {filename} загружен в MinIO [bucket: {bucket_name}, namespace: {namespace}, ID: {document_id}]")

        return {
            "document_id": document_id,
            "filename": filename,
            "bucket_name": bucket_name,
            "object_name": object_name,
            "namespace": namespace,
            "size": len(content),
            "uploaded_at": minio_metadata["uploaded_at"]
        }

    async def download_document(
        self,
        document_id: str,  # kept for API compatibility, not used in path
        filename: str,
        bucket_name: str = None
    ) -> bytes:
        bucket_name = bucket_name or self.corporate_bucket
        object_name = filename  # flat structure

        try:
            response = self.client.get_object(bucket_name, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error as e:
            raise FileNotFoundError(
                f"Документ {filename} не найден в bucket {bucket_name}"
            ) from e

    async def delete_document(
        self,
        document_id: str,
        filename: str,
        bucket_name: str = None
    ) -> None:
        bucket_name = bucket_name or self.corporate_bucket
        object_name = filename  # flat structure

        try:
            self.client.remove_object(bucket_name, object_name)
            print(f"Документ {filename} (ID: {document_id}) удален из MinIO [bucket: {bucket_name}]")
        except S3Error as e:
            print(f"Ошибка при удалении документа {filename} из bucket {bucket_name}: {e}")

    async def list_documents(
        self,
        namespaces: List[str] = None
    ) -> List[Dict]:
        documents = []

        if namespaces:
            buckets_to_read = [self._namespace_to_bucket(ns) for ns in namespaces]
        else:
            all_buckets = self.client.list_buckets()
            buckets_to_read = [b.name for b in all_buckets if b.name.startswith("documents-")]

        for bucket_name in buckets_to_read:
            if not self.client.bucket_exists(bucket_name):
                continue

            if bucket_name == self.corporate_bucket:
                namespace = "corporate"
            else:
                namespace = bucket_name.replace("documents-", "").replace("-", "_")

            try:
                objects = self.client.list_objects(bucket_name, recursive=True)
                for obj in objects:
                    object_name = obj.object_name

                    # Skip objects with subfolder paths (legacy format)
                    if '/' in object_name:
                        print(f"[MinIO] Пропуск legacy объекта: {object_name} (устаревший формат {bucket_name}/{object_name})")
                        continue

                    # Read document_id from object metadata
                    document_id = None
                    try:
                        stat = self.client.stat_object(bucket_name, object_name)
                        meta = stat.metadata or {}
                        document_id = (
                            meta.get("x-amz-meta-document-id")
                            or meta.get("document-id")
                            or meta.get("X-Amz-Meta-Document-Id")
                        )
                    except Exception as e:
                        print(f"[MinIO] Не удалось получить метаданные {object_name}: {e}")

                    documents.append({
                        "document_id": document_id,  # None = manually uploaded, needs hash
                        "filename": object_name,
                        "bucket_name": bucket_name,
                        "namespace": namespace,
                        "object_name": object_name,
                        "size": obj.size,
                        "last_modified": obj.last_modified.isoformat() if obj.last_modified else None
                    })

            except S3Error as e:
                print(f"[MinIO] Ошибка при чтении bucket {bucket_name}: {e}")

        print(f"[MinIO] Найдено документов: {len(documents)} (namespaces: {namespaces or 'all'})")
        return documents

    async def document_exists(
        self,
        document_id: str,  # kept for API compatibility, not used in path
        filename: str,
        bucket_name: str = None
    ) -> bool:
        bucket_name = bucket_name or self.corporate_bucket
        object_name = filename  # flat structure

        try:
            self.client.stat_object(bucket_name, object_name)
            return True
        except S3Error:
            return False

    async def write_document_id_metadata(
        self,
        filename: str,
        document_id: str,
        bucket_name: str = None
    ) -> None:
        """
        Записывает document_id в метаданные объекта (copy-to-self).
        Используется для вручную загруженных файлов без метаданных.
        """
        bucket_name = bucket_name or self.corporate_bucket
        try:
            from minio.commonconfig import CopySource, REPLACE

            stat = self.client.stat_object(bucket_name, filename)
            existing_meta = dict(stat.metadata or {})

            # Strip x-amz-meta- prefix for user metadata
            user_meta = {}
            for k, v in existing_meta.items():
                key = k.lower()
                if key.startswith("x-amz-meta-"):
                    user_meta[key[len("x-amz-meta-"):]] = v
                elif not key.startswith("x-amz-") and not key.startswith("content-"):
                    user_meta[key] = v

            user_meta["document-id"] = document_id

            self.client.copy_object(
                bucket_name,
                filename,
                CopySource(bucket_name, filename),
                metadata=user_meta,
                metadata_directive=REPLACE
            )
            print(f"[MinIO] document-id={document_id} записан в метаданные {filename}")
        except Exception as e:
            print(f"[MinIO] Не удалось записать метаданные для {filename}: {e}")

    async def cleanup(self):
        print("MinIO клиент очищен")
