from minio import Minio
from minio.error import S3Error
from typing import List, Dict, BinaryIO
import io
import hashlib
from datetime import datetime
from services.document_types import get_content_type


class MinioStorageService:

    def __init__(
        self,
        endpoint: str = "localhost:9000",
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin123",
        bucket_name: str = "documents",  # Legacy: will be replaced by corporate bucket
        secure: bool = False
    ):
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket_name = bucket_name  # Legacy bucket name
        self.corporate_bucket = "documents-corporate"
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

            if self.client.bucket_exists(self.bucket_name):
                print(f"[INFO] Legacy bucket '{self.bucket_name}' существует (будет использован для миграции)")

        except S3Error as e:
            print(f"Ошибка MinIO S3: {e}")
            raise
        except Exception as e:
            print(f"Ошибка подключения к MinIO ({self.endpoint}): {e}")
            raise

    def generate_document_id(self, filename: str, content: bytes = None) -> str:
        if content:
            return hashlib.sha256(content).hexdigest()[:16]
        else:
            return hashlib.sha256(filename.encode()).hexdigest()[:16]

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

        document_id = self.generate_document_id(filename, content)

        bucket_name = self._namespace_to_bucket(namespace)

        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)
            print(f"Создан bucket: {bucket_name}")

        object_name = f"{document_id}/{filename}"

        minio_metadata = metadata.copy() if metadata else {}
        minio_metadata["namespace"] = namespace
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
        document_id: str,
        filename: str,
        bucket_name: str = None
    ) -> bytes:

        bucket_name = bucket_name or self.corporate_bucket
        object_name = f"{document_id}/{filename}"

        try:
            response = self.client.get_object(bucket_name, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error:
            try:
                response = self.client.get_object(bucket_name, filename)
                data = response.read()
                response.close()
                response.release_conn()
                return data
            except S3Error as e:
                raise FileNotFoundError(
                    f"Документ {filename} (ID: {document_id}) не найден в bucket {bucket_name}"
                ) from e

    async def delete_document(
        self,
        document_id: str,
        filename: str,
        bucket_name: str = None
    ) -> None:

        bucket_name = bucket_name or self.corporate_bucket
        object_name = f"{document_id}/{filename}"

        try:
            self.client.remove_object(bucket_name, object_name)
            print(f"Документ {filename} (ID: {document_id}) удален из MinIO [bucket: {bucket_name}]")
            return
        except S3Error:
            pass

        try:
            self.client.remove_object(bucket_name, filename)
            print(f"Документ {filename} (ID: {document_id}) удален из MinIO (legacy формат) [bucket: {bucket_name}]")
        except S3Error as e:
            print(f"Ошибка при удалении документа из bucket {bucket_name}: {e}")

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
                    parts = obj.object_name.split('/', 1)
                    if len(parts) == 2:
                        document_id, filename = parts
                        documents.append({
                            "document_id": document_id,
                            "filename": filename,
                            "bucket_name": bucket_name,
                            "namespace": namespace,
                            "object_name": obj.object_name,
                            "size": obj.size,
                            "last_modified": obj.last_modified.isoformat() if obj.last_modified else None
                        })
            except S3Error as e:
                print(f"[MinIO] Ошибка при чтении bucket {bucket_name}: {e}")

        print(f"[MinIO] Найдено документов: {len(documents)} (namespaces: {namespaces or 'all'})")
        return documents

    async def document_exists(
        self,
        document_id: str,
        filename: str,
        bucket_name: str = None
    ) -> bool:

        bucket_name = bucket_name or self.corporate_bucket
        object_name = f"{document_id}/{filename}"

        try:
            self.client.stat_object(bucket_name, object_name)
            return True
        except S3Error:
            pass

        try:
            self.client.stat_object(bucket_name, filename)
            return True
        except S3Error:
            return False

    async def cleanup(self):
        print("MinIO клиент очищен")
