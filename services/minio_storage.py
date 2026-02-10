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
        bucket_name: str = "documents",
        secure: bool = False
    ):
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket_name = bucket_name
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

            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                print(f"Создан bucket MinIO: {self.bucket_name}")
            else:
                print(f"Bucket MinIO '{self.bucket_name}' уже существует")
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

    async def upload_document(self, file_data: BinaryIO, filename: str) -> Dict:
        content = file_data.read()
        file_data.seek(0)

        document_id = self.generate_document_id(filename, content)
        object_name = f"{document_id}/{filename}"

        self.client.put_object(
            bucket_name=self.bucket_name,
            object_name=object_name,
            data=io.BytesIO(content),
            length=len(content),
            content_type=get_content_type(filename)
        )

        print(f"Документ {filename} загружен в MinIO с ID: {document_id}")

        return {
            "document_id": document_id,
            "filename": filename,
            "object_name": object_name,
            "size": len(content),
            "uploaded_at": datetime.now().isoformat()
        }

    async def download_document(self, document_id: str, filename: str) -> bytes:
        object_name = f"{document_id}/{filename}"

        try:
            response = self.client.get_object(self.bucket_name, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error:
            try:
                response = self.client.get_object(self.bucket_name, filename)
                data = response.read()
                response.close()
                response.release_conn()
                return data
            except S3Error as e:
                raise FileNotFoundError(f"Документ {filename} (ID: {document_id}) не найден в MinIO") from e

    async def delete_document(self, document_id: str, filename: str) -> None:
        object_name = f"{document_id}/{filename}"

        try:
            self.client.remove_object(self.bucket_name, object_name)
            print(f"Документ {filename} (ID: {document_id}) удален из MinIO")
            return
        except S3Error:
            pass

        try:
            self.client.remove_object(self.bucket_name, filename)
            print(f"Документ {filename} (ID: {document_id}) удален из MinIO (legacy формат)")
        except S3Error as e:
            print(f"Ошибка при удалении документа: {e}")

    async def list_documents(self) -> List[Dict]:
        documents = []
        objects = self.client.list_objects(self.bucket_name, recursive=True)

        total_objects = 0
        for obj in objects:
            total_objects += 1
            parts = obj.object_name.split('/', 1)

            if total_objects <= 5:
                print(f"[MinIO Debug] Object {total_objects}: '{obj.object_name}' -> parts: {parts}")

            if len(parts) == 2:
                document_id, filename = parts
                documents.append({
                    "document_id": document_id,
                    "filename": filename,
                    "object_name": obj.object_name,
                    "size": obj.size,
                    "last_modified": obj.last_modified.isoformat() if obj.last_modified else None
                })
            elif len(parts) == 1:
                filename = parts[0]
                document_id = self.generate_document_id(filename)

                if total_objects <= 5:
                    print(f"[MinIO Debug] 📄 Legacy файл: '{filename}' -> ID: {document_id}")

                documents.append({
                    "document_id": document_id,
                    "filename": filename,
                    "object_name": obj.object_name,
                    "size": obj.size,
                    "last_modified": obj.last_modified.isoformat() if obj.last_modified else None,
                    "legacy": True
                })

        print(f"[MinIO Debug] Всего объектов в bucket: {total_objects}, распознано документов: {len(documents)}")
        legacy_count = sum(1 for doc in documents if doc.get('legacy'))
        if legacy_count > 0:
            print(f"[MinIO Debug] Legacy файлов (без document_id): {legacy_count}")
        return documents

    async def document_exists(self, document_id: str, filename: str) -> bool:
        object_name = f"{document_id}/{filename}"

        try:
            self.client.stat_object(self.bucket_name, object_name)
            return True
        except S3Error:
            pass

        try:
            self.client.stat_object(self.bucket_name, filename)
            return True
        except S3Error:
            return False

    async def cleanup(self):
        print("MinIO клиент очищен")
