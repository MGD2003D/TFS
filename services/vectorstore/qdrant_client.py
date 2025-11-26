from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from .base import BaseVectorStore
import uuid
import os


class QdrantVectorStore(BaseVectorStore):

    def __init__(
        self,
        collection_name: str = "documents",
        host: str = "localhost",
        port: int = 6333,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.embedding_model_name = embedding_model
        self.client = None
        self.embedding_model = None


    async def initialize(self):
        print(f"Подключаюсь к Qdrant на {self.host}:{self.port}")

        old_http_proxy = os.environ.get('HTTP_PROXY')
        old_https_proxy = os.environ.get('HTTPS_PROXY')
        if 'HTTP_PROXY' in os.environ:
            del os.environ['HTTP_PROXY']
        if 'HTTPS_PROXY' in os.environ:
            del os.environ['HTTPS_PROXY']

        try:
            self.client = QdrantClient(host=self.host, port=self.port, timeout=60)
        finally:
            if old_http_proxy:
                os.environ['HTTP_PROXY'] = old_http_proxy
            if old_https_proxy:
                os.environ['HTTPS_PROXY'] = old_https_proxy

        print(f"Загружаю embedding модель {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        vector_size = self.embedding_model.get_sentence_embedding_dimension()

        collections = self.client.get_collections().collections
        if not any(col.name == self.collection_name for col in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"Создана коллекция {self.collection_name}")
        else:
            print(f"Коллекция {self.collection_name} уже существует")


    async def add_documents(self, texts: List[str], metadata: List[Dict[str, Any]] = None) -> None:
        if not texts:
            return

        collections = self.client.get_collections().collections
        if not any(col.name == self.collection_name for col in collections):
            vector_size = self.embedding_model.get_sentence_embedding_dimension()
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"Создана коллекция {self.collection_name}")

        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

        points = []
        for idx, (text, embedding) in enumerate(zip(texts, embeddings)):
            point_metadata = metadata[idx] if metadata and idx < len(metadata) else {}
            point_metadata["text"] = text

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload=point_metadata
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        print(f"Добавлено {len(points)} документов в {self.collection_name}")


    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:

        query_vector = self.embedding_model.encode([query])[0]

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k
        )

        results = []
        for point in search_result.points:
            results.append({
                "text": point.payload.get("text", ""),
                "score": point.score,
                "metadata": {k: v for k, v in point.payload.items() if k != "text"}
            })

        return results


    async def delete_collection(self, collection_name: str = None) -> None:
        coll_name = collection_name or self.collection_name
        self.client.delete_collection(collection_name=coll_name)
        print(f"Коллекция {coll_name} удалена")

    async def cleanup(self) -> None:
        if self.client is not None:
            self.client.close()
            print("Qdrant клиент закрыт")
