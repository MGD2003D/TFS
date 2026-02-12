from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct, SparseVectorParams, SparseIndexParams
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from .base import BaseVectorStore
from services.sparse_encoder import BM25SparseEncoder
import uuid
import os
import torch


class QdrantVectorStore(BaseVectorStore):

    def __init__(
        self,
        collection_name: str = "documents",
        host: str = "localhost",
        port: int = 6333,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        enable_hybrid_search: bool = True
    ):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.embedding_model_name = embedding_model
        self.enable_hybrid_search = enable_hybrid_search
        self.client = None
        self.embedding_model = None
        self.sparse_encoder = None
        self.sparse_vocab_ready = False


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
        print("Это может занять несколько минут при первом запуске (скачивание ~500MB)...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding_model = SentenceTransformer(self.embedding_model_name, device=device)
        print(f"Модель {self.embedding_model_name} успешно загружена на {device.upper()}")

        vector_size = self.embedding_model.get_sentence_embedding_dimension()

        if self.enable_hybrid_search:
            print("Инициализация BM25 Sparse Encoder для Hybrid Search...")
            self.sparse_encoder = BM25SparseEncoder()
            self.sparse_vocab_ready = False
            print("BM25 Sparse Encoder готов")

        collections = self.client.get_collections().collections
        collection_exists = any(col.name == self.collection_name for col in collections)

        need_recreate = False
        if collection_exists and self.enable_hybrid_search:
            try:
                collection_info = self.client.get_collection(self.collection_name)
                if not isinstance(collection_info.config.params.vectors, dict):
                    print(f"Коллекция {self.collection_name} имеет старую структуру (без named vectors)")
                    print(f"Удаляю и пересоздаю коллекцию для Hybrid Search...")
                    self.client.delete_collection(self.collection_name)
                    collection_exists = False
                    need_recreate = True
            except Exception as e:
                print(f"Ошибка при проверке структуры коллекции: {e}")
                print(f"Пересоздаю коллекцию...")
                self.client.delete_collection(self.collection_name)
                collection_exists = False
                need_recreate = True

        if not collection_exists:
            if self.enable_hybrid_search:
                print(f"Создаю коллекцию {self.collection_name} с Hybrid Search (dense + sparse vectors)...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": VectorParams(size=vector_size, distance=Distance.COSINE),
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(
                            index=SparseIndexParams(
                                on_disk=False,
                            )
                        )
                    }
                )
                if need_recreate:
                    print(f"Коллекция {self.collection_name} пересоздана с Hybrid Search")
                else:
                    print(f"Создана коллекция {self.collection_name} с Hybrid Search")
            else:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                )
                print(f"Создана коллекция {self.collection_name} (только dense vectors)")
        else:
            print(f"Коллекция {self.collection_name} уже существует (структура совместима)")


    def build_sparse_vocab(self, texts: List[str]) -> None:
        if not self.enable_hybrid_search or not self.sparse_encoder:
            return
        if self.sparse_vocab_ready:
            return
        if not texts:
            return
        self.sparse_encoder.build_vocab(texts)
        self.sparse_vocab_ready = True

    async def add_documents(
        self,
        texts: List[str],
        metadata: List[Dict[str, Any]] = None,
        namespace: str = "default"
    ) -> None:
        if not texts:
            return

        collections = self.client.get_collections().collections
        if not any(col.name == self.collection_name for col in collections):
            vector_size = self.embedding_model.get_sentence_embedding_dimension()
            if self.enable_hybrid_search:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": VectorParams(size=vector_size, distance=Distance.COSINE),
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(
                            index=SparseIndexParams(on_disk=False)
                        )
                    }
                )
            else:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                )
            print(f"Создана коллекция {self.collection_name}")

        if self.enable_hybrid_search and self.sparse_encoder and not self.sparse_vocab_ready:
            self.sparse_encoder.build_vocab(texts)
            self.sparse_vocab_ready = True

        prefixed_texts = [f"passage: {text}" for text in texts] if self._use_prefixes() else texts
        embeddings = self.embedding_model.encode(prefixed_texts, show_progress_bar=True, normalize_embeddings=True)

        points = []
        for idx, (text, embedding) in enumerate(zip(texts, embeddings)):
            point_metadata = metadata[idx] if metadata and idx < len(metadata) else {}
            point_metadata["text"] = text

            if "namespace" not in point_metadata:
                point_metadata["namespace"] = namespace

            if self.enable_hybrid_search and self.sparse_encoder:
                sparse_vector = self.sparse_encoder.encode(text)

                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector={
                            "dense": embedding.tolist(),
                            "sparse": models.SparseVector(
                                indices=list(sparse_vector.keys()),
                                values=list(sparse_vector.values())
                            )
                        },
                        payload=point_metadata
                    )
                )
            else:
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

        if self.enable_hybrid_search:
            print(f"Добавлено {len(points)} чанков в {self.collection_name} (hybrid: dense + sparse) [namespace: {point_metadata.get('namespace', namespace)}]")
        else:
            print(f"Добавлено {len(points)} чанков в {self.collection_name} [namespace: {point_metadata.get('namespace', namespace)}]")


    def _build_namespace_filter(self, namespaces: List[str] = None):

        if not namespaces:
            return None

        if len(namespaces) == 1:
            return models.Filter(
                must=[
                    models.FieldCondition(
                        key="namespace",
                        match=models.MatchValue(value=namespaces[0])
                    )
                ]
            )
        else:
            return models.Filter(
                should=[
                    models.FieldCondition(
                        key="namespace",
                        match=models.MatchValue(value=ns)
                    )
                    for ns in namespaces
                ]
            )

    async def search(
        self,
        query: str,
        top_k: int = 5,
        namespaces: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Поиск документов. Автоматически использует hybrid search если включен.
        """
        if self.enable_hybrid_search and self.sparse_encoder:
            return await self.hybrid_search_rrf(query, top_k=top_k, namespaces=namespaces)
        else:
            return await self.dense_search(query, top_k=top_k, namespaces=namespaces)

    async def dense_search(
        self,
        query: str,
        top_k: int = 5,
        namespaces: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Обычный dense vector search (семантический поиск) с namespace filtering.
        """
        prefixed_query = f"query: {query}" if self._use_prefixes() else query
        query_vector = self.embedding_model.encode([prefixed_query], normalize_embeddings=True)[0]

        query_filter = self._build_namespace_filter(namespaces)

        if self.enable_hybrid_search:
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector.tolist(),
                using="dense",
                query_filter=query_filter,
                limit=top_k
            )
        else:
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector.tolist(),
                query_filter=query_filter,
                limit=top_k
            )

        results = []
        for point in search_result.points:
            results.append({
                "text": point.payload.get("text", ""),
                "score": point.score,
                "metadata": {k: v for k, v in point.payload.items() if k != "text"},
                "search_type": "dense"
            })

        return results

    async def sparse_search(
        self,
        query: str,
        top_k: int = 5,
        namespaces: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Sparse (BM25) search only with namespace filtering.
        """
        if not self.sparse_encoder:
            print("[WARN] Sparse encoder not initialized, returning empty results")
            return []

        sparse_vector = self.sparse_encoder.encode_query(query)
        if not sparse_vector:
            return []

        query_filter = self._build_namespace_filter(namespaces)

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=models.SparseVector(
                indices=list(sparse_vector.keys()),
                values=list(sparse_vector.values())
            ),
            using="sparse",
            query_filter=query_filter,
            limit=top_k
        )

        results = []
        for point in search_result.points:
            results.append({
                "text": point.payload.get("text", ""),
                "score": point.score,
                "metadata": {k: v for k, v in point.payload.items() if k != "text"},
                "search_type": "sparse"
            })

        return results

    async def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        namespaces: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search: комбинирует dense (семантический) и sparse (BM25) поиск с namespace filtering.

        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            dense_weight: Вес dense поиска (0.0-1.0)
            sparse_weight: Вес sparse поиска (0.0-1.0)
            namespaces: Список namespaces для фильтрации (None = все документы)

        Returns:
            Список результатов с объединенными скорами
        """
        if not self.sparse_encoder:
            print("[WARN] Sparse encoder not initialized, falling back to dense search")
            return await self.dense_search(query, top_k=top_k, namespaces=namespaces)

        query_filter = self._build_namespace_filter(namespaces)

        prefixed_query = f"query: {query}" if self._use_prefixes() else query
        dense_vector = self.embedding_model.encode([prefixed_query], normalize_embeddings=True)[0]

        dense_results = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vector.tolist(),
            using="dense",
            query_filter=query_filter,
            limit=top_k * 2
        )

        sparse_vector = self.sparse_encoder.encode_query(query)

        if sparse_vector:
            sparse_results = self.client.query_points(
                collection_name=self.collection_name,
                query=models.SparseVector(
                    indices=list(sparse_vector.keys()),
                    values=list(sparse_vector.values())
                ),
                using="sparse",
                query_filter=query_filter,
                limit=top_k * 2
            )
        else:
            class EmptyResults:
                points = []
            sparse_results = EmptyResults()

        results_map = {}

        for point in dense_results.points:
            point_id = point.id
            results_map[point_id] = {
                "text": point.payload.get("text", ""),
                "metadata": {k: v for k, v in point.payload.items() if k != "text"},
                "dense_score": point.score * dense_weight,
                "sparse_score": 0.0,
                "combined_score": point.score * dense_weight,
                "search_type": "dense"
            }

        for point in sparse_results.points:
            point_id = point.id
            if point_id in results_map:
                results_map[point_id]["sparse_score"] = point.score * sparse_weight
                results_map[point_id]["combined_score"] += point.score * sparse_weight
                results_map[point_id]["search_type"] = "hybrid"
            else:
                results_map[point_id] = {
                    "text": point.payload.get("text", ""),
                    "metadata": {k: v for k, v in point.payload.items() if k != "text"},
                    "dense_score": 0.0,
                    "sparse_score": point.score * sparse_weight,
                    "combined_score": point.score * sparse_weight,
                    "search_type": "sparse"
                }

        results = sorted(
            results_map.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )[:top_k]

        for result in results:
            result["score"] = result["combined_score"]

        return results

    async def hybrid_search_rrf(
        self,
        query: str,
        top_k: int = 5,
        rrf_k: int = 60,
        pool_k: int = None,
        namespaces: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Гибридный поиск с RRF и namespace filtering
        """
        if not self.sparse_encoder:
            print("[WARN] Sparse encoder not initialized, falling back to dense search")
            return await self.dense_search(query, top_k=top_k, namespaces=namespaces)

        pool_k = pool_k or max(top_k * 5, top_k)

        query_filter = self._build_namespace_filter(namespaces)

        prefixed_query = f"query: {query}" if self._use_prefixes() else query
        dense_vector = self.embedding_model.encode([prefixed_query], normalize_embeddings=True)[0]

        dense_results = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vector.tolist(),
            using="dense",
            query_filter=query_filter,
            limit=pool_k
        )

        sparse_vector = self.sparse_encoder.encode_query(query)

        if sparse_vector:
            sparse_results = self.client.query_points(
                collection_name=self.collection_name,
                query=models.SparseVector(
                    indices=list(sparse_vector.keys()),
                    values=list(sparse_vector.values())
                ),
                using="sparse",
                query_filter=query_filter,
                limit=pool_k
            )
        else:
            class EmptyResults:
                points = []
            sparse_results = EmptyResults()

        # 3. RRF merge
        results_map = {}

        for rank, point in enumerate(dense_results.points, 1):
            rrf_score = 1.0 / (rrf_k + rank)
            results_map[point.id] = {
                "text": point.payload.get("text", ""),
                "metadata": {k: v for k, v in point.payload.items() if k != "text"},
                "rrf_score": rrf_score,
                "search_type": "rrf",
            }

        for rank, point in enumerate(sparse_results.points, 1):
            rrf_score = 1.0 / (rrf_k + rank)
            if point.id in results_map:
                results_map[point.id]["rrf_score"] += rrf_score
            else:
                results_map[point.id] = {
                    "text": point.payload.get("text", ""),
                    "metadata": {k: v for k, v in point.payload.items() if k != "text"},
                    "rrf_score": rrf_score,
                    "search_type": "rrf",
                }

        results = sorted(
            results_map.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )[:top_k]

        for result in results:
            result["score"] = result["rrf_score"]

        return results


    async def delete_by_document_id(self, document_id: str) -> int:
        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id)
                        )
                    ]
                )
            )
        )
        print(f"Удалены чанки документа {document_id} из Qdrant")
        return result

    async def get_documents_list(self) -> List[Dict[str, Any]]:
        scroll_result = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )

        documents = {}
        for point in scroll_result[0]:
            doc_id = point.payload.get("document_id")
            if doc_id and doc_id not in documents:
                documents[doc_id] = {
                    "document_id": doc_id,
                    "source": point.payload.get("source", "unknown"),
                    "total_chunks": point.payload.get("total_chunks", 0)
                }

        return list(documents.values())

    async def delete_collection(self, collection_name: str = None) -> None:
        coll_name = collection_name or self.collection_name
        self.client.delete_collection(collection_name=coll_name)
        print(f"Коллекция {coll_name} удалена")

    def _use_prefixes(self) -> bool:
        """Проверяет, требуется ли использовать префиксы query:/passage: для модели"""
        e5_models = ["e5-small", "e5-base", "e5-large", "multilingual-e5"]
        return any(model_name in self.embedding_model_name.lower() for model_name in e5_models)

    async def cleanup(self) -> None:
        if self.client is not None:
            self.client.close()
            print("Qdrant клиент закрыт")
