"""
Knowledge Graph — семантический кеш связей, извлечённых из документов.

Строится как побочный продукт запросов: после каждого поиска LLM извлекает
триплеты из документов и сохраняет их в граф. При повторных запросах — сначала
проверяем граф, и если есть подходящий факт — используем его как дополнительный
контекст или пропускаем дорогой поиск.

Пишет только из документов — не из параметрической памяти LLM.
"""

from typing import List, Dict, Optional, Tuple
import hashlib
import time
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchText
)


class KnowledgeGraph:
    """
    Граф знаний в Qdrant. Хранит триплеты (subject, predicate, object)
    с векторными эмбеддингами для семантического поиска.

    Каждый триплет идемпотентен: повторная вставка одного факта
    увеличивает frequency, не создаёт дубль.
    """

    DEFAULT_COLLECTION = "knowledge_graph"

    def __init__(self, qdrant_client, embedding_model, collection_name: str = None):
        """
        Args:
            qdrant_client: инстанс QdrantClient (из vector_store.client)
            embedding_model: модель для эмбеддингов (из vector_store.embedding_model)
            collection_name: имя коллекции (по умолчанию "knowledge_graph")
        """
        self.client = qdrant_client
        self.embedding_model = embedding_model
        self.collection_name = collection_name or self.DEFAULT_COLLECTION
        self._initialized = False

    async def initialize(self):
        """Создаёт коллекцию если не существует."""
        try:
            collections = self.client.get_collections().collections
            names = [c.name for c in collections]

            if self.collection_name not in names:
                sample = self.embedding_model.encode(["test"], normalize_embeddings=True)
                vector_size = len(sample[0])

                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                )
                print(f"[GRAPH] Created collection '{self.collection_name}' (dim={vector_size})")
            else:
                count = self.client.get_collection(self.collection_name).points_count
                print(f"[GRAPH] Collection '{self.collection_name}' exists ({count} triplets)")

            self._initialized = True
        except Exception as e:
            print(f"[GRAPH] Init error: {e}")
            self._initialized = False

    def _triplet_id(self, subject: str, predicate: str, obj: str) -> str:
        """Детерминированный UUID-формат ID для триплета."""
        text = f"{subject.lower().strip()}|{predicate.lower().strip()}|{obj.lower().strip()}"
        md5 = hashlib.md5(text.encode()).hexdigest()
        return f"{md5[:8]}-{md5[8:12]}-{md5[12:16]}-{md5[16:20]}-{md5[20:32]}"

    def _encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts to embeddings."""
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
        return [e.tolist() for e in embeddings]

    def _get_existing_frequency(self, triplet_ids: List[str]) -> Dict[str, int]:
        """Получает текущие frequency для списка ID (batch retrieve)."""
        try:
            existing = self.client.retrieve(
                collection_name=self.collection_name,
                ids=triplet_ids,
                with_payload=True,
            )
            return {str(p.id): p.payload.get("frequency", 1) for p in existing}
        except Exception:
            return {}

    async def save_triplet(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 0.8,
        source_query: str = "",
        source_strategy: str = "",
    ) -> Optional[str]:
        """
        Сохраняет один триплет в граф.

        При повторном сохранении одного факта:
        - frequency увеличивается на 1
        - confidence пересчитывается: min(1.0, confidence + 0.05 * frequency)

        Args:
            subject: субъект (например "тур Золотая Анталья")
            predicate: отношение (например "located_in")
            obj: объект (например "Анталья")
            confidence: уверенность LLM [0-1]
            source_query: запрос, породивший этот триплет
            source_strategy: стратегия (multihop/decomposition/baseline)

        Returns:
            triplet_id или None при ошибке
        """
        if not self._initialized:
            return None

        # Порог: не сохраняем слабые извлечения
        if confidence < 0.7:
            return None

        try:
            triplet_id = self._triplet_id(subject, predicate, obj)
            triplet_text = f"{subject} {predicate} {obj}"
            embedding = self._encode([triplet_text])[0]

            # Получаем текущую frequency (если уже есть)
            freq_map = self._get_existing_frequency([triplet_id])
            current_freq = freq_map.get(triplet_id, 0)
            new_freq = current_freq + 1
            # Уверенность растёт с каждым независимым подтверждением
            final_confidence = min(1.0, confidence + 0.05 * (new_freq - 1))

            point = PointStruct(
                id=triplet_id,
                vector=embedding,
                payload={
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "confidence": final_confidence,
                    "frequency": new_freq,
                    "source_query": source_query,
                    "source_strategy": source_strategy,
                    "text": triplet_text,
                    "updated_at": time.time(),
                }
            )

            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            if current_freq == 0:
                print(f"[GRAPH] New: ({subject}) --[{predicate}]--> ({obj}) conf={final_confidence:.2f}")
            else:
                print(f"[GRAPH] Updated freq={new_freq}: ({subject}) --[{predicate}]--> ({obj})")
            return triplet_id

        except Exception as e:
            print(f"[GRAPH] Save error: {e}")
            return None

    async def save_triplets_batch(self, triplets: List[Dict]) -> int:
        """
        Batch сохранение триплетов с frequency tracking.

        Args:
            triplets: [{subject, predicate, object, confidence, source_query?, source_strategy?}, ...]

        Returns:
            Количество сохранённых триплетов
        """
        if not self._initialized or not triplets:
            return 0

        # Фильтруем слабые
        triplets = [t for t in triplets if t.get("confidence", 0.8) >= 0.7]
        if not triplets:
            return 0

        try:
            texts = [f"{t['subject']} {t['predicate']} {t['object']}" for t in triplets]
            embeddings = self._encode(texts)

            # Получаем текущие frequency одним batch-запросом
            ids = [self._triplet_id(t["subject"], t["predicate"], t["object"]) for t in triplets]
            freq_map = self._get_existing_frequency(ids)

            points = []
            for t, embedding, text, tid in zip(triplets, embeddings, texts, ids):
                current_freq = freq_map.get(tid, 0)
                new_freq = current_freq + 1
                base_conf = t.get("confidence", 0.8)
                final_confidence = min(1.0, base_conf + 0.05 * (new_freq - 1))

                points.append(PointStruct(
                    id=tid,
                    vector=embedding,
                    payload={
                        "subject": t["subject"],
                        "predicate": t["predicate"],
                        "object": t["object"],
                        "confidence": final_confidence,
                        "frequency": new_freq,
                        "source_query": t.get("source_query", ""),
                        "source_strategy": t.get("source_strategy", ""),
                        "text": text,
                        "updated_at": time.time(),
                    }
                ))

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            new_count = sum(1 for tid in ids if freq_map.get(tid, 0) == 0)
            updated_count = len(points) - new_count
            print(f"[GRAPH] Batch: {new_count} new, {updated_count} updated triplets")
            return len(points)

        except Exception as e:
            print(f"[GRAPH] Batch save error: {e}")
            return 0

    async def search_semantic(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.7,
    ) -> List[Tuple[Dict, float]]:
        """
        Семантический поиск по графу — находит триплеты, похожие на запрос.

        Args:
            query: поисковый запрос
            limit: макс. результатов
            min_score: минимальный cosine score

        Returns:
            [(triplet_payload, score), ...] отсортированные по score
        """
        if not self._initialized:
            return []

        try:
            embedding = self._encode([query])[0]

            results = self.client.query_points(
                collection_name=self.collection_name,
                query=embedding,
                limit=limit,
                score_threshold=min_score,
            )

            return [(hit.payload, hit.score) for hit in results.points]

        except Exception as e:
            print(f"[GRAPH] Semantic search error: {e}")
            return []

    async def lookup(
        self,
        query: str,
        min_score: float = 0.75,
        limit: int = 3,
    ) -> List[Dict]:
        """
        Ищет факты в графе по запросу.

        Возвращает только высокоуверенные результаты (score >= min_score).
        Используется для enrichment контекста перед поиском.

        Returns:
            Список doc-like dict: {text, score, metadata, from_graph=True}
        """
        results = await self.search_semantic(query, limit=limit, min_score=min_score)

        docs = []
        for triplet, score in results:
            # Конвертируем триплет в документ для использования как контекст
            triplet_text = (
                f"{triplet.get('subject', '')} "
                f"{triplet.get('predicate', '')} "
                f"{triplet.get('object', '')}"
            )
            docs.append({
                "text": triplet_text,
                "score": score,
                "metadata": {
                    "source": "knowledge_graph",
                    "chunk_id": 0,
                    "subject": triplet.get("subject", ""),
                    "predicate": triplet.get("predicate", ""),
                    "object": triplet.get("object", ""),
                    "confidence": triplet.get("confidence", 0.0),
                    "frequency": triplet.get("frequency", 1),
                },
                "from_graph": True,
            })

        if docs:
            print(f"[GRAPH LOOKUP] '{query[:60]}' → {len(docs)} hits")

        return docs

    async def resolve_hop(
        self,
        hop_query: str,
        extract_hint: str = "",
        min_score: float = 0.75,
    ) -> Optional[str]:
        """
        Пытается разрешить hop через граф (cache hit).

        Args:
            hop_query: запрос хопа
            extract_hint: подсказка что извлечь
            min_score: минимальный score для cache hit

        Returns:
            Resolved object (str) или None если cache miss
        """
        search_text = f"{hop_query} {extract_hint}".strip()
        results = await self.search_semantic(search_text, limit=3, min_score=min_score)

        if not results:
            return None

        best_triplet, best_score = results[0]
        resolved = best_triplet.get("object", "")

        if resolved:
            print(f"[GRAPH HIT] '{hop_query[:60]}' → '{resolved}' (score={best_score:.3f})")
            return resolved

        return None

    async def count(self) -> int:
        """Количество триплетов в графе."""
        if not self._initialized:
            return 0
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count
        except Exception:
            return 0

    async def clear(self):
        """Очищает все триплеты из коллекции."""
        try:
            self.client.delete_collection(self.collection_name)
            self._initialized = False
            print(f"[GRAPH] Collection '{self.collection_name}' cleared")
        except Exception as e:
            print(f"[GRAPH] Clear error: {e}")
