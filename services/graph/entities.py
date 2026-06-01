"""
EntityStore — хранилище узлов графа знаний в Qdrant.

Каждый узел — это уникальная именованная сущность с типом.
entity_id детерминирован: hash(normalize(name) + type).

Entity resolution — двухуровневый каскад:
  Level 1: typed hash (точное совпадение нормализованного имени + типа)
  Level 2: семантический поиск (fallback для вариантов написания)
"""

import hashlib
import time
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)


class EntityStore:
    COLLECTION = "entities"

    def __init__(self, qdrant_client: QdrantClient, embedding_model):
        self.client = qdrant_client
        self.embedding_model = embedding_model
        self._initialized = False

    def _entity_id(self, name: str, entity_type: str) -> str:
        key = f"{name.lower().strip()}|{entity_type.lower().strip()}"
        md5 = hashlib.md5(key.encode()).hexdigest()
        return f"{md5[:8]}-{md5[8:12]}-{md5[12:16]}-{md5[16:20]}-{md5[20:32]}"

    def _encode(self, text: str) -> list:
        return self.embedding_model.encode([text], normalize_embeddings=True)[0].tolist()

    async def initialize(self):
        try:
            collections = self.client.get_collections().collections
            names = [c.name for c in collections]
            if self.COLLECTION not in names:
                sample = self.embedding_model.encode(["test"], normalize_embeddings=True)
                vector_size = len(sample[0])
                self.client.create_collection(
                    collection_name=self.COLLECTION,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
                print(f"[ENTITIES] Created collection '{self.COLLECTION}' (dim={vector_size})")
            else:
                count = self.client.get_collection(self.COLLECTION).points_count
                print(f"[ENTITIES] Collection '{self.COLLECTION}' exists ({count} entities)")
            self._initialized = True
        except Exception as e:
            print(f"[ENTITIES] Init error: {e}")
            self._initialized = False

    async def upsert(self, name: str, entity_type: str) -> Optional[str]:
        """
        Upsert сущности с двухуровневым entity resolution.
        Возвращает entity_id (существующего или нового узла).
        """
        if not self._initialized:
            return None

        name = name.strip()
        entity_type = entity_type.lower().strip() if entity_type else "other"

        if not name:
            return None

        # Level 1: typed hash
        entity_id = self._entity_id(name, entity_type)
        existing = self._get_by_id(entity_id)

        if existing:
            self._increment_mentions(entity_id, existing)
            return entity_id

        # Level 2: semantic fallback — ищем похожие сущности того же типа
        resolved_id = await self._find_semantic(name, entity_type, threshold=0.92)
        if resolved_id:
            existing = self._get_by_id(resolved_id)
            if existing:
                self._increment_mentions(resolved_id, existing)
                return resolved_id

        # Новая сущность
        embedding = self._encode(name)
        point = PointStruct(
            id=entity_id,
            vector=embedding,
            payload={
                "name": name,
                "entity_type": entity_type,
                "mentions": 1,
                "updated_at": time.time(),
            },
        )
        self.client.upsert(collection_name=self.COLLECTION, points=[point])
        print(f"[ENTITIES] New: '{name}' ({entity_type}) id={entity_id[:8]}")
        return entity_id

    def _get_by_id(self, entity_id: str) -> Optional[dict]:
        try:
            results = self.client.retrieve(
                collection_name=self.COLLECTION,
                ids=[entity_id],
                with_payload=True,
            )
            return results[0].payload if results else None
        except Exception:
            return None

    def _increment_mentions(self, entity_id: str, payload: dict):
        try:
            new_mentions = payload.get("mentions", 1) + 1
            self.client.set_payload(
                collection_name=self.COLLECTION,
                payload={"mentions": new_mentions, "updated_at": time.time()},
                points=[entity_id],
            )
        except Exception as e:
            print(f"[ENTITIES] increment error: {e}")

    async def _find_semantic(
        self, name: str, entity_type: str, threshold: float = 0.92
    ) -> Optional[str]:
        """Семантический поиск сущности того же типа (Level 2 fallback)."""
        try:
            embedding = self._encode(name)
            results = self.client.query_points(
                collection_name=self.COLLECTION,
                query=embedding,
                query_filter=Filter(
                    must=[FieldCondition(key="entity_type", match=MatchValue(value=entity_type))]
                ),
                limit=1,
                score_threshold=threshold,
            )
            if results.points:
                hit = results.points[0]
                print(
                    f"[ENTITIES] Resolved '{name}' → '{hit.payload.get('name')}' "
                    f"(score={hit.score:.3f})"
                )
                return str(hit.id)
        except Exception as e:
            print(f"[ENTITIES] semantic search error: {e}")
        return None

    async def search_any(self, query: str, threshold: float = 0.75) -> Optional[str]:
        """Semantic search across all entity types — entry point for local graph traversal."""
        if not self._initialized:
            return None
        try:
            embedding = self._encode(query)
            results = self.client.query_points(
                collection_name=self.COLLECTION,
                query=embedding,
                limit=1,
                score_threshold=threshold,
            )
            if results.points:
                return str(results.points[0].id)
        except Exception as e:
            print(f"[ENTITIES] search_any error: {e}")
        return None

    async def count(self) -> int:
        if not self._initialized:
            return 0
        try:
            return self.client.get_collection(self.COLLECTION).points_count or 0
        except Exception:
            return 0

    async def get_all(self, limit: int = 10000) -> list:
        """Возвращает все узлы (для построения графа)."""
        results = []
        next_offset = None
        while True:
            batch, next_offset = self.client.scroll(
                collection_name=self.COLLECTION,
                limit=min(limit, 1000),
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            results.extend(batch)
            if next_offset is None or len(results) >= limit:
                break
        return results
