"""
CommunityManager — обнаружение тематических кластеров (Leiden) и генерация summaries.

Алгоритм:
  1. build_igraph() — собрать in-memory граф из Qdrant
  2. leidenalg.find_partition() — кластеризация (seed=42 для воспроизводимости)
  3. Для каждого сообщества — LLM summary
  4. Сохранить summaries в Qdrant (коллекция "communities")

Запускается периодически через CommunityScheduler, не на каждый запрос.
"""

import hashlib
import time
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

import prompts_config
from .builder import build_igraph
from .entities import EntityStore


class CommunityManager:
    COLLECTION = "communities"
    MIN_NODES = 10       # минимум узлов для запуска Leiden
    MIN_COMMUNITY = 3    # минимальный размер сообщества для генерации summary

    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedding_model,
        llm_client,
        entity_store: EntityStore,
        triplets_collection: str = "knowledge_graph",
    ):
        self.client = qdrant_client
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.entity_store = entity_store
        self.triplets_collection = triplets_collection
        self._initialized = False
        self._last_rebuild_at: float = 0.0

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
                print(f"[COMMUNITY] Created collection '{self.COLLECTION}'")
            else:
                count = self.client.get_collection(self.COLLECTION).points_count
                print(f"[COMMUNITY] Collection '{self.COLLECTION}' exists ({count} communities)")
            self._initialized = True
        except Exception as e:
            print(f"[COMMUNITY] Init error: {e}")
            self._initialized = False

    async def rebuild(self) -> int:
        """
        Полная пересборка сообществ.
        Возвращает количество сохранённых сообществ.
        """
        if not self._initialized:
            return 0

        try:
            import leidenalg
        except ImportError:
            print("[COMMUNITY] leidenalg not installed, skipping rebuild")
            return 0

        try:
            g = await build_igraph(self.entity_store, self.client, self.triplets_collection)

            if g.vcount() < self.MIN_NODES:
                print(f"[COMMUNITY] Graph too small ({g.vcount()} nodes < {self.MIN_NODES}), skipping")
                return 0

            # Refresh in-memory igraph (atomic: no await between the two assignments)
            try:
                import app_state as _app_state
                from .builder import build_entity_idx
                _app_state.igraph = g
                _app_state.igraph_entity_idx = build_entity_idx(g)
                print(f"[COMMUNITY] igraph refreshed: {g.vcount()} nodes, {g.ecount()} edges")
            except Exception as e:
                print(f"[COMMUNITY] igraph refresh error: {e}")

            # Leiden с фиксированным seed для воспроизводимости
            weights = g.es["weight"] if g.ecount() > 0 and "weight" in g.es.attributes() else None
            partition = leidenalg.find_partition(
                g,
                leidenalg.ModularityVertexPartition,
                weights=weights,
                seed=42,
            )

            # Группируем вершины по сообществам
            communities: Dict[int, List[int]] = {}
            for vertex_idx, community_id in enumerate(partition.membership):
                communities.setdefault(community_id, []).append(vertex_idx)

            print(f"[COMMUNITY] Leiden: {len(communities)} communities found")

            saved = 0
            for community_id, vertex_indices in communities.items():
                if len(vertex_indices) < self.MIN_COMMUNITY:
                    continue

                entity_names = [g.vs[i]["name"] for i in vertex_indices]
                entity_ids = [g.vs[i]["entity_id"] for i in vertex_indices]
                entity_types = [g.vs[i]["entity_type"] for i in vertex_indices]

                # Собираем рёбра внутри сообщества
                subgraph = g.subgraph(vertex_indices)
                triplet_texts = []
                for edge in subgraph.get_edgelist():
                    src_name = subgraph.vs[edge[0]]["name"]
                    tgt_name = subgraph.vs[edge[1]]["name"]
                    triplet_texts.append(f"{src_name} → {tgt_name}")

                # Проверяем: изменился ли состав с прошлой пересборки
                membership_hash = self._membership_hash(entity_ids)
                point_id = self._community_point_id(membership_hash)
                existing = self._get_community(point_id)
                if existing and existing.get("membership_hash") == membership_hash:
                    continue  # состав не изменился — пропускаем LLM вызов

                # Генерируем summary
                summary = await self._generate_summary(entity_names, triplet_texts)

                await self._save_community(
                    point_id=point_id,
                    entity_ids=entity_ids,
                    entity_names=entity_names,
                    entity_types=list(set(entity_types)),
                    summary=summary,
                    membership_hash=membership_hash,
                )
                saved += 1

            self._last_rebuild_at = time.time()
            print(f"[COMMUNITY] Rebuild done: {saved} communities updated")
            return saved

        except Exception as e:
            print(f"[COMMUNITY] Rebuild error: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def _membership_hash(self, entity_ids: List[str]) -> str:
        key = "|".join(sorted(entity_ids))
        return hashlib.md5(key.encode()).hexdigest()

    def _community_point_id(self, membership_hash: str) -> str:
        """UUID-формат из membership_hash — стабильный ID для Qdrant."""
        h = membership_hash
        return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"

    def _get_community(self, point_id: str) -> Optional[dict]:
        try:
            results = self.client.retrieve(
                collection_name=self.COLLECTION,
                ids=[point_id],
                with_payload=True,
            )
            return results[0].payload if results else None
        except Exception:
            return None

    async def _generate_summary(self, entity_names: List[str], triplet_texts: List[str]) -> str:
        try:
            prompt = prompts_config.build_community_summary_prompt(entity_names, triplet_texts)
            summary = await self.llm_client.simple_query(prompt)
            summary = summary.strip()
            # Убираем think-теги (Qwen3)
            if "<think>" in summary and "</think>" in summary:
                summary = summary.split("</think>", 1)[1].strip()
            return summary[:500] if summary else "Тематический кластер сущностей."
        except Exception as e:
            print(f"[COMMUNITY] Summary generation error: {e}")
            return f"Кластер из {len(entity_names)} сущностей: {', '.join(entity_names[:5])}."

    async def _save_community(
        self,
        point_id: str,
        entity_ids: List[str],
        entity_names: List[str],
        entity_types: List[str],
        summary: str,
        membership_hash: str,
    ):
        embedding = self._encode(summary)
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "summary": summary,
                "entity_ids": entity_ids,
                "entity_names": entity_names[:20],
                "entity_types": entity_types,
                "size": len(entity_ids),
                "membership_hash": membership_hash,
                "rebuilt_at": time.time(),
            },
        )
        self.client.upsert(collection_name=self.COLLECTION, points=[point])

    async def search(self, query: str, top_k: int = 3, min_score: float = 0.5) -> List[Dict]:
        """
        Семантический поиск по community summaries.
        Возвращает doc-like dicts для использования как контекст LLM.
        """
        if not self._initialized:
            return []
        try:
            embedding = self._encode(query)
            results = self.client.query_points(
                collection_name=self.COLLECTION,
                query=embedding,
                limit=top_k,
                score_threshold=min_score,
            )
            docs = []
            for hit in results.points:
                docs.append({
                    "text": hit.payload.get("summary", ""),
                    "score": hit.score,
                    "metadata": {
                        "source": "community",
                        "chunk_id": 0,
                        "community_size": hit.payload.get("size", 0),
                        "entity_types": hit.payload.get("entity_types", []),
                    },
                    "from_community": True,
                })
            if docs:
                print(f"[COMMUNITY LOOKUP] '{query[:60]}' → {len(docs)} hits")
            return docs
        except Exception as e:
            print(f"[COMMUNITY] Search error: {e}")
            return []

    async def count(self) -> int:
        if not self._initialized:
            return 0
        try:
            return self.client.get_collection(self.COLLECTION).points_count or 0
        except Exception:
            return 0

    async def triplets_since_last_rebuild(self) -> int:
        """Считает новые триплеты с момента последней пересборки."""
        if self._last_rebuild_at == 0:
            return 0
        try:
            from qdrant_client.models import Filter, FieldCondition, Range
            result = self.client.count(
                collection_name=self.triplets_collection,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="updated_at",
                            range=Range(gte=self._last_rebuild_at),
                        )
                    ]
                ),
            )
            return result.count
        except Exception:
            return 0
