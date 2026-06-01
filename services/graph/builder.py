"""
build_igraph — собирает in-memory igraph.Graph из Qdrant (entities + triplets).

Хранится в app_state.igraph и обновляется:
  - при старте (build_igraph из Qdrant)
  - инкрементально при save_triplets_batch
  - полностью при CommunityManager.rebuild() (Leiden требует полный граф)
"""

from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    import igraph as ig
    from .entities import EntityStore


async def build_igraph(entity_store: "EntityStore", qdrant_client, triplets_collection: str):
    """
    Строит undirected igraph.Graph из сущностей и триплетов.

    Returns:
        igraph.Graph с атрибутами вершин (entity_id, name, entity_type)
        и рёбер (weight, predicate)
    """
    import igraph as ig

    all_entities = await entity_store.get_all()
    if not all_entities:
        return ig.Graph(directed=False)

    entity_to_idx = {str(e.id): i for i, e in enumerate(all_entities)}

    g = ig.Graph(directed=False)
    g.add_vertices(len(all_entities))
    g.vs["entity_id"]   = [str(e.id) for e in all_entities]
    g.vs["name"]        = [e.payload.get("name", "") for e in all_entities]
    g.vs["entity_type"] = [e.payload.get("entity_type", "other") for e in all_entities]

    edges, weights, predicates = [], [], []
    next_offset = None

    while True:
        batch, next_offset = qdrant_client.scroll(
            collection_name=triplets_collection,
            limit=1000,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in batch:
            s_id = point.payload.get("subject_entity_id")
            o_id = point.payload.get("object_entity_id")
            if s_id and o_id and s_id in entity_to_idx and o_id in entity_to_idx:
                edges.append((entity_to_idx[s_id], entity_to_idx[o_id]))
                weights.append(float(point.payload.get("confidence", 1.0)))
                predicates.append(point.payload.get("predicate", "related_to"))

        if next_offset is None:
            break

    if edges:
        g.add_edges(edges)
        g.es["weight"]    = weights
        g.es["predicate"] = predicates

    print(f"[BUILDER] Graph: {g.vcount()} nodes, {g.ecount()} edges")
    return g


def build_entity_idx(g) -> Dict[str, int]:
    """Строит словарь entity_id → vertex_index для O(1) поиска."""
    return {g.vs[i]["entity_id"]: i for i in range(g.vcount())}


def find_neighbors(g, entity_idx: Dict[str, int], entity_id: str,
                   max_edges: int = 15) -> List[Dict]:
    """
    Возвращает соседей сущности как doc-like dicts для контекста LLM.

    Args:
        g: igraph.Graph
        entity_idx: словарь entity_id → vertex_index
        entity_id: ID сущности-точки входа
        max_edges: максимум рёбер для возврата
    """
    vertex_idx = entity_idx.get(entity_id)
    if vertex_idx is None:
        return []

    vertex = g.vs[vertex_idx]
    has_predicates = "predicate" in g.es.attributes()
    docs = []

    for edge_idx in g.incident(vertex_idx, mode="all")[:max_edges]:
        e = g.es[edge_idx]
        src, tgt = e.tuple
        other_idx = tgt if src == vertex_idx else src
        other = g.vs[other_idx]
        predicate = e["predicate"] if has_predicates else "→"
        fact = f"{vertex['name']} {predicate} {other['name']}"
        docs.append({
            "text": fact,
            "score": float(e["weight"]) if "weight" in g.es.attributes() else 0.8,
            "metadata": {
                "source": "knowledge_graph_local",
                "chunk_id": 0,
                "subject": vertex["name"],
                "predicate": predicate,
                "object": other["name"],
            },
            "from_graph": True,
        })

    return docs
