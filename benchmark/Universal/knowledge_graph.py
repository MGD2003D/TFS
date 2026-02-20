"""
Knowledge Graph storage in Qdrant.

Each point in the graph is a triplet: (subject, predicate, object)
Used as a cache for resolved DAG relationships and entity co-occurrences.

Architecture:
- Collection: knowledge_graph
- Vector: Embedding of triplet text for semantic search
- Payload: subject, predicate, object, confidence, source_chunks
"""

from typing import List, Dict, Optional, Tuple
import hashlib
from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


@dataclass
class Triplet:
    """Knowledge graph triplet (subject, predicate, object)."""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source_chunks: List[str] = None  # Document IDs where this relation was found

    def __post_init__(self):
        if self.source_chunks is None:
            self.source_chunks = []

    def to_text(self) -> str:
        """Convert triplet to text for embedding."""
        return f"{self.subject} {self.predicate} {self.object}"

    def to_id(self) -> str:
        """Generate unique ID for triplet."""
        text = f"{self.subject}|{self.predicate}|{self.object}".lower()
        return hashlib.md5(text.encode()).hexdigest()


class KnowledgeGraph:
    """
    Knowledge graph stored in Qdrant.

    Features:
    - Triplet storage with vector embeddings
    - Semantic search for related triplets
    - Exact filtering by subject/predicate/object
    - Source tracking for invalidation
    """

    def __init__(self, qdrant_client: QdrantClient, embedding_model, collection_name: str = "knowledge_graph"):
        self.client = qdrant_client
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        """Create knowledge_graph collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            # Get vector size from embedding model
            sample_embedding = self.embedding_model.encode(["test"], normalize_embeddings=True)
            vector_size = len(sample_embedding[0])

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"[GRAPH] Created collection '{self.collection_name}' with vector size {vector_size}")

    def add_triplet(self, triplet: Triplet) -> str:
        """
        Add or update a triplet in the graph.

        Args:
            triplet: Triplet to add

        Returns:
            Triplet ID
        """
        triplet_id = triplet.to_id()
        triplet_text = triplet.to_text()

        # Generate embedding
        embedding = self.embedding_model.encode([triplet_text], normalize_embeddings=True)[0]

        # Create point
        point = PointStruct(
            id=triplet_id,
            vector=embedding.tolist(),
            payload={
                "subject": triplet.subject,
                "predicate": triplet.predicate,
                "object": triplet.object,
                "confidence": triplet.confidence,
                "source_chunks": triplet.source_chunks,
                "text": triplet_text
            }
        )

        # Upsert (update if exists, create if not)
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )

        return triplet_id

    def add_triplets_batch(self, triplets: List[Triplet]) -> List[str]:
        """
        Add multiple triplets in batch.

        Args:
            triplets: List of triplets to add

        Returns:
            List of triplet IDs
        """
        if not triplets:
            return []

        triplet_ids = []
        points = []

        # Generate embeddings for all triplets
        triplet_texts = [t.to_text() for t in triplets]
        embeddings = self.embedding_model.encode(triplet_texts, normalize_embeddings=True)

        for triplet, embedding in zip(triplets, embeddings):
            triplet_id = triplet.to_id()
            triplet_ids.append(triplet_id)

            point = PointStruct(
                id=triplet_id,
                vector=embedding.tolist(),
                payload={
                    "subject": triplet.subject,
                    "predicate": triplet.predicate,
                    "object": triplet.object,
                    "confidence": triplet.confidence,
                    "source_chunks": triplet.source_chunks,
                    "text": triplet.to_text()
                }
            )
            points.append(point)

        # Batch upsert
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        return triplet_ids

    def find_by_subject(self, subject: str, limit: int = 10) -> List[Dict]:
        """
        Find triplets by exact subject match.

        Args:
            subject: Subject to search for
            limit: Max results

        Returns:
            List of triplets as dicts
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="subject",
                        match=MatchValue(value=subject)
                    )
                ]
            ),
            limit=limit
        )

        triplets = []
        for point in results[0]:  # scroll returns (points, next_page_offset)
            triplets.append(point.payload)

        return triplets

    def find_related(self, entities: List[str], limit: int = 10) -> List[Dict]:
        """
        Find triplets related to any of the given entities.

        Args:
            entities: List of entity names
            limit: Max results

        Returns:
            List of triplets as dicts
        """
        from qdrant_client.models import Filter, FieldCondition, MatchAny

        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                should=[
                    FieldCondition(key="subject", match=MatchAny(any=entities)),
                    FieldCondition(key="object", match=MatchAny(any=entities))
                ]
            ),
            limit=limit
        )

        triplets = []
        for point in results[0]:
            triplets.append(point.payload)

        return triplets

    def search_semantic(self, query: str, limit: int = 10) -> List[Tuple[Dict, float]]:
        """
        Semantic search for related triplets.

        Args:
            query: Search query
            limit: Max results

        Returns:
            List of (triplet, score) tuples
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)[0]

        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=limit
        )

        triplets_with_scores = []
        for hit in results:
            triplets_with_scores.append((hit.payload, hit.score))

        return triplets_with_scores

    def clear(self):
        """Clear all triplets from the graph."""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            self._ensure_collection()
            print(f"[GRAPH] Cleared collection '{self.collection_name}'")
        except Exception as e:
            print(f"[GRAPH] Error clearing collection: {e}")

    def count(self) -> int:
        """Get total number of triplets in the graph."""
        collection_info = self.client.get_collection(collection_name=self.collection_name)
        return collection_info.points_count


def extract_triplets_from_dag_results(
    dag: Dict,
    results_by_aspect: Dict[str, List[Dict]],
    resolved_answers: Dict[str, str]
) -> List[Triplet]:
    """
    Extract triplets from resolved DAG relationships.

    Example: If DAG resolved "director of Inception" -> "Christopher Nolan",
    creates triplet: ("Inception", "directed_by", "Christopher Nolan")

    Args:
        dag: DAG structure with aspects
        results_by_aspect: Search results for each aspect
        resolved_answers: Resolved answers for each aspect

    Returns:
        List of triplets extracted from DAG
    """
    triplets = []

    for aspect_key, aspect in dag["aspects"].items():
        if aspect_key == "original":
            continue  # Skip original query

        # Get dependencies
        dependencies = aspect.get("dependencies", [])
        if not dependencies:
            continue  # No dependencies = no relationship to store

        # For each dependency, create a triplet
        for dep_key in dependencies:
            if dep_key not in resolved_answers or aspect_key not in resolved_answers:
                continue

            subject = resolved_answers[dep_key][:100]  # Truncate long answers
            predicate = _infer_predicate(aspect.get("query", ""), dag["aspects"][dep_key].get("query", ""))
            obj = resolved_answers[aspect_key][:100]

            # Get source chunks
            source_chunks = []
            if aspect_key in results_by_aspect:
                source_chunks = [
                    r.get("metadata", {}).get("doc_id", "")
                    for r in results_by_aspect[aspect_key][:3]  # Top-3 sources
                    if r.get("metadata", {}).get("doc_id")
                ]

            triplet = Triplet(
                subject=subject.strip(),
                predicate=predicate,
                object=obj.strip(),
                confidence=0.8,  # From DAG pipeline, relatively high confidence
                source_chunks=source_chunks
            )

            triplets.append(triplet)

    return triplets


def _infer_predicate(dependent_query: str, base_query: str) -> str:
    """
    Infer predicate from query patterns.

    Examples:
    - "director of X" -> "directed_by"
    - "spouse of X" -> "spouse_of"
    - "works at X" -> "employed_by"

    Args:
        dependent_query: Query that depends on base
        base_query: Base query

    Returns:
        Inferred predicate
    """
    query_lower = dependent_query.lower()

    # Pattern matching for common predicates
    if "director" in query_lower or "directed" in query_lower:
        return "directed_by"
    elif "spouse" in query_lower or "married" in query_lower:
        return "spouse_of"
    elif "work" in query_lower or "employ" in query_lower:
        return "employed_by"
    elif "author" in query_lower or "wrote" in query_lower:
        return "authored_by"
    elif "located" in query_lower or "where" in query_lower:
        return "located_in"
    elif "founded" in query_lower or "created" in query_lower:
        return "founded_by"
    elif "parent" in query_lower or "child" in query_lower:
        return "parent_of"
    else:
        # Generic relationship
        return "related_to"
