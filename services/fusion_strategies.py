"""
Fusion strategies for multi-query retrieval.

Combines results from multiple query variants (original + decomposed aspects).
"""

from typing import List, Dict, Tuple
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Hit:
    """Single search result."""
    doc_id: str
    text: str
    score: float

    def __hash__(self):
        return hash(self.doc_id)


def max_score_fusion(
    variant_results: List[List[Dict]],
    top_k: int = 10,
) -> List[Hit]:
    """
    Max-Score Fusion: Take maximum score across all query variants.

    Pros:
    - Simple and interpretable
    - Guarantees non-degradation (original query always contributes)

    Args:
        variant_results: List of result lists, one per query variant.
                        Each result is Dict with keys: metadata.doc_id, text, score.
        top_k: Number of results to return.

    Returns:
        List of Hit objects sorted by max score.
    """
    doc_max_scores: Dict[str, Tuple[float, str]] = {}

    for results in variant_results:
        for d in results:
            doc_id = d.get("text", "")[:100]
            if not doc_id:
                continue

            score = float(d.get("score", 0.0))
            text = d.get("text", "")

            if doc_id not in doc_max_scores or score > doc_max_scores[doc_id][0]:
                doc_max_scores[doc_id] = (score, text)

    hits = [
        Hit(doc_id=doc_id, text=text, score=score)
        for doc_id, (score, text) in doc_max_scores.items()
    ]
    hits.sort(key=lambda x: x.score, reverse=True)
    return hits[:top_k]


def rrf_fusion(
    variant_results: List[List[Dict]],
    top_k: int = 10,
    rrf_k: int = 60,
    weights: List[float] = None,
) -> List[Hit]:
    """
    Reciprocal Rank Fusion (RRF): Combine rankings using reciprocal rank formula.

    Formula: RRF_score(doc) = Σ weight_i * (1 / (k + rank_i))

    Args:
        variant_results: List of result lists, one per query variant.
        top_k: Number of results to return.
        rrf_k: RRF constant (default 60, standard value from literature).
        weights: Optional per-variant weights. Default: equal weights.

    Returns:
        List of Hit objects sorted by RRF score.
    """
    if weights is None:
        weights = [1.0] * len(variant_results)

    assert len(weights) == len(variant_results), "weights must match number of variants"

    doc_rrf_scores: Dict[str, float] = defaultdict(float)
    doc_texts: Dict[str, str] = {}

    for variant_idx, results in enumerate(variant_results):
        weight = weights[variant_idx]

        for rank, d in enumerate(results, start=1):
            doc_id = d.get("text", "")[:100]
            if not doc_id:
                continue

            doc_rrf_scores[doc_id] += weight * (1.0 / (rrf_k + rank))

            if doc_id not in doc_texts:
                doc_texts[doc_id] = d.get("text", "")

    hits = [
        Hit(doc_id=doc_id, text=doc_texts[doc_id], score=score)
        for doc_id, score in doc_rrf_scores.items()
    ]
    hits.sort(key=lambda x: x.score, reverse=True)
    return hits[:top_k]


def weighted_rrf_fusion(
    variant_results: List[List[Dict]],
    top_k: int = 10,
    rrf_k: int = 60,
    original_weight: float = 3.0,
    aspect_weight: float = 1.0,
) -> List[Hit]:
    """
    Weighted RRF with coverage multiplier for decomposition-based retrieval.

    Gives higher weight to the original query, lower to decomposed aspects.
    Applies a coverage multiplier: documents appearing in more aspects rank higher.

    Args:
        variant_results: List of result lists. variant_results[0] MUST be original query.
        top_k: Number of results to return.
        rrf_k: RRF constant.
        original_weight: Weight for original query (default 3.0 — optimal per benchmarks).
        aspect_weight: Weight for each aspect query (default 1.0).

    Returns:
        List of Hit objects sorted by weighted RRF × coverage score.
    """
    if not variant_results:
        return []

    weights = [original_weight] + [aspect_weight] * (len(variant_results) - 1)
    total_aspects = len(variant_results) - 1  # Exclude original

    doc_rrf_scores: Dict[str, float] = defaultdict(float)
    doc_texts: Dict[str, str] = {}
    doc_aspect_hits: Dict[str, int] = defaultdict(int)  # How many aspects matched this doc

    for variant_idx, results in enumerate(variant_results):
        weight = weights[variant_idx]
        is_aspect = variant_idx > 0

        for rank, d in enumerate(results, start=1):
            # Use chunk text prefix as unique key (document_id + chunk_id would also work
            # but text prefix is simpler and avoids metadata key name mismatches)
            doc_id = d.get("text", "")[:100]
            if not doc_id:
                continue

            doc_rrf_scores[doc_id] += weight * (1.0 / (rrf_k + rank))

            if doc_id not in doc_texts:
                doc_texts[doc_id] = d.get("text", "")

            if is_aspect:
                doc_aspect_hits[doc_id] += 1

    # Apply coverage multiplier: 1.0 + (aspects_covered / total_aspects)
    hits = []
    for doc_id, rrf_score in doc_rrf_scores.items():
        coverage = doc_aspect_hits[doc_id] / total_aspects if total_aspects > 0 else 0.0
        final_score = rrf_score * (1.0 + coverage)
        hits.append(Hit(doc_id=doc_id, text=doc_texts[doc_id], score=final_score))

    hits.sort(key=lambda x: x.score, reverse=True)
    return hits[:top_k]


def conservative_fusion(
    variant_results: List[List[Dict]],
    top_k: int = 10,
    rrf_k: int = 60,
) -> List[Hit]:
    """
    Conservative Fusion: NEVER-WORSE guarantee.

    Locks original query top-K documents in final results, re-ranks by RRF within locked set.

    Args:
        variant_results: List of result lists. variant_results[0] MUST be original query.
        top_k: Number of results to return.
        rrf_k: RRF constant.

    Returns:
        List of Hit objects with guaranteed non-degradation vs baseline.
    """
    if not variant_results:
        return []

    original_results = variant_results[0]
    locked_doc_ids = {
        d.get("text", "")[:100]
        for d in original_results[:top_k]
        if d.get("text", "")[:100]
    }

    doc_rrf_scores: Dict[str, float] = defaultdict(float)
    doc_texts: Dict[str, str] = {}

    for results in variant_results:
        for rank, d in enumerate(results, start=1):
            doc_id = d.get("text", "")[:100]
            if not doc_id:
                continue

            doc_rrf_scores[doc_id] += 1.0 / (rrf_k + rank)

            if doc_id not in doc_texts:
                doc_texts[doc_id] = d.get("text", "")

    locked_hits = []
    unlocked_hits = []

    for doc_id, score in doc_rrf_scores.items():
        hit = Hit(doc_id=doc_id, text=doc_texts[doc_id], score=score)
        if doc_id in locked_doc_ids:
            locked_hits.append(hit)
        else:
            unlocked_hits.append(hit)

    locked_hits.sort(key=lambda x: x.score, reverse=True)
    unlocked_hits.sort(key=lambda x: x.score, reverse=True)

    return (locked_hits + unlocked_hits)[:top_k]


FUSION_STRATEGIES = {
    "max_score": max_score_fusion,
    "rrf": rrf_fusion,
    "weighted_rrf": weighted_rrf_fusion,
    "conservative": conservative_fusion,
}


def get_fusion_strategy(name: str):
    """Get fusion strategy function by name."""
    if name not in FUSION_STRATEGIES:
        raise ValueError(
            f"Unknown fusion strategy: '{name}'. "
            f"Available: {list(FUSION_STRATEGIES.keys())}"
        )
    return FUSION_STRATEGIES[name]
