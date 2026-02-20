#!/usr/bin/env python3
"""
Fusion strategies for multi-query retrieval.

Combines results from multiple query variants (original + LLM-enhanced alternatives).
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
    debug: bool = False
) -> List[Hit]:
    """
    Max-Score Fusion: Take maximum score across all query variants.

    Pros:
    - Simple and interpretable
    - Guarantees non-degradation (original query always contributes)

    Cons:
    - Ignores frequency information (doc in 1 query vs 5 queries same)
    - May miss documents that rank consistently high across variants

    Args:
        variant_results: List of result lists, one per query variant
                        Each result is Dict with keys: metadata.doc_id, text, score
        top_k: Number of results to return

    Returns:
        List of Hit objects sorted by max score
    """
    doc_max_scores: Dict[str, Tuple[float, str]] = {}  # doc_id -> (max_score, text)

    # DEBUG: Check what we're receiving (optional)
    if debug:
        total_docs = sum(len(results) for results in variant_results)
        if total_docs > 0:
            sample_doc = variant_results[0][0] if variant_results and variant_results[0] else None
            print(f"[FUSION DEBUG] Total docs: {total_docs}")
            print(f"[FUSION DEBUG] Sample doc keys: {list(sample_doc.keys()) if sample_doc else 'None'}")
            print(f"[FUSION DEBUG] Sample doc: {sample_doc}")

    for results in variant_results:
        for d in results:
            doc_id = d.get("metadata", {}).get("doc_id", "")
            if not doc_id:
                if debug:
                    print(f"[FUSION DEBUG] Skipping doc with empty doc_id. Keys: {list(d.keys())}")
                continue

            score = float(d.get("score", 0.0))
            text = d.get("text", "")

            if doc_id not in doc_max_scores or score > doc_max_scores[doc_id][0]:
                doc_max_scores[doc_id] = (score, text)

    # Convert to Hit objects
    hits = [
        Hit(doc_id=doc_id, text=text, score=score)
        for doc_id, (score, text) in doc_max_scores.items()
    ]

    # Sort by max score (descending)
    hits.sort(key=lambda x: x.score, reverse=True)

    return hits[:top_k]


def rrf_fusion(
    variant_results: List[List[Dict]],
    top_k: int = 10,
    rrf_k: int = 60,
    weights: List[float] = None,
    debug: bool = False
) -> List[Hit]:
    """
    Reciprocal Rank Fusion (RRF): Combine rankings using reciprocal rank formula.

    Formula: RRF_score(doc) = Σ weight_i * (1 / (k + rank_i))

    Pros:
    - Robust to score scale differences
    - Rewards documents appearing in multiple variants
    - Widely used in IR (e.g., Elasticsearch)

    Cons:
    - Can be hurt by noisy variants (multiple bad rankings accumulate)
    - Doesn't guarantee non-degradation

    Args:
        variant_results: List of result lists, one per query variant
        top_k: Number of results to return
        rrf_k: RRF constant (default 60, standard value from literature)
        weights: Optional weights for each variant (default: equal weights)
                 weights[0] should be for original query (higher weight recommended)

    Returns:
        List of Hit objects sorted by RRF score
    """
    if weights is None:
        weights = [1.0] * len(variant_results)

    assert len(weights) == len(variant_results), "weights must match number of variants"

    doc_rrf_scores: Dict[str, float] = defaultdict(float)
    doc_texts: Dict[str, str] = {}

    for variant_idx, results in enumerate(variant_results):
        weight = weights[variant_idx]

        for rank, d in enumerate(results, start=1):
            doc_id = d.get("metadata", {}).get("doc_id", "")
            if not doc_id:
                continue

            # RRF score contribution
            doc_rrf_scores[doc_id] += weight * (1.0 / (rrf_k + rank))

            # Store text (from first occurrence)
            if doc_id not in doc_texts:
                doc_texts[doc_id] = d.get("text", "")

    # Convert to Hit objects
    hits = [
        Hit(doc_id=doc_id, text=doc_texts[doc_id], score=score)
        for doc_id, score in doc_rrf_scores.items()
    ]

    # Sort by RRF score (descending)
    hits.sort(key=lambda x: x.score, reverse=True)

    return hits[:top_k]


def weighted_rrf_fusion(
    variant_results: List[List[Dict]],
    top_k: int = 10,
    rrf_k: int = 60,
    original_weight: float = 2.0,
    variant_weight: float = 0.5,
    debug: bool = False
) -> List[Hit]:
    """
    Weighted RRF: Give higher weight to original query, lower to variants.

    This is RRF with preset weights favoring the original query.

    Recommended weights:
    - original_weight: 2.0 (double weight for original)
    - variant_weight: 0.5 (half weight for variants)

    Args:
        variant_results: List of result lists, one per query variant
                        variant_results[0] MUST be original query
        top_k: Number of results to return
        rrf_k: RRF constant
        original_weight: Weight for original query (first variant)
        variant_weight: Weight for all other variants

    Returns:
        List of Hit objects sorted by weighted RRF score
    """
    weights = [original_weight] + [variant_weight] * (len(variant_results) - 1)
    return rrf_fusion(variant_results, top_k=top_k, rrf_k=rrf_k, weights=weights, debug=debug)


def hybrid_fusion(
    variant_results: List[List[Dict]],
    top_k: int = 10,
    original_boost: float = 1.5,
    variant_discount: float = 0.5,
    debug: bool = False
) -> List[Hit]:
    """
    Hybrid Fusion: Original query gets priority, variants add NEW documents.

    Strategy:
    1. Original query results get boosted scores
    2. Variant results contribute NEW documents (not in original) with discounted scores
    3. Merge and sort

    Pros:
    - Protects original query ranking
    - Variants can still contribute novel relevant documents
    - Good balance between safety and exploration

    Cons:
    - More complex logic
    - May underutilize variant information

    Args:
        variant_results: List of result lists, one per query variant
                        variant_results[0] MUST be original query
        top_k: Number of results to return
        original_boost: Score multiplier for original query results (>1.0 boosts)
        variant_discount: Score multiplier for variant results (<1.0 discounts)

    Returns:
        List of Hit objects sorted by hybrid score
    """
    if not variant_results:
        return []

    # Step 1: Get original query results and their IDs
    original_results = variant_results[0]
    original_doc_ids = set()
    doc_scores: Dict[str, float] = {}
    doc_texts: Dict[str, str] = {}

    for d in original_results:
        doc_id = d.get("metadata", {}).get("doc_id", "")
        if not doc_id:
            continue

        score = float(d.get("score", 0.0))
        text = d.get("text", "")

        # Boost original query scores
        doc_scores[doc_id] = score * original_boost
        doc_texts[doc_id] = text
        original_doc_ids.add(doc_id)

    # Step 2: Add NEW documents from variants (with discount)
    for variant_results_i in variant_results[1:]:
        for d in variant_results_i:
            doc_id = d.get("metadata", {}).get("doc_id", "")
            if not doc_id:
                continue

            # Skip documents already in original
            if doc_id in original_doc_ids:
                continue

            score = float(d.get("score", 0.0))
            text = d.get("text", "")

            # Take max score among variants (with discount)
            discounted_score = score * variant_discount
            if doc_id not in doc_scores or discounted_score > doc_scores[doc_id]:
                doc_scores[doc_id] = discounted_score
                doc_texts[doc_id] = text

    # Step 3: Convert to Hit objects and sort
    hits = [
        Hit(doc_id=doc_id, text=doc_texts[doc_id], score=score)
        for doc_id, score in doc_scores.items()
    ]

    hits.sort(key=lambda x: x.score, reverse=True)

    return hits[:top_k]


def conservative_fusion(
    variant_results: List[List[Dict]],
    top_k: int = 10,
    rrf_k: int = 60,
    debug: bool = False
) -> List[Hit]:
    """
    Conservative Fusion: NEVER-WORSE guarantee.

    Strategy:
    1. Lock original query top-K documents (they MUST be in final top-K)
    2. Compute RRF scores for ALL documents (including locked ones)
    3. Re-rank locked documents by RRF (allows improvement within top-K)
    4. Remaining positions filled by unlocked documents

    Guarantee: Original top-K documents always in final top-K (but may reorder).

    Pros:
    - Guarantees non-degradation at top-K level
    - Still allows variants to improve ranking within top-K
    - Very safe for production

    Cons:
    - May miss good documents outside original top-K
    - Conservative (less exploration)

    Args:
        variant_results: List of result lists, one per query variant
                        variant_results[0] MUST be original query
        top_k: Number of results to return
        rrf_k: RRF constant

    Returns:
        List of Hit objects with guaranteed non-degradation
    """
    if not variant_results:
        return []

    # Step 1: Get original top-K document IDs (LOCKED)
    original_results = variant_results[0]
    locked_doc_ids = set()

    for i, d in enumerate(original_results[:top_k]):
        doc_id = d.get("metadata", {}).get("doc_id", "")
        if doc_id:
            locked_doc_ids.add(doc_id)

    # Step 2: Compute RRF scores for ALL documents
    doc_rrf_scores: Dict[str, float] = defaultdict(float)
    doc_texts: Dict[str, str] = {}

    for results in variant_results:
        for rank, d in enumerate(results, start=1):
            doc_id = d.get("metadata", {}).get("doc_id", "")
            if not doc_id:
                continue

            doc_rrf_scores[doc_id] += 1.0 / (rrf_k + rank)

            if doc_id not in doc_texts:
                doc_texts[doc_id] = d.get("text", "")

    # Step 3: Separate locked and unlocked documents
    locked_hits = []
    unlocked_hits = []

    for doc_id, score in doc_rrf_scores.items():
        hit = Hit(doc_id=doc_id, text=doc_texts[doc_id], score=score)

        if doc_id in locked_doc_ids:
            locked_hits.append(hit)
        else:
            unlocked_hits.append(hit)

    # Step 4: Sort both groups
    locked_hits.sort(key=lambda x: x.score, reverse=True)
    unlocked_hits.sort(key=lambda x: x.score, reverse=True)

    # Step 5: Final ranking: locked top-K + unlocked rest
    # Note: If locked < top_k (some docs not found), fill with unlocked
    final_hits = locked_hits + unlocked_hits

    return final_hits[:top_k]


# Strategy registry for easy lookup
FUSION_STRATEGIES = {
    "max_score": max_score_fusion,
    "rrf": rrf_fusion,
    "weighted_rrf": weighted_rrf_fusion,
    "hybrid": hybrid_fusion,
    "conservative": conservative_fusion,
}


def get_fusion_strategy(name: str):
    """Get fusion strategy by name."""
    if name not in FUSION_STRATEGIES:
        raise ValueError(
            f"Unknown fusion strategy: {name}. "
            f"Available: {list(FUSION_STRATEGIES.keys())}"
        )
    return FUSION_STRATEGIES[name]
