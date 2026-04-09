"""
Fusion strategies для production RAG pipeline.

Объединяют результаты из нескольких поисковых запросов (аспекты, хопы, варианты).
Работают с форматом Dict результатов из vector_store.search().
"""

from typing import List, Dict
from collections import defaultdict


def weighted_rrf_fusion(
    variant_results: List[List[Dict]],
    top_k: int = 10,
    rrf_k: int = 60,
    original_weight: float = 3.0,
    variant_weight: float = 1.0,
) -> List[Dict]:
    """
    Weighted RRF: оригинальный запрос получает больший вес.

    Args:
        variant_results: [original_results, variant1_results, ...]
        top_k: количество результатов
        rrf_k: константа RRF (стандарт = 60)
        original_weight: вес оригинального запроса
        variant_weight: вес вариантов

    Returns:
        List[Dict] — отсортированные результаты с RRF score
    """
    if not variant_results:
        return []

    weights = [original_weight] + [variant_weight] * (len(variant_results) - 1)

    doc_scores: Dict[str, float] = defaultdict(float)
    doc_data: Dict[str, Dict] = {}

    for variant_idx, results in enumerate(variant_results):
        w = weights[variant_idx]
        for rank, doc in enumerate(results, start=1):
            text_key = doc['text'][:100]
            doc_scores[text_key] += w * (1.0 / (rrf_k + rank))

            if text_key not in doc_data:
                doc_data[text_key] = doc.copy()

    merged = []
    for text_key, rrf_score in doc_scores.items():
        doc = doc_data[text_key].copy()
        doc['rrf_score'] = rrf_score
        doc['score'] = rrf_score
        merged.append(doc)

    merged.sort(key=lambda x: x['score'], reverse=True)
    return merged[:top_k]


def aspect_fusion(
    original_results: List[Dict],
    aspect_results: Dict[str, List[Dict]],
    top_k: int = 10,
    rrf_k: int = 60,
    original_weight: float = 3.0,
    aspect_weight: float = 1.0,
    coverage_bonus: float = 0.5,
) -> List[Dict]:
    """
    Aspect-aware fusion для decomposition.

    Документы, найденные по НЕСКОЛЬКИМ аспектам, получают бонус (coverage).

    Args:
        original_results: результаты по оригинальному запросу
        aspect_results: {aspect_name: [results]} для каждого аспекта
        top_k: количество результатов
        rrf_k: константа RRF
        original_weight: вес оригинального запроса
        aspect_weight: вес каждого аспекта
        coverage_bonus: бонус за каждый дополнительный аспект, в котором найден документ

    Returns:
        List[Dict] — отсортированные результаты
    """
    doc_scores: Dict[str, float] = defaultdict(float)
    doc_aspects: Dict[str, set] = defaultdict(set)
    doc_data: Dict[str, Dict] = {}

    # Original query results
    for rank, doc in enumerate(original_results, start=1):
        text_key = doc['text'][:100]
        doc_scores[text_key] += original_weight * (1.0 / (rrf_k + rank))
        doc_aspects[text_key].add("original")
        if text_key not in doc_data:
            doc_data[text_key] = doc.copy()

    # Aspect results
    for aspect_name, results in aspect_results.items():
        for rank, doc in enumerate(results, start=1):
            text_key = doc['text'][:100]
            doc_scores[text_key] += aspect_weight * (1.0 / (rrf_k + rank))
            doc_aspects[text_key].add(aspect_name)
            if text_key not in doc_data:
                doc_data[text_key] = doc.copy()

    # Coverage multiplier: документы из нескольких аспектов получают бонус
    total_aspects = len(aspect_results) + 1  # +1 for original
    for text_key in doc_scores:
        coverage = len(doc_aspects[text_key]) / total_aspects
        if coverage > 0.5:
            doc_scores[text_key] *= (1.0 + coverage_bonus * coverage)

    merged = []
    for text_key, score in doc_scores.items():
        doc = doc_data[text_key].copy()
        doc['score'] = score
        doc['rrf_score'] = score
        doc['covered_aspects'] = list(doc_aspects[text_key])
        merged.append(doc)

    merged.sort(key=lambda x: x['score'], reverse=True)
    return merged[:top_k]


def multihop_merge(
    hop_results: List[List[Dict]],
    top_k: int = 10,
    later_hop_boost: float = 1.5,
) -> List[Dict]:
    """
    Merge результатов из sequential hops.

    Последние хопы (ближе к финальному ответу) получают больший приоритет.

    Args:
        hop_results: [hop1_results, hop2_results, ...] — в порядке выполнения
        top_k: количество результатов
        later_hop_boost: множитель для каждого следующего хопа

    Returns:
        List[Dict] — объединённые результаты с приоритетом поздних хопов
    """
    if not hop_results:
        return []

    seen_texts = set()
    merged = []

    # Проходим от последнего хопа к первому (поздние важнее)
    for hop_idx in range(len(hop_results) - 1, -1, -1):
        boost = later_hop_boost ** (hop_idx)
        for doc in hop_results[hop_idx]:
            text_key = doc['text'][:100]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                doc_copy = doc.copy()
                doc_copy['score'] = doc.get('score', 0) * boost
                doc_copy['hop_index'] = hop_idx
                merged.append(doc_copy)

    merged.sort(key=lambda x: x['score'], reverse=True)
    return merged[:top_k]
