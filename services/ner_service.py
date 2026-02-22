from typing import List, Dict

_ner_model = None

# Entity labels to keep (union of Russian and English spaCy models)
_IMPORTANT_LABELS = {
    "PER", "PERSON",          # persons
    "ORG",                    # organizations
    "GPE", "LOC",             # geo-political entities, locations
    "FAC",                    # facilities
    "LAW",                    # laws / legal documents
    "NORP",                   # nationalities / groups
    "PRODUCT", "EVENT", "WORK_OF_ART",
}


def load_ner_model():
    """Lazy-load spaCy NER model. Tries Russian first, then English."""
    global _ner_model
    if _ner_model is not None:
        return _ner_model if _ner_model is not False else None

    try:
        import spacy
        for model_name in ["ru_core_news_sm", "en_core_web_sm"]:
            try:
                _ner_model = spacy.load(model_name)
                print(f"[NER] Loaded model: {model_name}")
                return _ner_model
            except OSError:
                print(f"[NER] Model '{model_name}' not found, trying next...")

        print("[NER] No model available — NER disabled")
        _ner_model = False
        return None

    except ImportError:
        print("[NER] spaCy not installed — NER disabled")
        _ner_model = False
        return None
    except Exception as e:
        print(f"[NER] Failed to load model: {e}")
        _ner_model = False
        return None


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract named entities from text.

    Returns:
        {"entity_texts": [...], "entity_labels": [...]}
    """
    nlp = load_ner_model()
    if nlp is None:
        return {"entity_texts": [], "entity_labels": []}

    try:
        doc = nlp(text[:100_000])
        entity_texts = []
        entity_labels = []
        seen = set()

        for ent in doc.ents:
            if ent.label_ in _IMPORTANT_LABELS:
                key = (ent.text.lower(), ent.label_)
                if key not in seen:
                    entity_texts.append(ent.text)
                    entity_labels.append(ent.label_)
                    seen.add(key)

        return {"entity_texts": entity_texts, "entity_labels": entity_labels}

    except Exception as e:
        print(f"[NER] Error extracting entities: {e}")
        return {"entity_texts": [], "entity_labels": []}


def compute_entity_overlap_score(query_entities: List[str], doc_entities: List[str]) -> float:
    """Jaccard similarity between query and document entity sets."""
    if not query_entities or not doc_entities:
        return 0.0
    q = set(e.lower() for e in query_entities)
    d = set(e.lower() for e in doc_entities)
    intersection = len(q & d)
    union = len(q | d)
    return intersection / union if union > 0 else 0.0


def boost_results_by_entities(
    results: List[Dict],
    query_entities: List[str],
    boost_weight: float = 0.3
) -> List[Dict]:
    """
    Rerank results using combined score:
        final = (1 - boost_weight) * retrieval_score + boost_weight * entity_overlap

    Args:
        results: search results with 'score' and 'metadata.entity_texts'
        query_entities: entity texts extracted from query
        boost_weight: weight for entity overlap (β)
    """
    if not query_entities or not results:
        return results

    boosted = []
    for result in results:
        doc_entities = result.get("metadata", {}).get("entity_texts", [])
        retrieval_score = result.get("score", 0.0)
        entity_overlap = compute_entity_overlap_score(query_entities, doc_entities)
        final_score = (1 - boost_weight) * retrieval_score + boost_weight * entity_overlap

        r = result.copy()
        r["score"] = final_score
        r["entity_overlap"] = entity_overlap
        r["original_score"] = retrieval_score
        boosted.append(r)

    boosted.sort(key=lambda x: x["score"], reverse=True)
    return boosted
