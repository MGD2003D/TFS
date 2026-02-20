import argparse
import asyncio
import math
import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json

import ir_datasets

# Import fusion strategies
from fusion_strategies import get_fusion_strategy, Hit as FusionHit

# Import knowledge graph for caching resolved DAG relationships
from knowledge_graph import KnowledgeGraph, Triplet, extract_triplets_from_dag_results

# =============================================================================
# Global Configuration
# =============================================================================

# Entity-aware retrieval config (set from command-line args)
_entity_boost_enabled = False
_entity_boost_weight = 0.3


# =============================================================================
# Reranker (lazy loaded)
# =============================================================================

_reranker = None

def get_reranker():
    """Lazy load cross-encoder reranker (only when --rerank flag is used)."""
    global _reranker
    if _reranker is None:
        import torch
        from sentence_transformers import CrossEncoder

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[RERANKER] Loading cross-encoder/ms-marco-MiniLM-L-6-v2 on {device.upper()}...")
        _reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
        print(f"[RERANKER] Model loaded successfully on {device.upper()}")
    return _reranker


def rerank_hits(query: str, hits: List, top_k: int = 10, logger: Optional = None) -> List:
    """
    Rerank hits using cross-encoder.

    Args:
        query: Original query
        hits: List of Hit objects from retrieval
        top_k: Number of results to return after reranking
        logger: Optional logger

    Returns:
        Reranked list of Hit objects
    """
    if not hits:
        return hits

    reranker = get_reranker()

    # Prepare query-document pairs
    pairs = [[query, hit.text] for hit in hits]

    if logger:
        logger.log(f"\n[RERANK] Reranking {len(hits)} candidates...")

    # Get cross-encoder scores (with batching for performance)
    t0 = time.perf_counter()
    scores = reranker.predict(pairs, batch_size=128, show_progress_bar=False)
    rerank_time = (time.perf_counter() - t0) * 1000

    # Create new Hit objects with reranker scores
    from dataclasses import replace
    reranked_hits = []
    for hit, score in zip(hits, scores):
        reranked_hits.append(replace(hit, score=float(score)))

    # Sort by reranker scores
    reranked_hits.sort(key=lambda x: x.score, reverse=True)
    result = reranked_hits[:top_k]

    if logger:
        logger.log(f"[RERANK] Completed in {rerank_time:.1f}ms")
        logger.log(f"[RERANK] Top-3 scores after reranking:")
        for i, hit in enumerate(result[:3], 1):
            logger.log(f"  {i}. {hit.doc_id[:40]}... (score: {hit.score:.4f})")

    return result


def batch_rerank_all_queries(queries: Dict[str, str], all_hits: Dict[str, List], top_k: int = 10) -> Dict[str, List]:
    """
    Batch rerank ALL queries at once for maximum performance (benchmark mode only).

    Instead of reranking each query separately (648 × 6s = 65 min),
    collect all pairs and rerank in one batch (~1-2 min for 64,800 pairs).

    Args:
        queries: Dict mapping qid → query text
        all_hits: Dict mapping qid → list of Hit objects from retrieval
        top_k: Number of results to return after reranking

    Returns:
        Dict mapping qid → reranked list of Hit objects
    """
    from dataclasses import replace

    if not queries:
        return {}

    print(f"\n[BATCH RERANK] Collecting query-document pairs from {len(queries)} queries...")

    # Step 1: Collect ALL (query, doc) pairs
    all_pairs = []
    query_map = {}  # {qid: (offset, count, hits)}

    for qid, qtext in queries.items():
        hits = all_hits.get(qid, [])
        if not hits:
            query_map[qid] = (0, 0, [])
            continue

        # Record offset, count, AND hits for this query
        offset = len(all_pairs)
        count = len(hits)
        query_map[qid] = (offset, count, hits)

        # Add pairs to batch
        for hit in hits:
            all_pairs.append([qtext, hit.text])

    print(f"[BATCH RERANK] Collected {len(all_pairs)} pairs from {len(queries)} queries")

    if not all_pairs:
        return {qid: [] for qid in queries}

    # Step 2: ONE batch reranking call
    reranker = get_reranker()

    print(f"[BATCH RERANK] Reranking {len(all_pairs)} pairs in batch (this may take 1-2 minutes)...")
    t0 = time.perf_counter()
    scores = reranker.predict(all_pairs, batch_size=512, show_progress_bar=True)
    rerank_time = (time.perf_counter() - t0)

    print(f"[BATCH RERANK] Completed in {rerank_time:.1f}s ({len(all_pairs)/rerank_time:.0f} pairs/s)")

    # Step 3: Split scores back to queries and rerank each
    results = {}
    for qid in queries:
        offset, count, hits = query_map[qid]

        if count == 0:
            results[qid] = []
            continue

        # Extract scores for this query
        query_scores = scores[offset:offset+count]

        # Create new Hit objects with reranker scores
        reranked_hits = []
        for hit, score in zip(hits, query_scores):
            reranked_hits.append(replace(hit, score=float(score)))

        # Sort by reranker scores and take top-k
        reranked_hits.sort(key=lambda x: x.score, reverse=True)
        results[qid] = reranked_hits[:top_k]

    return results


# =============================================================================
# Project setup
# =============================================================================

def add_project_root_to_syspath(project_root: Optional[str] = None):
    if project_root:
        root = Path(project_root).resolve()
    else:
        root = Path(__file__).resolve().parents[2]

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        from dotenv import load_dotenv
        env_path = root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except Exception:
        pass

    print(f"[PATH] project_root = {root}")
    return root


# =============================================================================
# Logging utility
# =============================================================================

class Logger:
    """Dual logger that writes to both console and file."""
    def __init__(self, filepath: Optional[str] = None, console_details: bool = False):
        self.filepath = filepath
        self.file = None
        self.console_details = console_details
        if filepath:
            # Create directory if it doesn't exist
            import os
            dirpath = os.path.dirname(filepath)
            if dirpath:  # Only create if there's a directory path
                os.makedirs(dirpath, exist_ok=True)
            self.file = open(filepath, 'w', encoding='utf-8')
            print(f"[LOG] Detailed logs will be saved to: {filepath}")
    
    def log(self, message: str):
        """Write message to console (if enabled) and file."""
        if self.console_details:
            print(message)
        if self.file:
            self.file.write(message + '\n')
            self.file.flush()
    
    def close(self):
        if self.file:
            self.file.close()


# =============================================================================
# API Key management with retry
# =============================================================================

async def init_llm_with_key(api_key: Optional[str] = None):
    """Initialize LLM with optional API key override."""
    import app_state
    from services.llm.caila_client import CailaClient

    if api_key:
        os.environ["CAILA_TOKEN"] = api_key

    client = CailaClient()
    app_state.llm_client = client
    await client.initialize()
    return client


def prompt_for_api_key():
    """Prompt user for new API key."""
    print("\n" + "="*60)
    print("[INPUT REQUIRED] Please enter new API key:")
    print("="*60)
    api_key = input("API Key: ").strip()
    return api_key


async def call_with_retry(func, *args, max_retries=7, **kwargs):
    """
    Call async function with retry logic.
    After max_retries, prompt for new API key.
    """
    retries = 0
    current_api_key = None
    
    while True:
        try:
            result = await func(*args, **kwargs)
            return result
        
        except Exception as e:
            retries += 1
            error_msg = str(e)
            
            # Check if it's an API error (rate limit, auth, etc.)
            is_api_error = any(keyword in error_msg.lower() for keyword in 
                             ['api', 'auth', 'key', 'rate', 'quota', 'limit', 'timeout'])
            
            if retries <= max_retries:
                print(f"[RETRY {retries}/{max_retries}] Error: {error_msg}")
                print(f"[RETRY] Waiting 3 seconds before retry...")
                await asyncio.sleep(3)
            else:
                print(f"\n[ERROR] Max retries ({max_retries}) reached!")
                print(f"[ERROR] Last error: {error_msg}")
                
                if is_api_error:
                    print("\n[SOLUTION] This looks like an API error.")
                    print("Possible causes:")
                    print("  - API key expired or out of credits")
                    print("  - Rate limit exceeded")
                    print("  - Network issues")
                    
                    # Prompt for new API key
                    current_api_key = prompt_for_api_key()
                    
                    if not current_api_key:
                        print("[EXIT] No API key provided. Exiting...")
                        raise
                    
                    print("[REINIT] Reinitializing LLM with new API key...")
                    
                    # Reinitialize LLM with new key
                    import app_state
                    if hasattr(app_state, 'llm_client') and app_state.llm_client:
                        await app_state.llm_client.cleanup()
                    
                    await init_llm_with_key(current_api_key)
                    
                    # Also update enhancer if needed
                    from services.query_enhancer import QueryEnhancerService
                    app_state.query_enhancer = QueryEnhancerService()
                    
                    # Reset retry counter and try again
                    retries = 0
                    print("[RETRY] Continuing with new API key...")
                    continue
                else:
                    # Not an API error - just raise
                    raise


# =============================================================================
# Vector store and services
# =============================================================================

async def init_vector_store(
    collection_name: str,
    enable_hybrid: bool = True,
    embedding_model: str = "intfloat/multilingual-e5-base"
):
    from services.vectorstore.qdrant_client import QdrantVectorStore

    store = QdrantVectorStore(
        collection_name=collection_name,
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
        embedding_model=embedding_model,
        enable_hybrid_search=enable_hybrid,
    )
    await store.initialize()
    return store


async def init_query_enhancer():
    import app_state
    from services.query_enhancer import QueryEnhancerService

    enhancer = QueryEnhancerService()
    app_state.query_enhancer = enhancer
    return enhancer


# =============================================================================
# Dataset loading
# =============================================================================

def load_ds(dataset_id: str):
    return ir_datasets.load(dataset_id)


def take_queries(ds, max_queries: int) -> Dict[str, str]:
    """Load queries from dataset."""
    queries = {}
    
    # BEIR datasets use different structure
    if hasattr(ds, 'queries_iter'):
        for i, q in enumerate(ds.queries_iter()):
            queries[q.query_id] = q.text if hasattr(q, 'text') else q.default_text()
            if max_queries and i + 1 >= max_queries:
                break
    else:
        raise AttributeError(f"Dataset {ds} does not have queries_iter()")
    
    return queries


def build_qrels(ds, allowed_qids: set) -> Dict[str, List[str]]:
    """Build relevance judgments (qrels) from dataset."""
    qrels = defaultdict(list)
    
    # Try different qrels access methods
    if hasattr(ds, 'qrels_iter'):
        for qr in ds.qrels_iter():
            if qr.query_id in allowed_qids and qr.relevance > 0:
                qrels[qr.query_id].append(qr.doc_id)
    elif hasattr(ds, 'qrels'):
        # BEIR datasets might have qrels as dict
        for qr in ds.qrels:
            if qr.query_id in allowed_qids and qr.relevance > 0:
                qrels[qr.query_id].append(qr.doc_id)
    else:
        print(f"[WARNING] Dataset does not have qrels, using empty qrels")
        return {}
    
    return dict(qrels)


def build_full_corpus(ds) -> Dict[str, str]:
    """Load entire corpus from dataset."""
    print("[CORPUS] Loading full corpus (this may take a while)...")
    docs = {}
    count = 0
    t0 = time.perf_counter()

    for d in ds.docs_iter():
        # Handle different text field names
        text = ""
        if hasattr(d, 'text'):
            text = d.text
        elif hasattr(d, 'default_text'):
            text = d.default_text()
        elif hasattr(d, 'title') and hasattr(d, 'text'):
            text = f"{d.title} {d.text}"
        else:
            # Try to get any text field
            text = str(d)

        docs[d.doc_id] = text
        count += 1

        if count % 100000 == 0:
            dt = time.perf_counter() - t0
            speed = count / dt if dt > 0 else 0
            print(f"[CORPUS] Loaded {count} docs | {speed:.1f} docs/s")

    print(f"[CORPUS] Full corpus loaded: {len(docs)} documents in {time.perf_counter()-t0:.1f}s")
    return docs


# =============================================================================
# NER (Named Entity Recognition) for entity-aware indexing
# =============================================================================

_ner_model = None

def get_ner_model():
    """Lazy load spaCy NER model."""
    global _ner_model
    if _ner_model is None:
        try:
            import spacy
            print("[NER] Loading spaCy model (en_core_web_sm)...")
            try:
                _ner_model = spacy.load("en_core_web_sm")
                print("[NER] spaCy model loaded successfully")
            except OSError:
                print("[NER] Model not found, downloading en_core_web_sm...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                _ner_model = spacy.load("en_core_web_sm")
                print("[NER] spaCy model downloaded and loaded")
        except Exception as e:
            print(f"[NER] Warning: Failed to load spaCy: {e}")
            print("[NER] NER extraction will be disabled")
            _ner_model = False  # Mark as failed to avoid repeated attempts
    return _ner_model if _ner_model is not False else None


def extract_entities(text: str, max_length: int = 1000000) -> List[Dict[str, str]]:
    """
    Extract named entities from text using spaCy.

    Args:
        text: Input text
        max_length: Max text length (spaCy has limits)

    Returns:
        List of entities: [{"text": "Apple", "label": "ORG"}, ...]
    """
    nlp = get_ner_model()
    if nlp is None:
        return []

    try:
        # Truncate text if too long (spaCy has limits)
        if len(text) > max_length:
            text = text[:max_length]

        doc = nlp(text)
        entities = []
        seen = set()  # Deduplicate entities

        for ent in doc.ents:
            # Filter to important entity types
            if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"]:
                entity_key = (ent.text.lower(), ent.label_)
                if entity_key not in seen:
                    entities.append({
                        "text": ent.text,
                        "label": ent.label_
                    })
                    seen.add(entity_key)

        return entities
    except Exception as e:
        print(f"[NER] Error extracting entities: {e}")
        return []


def compute_entity_overlap_score(query_entities: List[str], doc_entities: List[str]) -> float:
    """
    Compute entity overlap score between query and document.

    Args:
        query_entities: List of entity texts from query
        doc_entities: List of entity texts from document

    Returns:
        Overlap score [0, 1]
    """
    if not query_entities or not doc_entities:
        return 0.0

    # Normalize to lowercase for comparison
    query_set = set(e.lower() for e in query_entities)
    doc_set = set(e.lower() for e in doc_entities)

    # Jaccard similarity
    intersection = len(query_set & doc_set)
    union = len(query_set | doc_set)

    return intersection / union if union > 0 else 0.0


def boost_results_by_entities(
    results: List[Dict],
    query_entities: List[str],
    boost_weight: float = 0.3
) -> List[Dict]:
    """
    Boost search results based on entity overlap with query.

    Reranks results using combined score: final_score = (1-β) × retrieval_score + β × entity_overlap

    Args:
        results: List of search results with score and metadata
        query_entities: List of entity texts extracted from query
        boost_weight: Weight for entity overlap component (β)

    Returns:
        Reranked results with updated scores
    """
    if not query_entities or not results:
        return results

    boosted_results = []

    for result in results:
        doc_entities = result.get("metadata", {}).get("entity_texts", [])
        retrieval_score = result.get("score", 0.0)

        # Compute entity overlap
        entity_overlap = compute_entity_overlap_score(query_entities, doc_entities)

        # Combined score
        final_score = (1 - boost_weight) * retrieval_score + boost_weight * entity_overlap

        # Create new result with updated score
        boosted_result = result.copy()
        boosted_result["score"] = final_score
        boosted_result["entity_overlap"] = entity_overlap  # Store for debugging
        boosted_result["original_score"] = retrieval_score

        boosted_results.append(boosted_result)

    # Re-sort by new score
    boosted_results.sort(key=lambda x: x["score"], reverse=True)

    return boosted_results


def build_head_corpus(ds, max_docs: int) -> Dict[str, str]:
    """Load first N documents from corpus."""
    print(f"[CORPUS] Loading first {max_docs} documents...")
    docs = {}
    
    for i, d in enumerate(ds.docs_iter()):
        # Handle different text field names
        text = ""
        if hasattr(d, 'text'):
            text = d.text
        elif hasattr(d, 'default_text'):
            text = d.default_text()
        elif hasattr(d, 'title') and hasattr(d, 'text'):
            text = f"{d.title} {d.text}"
        else:
            text = str(d)
        
        docs[d.doc_id] = text
        
        if max_docs and i + 1 >= max_docs:
            break
    
    print(f"[CORPUS] Head corpus loaded: {len(docs)} documents")
    return docs


def build_candidate_corpus(ds, qids: set, include_scoreddocs: bool = True) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Build corpus containing:
    1. All relevant documents for given queries
    2. Optionally scoreddocs (BM25 top-1000) if available
    
    Returns (documents, qrels)
    """
    print("[CORPUS] Building candidate corpus...")
    qrels = build_qrels(ds, qids)
    
    # Collect all doc IDs we need
    doc_ids = set()
    
    # Add all relevant documents
    for rels in qrels.values():
        doc_ids.update(rels)
    
    print(f"[CORPUS] Relevant documents: {len(doc_ids)}")
    
    # Add scoreddocs if available
    if include_scoreddocs and hasattr(ds, "scoreddocs_iter"):
        scoreddocs_count = 0
        for sd in ds.scoreddocs_iter():
            if sd.query_id in qids:
                doc_ids.add(sd.doc_id)
                scoreddocs_count += 1
        print(f"[CORPUS] Added {scoreddocs_count} scoreddocs")
    
    print(f"[CORPUS] Total candidate doc IDs: {len(doc_ids)}")
    
    # Collect actual documents
    docs = {}
    for d in ds.docs_iter():
        if d.doc_id in doc_ids:
            # Handle different text field names
            text = ""
            if hasattr(d, 'text'):
                text = d.text
            elif hasattr(d, 'default_text'):
                text = d.default_text()
            elif hasattr(d, 'title') and hasattr(d, 'text'):
                text = f"{d.title} {d.text}"
            else:
                text = str(d)
            
            docs[d.doc_id] = text
            
            if len(docs) % 1000 == 0:
                print(f"[CORPUS] Collected {len(docs)}/{len(doc_ids)} documents...")
    
    print(f"[CORPUS] Candidate corpus loaded: {len(docs)} documents")
    return docs, qrels


# =============================================================================
# Indexing
# =============================================================================

async def index_documents(store, documents: Dict[str, str], batch_size: int = 256, enable_ner: bool = True, max_workers: int = 1):
    """
    Index documents with optional NER entity extraction and parallel processing.

    Args:
        store: Vector store
        documents: Dict mapping doc_id -> text
        batch_size: Batch size for indexing
        enable_ner: If True, extract entities and add to metadata
        max_workers: Number of parallel workers for indexing (default: 1 = sequential)
    """
    items = list(documents.items())
    t0 = time.perf_counter()

    # Preload NER model if needed
    if enable_ner:
        get_ner_model()
        print(f"[INDEX] NER enabled - extracting entities for {len(items)} documents")
    else:
        print(f"[INDEX] NER disabled - indexing without entity extraction")

    if max_workers > 1:
        print(f"[INDEX] Using {max_workers} parallel workers for indexing")
        await index_documents_parallel(store, items, batch_size, enable_ner, max_workers, t0)
    else:
        print(f"[INDEX] Sequential indexing")
        await index_documents_sequential(store, items, batch_size, enable_ner, t0)

    print(f"[INDEX] done: {len(items)} docs in {time.perf_counter()-t0:.1f}s")


async def index_documents_sequential(store, items: List[Tuple[str, str]], batch_size: int, enable_ner: bool, t0: float):
    """Sequential indexing (original behavior)."""
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        texts = [t for _, t in batch]

        # Extract entities if NER enabled
        metadata = []
        for doc_id, text in batch:
            meta = {"doc_id": doc_id, "source": "ir_dataset"}

            if enable_ner:
                entities = extract_entities(text)
                if entities:
                    meta["entities"] = entities
                    meta["entity_texts"] = [e["text"] for e in entities]
                    meta["entity_labels"] = [e["label"] for e in entities]

            metadata.append(meta)

        # Infinite retry with exponential backoff
        attempt = 0
        while True:
            try:
                await store.add_documents(texts, metadata)
                break
            except Exception as e:
                attempt += 1
                wait_time = min(2 ** attempt, 60)
                print(f"[RETRY] Attempt {attempt} failed: {e}")
                print(f"[RETRY] Waiting {wait_time}s before retry... (Ctrl+C to abort)")
                time.sleep(wait_time)

        if (i // batch_size) % 5 == 0:
            done = min(i + batch_size, len(items))
            dt = time.perf_counter() - t0
            speed = done / dt if dt > 0 else 0

            if enable_ner and metadata and metadata[0].get("entities"):
                sample_entities = metadata[0]["entity_texts"][:3]
                entity_preview = ", ".join(sample_entities)
                print(f"[INDEX] {done}/{len(items)} ({done*100//len(items)}%) | {speed:.1f} docs/s | Sample entities: {entity_preview}")
            else:
                print(f"[INDEX] {done}/{len(items)} ({done*100//len(items)}%) | {speed:.1f} docs/s")


async def index_documents_parallel(store, items: List[Tuple[str, str]], batch_size: int, enable_ner: bool, max_workers: int, t0: float):
    """Parallel indexing with semaphore control."""
    semaphore = asyncio.Semaphore(max_workers)
    completed_count = 0
    failed_count = 0
    lock = asyncio.Lock()

    async def process_batch(batch_idx: int, batch: List[Tuple[str, str]]):
        nonlocal completed_count, failed_count

        async with semaphore:
            texts = [t for _, t in batch]

            # Extract entities if NER enabled
            metadata = []
            for doc_id, text in batch:
                meta = {"doc_id": doc_id, "source": "ir_dataset"}

                if enable_ner:
                    entities = extract_entities(text)
                    if entities:
                        meta["entities"] = entities
                        meta["entity_texts"] = [e["text"] for e in entities]
                        meta["entity_labels"] = [e["label"] for e in entities]

                metadata.append(meta)

            # Retry with exponential backoff
            attempt = 0
            while True:
                try:
                    await store.add_documents(texts, metadata)
                    break
                except Exception as e:
                    attempt += 1
                    if attempt > 5:
                        async with lock:
                            failed_count += len(batch)
                        print(f"[RETRY] Batch {batch_idx} failed after 5 attempts: {e}")
                        return
                    wait_time = min(2 ** attempt, 60)
                    await asyncio.sleep(wait_time)

            async with lock:
                completed_count += len(batch)

                # Progress reporting
                if completed_count % (batch_size * 5) < len(batch) or completed_count == len(items):
                    dt = time.perf_counter() - t0
                    speed = completed_count / dt if dt > 0 else 0
                    progress = completed_count * 100 // len(items)
                    print(f"[INDEX] {completed_count}/{len(items)} ({progress}%) | {speed:.1f} docs/s | Failed: {failed_count}")

    # Create tasks for all batches
    tasks = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_idx = i // batch_size
        tasks.append(asyncio.create_task(process_batch(batch_idx, batch)))

    # Wait for all tasks to complete
    await asyncio.gather(*tasks, return_exceptions=True)


# =============================================================================
# Retrieval
# =============================================================================

@dataclass
class Hit:
    doc_id: str
    text: str
    score: float


async def search_store(store, query: str, top_k: int, fusion: str, search_mode: str, enable_entity_boost: bool = None, entity_boost_weight: float = None):
    """
    Search with optional entity-aware boosting.

    Args:
        store: Vector store
        query: Search query
        top_k: Number of results
        fusion: Fusion strategy
        search_mode: dense/sparse/hybrid
        enable_entity_boost: If True, boost results by entity overlap (default: use global config)
        entity_boost_weight: Weight for entity boosting (default: use global config)

    Returns:
        List of search results
    """
    # Use global config if not specified
    if enable_entity_boost is None:
        enable_entity_boost = _entity_boost_enabled
    if entity_boost_weight is None:
        entity_boost_weight = _entity_boost_weight

    # Standard retrieval
    if search_mode == "dense":
        results = await store.dense_search(query, top_k=top_k * 2 if enable_entity_boost else top_k)
    elif search_mode == "sparse":
        if getattr(store, "sparse_search", None):
            results = await store.sparse_search(query, top_k=top_k * 2 if enable_entity_boost else top_k)
        else:
            results = []
    else:  # hybrid
        if fusion == "rrf" and getattr(store, "hybrid_search_rrf", None) and store.enable_hybrid_search:
            results = await store.hybrid_search_rrf(query, top_k=top_k * 2 if enable_entity_boost else top_k)
        else:
            results = await store.search(query, top_k=top_k * 2 if enable_entity_boost else top_k)

    # Entity-aware boosting (optional)
    if enable_entity_boost and results:
        query_entities_data = extract_entities(query)
        query_entity_texts = [e["text"] for e in query_entities_data]

        if query_entity_texts:
            results = boost_results_by_entities(results, query_entity_texts, boost_weight=entity_boost_weight)
            results = results[:top_k]  # Trim to top_k after boosting

    return results


async def retrieve_baseline(
    store,
    query: str,
    top_k: int,
    fusion: str,
    search_mode: str,
    logger: Optional[Logger] = None
):
    """Baseline retrieval with optional detailed logging.

    Returns:
        (hits, debug_info) tuple for consistency with other strategies
    """

    debug = {"strategy": "baseline"}

    if logger:
        logger.log("\n" + "="*80)
        logger.log(f"[QUERY] {query}")
        logger.log("="*80)

    res = await search_store(store, query, top_k=top_k, fusion=fusion, search_mode=search_mode)
    hits = []

    for d in res:
        hit = Hit(
            doc_id=d.get("metadata", {}).get("doc_id", ""),
            text=d.get("text", ""),
            score=float(d.get("score", 0.0)),
        )
        hits.append(hit)

    if logger:
        logger.log(f"\n[RESULTS] Found {len(hits)} results:")
        for i, hit in enumerate(hits, 1):
            logger.log(f"  {i}. {hit.doc_id[:40]}... (score: {hit.score:.4f})")
        logger.log("="*80 + "\n")

    return hits, debug


async def retrieve_agentic(
    store,
    enhancer,
    query: str,
    top_k: int,
    max_variants: int,
    fusion: str,
    search_mode: str,
    logger: Optional[Logger] = None,
    adaptive_fallback: bool = False,
    include_original: bool = False,
    fusion_strategy: str = None
) -> Tuple[List[Hit], Dict]:
    """
    Agentic retrieval with optional detailed logging and adaptive fallback.

    Args:
        adaptive_fallback: If True, compares agentic vs baseline and uses baseline if it's better
        include_original: If True, includes original query alongside enhanced variants (prevents degradation)
        fusion_strategy: How to combine results from multiple query variants
                        Options: max_score, rrf, weighted_rrf, hybrid, conservative
    """
    debug = {"variants": 1, "enhance_ms": 0.0, "used_fallback": False}

    if logger:
        logger.log("\n" + "="*80)
        logger.log(f"[QUERY] Original: {query}")
        logger.log("="*80)

    # If adaptive fallback enabled, get baseline results first
    baseline_hits = []
    baseline_top_score = 0.0
    if adaptive_fallback:
        baseline_res = await search_store(store, query, top_k=top_k, fusion=fusion, search_mode=search_mode)
        for d in baseline_res:
            doc_id = d.get("metadata", {}).get("doc_id", "")
            if doc_id:
                score = float(d.get("score", 0.0))
                baseline_hits.append(Hit(doc_id=doc_id, text=d.get("text", ""), score=score))

        if baseline_hits:
            baseline_top_score = baseline_hits[0].score
            if logger:
                logger.log(f"[BASELINE] Top score: {baseline_top_score:.4f}")

    t0 = time.perf_counter()

    # Call with retry wrapper and store enhanced result
    enhanced_result = None
    async def enhance_with_fallback():
        nonlocal enhanced_result
        enhanced_result = await enhancer.enhance_query(query)
        return enhancer.build_search_queries(enhanced_result) or [query]

    try:
        variants = await call_with_retry(enhance_with_fallback, max_retries=7)
    except Exception as e:
        print(f"[FALLBACK] Query enhancement failed: {e}")
        if logger:
            logger.log(f"[FALLBACK] Query enhancement failed: {e}")
        variants = [query]

    debug["enhance_ms"] = (time.perf_counter() - t0) * 1000.0
    variants = variants[:max(1, max_variants)]

    # Include original query if requested (prevents degradation)
    if include_original and query not in variants:
        variants.insert(0, query)  # Add original as first variant
        if logger:
            logger.log(f"[INCLUDE-ORIGINAL] Added original query as safety variant")

    debug["variants"] = len(variants)

    if logger:
        logger.log(f"\n[ENHANCEMENT] Completed in {debug['enhance_ms']:.1f}ms")

        if enhanced_result:
            logger.log(f"  Intent: {enhanced_result.get('intent', 'N/A')}")
            logger.log(f"  Rewritten: {enhanced_result.get('rewritten_query', 'N/A')}")

            alternatives = enhanced_result.get('alternative_queries', [])
            if alternatives:
                logger.log(f"  Alternatives ({len(alternatives)}):")
                for i, alt in enumerate(alternatives, 1):
                    logger.log(f"    {i}. {alt}")

            entities = enhanced_result.get('entities', {})
            if entities:
                logger.log(f"  Entities:")
                for key, values in entities.items():
                    if values:
                        logger.log(f"    - {key}: {values}")

        logger.log(f"\n[SEARCH QUERIES] Using {len(variants)} variants:")
        for i, v in enumerate(variants, 1):
            logger.log(f"  [{i}] {v}")

    # =============================================================================
    # PHASE 2: Parallel Variant Retrieval (4x speedup)
    # =============================================================================
    # Search all variants in parallel instead of sequentially
    variant_tasks = [
        search_store(store, vq, top_k=top_k, fusion=fusion, search_mode=search_mode)
        for vq in variants
    ]
    variant_results = await asyncio.gather(*variant_tasks)

    if logger:
        for i, (vq, res) in enumerate(zip(variants, variant_results), 1):
            logger.log(f"\n[SEARCH] Variant {i}/{len(variants)}: found {len(res)} results")

    # =============================================================================
    # PHASE 3: Multi-Query Fusion
    # =============================================================================

    if fusion_strategy is None:
        # OLD CODE (default, proven to work)
        all_hits: Dict[str, Hit] = {}

        for i, (vq, res) in enumerate(zip(variants, variant_results), 1):
            if logger:
                logger.log(f"\n[SEARCH] Variant {i}/{len(variants)}: found {len(res)} results")

            for d in res:
                doc_id = d.get("metadata", {}).get("doc_id", "")
                if not doc_id:
                    continue
                score = float(d.get("score", 0.0))

                if logger and doc_id not in all_hits:
                    logger.log(f"  + New: {doc_id[:40]}... (score: {score:.4f})")

                if (doc_id not in all_hits) or (score > all_hits[doc_id].score):
                    all_hits[doc_id] = Hit(doc_id=doc_id, text=d.get("text", ""), score=score)

        merged = sorted(all_hits.values(), key=lambda x: x.score, reverse=True)[:top_k]
    else:
        # NEW CODE (experimental fusion strategies)
        if logger:
            logger.log(f"\n[FUSION] Using strategy: {fusion_strategy}")

        fusion_fn = get_fusion_strategy(fusion_strategy)
        fusion_hits = fusion_fn(variant_results, top_k=top_k)

        # Convert FusionHit back to Hit (benchmark Hit type)
        merged = [
            Hit(doc_id=h.doc_id, text=h.text, score=h.score)
            for h in fusion_hits
        ]

    # Adaptive fallback: compare agentic vs baseline
    if adaptive_fallback and baseline_hits:
        agentic_top_score = merged[0].score if merged else 0.0

        # Decision heuristics
        use_baseline = False
        reason = ""

        # Heuristic 1: Baseline top score significantly better (>10% improvement)
        if baseline_top_score > agentic_top_score * 1.1:
            use_baseline = True
            reason = f"Baseline top score ({baseline_top_score:.2f}) > Agentic ({agentic_top_score:.2f}) by >10%"

        # Heuristic 2: Agentic variants generated poor results (top score too low)
        elif agentic_top_score < baseline_top_score * 0.7:
            use_baseline = True
            reason = f"Agentic top score ({agentic_top_score:.2f}) < 70% of baseline ({baseline_top_score:.2f})"

        if use_baseline:
            merged = baseline_hits[:top_k]
            debug["used_fallback"] = True
            if logger:
                logger.log(f"\n[ADAPTIVE FALLBACK] Using BASELINE results")
                logger.log(f"  Reason: {reason}")

        if logger:
            logger.log(f"\n[COMPARISON]")
            logger.log(f"  Baseline top score: {baseline_top_score:.4f}")
            logger.log(f"  Agentic top score: {agentic_top_score:.4f}")
            logger.log(f"  Decision: {'BASELINE' if use_baseline else 'AGENTIC'}")

    if logger:
        logger.log(f"\n[MERGED] Final top-{top_k} results:")
        for i, hit in enumerate(merged, 1):
            logger.log(f"  {i}. {hit.doc_id[:40]}... (score: {hit.score:.4f})")
        logger.log("="*80 + "\n")

    return merged, debug


# =============================================================================
# Decomposition-based Retrieval
# =============================================================================

async def retrieve_decomposition(
    store,
    llm,
    query: str,
    top_k: int = 10,
    fusion: str = "rrf",
    search_mode: str = "hybrid",
    original_weight: float = 1.5,
    aspect_weight: float = 1.0,
    humanfactor: bool = False,
    humanfactor_weight: float = 2.0,
    logger=None
):
    """
    Decomposition-based retrieval using aspect extraction.

    Strategy:
    1. LLM extracts aspects from query
    2. If only 1 aspect (simple query) → fallback to baseline
    3. Otherwise, parallel search per aspect
    4. Fusion with weighted RRF × Coverage multiplier

    Args:
        store: Vector store
        llm: LLM client with extract_aspects() method
        query: User query
        top_k: Number of results to return
        fusion: Fusion strategy for hybrid search
        search_mode: dense/sparse/hybrid
        original_weight: Weight for "original" aspect (1.0-2.0)
        aspect_weight: Weight for extracted aspects (default 1.0)
        logger: Optional logger

    Returns:
        (hits, debug_info)
    """

    debug = {"aspects_count": 0, "extraction_ms": 0.0, "used_baseline": False}

    if logger:
        logger.log("\n" + "="*80)
        logger.log(f"[DECOMPOSITION] Query: {query}")
        logger.log("="*80)

    # =============================================================================
    # PHASE 1: Aspect Extraction
    # =============================================================================

    import time
    t0 = time.perf_counter()

    aspects = await llm.extract_aspects(query)

    debug["extraction_ms"] = (time.perf_counter() - t0) * 1000.0

    # =============================================================================
    # HUMANFACTOR: Add reformulated query variant
    # =============================================================================
    if humanfactor and aspects and "original" in aspects:
        try:
            # Ask LLM to reformulate the original query (fix typos, improve clarity)
            reformulation_prompt = f"""Rephrase this query to improve clarity and fix any typos or awkward phrasing. Return ONLY the rephrased query, nothing else.

Original query: {query}

Rephrased query:"""
            reformulated = await llm.simple_query(reformulation_prompt)
            reformulated = reformulated.strip()

            # Add reformulated as additional aspect with intermediate weight
            aspects["reformulated"] = reformulated

            if logger:
                logger.log(f"\n[HUMANFACTOR] Reformulated query added:")
                logger.log(f"  Original: {query}")
                logger.log(f"  Reformulated: {reformulated}")
                logger.log(f"  Weight: {humanfactor_weight}")
        except Exception as e:
            if logger:
                logger.log(f"\n[HUMANFACTOR] Reformulation failed: {e}")
            # Continue without reformulated query

    # Fallback если extraction failed
    if aspects is None:
        print(f"[DECOMPOSITION] Extraction failed, fallback to baseline")
        if logger:
            logger.log(f"[DECOMPOSITION] Extraction failed, fallback to baseline")
        debug["used_baseline"] = True
        hits, _ = await retrieve_baseline(store, query, top_k=top_k, fusion=fusion, search_mode=search_mode, logger=logger)
        return hits, debug

    # Fallback если только 1 аспект (простой запрос)
    if len(aspects) == 1:
        print(f"[DECOMPOSITION] Only 1 aspect (simple query), fallback to baseline")
        if logger:
            logger.log(f"[DECOMPOSITION] Only 1 aspect, fallback to baseline")
        debug["used_baseline"] = True
        hits, _ = await retrieve_baseline(store, query, top_k=top_k, fusion=fusion, search_mode=search_mode, logger=logger)
        return hits, debug

    debug["aspects_count"] = len(aspects)

    if logger:
        logger.log(f"\n[ASPECTS] Extracted {len(aspects)} aspects:")
        for aspect_name, aspect_query in aspects.items():
            # Assign weights: original > reformulated > aspects
            if aspect_name == "original":
                weight = original_weight
            elif aspect_name == "reformulated":
                weight = humanfactor_weight
            else:
                weight = aspect_weight
            logger.log(f"  [{aspect_name}] (weight={weight}): {aspect_query}")

    # =============================================================================
    # PHASE 2: Parallel Aspect Search
    # =============================================================================

    results_by_aspect = {}

    # Parallel search для всех аспектов
    aspect_names = list(aspects.keys())
    aspect_queries = list(aspects.values())

    # Create coroutines for parallel execution
    tasks = [
        search_store(store, aspect_query, top_k=20, fusion=fusion, search_mode=search_mode)
        for aspect_query in aspect_queries
    ]

    # Execute all searches in parallel
    results_list = await asyncio.gather(*tasks)

    # Map results back to aspect names
    for aspect_name, results in zip(aspect_names, results_list):
        results_by_aspect[aspect_name] = results

        if logger:
            logger.log(f"\n[ASPECT SEARCH] {aspect_name}: found {len(results)} results")

    # =============================================================================
    # PHASE 3: Decomposition Fusion (Weighted RRF × Coverage)
    # =============================================================================

    from collections import defaultdict

    # Структура: doc_id -> {rrf_score, coverage_count, aspects_set, text}
    doc_data = defaultdict(lambda: {"rrf": 0.0, "count": 0, "aspects": set(), "text": ""})

    # Weighted RRF calculation
    rrf_k = 60

    for aspect_name, results in results_by_aspect.items():
        # Assign weights: original > reformulated > aspects
        if aspect_name == "original":
            weight = original_weight
        elif aspect_name == "reformulated":
            weight = humanfactor_weight
        else:
            weight = aspect_weight

        for rank, doc in enumerate(results, start=0):
            doc_id = doc.get("metadata", {}).get("doc_id", "")
            if not doc_id:
                continue

            # Weighted RRF score
            rrf_contribution = weight * (1.0 / (rrf_k + rank))
            doc_data[doc_id]["rrf"] += rrf_contribution
            doc_data[doc_id]["count"] += 1
            doc_data[doc_id]["aspects"].add(aspect_name)

            # Store text from first occurrence
            if not doc_data[doc_id]["text"]:
                doc_data[doc_id]["text"] = doc.get("text", "")

    # Coverage multiplier
    total_aspects = len(aspects)

    # Final scoring: RRF × Coverage
    final_hits = []
    for doc_id, data in doc_data.items():
        coverage_ratio = data["count"] / total_aspects
        coverage_multiplier = 1.0 + coverage_ratio

        final_score = data["rrf"] * coverage_multiplier

        final_hits.append(Hit(
            doc_id=doc_id,
            text=data["text"],
            score=final_score
        ))

    # Sort and take top-k
    final_hits.sort(key=lambda h: h.score, reverse=True)
    merged = final_hits[:top_k]

    if logger:
        logger.log(f"\n[FUSION] Decomposition Fusion (RRF × Coverage):")
        logger.log(f"  Total unique docs: {len(doc_data)}")
        logger.log(f"  Total aspects: {total_aspects}")
        logger.log(f"\n[MERGED] Final top-{top_k} results:")
        for i, hit in enumerate(merged, 1):
            aspects_covered = doc_data[hit.doc_id]["aspects"]
            coverage = len(aspects_covered) / total_aspects * 100
            logger.log(f"  {i}. {hit.doc_id[:40]}... (score: {hit.score:.4f}, coverage: {coverage:.0f}%)")
            logger.log(f"      aspects: {', '.join(aspects_covered)}")
        logger.log("="*80 + "\n")

    return merged, debug


async def retrieve_multihop(
    store,
    llm,
    query: str,
    top_k: int = 10,
    fusion: str = "rrf",
    search_mode: str = "hybrid",
    logger=None
):
    """
    Multihop retrieval using PankRAG DAG execution (paper 2506.11106v2).

    Strategy:
    1. Extract DAG with dependencies via LLM
    2. Bottom-up execution: resolve dependencies level-by-level
    3. Top-down refinement: refine original query with resolved answers
    4. Dependency-aware reranking: α × Ri + β × Mi

    Args:
        store: Vector store
        llm: LLM client with extract_aspects() method
        query: User query
        top_k: Number of results to return
        fusion: Fusion strategy for hybrid search
        search_mode: dense/sparse/hybrid
        logger: Optional logger

    Returns:
        (hits, debug_info)
    """

    debug = {"aspects_count": 0, "dag_levels": 0, "extraction_ms": 0.0}

    if logger:
        logger.log("\n" + "="*80)
        logger.log(f"[MULTIHOP] Query: {query}")
        logger.log("="*80)

    # =============================================================================
    # PHASE 1: DAG Extraction
    # =============================================================================

    import time
    t0 = time.perf_counter()

    # Extract DAG with dependencies
    dag = await llm.extract_aspects(query)  # Returns dict with aspects + execution_order

    debug["extraction_ms"] = (time.perf_counter() - t0) * 1000.0

    # Fallback if extraction failed or no dependencies
    if dag is None or len(dag.get("aspects", {})) == 1:
        if logger:
            logger.log(f"[MULTIHOP] No dependencies, fallback to baseline")
        debug["fallback"] = "baseline"
        hits, _ = await retrieve_baseline(store, query, top_k=top_k, fusion=fusion, search_mode=search_mode, logger=logger)
        return hits, debug

    aspects = dag["aspects"]
    execution_order = dag.get("execution_order", [[]])

    debug["aspects_count"] = len(aspects)
    debug["dag_levels"] = len(execution_order)

    if logger:
        logger.log(f"\n[DAG] Extracted {len(aspects)} aspects in {len(execution_order)} levels:")
        for level_idx, batch in enumerate(execution_order):
            logger.log(f"  Level {level_idx}: {batch}")

    # =============================================================================
    # PHASE 2: Bottom-up DAG Execution
    # =============================================================================

    results_by_aspect = {}
    resolved_answers = {}  # aspect_key -> top-1 result text

    for level_idx, batch in enumerate(execution_order):
        if logger:
            logger.log(f"\n[LEVEL {level_idx}] Executing batch: {batch}")

        tasks = []
        for aspect_key in batch:
            aspect = aspects[aspect_key]
            aspect_query = aspect.get("query", query)

            # Substitute placeholders with resolved answers from dependencies
            for dep_key in aspect.get("dependencies", []):
                if dep_key in resolved_answers:
                    placeholder = f"[{dep_key}]"
                    aspect_query = aspect_query.replace(placeholder, resolved_answers[dep_key])

            tasks.append(search_store(store, aspect_query, top_k=20, fusion=fusion, search_mode=search_mode))

        # Parallel execution within level
        batch_results = await asyncio.gather(*tasks)

        for aspect_key, results in zip(batch, batch_results):
            results_by_aspect[aspect_key] = results

            # Extract answer from top-1 result
            if results:
                resolved_answers[aspect_key] = results[0].get("text", "")[:200]  # First 200 chars

            if logger:
                logger.log(f"  [{aspect_key}] found {len(results)} results")

    # =============================================================================
    # KNOWLEDGE GRAPH: Cache resolved relationships
    # =============================================================================

    try:
        # Initialize knowledge graph (stores triplets in Qdrant)
        kg = KnowledgeGraph(
            qdrant_client=store.client,
            embedding_model=store.embedding_model,
            collection_name="knowledge_graph"
        )

        # Extract triplets from resolved DAG relationships
        triplets = extract_triplets_from_dag_results(dag, results_by_aspect, resolved_answers)

        if triplets:
            # Add triplets to graph (batch operation)
            triplet_ids = kg.add_triplets_batch(triplets)

            if logger:
                logger.log(f"\n[GRAPH] Stored {len(triplet_ids)} triplets:")
                for triplet in triplets[:3]:  # Show first 3
                    logger.log(f"  ({triplet.subject[:30]}..., {triplet.predicate}, {triplet.object[:30]}...)")
                if len(triplets) > 3:
                    logger.log(f"  ... and {len(triplets) - 3} more")

            debug["graph_triplets"] = len(triplets)
        else:
            if logger:
                logger.log(f"\n[GRAPH] No triplets extracted (no dependencies)")
            debug["graph_triplets"] = 0

    except Exception as e:
        # Don't fail the entire pipeline if graph storage fails
        if logger:
            logger.log(f"\n[GRAPH] Error storing triplets: {e}")
        debug["graph_error"] = str(e)

    # =============================================================================
    # PHASE 3: Top-down Refinement
    # =============================================================================

    # Refine original query with all resolved answers
    refined_query = query
    for aspect_key, answer in resolved_answers.items():
        if aspect_key != "original":
            refined_query += f" {answer}"

    if logger:
        logger.log(f"\n[TOP-DOWN] Refined query: {refined_query[:150]}...")

    # Final retrieval with refined query
    final_results = await search_store(store, refined_query, top_k=top_k*2, fusion=fusion, search_mode=search_mode)
    results_by_aspect["refined"] = final_results

    # =============================================================================
    # PHASE 4: Dependency-Aware Reranking (PankRAG Equation 5)
    # =============================================================================

    # Determine query type (SCQ vs ACQ) based on DAG structure
    has_sequential_deps = len(execution_order) > 1
    query_type = "SCQ" if has_sequential_deps else "ACQ"

    # PankRAG weights (Section 3.4)
    if query_type == "SCQ":
        alpha, beta = 0.6, 0.4  # Multihop queries
    else:
        alpha, beta = 0.75, 0.25  # Abstract queries

    if logger:
        logger.log(f"\n[RERANKING] Query type: {query_type}, weights: α={alpha}, β={beta}")

    # Combined scoring: final_score = α × Ri + β × Mi
    combined_results = {}

    for aspect_key, results in results_by_aspect.items():
        aspect = aspects.get(aspect_key, {})
        dep_keys = aspect.get("dependencies", [])

        # Get resolved dependency answers for Mi computation
        dep_answers = {k: resolved_answers.get(k, "") for k in dep_keys if k in resolved_answers}

        for result in results:
            doc_id = result.get("metadata", {}).get("doc_id", "")
            if not doc_id:
                continue

            text = result.get("text", "")

            # Ri: Intrinsic retrieval quality (Qdrant score)
            Ri = result.get("score", 0.0)

            # Mi: Dependency similarity (PankRAG Equation 4)
            Mi = 0.0
            if dep_answers:
                # Compute cosine similarity with dependency answers
                # Simplified: use text overlap ratio (placeholder - proper embedding would be better)
                Mi = compute_text_similarity(text, list(dep_answers.values()))

            # Combined score (Equation 5)
            final_score = alpha * Ri + beta * Mi

            if doc_id not in combined_results or final_score > combined_results[doc_id]["score"]:
                combined_results[doc_id] = {
                    "doc_id": doc_id,
                    "text": text,
                    "score": final_score,
                    "Ri": Ri,
                    "Mi": Mi,
                }

    # Sort by final score
    final_hits = [
        Hit(doc_id=data["doc_id"], text=data["text"], score=data["score"])
        for data in combined_results.values()
    ]
    final_hits.sort(key=lambda h: h.score, reverse=True)
    merged = final_hits[:top_k]

    if logger:
        logger.log(f"\n[MERGED] Final top-{top_k} results:")
        for i, hit in enumerate(merged, 1):
            data = combined_results[hit.doc_id]
            logger.log(f"  {i}. {hit.doc_id[:40]}... (score: {hit.score:.4f}, Ri: {data['Ri']:.4f}, Mi: {data['Mi']:.4f})")
        logger.log("="*80 + "\n")

    return merged, debug


def compute_text_similarity(text: str, dep_answers: List[str]) -> float:
    """
    Simplified text similarity (placeholder for proper embedding similarity).

    In production: use embedding_model.encode() + cosine_similarity.
    For benchmark: use token overlap as proxy.

    Args:
        text: Candidate chunk text
        dep_answers: List of resolved dependency answer texts

    Returns:
        Similarity score [0, 1]
    """
    if not dep_answers:
        return 0.0

    text_tokens = set(text.lower().split())
    similarities = []

    for answer in dep_answers:
        answer_tokens = set(answer.lower().split())
        if not answer_tokens:
            continue

        overlap = len(text_tokens & answer_tokens)
        union = len(text_tokens | answer_tokens)

        if union > 0:
            jaccard = overlap / union
            similarities.append(jaccard)

    return sum(similarities) / len(similarities) if similarities else 0.0


async def retrieve_adaptive(
    store,
    llm,
    query: str,
    top_k: int = 10,
    fusion: str = "rrf",
    search_mode: str = "hybrid",
    original_weight: float = 3.0,
    humanfactor: bool = False,
    humanfactor_weight: float = 2.0,
    logger=None
):
    """
    Adaptive retrieval: LLM router selects strategy per query.

    Strategy selection:
    1. Extract aspects via LLM
    2. Classify based on:
       - Aspect count (1 = baseline, 2+ = decomposition/multihop)
       - Dependencies (sequential = multihop, parallel = decomposition)
       - Query complexity
    3. Execute selected strategy
    4. Fallback: baseline on errors

    Args:
        store: Vector store
        llm: LLM client
        query: User query
        top_k: Number of results
        fusion: Fusion strategy
        search_mode: dense/sparse/hybrid
        original_weight: Weight for original query (decomposition)
        logger: Optional logger

    Returns:
        (hits, debug_info)
    """

    debug = {"strategy": "baseline", "reasoning": ""}

    if logger:
        logger.log("\n" + "="*80)
        logger.log(f"[ADAPTIVE] Query: {query}")
        logger.log("="*80)

    # =============================================================================
    # PHASE 1: Extract Aspects for Classification
    # =============================================================================

    import time
    t0 = time.perf_counter()

    aspects = await llm.extract_aspects(query)
    extraction_ms = (time.perf_counter() - t0) * 1000.0

    # Fallback to baseline if extraction failed
    if aspects is None:
        debug["strategy"] = "baseline"
        debug["reasoning"] = "Aspect extraction failed"
        if logger:
            logger.log(f"[ADAPTIVE] Strategy: baseline (extraction failed)")
        hits, _ = await retrieve_baseline(store, query, top_k=top_k, fusion=fusion, search_mode=search_mode, logger=logger)
        return hits, debug

    aspect_count = len(aspects)

    # Build simple execution order (all aspects in parallel)
    execution_order = [list(aspects.keys())]

    # =============================================================================
    # PHASE 2: Strategy Classification
    # =============================================================================

    # Rule-based classification (simplified for flat aspects format)
    # Note: Flat format from extract_aspects doesn't include dependency info,
    # so we can only choose between baseline and decomposition

    if aspect_count == 1:
        # Simple query → baseline
        strategy = "baseline"
        reasoning = "Single aspect (simple query)"
    elif aspect_count >= 2:
        # Multiple independent aspects → decomposition
        strategy = "decomposition"
        reasoning = f"Multiple parallel aspects ({aspect_count})"
    else:
        # Default fallback
        strategy = "baseline"
        reasoning = "Default fallback"

    debug["strategy"] = strategy
    debug["reasoning"] = reasoning
    debug["aspect_count"] = aspect_count
    debug["extraction_ms"] = extraction_ms

    if logger:
        logger.log(f"\n[ADAPTIVE] Classification:")
        logger.log(f"  Aspect count: {aspect_count}")
        logger.log(f"  → Strategy: {strategy} ({reasoning})")

    # =============================================================================
    # PHASE 3: Execute Selected Strategy
    # =============================================================================

    try:
        if strategy == "baseline":
            hits, _ = await retrieve_baseline(store, query, top_k=top_k, fusion=fusion, search_mode=search_mode, logger=logger)
        elif strategy == "decomposition":
            hits, _ = await retrieve_decomposition(
                store, llm, query, top_k=top_k, fusion=fusion, search_mode=search_mode,
                original_weight=original_weight,
                humanfactor=humanfactor,
                humanfactor_weight=humanfactor_weight,
                logger=logger
            )
        else:
            # Unknown strategy → fallback to baseline
            debug["fallback"] = "unknown_strategy"
            hits, _ = await retrieve_baseline(store, query, top_k=top_k, fusion=fusion, search_mode=search_mode, logger=logger)

    except Exception as e:
        # Error in strategy execution → fallback to baseline
        if logger:
            logger.log(f"\n[ADAPTIVE] ERROR in {strategy}: {e}")
            logger.log(f"[ADAPTIVE] Falling back to baseline")
        debug["fallback"] = f"{strategy}_error"
        debug["error"] = str(e)
        hits, _ = await retrieve_baseline(store, query, top_k=top_k, fusion=fusion, search_mode=search_mode, logger=logger)

    return hits, debug


# =============================================================================
# Metrics
# =============================================================================

def compute_retrieval_metrics(
    retrieval_results: Dict[str, List[str]],
    qrels: Dict[str, List[str]],
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    valid_qids = [qid for qid in retrieval_results.keys() if qid in qrels and qrels[qid]]
    n = len(valid_qids)
    if n == 0:
        return {}

    out = {}
    for k in k_values:
        precisions, recalls, hitrates, ndcgs, mrrs = [], [], [], [], []

        for qid in valid_qids:
            retrieved = retrieval_results[qid][:k]
            relevant = set(qrels[qid])

            hits = sum(1 for did in retrieved if did in relevant)

            precisions.append(hits / k)
            recalls.append(hits / len(relevant) if relevant else 0.0)
            hitrates.append(1.0 if hits > 0 else 0.0)

            rr = 0.0
            for rank, did in enumerate(retrieved, 1):
                if did in relevant:
                    rr = 1.0 / rank
                    break
            mrrs.append(rr)

            dcg = 0.0
            for rank, did in enumerate(retrieved, 1):
                if did in relevant:
                    dcg += 1.0 / math.log2(rank + 1)
            ideal_hits = min(len(relevant), k)
            idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

        out[f"Precision@{k}"] = sum(precisions) / n
        out[f"Recall@{k}"] = sum(recalls) / n
        out[f"HitRate@{k}"] = sum(hitrates) / n
        out[f"nDCG@{k}"] = sum(ndcgs) / n
        out[f"MRR@{k}"] = sum(mrrs) / n

    return out


# =============================================================================
# Main
# =============================================================================

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--dataset-id", default="beir/scifact", help="Dataset ID (e.g., beir/scifact, beir/nq)")
    ap.add_argument("--collection", default="benchmark_scifact", help="Qdrant collection name")
    ap.add_argument("--embedding", default="intfloat/multilingual-e5-base",
                    help="Embedding model: intfloat/multilingual-e5-base (default, BM25 sparse) or BAAI/bge-m3 (dense+learned sparse)")
    ap.add_argument("--variant", choices=["baseline", "agentic", "decomposition", "multihop", "adaptive"], default="baseline")
    ap.add_argument("--corpus", choices=["full", "candidate", "head"], default="full",
                    help="Corpus type: full (entire dataset), candidate (relevant+scoreddocs), head (first N docs)")
    ap.add_argument("--max-queries", type=int, default=None, help="Limit number of queries (None = all)")
    ap.add_argument("--max-docs", type=int, default=10000, help="Max docs for 'head' corpus mode")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-variants", type=int, default=4)
    ap.add_argument("--include-original", action="store_true",
                    help="Include original query alongside enhanced variants (agentic only, prevents degradation)")
    ap.add_argument("--fusion-strategy",
                    choices=["max_score", "rrf", "weighted_rrf", "hybrid", "conservative"],
                    default=None,
                    help="Multi-query fusion strategy (agentic only, EXPERIMENTAL): "
                         "If not specified, uses proven max-score implementation. "
                         "max_score (take max score per doc), "
                         "rrf (reciprocal rank fusion), "
                         "weighted_rrf (RRF with original query priority), "
                         "hybrid (boost original + new docs from variants), "
                         "conservative (never-worse guarantee)")
    ap.add_argument("--search-mode", choices=["hybrid", "dense", "sparse"], default="hybrid",
                    help="Retrieval mode")
    ap.add_argument("--fusion", choices=["weighted", "rrf"], default="weighted",
                    help="Fusion method for hybrid search (dense+sparse within each query)")
    ap.add_argument("--reindex", action="store_true")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--output", type=str, default=None, 
                    help="Output file for detailed logs (e.g., logs.txt)")
    ap.add_argument("--log-console", action="store_true",
                    help="Show detailed logs in console (in addition to file)")
    ap.add_argument("--results-out", type=str, default=None,
                    help="Write summary metrics and run info as JSONL")
    ap.add_argument("--adaptive-fallback", action="store_true",
                    help="Enable adaptive fallback: use baseline if agentic performs worse")
    ap.add_argument("--rerank", action="store_true",
                    help="Enable cross-encoder reranking (ms-marco-MiniLM-L-6-v2)")
    ap.add_argument("--rerank-top-k", type=int, default=100,
                    help="Number of candidates to retrieve before reranking (default: 100)")
    ap.add_argument("--max-workers", type=int, default=1,
                    help="Number of parallel workers for query processing (default: 1 = sequential)")
    ap.add_argument("--batch-rerank", action="store_true",
                    help="Batch rerank ALL queries at once (benchmark mode only, 50x speedup)")
    ap.add_argument("--original-weight", type=float, default=1.5,
                    help="Weight for original query in decomposition mode (default: 1.5, try 2.0-3.0)")
    ap.add_argument("--enable-ner", action="store_true",
                    help="Extract named entities during indexing and add to metadata (requires spaCy)")
    ap.add_argument("--entity-boost", action="store_true",
                    help="Boost retrieval results based on entity overlap with query (requires NER-indexed data)")
    ap.add_argument("--entity-boost-weight", type=float, default=0.3,
                    help="Weight for entity boosting component (default: 0.3)")
    ap.add_argument("--humanfactor", action="store_true",
                    help="Add reformulated query variant to decomposition (helps with poorly written queries, typos, human factor)")
    ap.add_argument("--humanfactor-weight", type=float, default=2.0,
                    help="Weight for reformulated query in humanfactor mode (default: 2.0, between original and aspects)")
    args = ap.parse_args()

    # Set global entity boosting config
    global _entity_boost_enabled, _entity_boost_weight
    _entity_boost_enabled = args.entity_boost
    _entity_boost_weight = args.entity_boost_weight

    # Setup logger
    logger = None
    if args.output:
        logger = Logger(args.output, console_details=args.log_console)

    # Setup
    add_project_root_to_syspath(args.project_root)

    print(f"[RUN] dataset={args.dataset_id} | variant={args.variant} | corpus={args.corpus}")
    print(f"[RUN] max_queries={args.max_queries or 'all'} | collection={args.collection}")
    print(f"[RUN] mode={args.search_mode} | fusion={args.fusion}")

    # Validate flag compatibility
    if args.batch_rerank and args.adaptive_fallback:
        print("\n" + "="*80)
        print("[ERROR] --batch-rerank and --adaptive-fallback are incompatible")
        print("="*80)
        print("Reason:")
        print("  • Batch reranking collects ALL queries first, then reranks in one batch")
        print("  • Adaptive fallback requires per-query comparison (baseline vs agentic)")
        print("  • Cannot compare per-query when reranking happens after all retrieval")
        print("\nSolution:")
        print("  • Use --batch-rerank for SPEED (50x faster reranking)")
        print("  • Use --adaptive-fallback for QUALITY (prevent degradation)")
        print("  • Cannot use both simultaneously")
        print("="*80)
        sys.exit(1)

    # Load dataset
    ds = load_ds(args.dataset_id)
    queries = take_queries(ds, max_queries=args.max_queries)
    qids = set(queries.keys())
    
    print(f"[DATA] queries: {len(queries)}")
    
    # Build corpus based on mode
    if args.corpus == "full":
        documents = build_full_corpus(ds)
        qrels = build_qrels(ds, qids)
    elif args.corpus == "candidate":
        documents, qrels = build_candidate_corpus(ds, qids, include_scoreddocs=True)
    else:  # head
        documents = build_head_corpus(ds, max_docs=args.max_docs)
        qrels = build_qrels(ds, qids)
    
    print(f"[DATA] qrels: {len(qrels)}")
    print(f"[DATA] docs: {len(documents)}")

    # Initialize vector store
    store = await init_vector_store(
        collection_name=args.collection,
        embedding_model=args.embedding
    )

    if args.reindex:
        print("[INDEX] Reindexing...")
        await store.delete_collection()
        await store.initialize()
        # Build vocab only for BM25 (not BGE-M3 which has learned sparse)
        if store.enable_hybrid_search and getattr(store, 'sparse_encoder', None) and not store.sparse_vocab_ready:
            print('[BM25] Building vocabulary from full corpus...')
            store.build_sparse_vocab(list(documents.values()))
        elif store.enable_hybrid_search and store.sparse_vocab_ready:
            print('[BGE-M3] Using learned sparse (vocab not needed)')
        await index_documents(store, documents, batch_size=args.batch_size, enable_ner=args.enable_ner, max_workers=args.max_workers)
    else:
        print("[INDEX] Using existing collection")
        # Build sparse vocab if not already built (needed for BM25 sparse, not BGE-M3)
        if store.enable_hybrid_search and getattr(store, 'sparse_encoder', None):
            if not store.sparse_vocab_ready:
                print('[BM25] Building vocabulary from corpus for existing collection...')
                store.build_sparse_vocab(list(documents.values()))
            else:
                print('[BGE-M3] Using learned sparse (vocab already ready)')

    # Initialize LLM if needed
    llm = None
    enhancer = None
    if args.variant == "agentic":
        # Prompt for API key at the start
        print("\n" + "="*60)
        print("[SETUP] Agentic mode requires API key")
        print("="*60)
        use_env = input("Use API key from .env file? (y/n): ").strip().lower()

        if use_env == 'y':
            llm = await init_llm_with_key()
        else:
            api_key = prompt_for_api_key()
            llm = await init_llm_with_key(api_key)

        enhancer = await init_query_enhancer()

    elif args.variant in ["decomposition", "multihop", "adaptive"]:
        # These modes require LLM for aspect extraction / DAG planning
        mode_name = args.variant.capitalize()
        print("\n" + "="*60)
        print(f"[SETUP] {mode_name} mode requires API key")
        print("="*60)
        use_env = input("Use API key from .env file? (y/n): ").strip().lower()

        if use_env == 'y':
            llm = await init_llm_with_key()
        else:
            api_key = prompt_for_api_key()
            llm = await init_llm_with_key(api_key)

    # =============================================================================
    # PHASE 1: Query Processing (Sequential or Parallel)
    # =============================================================================

    async def process_single_query(qid: str, qtext: str, idx: int) -> Tuple[str, List, float, Dict]:
        """
        Process a single query: retrieval + optional reranking + adaptive fallback.

        Returns:
            (qid, hits, query_time_ms, debug_info)
        """
        query_start = time.perf_counter()
        debug_info = {}

        # Determine retrieval top_k (more if reranking)
        retrieval_k = args.rerank_top_k if args.rerank else args.top_k

        if args.variant == "baseline":
            hits, _ = await retrieve_baseline(
                store,
                qtext,
                top_k=retrieval_k,
                fusion=args.fusion,
                search_mode=args.search_mode,
                logger=logger if idx <= 5 else None  # Detailed logging for first 5 queries only
            )

            # Apply reranking if enabled (skip if batch_rerank mode)
            if args.rerank and not args.batch_rerank:
                hits = rerank_hits(qtext, hits, top_k=args.top_k,
                                 logger=logger if idx <= 5 else None)

        elif args.variant == "agentic":
            # Agentic mode with optional adaptive reranking
            agentic_start = time.perf_counter()
            agentic_hits, dbg = await retrieve_agentic(store, enhancer, qtext,
                                              top_k=retrieval_k,
                                              max_variants=args.max_variants,
                                              fusion=args.fusion,
                                              search_mode=args.search_mode,
                                              logger=logger if idx <= 5 else None,
                                              adaptive_fallback=False,  # Handle fallback after reranking
                                              include_original=args.include_original,
                                              fusion_strategy=args.fusion_strategy)
            agentic_time = (time.perf_counter() - agentic_start) * 1000

            debug_info["variants"] = dbg["variants"]
            debug_info["enhance_ms"] = dbg["enhance_ms"]

            if logger and idx <= 5:  # Detailed timing for first 5 queries
                logger.log(f"[TIMING] Agentic retrieval: {agentic_time:.1f}ms")

            # Adaptive reranking: compare agentic+rerank vs baseline+rerank
            # Skip if batch_rerank mode (will rerank all queries later)
            if args.rerank and args.adaptive_fallback and not args.batch_rerank:
                # Get baseline for comparison
                baseline_start = time.perf_counter()
                baseline_hits, _ = await retrieve_baseline(store, qtext, top_k=retrieval_k,
                                                       fusion=args.fusion, search_mode=args.search_mode)
                baseline_retr_time = (time.perf_counter() - baseline_start) * 1000

                if logger and idx <= 5:
                    logger.log(f"[TIMING] Baseline retrieval (for adaptive): {baseline_retr_time:.1f}ms")

                # Rerank both
                agentic_rerank_start = time.perf_counter()
                agentic_reranked = rerank_hits(qtext, agentic_hits, top_k=args.top_k,
                                             logger=logger if idx <= 5 else None)
                agentic_rerank_time = (time.perf_counter() - agentic_rerank_start) * 1000

                baseline_rerank_start = time.perf_counter()
                baseline_reranked = rerank_hits(qtext, baseline_hits, top_k=args.top_k, logger=None)
                baseline_rerank_time = (time.perf_counter() - baseline_rerank_start) * 1000

                if logger and idx <= 5:
                    logger.log(f"[TIMING] Agentic reranking: {agentic_rerank_time:.1f}ms")
                    logger.log(f"[TIMING] Baseline reranking (for adaptive): {baseline_rerank_time:.1f}ms")

                # Compare top-1 scores
                agentic_score = agentic_reranked[0].score if agentic_reranked else 0.0
                baseline_score = baseline_reranked[0].score if baseline_reranked else 0.0

                if baseline_score > agentic_score * 1.05:  # 5% threshold
                    hits = baseline_reranked
                    debug_info["fallback_used"] = True
                    if logger:
                        logger.log(f"[ADAPTIVE RERANK] Using baseline (score: {baseline_score:.4f} vs {agentic_score:.4f})")
                else:
                    hits = agentic_reranked
                    debug_info["fallback_used"] = False
            elif args.rerank and not args.batch_rerank:
                # Reranking without adaptive comparison (skip if batch_rerank mode)
                hits = rerank_hits(qtext, agentic_hits, top_k=args.top_k,
                                 logger=logger if idx <= 5 else None)
            else:
                # No reranking, use adaptive_fallback from retrieve_agentic if enabled
                hits = agentic_hits

        elif args.variant == "decomposition":
            # Decomposition mode with aspect extraction
            decomp_start = time.perf_counter()
            decomp_hits, dbg = await retrieve_decomposition(
                store, llm, qtext,
                top_k=retrieval_k,
                fusion=args.fusion,
                search_mode=args.search_mode,
                original_weight=getattr(args, 'original_weight', 1.5),  # Default 1.5
                aspect_weight=getattr(args, 'aspect_weight', 1.0),      # Default 1.0
                humanfactor=getattr(args, 'humanfactor', False),        # Default False
                humanfactor_weight=getattr(args, 'humanfactor_weight', 2.0),  # Default 2.0
                logger=logger if idx <= 5 else None
            )
            decomp_time = (time.perf_counter() - decomp_start) * 1000

            debug_info["aspects_count"] = dbg["aspects_count"]
            debug_info["extraction_ms"] = dbg["extraction_ms"]
            debug_info["used_baseline"] = dbg["used_baseline"]

            if logger and idx <= 5:
                logger.log(f"[TIMING] Decomposition retrieval: {decomp_time:.1f}ms")

            # Apply reranking if enabled
            if args.rerank and not args.batch_rerank:
                hits = rerank_hits(qtext, decomp_hits, top_k=args.top_k,
                                 logger=logger if idx <= 5 else None)
            else:
                hits = decomp_hits

        elif args.variant == "multihop":
            # Multihop mode with PankRAG DAG execution
            multihop_start = time.perf_counter()
            multihop_hits, dbg = await retrieve_multihop(
                store, llm, qtext,
                top_k=retrieval_k,
                fusion=args.fusion,
                search_mode=args.search_mode,
                logger=logger if idx <= 5 else None
            )
            multihop_time = (time.perf_counter() - multihop_start) * 1000

            debug_info["aspects_count"] = dbg.get("aspects_count", 0)
            debug_info["dag_levels"] = dbg.get("dag_levels", 0)
            debug_info["extraction_ms"] = dbg.get("extraction_ms", 0.0)
            debug_info["fallback"] = dbg.get("fallback", None)

            if logger and idx <= 5:
                logger.log(f"[TIMING] Multihop retrieval: {multihop_time:.1f}ms")

            # Apply reranking if enabled
            if args.rerank and not args.batch_rerank:
                hits = rerank_hits(qtext, multihop_hits, top_k=args.top_k,
                                 logger=logger if idx <= 5 else None)
            else:
                hits = multihop_hits

        elif args.variant == "adaptive":
            # Adaptive mode with automatic strategy selection
            adaptive_start = time.perf_counter()
            adaptive_hits, dbg = await retrieve_adaptive(
                store, llm, qtext,
                top_k=retrieval_k,
                fusion=args.fusion,
                search_mode=args.search_mode,
                original_weight=getattr(args, 'original_weight', 3.0),
                humanfactor=getattr(args, 'humanfactor', False),
                humanfactor_weight=getattr(args, 'humanfactor_weight', 2.0),
                logger=logger if idx <= 5 else None
            )
            adaptive_time = (time.perf_counter() - adaptive_start) * 1000

            debug_info["strategy"] = dbg.get("strategy", "baseline")
            debug_info["reasoning"] = dbg.get("reasoning", "")
            debug_info["aspect_count"] = dbg.get("aspect_count", 0)
            debug_info["has_dependencies"] = dbg.get("has_dependencies", False)
            debug_info["extraction_ms"] = dbg.get("extraction_ms", 0.0)
            debug_info["fallback"] = dbg.get("fallback", None)

            if logger and idx <= 5:
                logger.log(f"[TIMING] Adaptive retrieval: {adaptive_time:.1f}ms")
                logger.log(f"[ADAPTIVE] Selected strategy: {dbg.get('strategy')} ({dbg.get('reasoning')})")

            # Apply reranking if enabled
            if args.rerank and not args.batch_rerank:
                hits = rerank_hits(qtext, adaptive_hits, top_k=args.top_k,
                                 logger=logger if idx <= 5 else None)
            else:
                hits = adaptive_hits

        query_total_time = (time.perf_counter() - query_start) * 1000
        if logger and idx <= 5:
            logger.log(f"[TIMING] ===== Total query time: {query_total_time:.1f}ms ({query_total_time/1000:.2f}s) =====\n")

        return qid, hits, query_total_time, debug_info

    # Run retrieval
    retrieval_doc_ids: Dict[str, List[str]] = {}
    retrieval_hits: Dict[str, List] = {}  # Store hits for batch reranking
    agentic_debug = {"avg_variants": 0.0, "avg_enh_ms": 0.0}
    decomp_debug = {"avg_aspects": 0.0, "avg_extract_ms": 0.0, "baseline_fallback_count": 0}
    multihop_debug = {"avg_aspects": 0.0, "avg_dag_levels": 0.0, "avg_extract_ms": 0.0, "fallback_count": 0}
    adaptive_debug = {"strategy_counts": defaultdict(int), "avg_aspect_count": 0.0, "avg_extract_ms": 0.0, "fallback_count": 0}
    dbg_count = 0

    print(f"\n[RETRIEVAL] Processing {len(queries)} queries...")
    if args.max_workers > 1:
        print(f"[PARALLEL] Using {args.max_workers} workers for concurrent query processing")
    if args.batch_rerank:
        print(f"[BATCH RERANK] Will rerank all queries at once after retrieval (50x speedup!)")

    t0 = time.perf_counter()

    if args.max_workers == 1:
        # Sequential processing (original behavior)
        for idx, (qid, qtext) in enumerate(queries.items(), 1):
            try:
                qid, hits, query_time, debug_info = await process_single_query(qid, qtext, idx)
                retrieval_doc_ids[qid] = [h.doc_id for h in hits]
                retrieval_hits[qid] = hits  # Store hits for batch reranking

                # Update debug stats
                if args.variant == "agentic":
                    agentic_debug["avg_variants"] += debug_info.get("variants", 0)
                    agentic_debug["avg_enh_ms"] += debug_info.get("enhance_ms", 0)
                    dbg_count += 1
                    if debug_info.get("fallback_used"):
                        agentic_debug["rerank_fallback_count"] = agentic_debug.get("rerank_fallback_count", 0) + 1
                elif args.variant == "decomposition":
                    decomp_debug["avg_aspects"] += debug_info.get("aspects_count", 0)
                    decomp_debug["avg_extract_ms"] += debug_info.get("extraction_ms", 0)
                    dbg_count += 1
                    if debug_info.get("used_baseline"):
                        decomp_debug["baseline_fallback_count"] += 1
                elif args.variant == "multihop":
                    multihop_debug["avg_aspects"] += debug_info.get("aspects_count", 0)
                    multihop_debug["avg_dag_levels"] += debug_info.get("dag_levels", 0)
                    multihop_debug["avg_extract_ms"] += debug_info.get("extraction_ms", 0)
                    dbg_count += 1
                    if debug_info.get("fallback"):
                        multihop_debug["fallback_count"] += 1
                elif args.variant == "adaptive":
                    strategy = debug_info.get("strategy", "baseline")
                    adaptive_debug["strategy_counts"][strategy] += 1
                    adaptive_debug["avg_aspect_count"] += debug_info.get("aspect_count", 0)
                    adaptive_debug["avg_extract_ms"] += debug_info.get("extraction_ms", 0)
                    dbg_count += 1
                    if debug_info.get("fallback"):
                        adaptive_debug["fallback_count"] += 1

            except Exception as e:
                print(f"[ERROR] Query {qid} failed: {e}")
                retrieval_doc_ids[qid] = []
                retrieval_hits[qid] = []

            # Progress tracking
            if idx % 10 == 0 or idx == len(queries):
                dt = time.perf_counter() - t0
                speed = idx / dt if dt > 0 else 0
                eta = (len(queries) - idx) / speed if speed > 0 else 0
                print(f"[PROGRESS] {idx}/{len(queries)} ({idx*100//len(queries)}%) | "
                      f"{speed:.2f} q/s | ETA: {eta/60:.1f} min")

    else:
        # Parallel processing with asyncio.Semaphore
        semaphore = asyncio.Semaphore(args.max_workers)
        completed_count = 0
        failed_count = 0

        async def process_with_semaphore(qid: str, qtext: str, idx: int):
            async with semaphore:
                return await process_single_query(qid, qtext, idx)

        # Create tasks for all queries
        query_list = list(queries.items())
        tasks = [
            asyncio.create_task(process_with_semaphore(qid, qtext, idx))
            for idx, (qid, qtext) in enumerate(query_list, 1)
        ]

        # Process results as they complete (not in original order)
        for coro in asyncio.as_completed(tasks):
            try:
                qid, hits, query_time, debug_info = await coro
                retrieval_doc_ids[qid] = [h.doc_id for h in hits]
                retrieval_hits[qid] = hits  # Store hits for batch reranking

                # Update debug stats
                if args.variant == "agentic":
                    agentic_debug["avg_variants"] += debug_info.get("variants", 0)
                    agentic_debug["avg_enh_ms"] += debug_info.get("enhance_ms", 0)
                    dbg_count += 1
                    if debug_info.get("fallback_used"):
                        agentic_debug["rerank_fallback_count"] = agentic_debug.get("rerank_fallback_count", 0) + 1
                elif args.variant == "decomposition":
                    decomp_debug["avg_aspects"] += debug_info.get("aspects_count", 0)
                    decomp_debug["avg_extract_ms"] += debug_info.get("extraction_ms", 0)
                    dbg_count += 1
                    if debug_info.get("used_baseline"):
                        decomp_debug["baseline_fallback_count"] += 1
                elif args.variant == "multihop":
                    multihop_debug["avg_aspects"] += debug_info.get("aspects_count", 0)
                    multihop_debug["avg_dag_levels"] += debug_info.get("dag_levels", 0)
                    multihop_debug["avg_extract_ms"] += debug_info.get("extraction_ms", 0)
                    dbg_count += 1
                    if debug_info.get("fallback"):
                        multihop_debug["fallback_count"] += 1
                elif args.variant == "adaptive":
                    strategy = debug_info.get("strategy", "baseline")
                    adaptive_debug["strategy_counts"][strategy] += 1
                    adaptive_debug["avg_aspect_count"] += debug_info.get("aspect_count", 0)
                    adaptive_debug["avg_extract_ms"] += debug_info.get("extraction_ms", 0)
                    dbg_count += 1
                    if debug_info.get("fallback"):
                        adaptive_debug["fallback_count"] += 1

                completed_count += 1

            except Exception as e:
                print(f"[ERROR] Query failed: {e}")
                failed_count += 1
                completed_count += 1

            # Progress tracking
            if completed_count % 10 == 0 or completed_count == len(queries):
                dt = time.perf_counter() - t0
                speed = completed_count / dt if dt > 0 else 0
                eta = (len(queries) - completed_count) / speed if speed > 0 else 0
                print(f"[PROGRESS] {completed_count}/{len(queries)} ({completed_count*100//len(queries)}%) | "
                      f"{speed:.2f} q/s | ETA: {eta/60:.1f} min | Failed: {failed_count}")

        if failed_count > 0:
            print(f"[WARNING] {failed_count} queries failed during processing")

    # =============================================================================
    # PHASE 5: Batch Reranking (if enabled)
    # =============================================================================
    if args.batch_rerank and args.rerank:
        print(f"\n{'='*80}")
        print(f"[BATCH RERANK] Starting batch reranking for {len(queries)} queries...")
        print(f"{'='*80}")

        # Batch rerank all queries at once
        reranked_hits = batch_rerank_all_queries(queries, retrieval_hits, top_k=args.top_k)

        # Update retrieval_doc_ids with reranked results
        for qid, hits in reranked_hits.items():
            retrieval_doc_ids[qid] = [h.doc_id for h in hits]

        print(f"[BATCH RERANK] All queries reranked successfully!\n")

    # Compute metrics
    metrics = compute_retrieval_metrics(retrieval_doc_ids, qrels, k_values=[1, 3, 5, 10])
    run_seconds = time.perf_counter() - t0
    
    results_str = "\n" + "="*60 + "\n"
    results_str += f"[RESULTS] Retrieval Metrics ({args.variant})\n"
    results_str += "="*60 + "\n"
    for k, v in sorted(metrics.items()):
        results_str += f"  {k:20s}: {v:.4f}\n"
    
    print(results_str)
    if logger:
        logger.log(results_str)

    if args.results_out:
        run_info = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_id": args.dataset_id,
            "variant": args.variant,
            "corpus": args.corpus,
            "search_mode": args.search_mode,
            "fusion": args.fusion,
            "fusion_strategy": args.fusion_strategy if args.variant == "agentic" else None,
            "original_weight": args.original_weight if args.variant in ["decomposition", "adaptive"] else None,
            "top_k": args.top_k,
            "max_variants": args.max_variants,
            "include_original": args.include_original,
            "max_queries": args.max_queries,
            "max_docs": args.max_docs,
            "reindex": args.reindex,
            "batch_size": args.batch_size,
            "queries_count": len(queries),
            "docs_count": len(documents),
            "qrels_count": len(qrels),
            "runtime_sec": round(run_seconds, 3),
            "metrics": metrics,
        }
        with open(args.results_out, "a", encoding="utf-8") as f:
            f.write(json.dumps(run_info, ensure_ascii=False) + "\n")
    
    if dbg_count:
        agentic_debug["avg_variants"] /= dbg_count
        agentic_debug["avg_enh_ms"] /= dbg_count
        debug_str = f"\n[AGENTIC] avg_variants: {agentic_debug['avg_variants']:.2f}\n"
        debug_str += f"[AGENTIC] avg_enhance_time: {agentic_debug['avg_enh_ms']:.1f} ms\n"
        if args.adaptive_fallback:
            fallback_count = agentic_debug.get("fallback_count", 0)
            fallback_rate = (fallback_count / dbg_count) * 100
            debug_str += f"[AGENTIC] fallback_used: {fallback_count}/{dbg_count} ({fallback_rate:.1f}%)\n"
        if args.rerank and args.adaptive_fallback:
            rerank_fallback_count = agentic_debug.get("rerank_fallback_count", 0)
            rerank_fallback_rate = (rerank_fallback_count / dbg_count) * 100
            debug_str += f"[AGENTIC] rerank_fallback_used: {rerank_fallback_count}/{dbg_count} ({rerank_fallback_rate:.1f}%)\n"
        print(debug_str)
        if logger:
            logger.log(debug_str)

    if dbg_count and args.variant == "multihop":
        multihop_debug["avg_aspects"] /= dbg_count
        multihop_debug["avg_dag_levels"] /= dbg_count
        multihop_debug["avg_extract_ms"] /= dbg_count
        debug_str = f"\n[MULTIHOP] avg_aspects: {multihop_debug['avg_aspects']:.2f}\n"
        debug_str += f"[MULTIHOP] avg_dag_levels: {multihop_debug['avg_dag_levels']:.2f}\n"
        debug_str += f"[MULTIHOP] avg_extract_time: {multihop_debug['avg_extract_ms']:.1f} ms\n"
        fallback_count = multihop_debug.get("fallback_count", 0)
        fallback_rate = (fallback_count / dbg_count) * 100
        debug_str += f"[MULTIHOP] fallback_to_baseline: {fallback_count}/{dbg_count} ({fallback_rate:.1f}%)\n"
        print(debug_str)
        if logger:
            logger.log(debug_str)

    if dbg_count and args.variant == "adaptive":
        adaptive_debug["avg_aspect_count"] /= dbg_count
        adaptive_debug["avg_extract_ms"] /= dbg_count
        debug_str = f"\n[ADAPTIVE] Strategy distribution:\n"
        for strategy, count in sorted(adaptive_debug["strategy_counts"].items()):
            percentage = (count / dbg_count) * 100
            debug_str += f"  {strategy:15s}: {count:4d} ({percentage:5.1f}%)\n"
        debug_str += f"[ADAPTIVE] avg_aspect_count: {adaptive_debug['avg_aspect_count']:.2f}\n"
        debug_str += f"[ADAPTIVE] avg_extract_time: {adaptive_debug['avg_extract_ms']:.1f} ms\n"
        fallback_count = adaptive_debug.get("fallback_count", 0)
        fallback_rate = (fallback_count / dbg_count) * 100
        debug_str += f"[ADAPTIVE] fallback_errors: {fallback_count}/{dbg_count} ({fallback_rate:.1f}%)\n"
        print(debug_str)
        if logger:
            logger.log(debug_str)

    # Cleanup
    if llm:
        await llm.cleanup()
    await store.cleanup()
    
    if logger:
        logger.close()
        print(f"\n[LOG] Detailed logs saved to: {args.output}")
    
    print("\n[DONE] Experiment completed!")


if __name__ == "__main__":
    asyncio.run(main())
