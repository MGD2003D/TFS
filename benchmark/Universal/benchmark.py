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

# LLM backend (set from --llm arg)
_llm_backend = "qwen"


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

async def init_llm_with_key(api_key: Optional[str] = None, llm: str = "qwen"):
    """
    Initialize LLM with optional API key override.

    Args:
        api_key: override CAILA_TOKEN
        llm: which LLM to use — "qwen" (default) or "gemini"
              gemini model can be specified as "gemini:gemini-2.5-flash"
    """
    import app_state

    if api_key:
        os.environ["CAILA_TOKEN"] = api_key

    if llm.startswith("gemini"):
        from services.llm.gemini_client import GeminiClient
        # Support "gemini" or "gemini:gemini-2.5-flash"
        model = "gemini-2.5-flash"
        if ":" in llm:
            model = llm.split(":", 1)[1]
        client = GeminiClient(model=model)
        print(f"[LLM] Using Gemini ({model})")
    else:
        from services.llm.caila_client import CailaClient
        client = CailaClient()
        print(f"[LLM] Using Qwen3-30B (Caila)")

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


async def call_with_retry(func, *args, max_retries=7, llm_backend: str = "qwen", **kwargs):
    """
    Call async function with retry logic.
    After max_retries, prompt for new API key.

    Args:
        llm_backend: passed to init_llm_with_key on reinit after key change
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
                    
                    await init_llm_with_key(current_api_key, llm=llm_backend)
                    
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

# =============================================================================
# HuggingFace dataset adapters
# Wrap HF datasets to expose the same interface as ir_datasets:
#   .queries_iter() → namedtuple-like with .query_id, .text
#   .docs_iter()    → namedtuple-like with .doc_id, .text
#   .qrels_iter()   → namedtuple-like with .query_id, .doc_id, .relevance
# =============================================================================

class _Q:
    """Minimal query container."""
    def __init__(self, query_id, text):
        self.query_id = query_id
        self.text = text

class _D:
    """Minimal document container."""
    def __init__(self, doc_id, text):
        self.doc_id = doc_id
        self.text = text

class _R:
    """Minimal qrel container."""
    def __init__(self, query_id, doc_id, relevance=1):
        self.query_id = query_id
        self.doc_id = doc_id
        self.relevance = relevance


class HFDatasetAdapter:
    """
    Wraps a HuggingFace dataset to expose ir_datasets-compatible interface.
    Each subclass implements _iter_queries, _iter_docs, _iter_qrels.
    """
    def queries_iter(self):
        return self._iter_queries()

    def docs_iter(self):
        return self._iter_docs()

    def qrels_iter(self):
        return self._iter_qrels()


class PopQAAdapter(HFDatasetAdapter):
    """
    PopQA: akariasai/PopQA, split=test
    Fields: id, question, possible_answers (list), prop, subj, obj, s_pop, o_pop
    No separate corpus — answers are inline. We create synthetic single-sentence docs.
    """
    def __init__(self, hf_ds):
        self._ds = hf_ds

    def _iter_queries(self):
        for row in self._ds:
            yield _Q(str(row["id"]), row["question"])

    def _iter_docs(self):
        # Each answer becomes a minimal document
        for row in self._ds:
            qid = str(row["id"])
            answers = row.get("possible_answers") or []
            if isinstance(answers, str):
                import json as _json
                try:
                    answers = _json.loads(answers)
                except Exception:
                    answers = [answers]
            for i, ans in enumerate(answers):
                doc_id = f"{qid}_ans_{i}"
                text = f"{row['question']} {ans}"
                yield _D(doc_id, text)

    def _iter_qrels(self):
        for row in self._ds:
            qid = str(row["id"])
            answers = row.get("possible_answers") or []
            if isinstance(answers, str):
                import json as _json
                try:
                    answers = _json.loads(answers)
                except Exception:
                    answers = [answers]
            for i in range(len(answers)):
                yield _R(qid, f"{qid}_ans_{i}", relevance=1)


class HotpotQAAdapter(HFDatasetAdapter):
    """
    HotpotQA: hotpot_qa, fullwiki, split=validation
    Fields: id, question, answer, context (dict: title→sentences), type, level
    """
    def __init__(self, hf_ds):
        self._ds = hf_ds

    def _iter_queries(self):
        for row in self._ds:
            yield _Q(row["id"], row["question"])

    def _iter_docs(self):
        seen = set()
        for row in self._ds:
            context = row.get("context", {})
            titles = context.get("title", [])
            sentences_list = context.get("sentences", [])
            for title, sentences in zip(titles, sentences_list):
                doc_id = f"hotpot_{title.replace(' ', '_')[:80]}"
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                text = title + " " + " ".join(sentences)
                yield _D(doc_id, text)

    def _iter_qrels(self):
        for row in self._ds:
            qid = row["id"]
            context = row.get("context", {})
            titles = context.get("title", [])
            supporting = row.get("supporting_facts", {})
            sup_titles = set(supporting.get("title", [])) if supporting else set(titles)
            for title in sup_titles:
                doc_id = f"hotpot_{title.replace(' ', '_')[:80]}"
                yield _R(qid, doc_id, relevance=1)


class TwoWikiAdapter(HFDatasetAdapter):
    """
    2WikiMultiHopQA: framolfese/2WikiMultihopQA, split=validation
    Fields: id, question, answer, type, supporting_facts {title, sent_id}, context {title, sentences}
    Same structure as HotpotQA.
    """
    def __init__(self, hf_ds, dataset_name: str):
        self._ds = hf_ds
        self._name = dataset_name

    def _iter_queries(self):
        for row in self._ds:
            yield _Q(str(row["id"]), row["question"])

    def _iter_docs(self):
        seen = set()
        for row in self._ds:
            context = row.get("context", {})
            titles = context.get("title", [])
            sentences_list = context.get("sentences", [])
            for title, sentences in zip(titles, sentences_list):
                doc_id = f"2wiki_{title.replace(' ', '_')[:80]}"
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                text = title + " " + " ".join(sentences)
                yield _D(doc_id, text)

    def _iter_qrels(self):
        for row in self._ds:
            qid = str(row["id"])
            # supporting_facts contains the titles of relevant docs
            sup = row.get("supporting_facts", {})
            for title in sup.get("title", []):
                yield _R(qid, f"2wiki_{title.replace(' ', '_')[:80]}", relevance=1)


class AmbigNQAdapter(HFDatasetAdapter):
    """
    AmbigNQ: ambig_qa, light, split=validation
    Fields: id, question, annotations (dict with answers)
    """
    def __init__(self, hf_ds):
        self._ds = hf_ds

    def _iter_queries(self):
        for row in self._ds:
            yield _Q(str(row["id"]), row["question"])

    @staticmethod
    def _extract_answers(annotations) -> list:
        """Extract answer strings from ambig_qa annotations field.
        HuggingFace returns annotations as dict-of-lists:
          {"type": ["singleAnswer", ...], "answer": [["ans"], {"answer": ["ans2"]}, ...]}
        """
        answers = []
        if not annotations:
            return answers
        if isinstance(annotations, dict):
            raw = annotations.get("answer") or []
        elif isinstance(annotations, list):
            # List of annotation dicts: [{"type": "singleAnswer", "answer": ["ans"]}, ...]
            raw = []
            for item in annotations:
                if isinstance(item, dict):
                    raw.extend(item.get("answer") or [])
                elif isinstance(item, str):
                    raw.append(item)
            return [a for a in raw if isinstance(a, str)]
        else:
            return answers
        for a in raw:
            if isinstance(a, str):
                answers.append(a)
            elif isinstance(a, list):
                answers.extend(x for x in a if isinstance(x, str))
            elif isinstance(a, dict):
                sub = a.get("answer") or []
                if isinstance(sub, list):
                    answers.extend(x for x in sub if isinstance(x, str))
                elif isinstance(sub, str):
                    answers.append(sub)
        return answers

    def _iter_docs(self):
        seen = set()
        for row in self._ds:
            qid = str(row["id"])
            answers = self._extract_answers(row.get("annotations"))
            for i, ans in enumerate(answers):
                doc_id = f"{qid}_ans_{i}"
                if doc_id not in seen:
                    seen.add(doc_id)
                    yield _D(doc_id, f"{row['question']} {ans}")

    def _iter_qrels(self):
        for row in self._ds:
            qid = str(row["id"])
            answers = self._extract_answers(row.get("annotations"))
            for i in range(len(answers)):
                yield _R(qid, f"{qid}_ans_{i}", relevance=1)


# HF dataset IDs that are handled via adapters (not ir_datasets)
HF_DATASETS = {
    "popqa":        ("akariasai/PopQA",            {"split": "test"}),
    "hotpotqa_hf":  ("hotpotqa/hotpot_qa",         {"name": "fullwiki", "split": "validation"}),
    "2wikimultihop":("framolfese/2WikiMultihopQA",  {"split": "validation"}),
    "hover":        ("hover",                       {"split": "validation"}),
    "ambignq":      ("sewon/ambig_qa",             {"name": "light", "split": "validation"}),
}

# Expected strategy per dataset (ground truth for confusion matrix)
# Based on query complexity characteristics of each dataset
DATASET_EXPECTED_STRATEGY = {
    # Single-hop factoid: no decomposition needed
    "popqa":             "baseline",
    "beir/nq":           "baseline",
    "beir/msmarco":      "baseline",
    "beir/trec-covid":   "baseline",
    "beir/fiqa":         "baseline",
    "beir/scifact":      "baseline",
    "beir/arguana":      "baseline",
    # Multi-hop sequential: each step depends on previous result
    "hotpotqa_hf":       "multihop",
    "beir/hotpotqa":     "multihop",
    "hover":             "multihop",
    # Multi-hop parallel: multiple independent aspects/constraints
    "2wikimultihop":     "decomposition",
    # Ambiguous questions: benefit from query expansion / multiple angles
    "ambignq":           "decomposition",
    "beir/nfcorpus":     "decomposition",
}


def load_ds(dataset_id: str):
    """
    Load dataset by ID. Supports:
    - ir_datasets IDs: "beir/hotpotqa", "beir/nq", etc.
    - HF shorthand IDs: "popqa", "hotpotqa_hf", "2wikimultihop", "hover", "ambignq"
    """
    if dataset_id in HF_DATASETS:
        from datasets import load_dataset as hf_load
        hf_name, hf_kwargs = HF_DATASETS[dataset_id]
        print(f"[DATASET] Loading HuggingFace dataset: {hf_name} ({hf_kwargs})")
        hf_ds = hf_load(hf_name, **hf_kwargs)

        if dataset_id == "popqa":
            return PopQAAdapter(hf_ds)
        elif dataset_id == "hotpotqa_hf":
            return HotpotQAAdapter(hf_ds)
        elif dataset_id in ("2wikimultihop", "hover"):
            return TwoWikiAdapter(hf_ds, dataset_id)
        elif dataset_id == "ambignq":
            return AmbigNQAdapter(hf_ds)

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


def build_gold_answers(ds, allowed_qids: set) -> Dict[str, List[str]]:
    """Extract gold answer strings for QA datasets (not available for BEIR)."""
    gold = {}
    if not hasattr(ds, '_ds'):
        return gold  # BEIR / ir_datasets — no answer strings
    for row in ds._ds:
        qid = str(row.get("id", ""))
        if qid not in allowed_qids:
            continue
        answers = []
        # popqa
        if "possible_answers" in row:
            raw = row["possible_answers"] or []
            if isinstance(raw, str):
                import json as _j
                try:
                    raw = _j.loads(raw)
                except Exception:
                    raw = [raw]
            answers = [a for a in raw if isinstance(a, str)]
        # hotpotqa / 2wiki
        elif "answer" in row and isinstance(row["answer"], str):
            answers = [row["answer"]]
        # ambignq
        elif "annotations" in row:
            answers = AmbigNQAdapter._extract_answers(row.get("annotations"))
        if answers:
            gold[qid] = answers
    return gold


def _normalize_answer(s: str) -> str:
    import re, string
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())


def compute_em(prediction: str, gold_answers: List[str]) -> float:
    pred = _normalize_answer(prediction)
    return float(any(_normalize_answer(g) == pred for g in gold_answers))


def compute_f1_score(prediction: str, gold_answers: List[str]) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    best = 0.0
    for gold in gold_answers:
        gold_tokens = _normalize_answer(gold).split()
        common = set(pred_tokens) & set(gold_tokens)
        if not common:
            continue
        p = len(common) / len(pred_tokens)
        r = len(common) / len(gold_tokens)
        f1 = 2 * p * r / (p + r)
        best = max(best, f1)
    return best


def compute_contains(prediction: str, gold_answers: List[str]) -> float:
    pred = _normalize_answer(prediction)
    return float(any(_normalize_answer(g) in pred for g in gold_answers))


async def generate_rag_answer(query: str, hits: List, llm_client) -> str:
    """Generate answer from retrieved docs using LLM."""
    context_parts = []
    for i, hit in enumerate(hits[:5], 1):
        context_parts.append(f"[{i}] {hit.text[:300]}")
    context = "\n".join(context_parts)
    prompt = (
        f"Answer the question based on the context below. "
        f"Give a short, direct answer (1-5 words).\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    return await llm_client.simple_query(prompt)


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

    # Indexing is GPU-bound (single encoder), parallel workers cause deadlock on CUDA
    # max_workers is still used for query processing later
    print(f"[INDEX] Sequential indexing (GPU encoder is single-threaded)")
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
        variants = await call_with_retry(enhance_with_fallback, max_retries=7, llm_backend=_llm_backend)
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

    raw_aspects = await llm.extract_aspects(query)

    debug["extraction_ms"] = (time.perf_counter() - t0) * 1000.0

    # Normalize new format (type/aspects/hops) to flat dict
    if raw_aspects and isinstance(raw_aspects, dict) and "type" in raw_aspects:
        if raw_aspects["type"] == "parallel" and "aspects" in raw_aspects:
            aspects = {"original": raw_aspects["original"]}
            aspects.update(raw_aspects["aspects"])
        else:
            aspects = {"original": raw_aspects.get("original", query)}
    else:
        aspects = raw_aspects

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
    graph=None,
    logger=None
):
    """
    Multihop retrieval with sequential hops and optional graph cache.

    Strategy:
    1. Extract hops via LLM (new sequential format)
    2. For each hop: graph lookup → search → LLM extract → graph save
    3. Top-down refinement with all resolved answers
    4. Dependency-aware reranking

    Args:
        store: Vector store
        llm: LLM client with extract_aspects() method
        query: User query
        top_k: Number of results to return
        fusion: Fusion strategy for hybrid search
        search_mode: dense/sparse/hybrid
        graph: Optional KnowledgeGraph for lazy caching
        logger: Optional logger

    Returns:
        (hits, debug_info)
    """

    debug = {"aspects_count": 0, "dag_levels": 0, "extraction_ms": 0.0, "graph_hits": 0, "graph_saves": 0}

    if logger:
        logger.log("\n" + "="*80)
        logger.log(f"[MULTIHOP] Query: {query}")
        logger.log(f"[MULTIHOP] Graph: {'ON' if graph else 'OFF'}")
        logger.log("="*80)

    # =============================================================================
    # PHASE 1: Extraction
    # =============================================================================

    import time
    t0 = time.perf_counter()

    raw = await llm.extract_aspects(query)

    debug["extraction_ms"] = (time.perf_counter() - t0) * 1000.0

    # Normalize format
    if raw is None:
        debug["fallback"] = "baseline"
        hits, _ = await retrieve_baseline(store, query, top_k=top_k, fusion=fusion, search_mode=search_mode, logger=logger)
        return hits, debug

    query_type = raw.get("type", "simple")
    hops = raw.get("hops", [])

    # If not sequential or <2 hops, fallback
    if query_type != "sequential" or len(hops) < 2:
        # Try as decomposition if parallel
        if query_type == "parallel" and raw.get("aspects"):
            if logger:
                logger.log(f"[MULTIHOP] Got parallel type, delegating to decomposition")
            # Convert to flat format for decomposition
            flat_aspects = {"original": raw["original"]}
            flat_aspects.update(raw["aspects"])
            debug["fallback"] = "decomposition"
            # Inline decomposition (simplified)
            hits, _ = await retrieve_baseline(store, query, top_k=top_k, fusion=fusion, search_mode=search_mode, logger=logger)
            return hits, debug

        if logger:
            logger.log(f"[MULTIHOP] Simple/no hops, fallback to baseline")
        debug["fallback"] = "baseline"
        hits, _ = await retrieve_baseline(store, query, top_k=top_k, fusion=fusion, search_mode=search_mode, logger=logger)
        return hits, debug

    debug["aspects_count"] = len(hops)
    debug["dag_levels"] = len(hops)

    if logger:
        logger.log(f"\n[HOPS] {len(hops)} sequential hops:")
        for i, hop in enumerate(hops, 1):
            logger.log(f"  Hop {i}: {hop.get('query', '?')} → extract: {hop.get('extract', '?')}")

    # =============================================================================
    # PHASE 2: Sequential Hop Execution with Graph Cache
    # =============================================================================

    results_by_hop = {}
    resolved_answers = {}
    prev_context = ""
    triplets_to_save = []

    for hop_idx, hop in enumerate(hops):
        hop_query_template = hop.get("query", "")
        extract_hint = hop.get("extract", "")

        # Substitute {prev} with resolved context
        hop_query = hop_query_template
        if prev_context and "{prev}" in hop_query:
            hop_query = hop_query.replace("{prev}", prev_context)

        hop_key = f"hop_{hop_idx}"

        if logger:
            logger.log(f"\n[HOP {hop_idx + 1}] Query: {hop_query}")

        # === GRAPH LOOKUP ===
        graph_hit = None
        if graph and hop_idx < len(hops) - 1:
            try:
                graph_results = graph.search_semantic(hop_query, limit=3)
                if graph_results:
                    best_triplet, best_score = graph_results[0]
                    if best_score >= 0.75:
                        graph_hit = best_triplet.get("object", "")
                        debug["graph_hits"] += 1
                        if logger:
                            logger.log(f"  [GRAPH HIT] '{graph_hit}' (score={best_score:.3f})")
            except Exception as e:
                if logger:
                    logger.log(f"  [GRAPH] Lookup error: {e}")

        # Search
        results = await search_store(store, hop_query, top_k=20, fusion=fusion, search_mode=search_mode)
        results_by_hop[hop_key] = results

        if logger:
            logger.log(f"  Found {len(results)} results")

        # Extract context for next hop
        if hop_idx < len(hops) - 1:
            if graph_hit:
                prev_context = graph_hit
            elif results:
                prev_context = results[0].get("text", "")[:200]
            resolved_answers[hop_key] = prev_context

            # Save triplet for graph (if no graph hit — it's new knowledge)
            if not graph_hit and prev_context and len(prev_context) < 200:
                triplets_to_save.append({
                    "subject": hop_query,
                    "predicate": extract_hint or "resolves_to",
                    "object": prev_context,
                })

            if logger:
                source = "GRAPH" if graph_hit else "TOP-1"
                logger.log(f"  [{source}] Context: {prev_context[:100]}...")

    # === GRAPH SAVE ===
    if graph and triplets_to_save:
        try:
            for t in triplets_to_save:
                triplet = Triplet(
                    subject=t["subject"],
                    predicate=t["predicate"],
                    object=t["object"],
                    confidence=0.8,
                )
                graph.add_triplet(triplet)
            debug["graph_saves"] = len(triplets_to_save)
            if logger:
                logger.log(f"\n[GRAPH] Saved {len(triplets_to_save)} triplets")
                for t in triplets_to_save[:3]:
                    logger.log(f"  ({t['subject'][:40]}) --[{t['predicate']}]--> ({t['object'][:40]})")
        except Exception as e:
            if logger:
                logger.log(f"\n[GRAPH] Save error: {e}")

    # =============================================================================
    # PHASE 3: Top-down Refinement
    # =============================================================================

    refined_query = query
    for hop_key, answer in resolved_answers.items():
        refined_query += f" {answer}"

    if logger:
        logger.log(f"\n[TOP-DOWN] Refined query: {refined_query[:150]}...")

    final_results = await search_store(store, refined_query, top_k=top_k*2, fusion=fusion, search_mode=search_mode)
    results_by_hop["refined"] = final_results

    # =============================================================================
    # PHASE 4: Reranking
    # =============================================================================

    alpha, beta = 0.6, 0.4  # Sequential queries

    if logger:
        logger.log(f"\n[RERANKING] weights: α={alpha}, β={beta}")

    combined_results = {}

    for hop_key, results in results_by_hop.items():
        dep_answers_list = [v for k, v in resolved_answers.items() if v]

        for result in results:
            doc_id = result.get("metadata", {}).get("doc_id", "")
            if not doc_id:
                continue

            text = result.get("text", "")
            Ri = result.get("score", 0.0)
            Mi = compute_text_similarity(text, dep_answers_list) if dep_answers_list else 0.0
            final_score = alpha * Ri + beta * Mi

            if doc_id not in combined_results or final_score > combined_results[doc_id]["score"]:
                combined_results[doc_id] = {
                    "doc_id": doc_id,
                    "text": text,
                    "score": final_score,
                    "Ri": Ri,
                    "Mi": Mi,
                }

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
    graph=None,
    logger=None
):
    """
    Adaptive retrieval: LLM classifies query type → routes to strategy.

    Uses new extract_aspects format (type: simple/parallel/sequential):
    - simple → baseline
    - parallel → decomposition
    - sequential → multihop (with optional graph cache)

    Args:
        store: Vector store
        llm: LLM client
        query: User query
        top_k: Number of results
        fusion: Fusion strategy
        search_mode: dense/sparse/hybrid
        original_weight: Weight for original query (decomposition)
        humanfactor: Add reformulated query variant
        humanfactor_weight: Weight for reformulated query
        graph: Optional KnowledgeGraph for lazy caching (--lazygraph)
        logger: Optional logger

    Returns:
        (hits, debug_info)
    """

    debug = {"strategy": "baseline", "reasoning": ""}

    if logger:
        logger.log("\n" + "="*80)
        logger.log(f"[ADAPTIVE] Query: {query}")
        logger.log(f"[ADAPTIVE] Graph: {'ON' if graph else 'OFF'}")
        logger.log("="*80)

    # =============================================================================
    # PHASE 1: Extract & Classify
    # =============================================================================

    import time
    t0 = time.perf_counter()

    raw = await llm.extract_aspects(query)
    extraction_ms = (time.perf_counter() - t0) * 1000.0

    if raw is None:
        debug["strategy"] = "baseline"
        debug["reasoning"] = "Aspect extraction failed"
        debug["extraction_ms"] = extraction_ms
        if logger:
            logger.log(f"[ADAPTIVE] Strategy: baseline (extraction failed)")
        hits, _ = await retrieve_baseline(store, query, top_k=top_k, fusion=fusion, search_mode=search_mode, logger=logger)
        return hits, debug

    query_type = raw.get("type", "simple")

    # =============================================================================
    # PHASE 2: Strategy Classification (based on type field)
    # =============================================================================

    if query_type == "sequential" and len(raw.get("hops", [])) >= 2:
        strategy = "multihop"
        reasoning = f"Sequential query ({len(raw['hops'])} hops)"
        aspect_count = len(raw["hops"])
    elif query_type == "parallel" and len(raw.get("aspects", {})) >= 2:
        strategy = "decomposition"
        reasoning = f"Parallel aspects ({len(raw['aspects'])})"
        aspect_count = len(raw["aspects"]) + 1  # +1 for original
    else:
        strategy = "baseline"
        reasoning = f"Simple query (type={query_type})"
        aspect_count = 1

    debug["strategy"] = strategy
    debug["reasoning"] = reasoning
    debug["aspect_count"] = aspect_count
    debug["has_dependencies"] = query_type == "sequential"
    debug["extraction_ms"] = extraction_ms

    if logger:
        logger.log(f"\n[ADAPTIVE] Classification:")
        logger.log(f"  Type: {query_type}")
        logger.log(f"  → Strategy: {strategy} ({reasoning})")

    # =============================================================================
    # PHASE 3: Execute Selected Strategy
    # =============================================================================

    try:
        if strategy == "baseline":
            hits, _ = await retrieve_baseline(store, query, top_k=top_k, fusion=fusion, search_mode=search_mode, logger=logger)

        elif strategy == "decomposition":
            hits, sub_debug = await retrieve_decomposition(
                store, llm, query, top_k=top_k, fusion=fusion, search_mode=search_mode,
                original_weight=original_weight,
                humanfactor=humanfactor,
                humanfactor_weight=humanfactor_weight,
                logger=logger
            )
            debug["decomp_debug"] = sub_debug

        elif strategy == "multihop":
            hits, sub_debug = await retrieve_multihop(
                store, llm, query, top_k=top_k, fusion=fusion, search_mode=search_mode,
                graph=graph,
                logger=logger
            )
            debug["multihop_debug"] = sub_debug
            debug["graph_hits"] = sub_debug.get("graph_hits", 0)
            debug["graph_saves"] = sub_debug.get("graph_saves", 0)

        else:
            debug["fallback"] = "unknown_strategy"
            hits, _ = await retrieve_baseline(store, query, top_k=top_k, fusion=fusion, search_mode=search_mode, logger=logger)

    except Exception as e:
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
    ap.add_argument("--lazygraph", action="store_true",
                    help="Enable lazy knowledge graph: cache resolved multi-hop relationships in Qdrant. "
                         "Graph builds up as side-effect of queries. Speeds up repeated/similar multi-hop queries.")
    ap.add_argument("--llm", type=str, default="qwen",
                    help="LLM backend to use: 'qwen' (default, Qwen3-30B via Caila) or "
                         "'gemini' / 'gemini:gemini-2.5-flash' / 'gemini:gemini-2.5-pro' etc.")
    ap.add_argument("--query-log", type=str, default=None,
                    help="Per-query JSONL log with predicted/expected strategy for confusion matrix analysis. "
                         "Only written in --variant adaptive. Example: logs/queries_hotpotqa.jsonl")
    ap.add_argument("--generate", action="store_true",
                    help="After retrieval, generate answer with LLM and compute F1/EM (QA datasets only)")
    args = ap.parse_args()

    # Set global entity boosting config
    global _entity_boost_enabled, _entity_boost_weight, _llm_backend
    _entity_boost_enabled = args.entity_boost
    _entity_boost_weight = args.entity_boost_weight
    _llm_backend = args.llm

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

    gold_answers: Dict[str, List[str]] = {}
    if args.generate:
        gold_answers = build_gold_answers(ds, set(qids.keys()))
        print(f"[DATA] gold_answers: {len(gold_answers)} (for generation eval)")

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
            llm = await init_llm_with_key(llm=args.llm)
        else:
            api_key = prompt_for_api_key()
            llm = await init_llm_with_key(api_key, llm=args.llm)

        enhancer = await init_query_enhancer()

    elif args.variant in ["decomposition", "multihop", "adaptive"]:
        # These modes require LLM for aspect extraction / DAG planning
        mode_name = args.variant.capitalize()
        print("\n" + "="*60)
        print(f"[SETUP] {mode_name} mode requires API key")
        print("="*60)
        use_env = input("Use API key from .env file? (y/n): ").strip().lower()

        if use_env == 'y':
            llm = await init_llm_with_key(llm=args.llm)
        else:
            api_key = prompt_for_api_key()
            llm = await init_llm_with_key(api_key, llm=args.llm)

    # =============================================================================
    # LAZY KNOWLEDGE GRAPH
    # =============================================================================

    graph = None
    if args.lazygraph and args.variant in ["multihop", "adaptive"]:
        # Уникальная коллекция графа для каждого бенчмарка (привязана к --collection)
        graph_collection = f"kg_{args.collection}"
        print(f"\n[GRAPH] Initializing lazy knowledge graph (collection: {graph_collection})...")
        graph = KnowledgeGraph(
            qdrant_client=store.client,
            embedding_model=store.embedding_model,
            collection_name=graph_collection
        )
        graph_count = graph.count()
        print(f"[GRAPH] Ready ({graph_count} existing triplets)")
    elif args.lazygraph:
        print("[GRAPH] --lazygraph ignored (only works with --variant multihop/adaptive)")

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
                graph=graph,
                logger=logger if idx <= 5 else None
            )
            multihop_time = (time.perf_counter() - multihop_start) * 1000

            debug_info["aspects_count"] = dbg.get("aspects_count", 0)
            debug_info["dag_levels"] = dbg.get("dag_levels", 0)
            debug_info["extraction_ms"] = dbg.get("extraction_ms", 0.0)
            debug_info["fallback"] = dbg.get("fallback", None)
            debug_info["graph_hits"] = dbg.get("graph_hits", 0)
            debug_info["graph_saves"] = dbg.get("graph_saves", 0)

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
                graph=graph,
                logger=logger if idx <= 5 else None
            )
            adaptive_time = (time.perf_counter() - adaptive_start) * 1000

            debug_info["strategy"] = dbg.get("strategy", "baseline")
            debug_info["reasoning"] = dbg.get("reasoning", "")
            debug_info["aspect_count"] = dbg.get("aspect_count", 0)
            debug_info["has_dependencies"] = dbg.get("has_dependencies", False)
            debug_info["extraction_ms"] = dbg.get("extraction_ms", 0.0)
            debug_info["fallback"] = dbg.get("fallback", None)
            debug_info["graph_hits"] = dbg.get("graph_hits", 0)
            debug_info["graph_saves"] = dbg.get("graph_saves", 0)

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

        # Generation + F1/EM evaluation (if --generate and gold answers available)
        gen_record = {}
        if args.generate and qid in gold_answers:
            import app_state as _app_state
            print(f"[GENERATE] Generating answer for query: {qtext[:80]}")
            try:
                generated = await call_with_retry(
                    generate_rag_answer, qtext, hits, _app_state.llm_client,
                    max_retries=3, llm_backend=_llm_backend
                )
                generated = generated.strip()
                golds = gold_answers[qid]
                em = compute_em(generated, golds)
                f1 = compute_f1_score(generated, golds)
                contains = compute_contains(generated, golds)
                print(f"[ANSWER] generated: \"{generated}\"")
                print(f"[EVAL] F1: {f1:.2f}  EM: {em:.2f}  Contains: {contains:.2f}  gold={golds[:2]}")
                gen_record = {
                    "generated": generated,
                    "gold": golds,
                    "em": em,
                    "f1": f1,
                    "contains": contains,
                }
            except Exception as e:
                print(f"[GENERATE] Failed: {e}")

        # Per-query log (adaptive or when --generate is used)
        if (args.variant == "adaptive" or args.generate) and args.query_log:
            expected = DATASET_EXPECTED_STRATEGY.get(args.dataset_id, "unknown")
            predicted = debug_info.get("strategy", "simple")
            query_record = {
                "qid": qid,
                "query": qtext,
                "dataset": args.dataset_id,
                "expected_strategy": expected,
                "predicted_strategy": predicted,
                "reasoning": debug_info.get("reasoning", ""),
                "aspect_count": debug_info.get("aspect_count", 0),
                "has_dependencies": debug_info.get("has_dependencies", False),
                "query_time_ms": round(query_total_time, 1),
                "hits_count": len(hits),
                "relevant": bool(qrels.get(qid)),
                **gen_record,
            }
            with open(args.query_log, "a", encoding="utf-8") as qf:
                qf.write(json.dumps(query_record, ensure_ascii=False) + "\n")

        return qid, hits, query_total_time, debug_info

    # Run retrieval
    retrieval_doc_ids: Dict[str, List[str]] = {}
    retrieval_hits: Dict[str, List] = {}  # Store hits for batch reranking
    agentic_debug = {"avg_variants": 0.0, "avg_enh_ms": 0.0}
    decomp_debug = {"avg_aspects": 0.0, "avg_extract_ms": 0.0, "baseline_fallback_count": 0}
    multihop_debug = {"avg_aspects": 0.0, "avg_dag_levels": 0.0, "avg_extract_ms": 0.0, "fallback_count": 0, "graph_hits": 0, "graph_saves": 0}
    adaptive_debug = {"strategy_counts": defaultdict(int), "avg_aspect_count": 0.0, "avg_extract_ms": 0.0, "fallback_count": 0, "graph_hits": 0, "graph_saves": 0}
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
                    multihop_debug["graph_hits"] += debug_info.get("graph_hits", 0)
                    multihop_debug["graph_saves"] += debug_info.get("graph_saves", 0)
                    dbg_count += 1
                    if debug_info.get("fallback"):
                        multihop_debug["fallback_count"] += 1
                elif args.variant == "adaptive":
                    strategy = debug_info.get("strategy", "baseline")
                    adaptive_debug["strategy_counts"][strategy] += 1
                    adaptive_debug["avg_aspect_count"] += debug_info.get("aspect_count", 0)
                    adaptive_debug["avg_extract_ms"] += debug_info.get("extraction_ms", 0)
                    adaptive_debug["graph_hits"] += debug_info.get("graph_hits", 0)
                    adaptive_debug["graph_saves"] += debug_info.get("graph_saves", 0)
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
                    multihop_debug["graph_hits"] += debug_info.get("graph_hits", 0)
                    multihop_debug["graph_saves"] += debug_info.get("graph_saves", 0)
                    dbg_count += 1
                    if debug_info.get("fallback"):
                        multihop_debug["fallback_count"] += 1
                elif args.variant == "adaptive":
                    strategy = debug_info.get("strategy", "baseline")
                    adaptive_debug["strategy_counts"][strategy] += 1
                    adaptive_debug["avg_aspect_count"] += debug_info.get("aspect_count", 0)
                    adaptive_debug["avg_extract_ms"] += debug_info.get("extraction_ms", 0)
                    adaptive_debug["graph_hits"] += debug_info.get("graph_hits", 0)
                    adaptive_debug["graph_saves"] += debug_info.get("graph_saves", 0)
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

    # Aggregate generation metrics from query log if --generate was used
    if args.generate and args.query_log and os.path.exists(args.query_log):
        em_scores, f1_scores, contains_scores = [], [], []
        with open(args.query_log, encoding="utf-8") as _qf:
            for _line in _qf:
                try:
                    _rec = json.loads(_line)
                    if "em" in _rec:
                        em_scores.append(_rec["em"])
                        f1_scores.append(_rec["f1"])
                        contains_scores.append(_rec["contains"])
                except Exception:
                    pass
        if em_scores:
            metrics["gen_em"] = sum(em_scores) / len(em_scores)
            metrics["gen_f1"] = sum(f1_scores) / len(f1_scores)
            metrics["gen_contains"] = sum(contains_scores) / len(contains_scores)

    run_seconds = time.perf_counter() - t0

    results_str = "\n" + "="*60 + "\n"
    results_str += f"[RESULTS] Retrieval Metrics ({args.variant})\n"
    results_str += "="*60 + "\n"
    for k, v in sorted(metrics.items()):
        results_str += f"  {k:20s}: {v:.4f}\n"
    
    print(results_str)
    if logger:
        logger.log(results_str)

    # =============================================================================
    # SAVE RESULTS (JSONL)
    # =============================================================================

    results_file = args.results_out or "benchmark/Universal/results.jsonl"

    # Build complete run info with ALL parameters
    run_info = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        # Dataset
        "dataset_id": args.dataset_id,
        "corpus": args.corpus,
        "queries_count": len(queries),
        "docs_count": len(documents),
        "qrels_count": len(qrels),
        "expected_strategy": DATASET_EXPECTED_STRATEGY.get(args.dataset_id, "unknown"),
        # LLM
        "llm": args.llm,
        # Strategy
        "variant": args.variant,
        "search_mode": args.search_mode,
        "fusion": args.fusion,
        "top_k": args.top_k,
        "collection": args.collection,
        "embedding": args.embedding,
        # Variant-specific params
        "fusion_strategy": args.fusion_strategy if args.variant == "agentic" else None,
        "max_variants": args.max_variants if args.variant == "agentic" else None,
        "include_original": args.include_original if args.variant == "agentic" else None,
        "original_weight": args.original_weight if args.variant in ["decomposition", "adaptive"] else None,
        "humanfactor": args.humanfactor if args.variant in ["decomposition", "adaptive"] else None,
        "humanfactor_weight": args.humanfactor_weight if args.humanfactor else None,
        # Reranking
        "rerank": args.rerank,
        "rerank_top_k": args.rerank_top_k if args.rerank else None,
        "batch_rerank": args.batch_rerank if args.rerank else None,
        "adaptive_fallback": args.adaptive_fallback,
        # Entity
        "enable_ner": args.enable_ner,
        "entity_boost": args.entity_boost,
        "entity_boost_weight": args.entity_boost_weight if args.entity_boost else None,
        # Graph
        "lazygraph": args.lazygraph,
        # Index
        "reindex": args.reindex,
        "batch_size": args.batch_size,
        "max_queries": args.max_queries,
        "max_docs": args.max_docs,
        "max_workers": args.max_workers,
        # Timing
        "runtime_sec": round(run_seconds, 3),
        # Metrics
        "metrics": metrics,
    }

    # Add variant-specific debug stats
    if dbg_count:
        if args.variant == "agentic":
            run_info["debug"] = {
                "avg_variants": round(agentic_debug["avg_variants"] / dbg_count, 2) if dbg_count else 0,
                "avg_enhance_ms": round(agentic_debug["avg_enh_ms"] / dbg_count, 1) if dbg_count else 0,
            }
        elif args.variant == "decomposition":
            run_info["debug"] = {
                "avg_aspects": round(decomp_debug["avg_aspects"] / dbg_count, 2),
                "avg_extract_ms": round(decomp_debug["avg_extract_ms"] / dbg_count, 1),
                "baseline_fallback_count": decomp_debug["baseline_fallback_count"],
                "baseline_fallback_rate": round(decomp_debug["baseline_fallback_count"] / dbg_count * 100, 1),
            }
        elif args.variant == "multihop":
            run_info["debug"] = {
                "avg_aspects": round(multihop_debug["avg_aspects"] / dbg_count, 2),
                "avg_dag_levels": round(multihop_debug["avg_dag_levels"] / dbg_count, 2),
                "avg_extract_ms": round(multihop_debug["avg_extract_ms"] / dbg_count, 1),
                "fallback_count": multihop_debug["fallback_count"],
                "graph_hits": multihop_debug["graph_hits"],
                "graph_saves": multihop_debug["graph_saves"],
                "graph_total": graph.count() if graph else 0,
            }
        elif args.variant == "adaptive":
            run_info["debug"] = {
                "strategy_distribution": dict(adaptive_debug["strategy_counts"]),
                "avg_aspect_count": round(adaptive_debug["avg_aspect_count"] / dbg_count, 2),
                "avg_extract_ms": round(adaptive_debug["avg_extract_ms"] / dbg_count, 1),
                "fallback_count": adaptive_debug["fallback_count"],
                "graph_hits": adaptive_debug["graph_hits"],
                "graph_saves": adaptive_debug["graph_saves"],
                "graph_total": graph.count() if graph else 0,
            }

    # Remove None values for cleaner JSONL
    run_info = {k: v for k, v in run_info.items() if v is not None}

    with open(results_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(run_info, ensure_ascii=False) + "\n")
    print(f"\n[RESULTS] Saved to {results_file}")
    
    if dbg_count and args.variant == "agentic":
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
        if graph:
            debug_str += f"[MULTIHOP] graph_cache_hits: {multihop_debug['graph_hits']}\n"
            debug_str += f"[MULTIHOP] graph_saves: {multihop_debug['graph_saves']}\n"
            graph_count = graph.count()
            debug_str += f"[MULTIHOP] graph_total_triplets: {graph_count}\n"
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
        if graph:
            debug_str += f"[ADAPTIVE] graph_cache_hits: {adaptive_debug['graph_hits']}\n"
            debug_str += f"[ADAPTIVE] graph_saves: {adaptive_debug['graph_saves']}\n"
            graph_count = graph.count()
            debug_str += f"[ADAPTIVE] graph_total_triplets: {graph_count}\n"
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
