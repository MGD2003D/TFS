"""
Shared runner for BEIR retrieval benchmarks.
Each runs/beir_*.py calls run_beir() with dataset-specific defaults.
"""
import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# Make sure project root and Universal/ are importable
_UNIVERSAL = Path(__file__).resolve().parents[1]
_PROJECT   = _UNIVERSAL.parents[1]
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_UNIVERSAL))

from core.infra import add_project_root_to_syspath, init_llm_with_key, init_vector_store, init_query_enhancer, call_with_retry
from core.indexer import index_documents
from core.retriever import retrieve_baseline, retrieve_agentic, retrieve_decomposition, retrieve_multihop, retrieve_adaptive, Hit
from core.reranker import rerank_hits, batch_rerank_all_queries
from core.metrics import compute_retrieval_metrics
from core.logger import Logger
from bench_datasets.loader import load_ds, take_queries, build_qrels, build_full_corpus, build_head_corpus, build_candidate_corpus, DATASET_EXPECTED_STRATEGY


LOGS_DIR = _UNIVERSAL / "logs"
RESULTS_FILE = _UNIVERSAL / "results.jsonl"


def make_parser(dataset_id: str, collection: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id",   default=dataset_id)
    ap.add_argument("--collection",   default=collection)
    ap.add_argument("--embedding",    default="intfloat/multilingual-e5-base")
    ap.add_argument("--variant",      choices=["baseline", "agentic", "decomposition", "multihop", "adaptive"], default="baseline")
    ap.add_argument("--corpus",       choices=["full", "head", "candidate"], default="full")
    ap.add_argument("--max-queries",  type=int, default=None)
    ap.add_argument("--max-docs",     type=int, default=10_000)
    ap.add_argument("--top-k",        type=int, default=10)
    ap.add_argument("--max-variants", type=int, default=4)
    ap.add_argument("--include-original", action="store_true")
    ap.add_argument("--search-mode",  choices=["hybrid", "dense", "sparse"], default="hybrid")
    ap.add_argument("--fusion",       choices=["weighted", "rrf"], default="rrf")
    ap.add_argument("--reindex",      action="store_true")
    ap.add_argument("--batch-size",   type=int, default=128)
    ap.add_argument("--max-workers",  type=int, default=1)
    ap.add_argument("--rerank",       action="store_true")
    ap.add_argument("--rerank-top-k", type=int, default=100)
    ap.add_argument("--batch-rerank", action="store_true")
    ap.add_argument("--original-weight",   type=float, default=3.0)
    ap.add_argument("--humanfactor",       action="store_true")
    ap.add_argument("--humanfactor-weight",type=float, default=2.0)
    ap.add_argument("--lazygraph",    action="store_true")
    ap.add_argument("--llm",          default="gemini",
                    help="LLM backend: 'qwen' or 'gemini' / 'gemini:gemini-2.5-flash'")
    ap.add_argument("--output",       default=None, help="Detailed log file path")
    ap.add_argument("--log-console",  action="store_true")
    ap.add_argument("--checkpoint",   default=None,
                    help="JSONL file to save/resume per-query results (for long runs)")
    return ap


async def run_beir(args: argparse.Namespace):
    add_project_root_to_syspath()

    logger = Logger(args.output, console_details=args.log_console) if args.output else None

    print(f"[RUN] dataset={args.dataset_id} variant={args.variant} mode={args.search_mode} fusion={args.fusion}")

    # --- Load dataset ---
    ds = load_ds(args.dataset_id)
    queries = take_queries(ds, max_queries=args.max_queries)
    qids = set(queries.keys())

    if args.corpus == "full":
        documents = build_full_corpus(ds)
        qrels = build_qrels(ds, qids)
    elif args.corpus == "candidate":
        documents, qrels = build_candidate_corpus(ds, qids)
    else:
        documents = build_head_corpus(ds, max_docs=args.max_docs)
        qrels = build_qrels(ds, qids)

    print(f"[DATA] queries={len(queries)}  docs={len(documents)}  qrels={len(qrels)}")

    # --- Vector store ---
    store = await init_vector_store(collection_name=args.collection, embedding_model=args.embedding)

    if args.reindex:
        await store.delete_collection()
        await store.initialize()
        if store.enable_hybrid_search and getattr(store, "sparse_encoder", None) and not store.sparse_vocab_ready:
            store.build_sparse_vocab(list(documents.values()))
        await index_documents(store, documents, batch_size=args.batch_size)
    else:
        print("[INDEX] Using existing collection")
        if store.enable_hybrid_search and getattr(store, "sparse_encoder", None) and not store.sparse_vocab_ready:
            store.build_sparse_vocab(list(documents.values()))

    # --- LLM (if needed) ---
    llm = None
    enhancer = None
    if args.variant != "baseline":
        print("\n" + "=" * 60)
        print(f"[SETUP] {args.variant.capitalize()} mode requires API key")
        print("=" * 60)
        use_env = input("Use API key from .env file? (y/n): ").strip().lower()
        if use_env == "y":
            llm = await init_llm_with_key(llm=args.llm)
        else:
            api_key = input("API Key: ").strip()
            llm = await init_llm_with_key(api_key, llm=args.llm)
        if args.variant == "agentic":
            enhancer = await init_query_enhancer()

    # --- Graph (optional) ---
    graph = None
    if args.lazygraph and args.variant in ("multihop", "adaptive"):
        from knowledge_graph import KnowledgeGraph
        graph = KnowledgeGraph(
            qdrant_client=store.client,
            embedding_model=store.embedding_model,
            collection_name=f"kg_{args.collection}",
        )
        print(f"[GRAPH] Ready ({graph.count()} triplets)")

    # --- Load checkpoint ---
    done_qids: set = set()
    checkpoint_results: dict = {}
    if args.checkpoint and Path(args.checkpoint).exists():
        with open(args.checkpoint, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                done_qids.add(rec["qid"])
                checkpoint_results[rec["qid"]] = rec["doc_ids"]
        print(f"[CHECKPOINT] Resuming: {len(done_qids)} queries already done")

    # --- Per-query detail log (automatic, separate from results summary) ---
    ts = time.strftime("%Y%m%d_%H%M%S")
    ds_slug = args.dataset_id.replace("/", "_")
    detail_log_path = LOGS_DIR / f"{ds_slug}_{args.variant}_{ts}.jsonl"
    LOGS_DIR.mkdir(exist_ok=True)
    detail_fh = open(detail_log_path, "w", encoding="utf-8")
    print(f"[LOG] Per-query details → {detail_log_path}")

    # --- Query processing ---
    retrieval_doc_ids: dict = dict(checkpoint_results)
    retrieval_hits: dict = {}

    pending = [(qid, qt) for qid, qt in queries.items() if qid not in done_qids]
    t0 = time.perf_counter()
    llm_backend = args.llm

    async def process_one(qid: str, qtext: str, idx: int):
        retrieval_k = args.rerank_top_k if args.rerank else args.top_k

        if args.variant == "baseline":
            hits, dbg = await retrieve_baseline(store, qtext, retrieval_k, args.fusion, args.search_mode)
        elif args.variant == "agentic":
            hits, dbg = await retrieve_agentic(
                store, enhancer, qtext, retrieval_k, args.max_variants,
                args.fusion, args.search_mode,
                include_original=args.include_original,
                llm_backend=llm_backend,
            )
        elif args.variant == "decomposition":
            hits, dbg = await retrieve_decomposition(
                store, llm, qtext, retrieval_k, args.fusion, args.search_mode,
                original_weight=args.original_weight,
                humanfactor=args.humanfactor,
                humanfactor_weight=args.humanfactor_weight,
            )
        elif args.variant == "multihop":
            hits, dbg = await retrieve_multihop(store, llm, qtext, retrieval_k, args.fusion, args.search_mode, graph=graph)
        else:  # adaptive
            hits, dbg = await retrieve_adaptive(
                store, llm, qtext, retrieval_k, args.fusion, args.search_mode,
                original_weight=args.original_weight,
                humanfactor=args.humanfactor,
                humanfactor_weight=args.humanfactor_weight,
                graph=graph,
            )

        if args.rerank and not args.batch_rerank:
            hits = rerank_hits(qtext, hits, top_k=args.top_k)

        return qid, hits, dbg

    sem = asyncio.Semaphore(args.max_workers)

    async def _process_with_sem(qid, qtext, idx):
        async with sem:
            return await process_one(qid, qtext, idx)

    tasks = [asyncio.create_task(_process_with_sem(qid, qt, i)) for i, (qid, qt) in enumerate(pending, 1)]

    checkpoint_fh = None
    if args.checkpoint:
        checkpoint_fh = open(args.checkpoint, "a", encoding="utf-8")

    completed = 0
    for coro in asyncio.as_completed(tasks):
        try:
            qid, hits, dbg = await coro
            doc_ids = [h.doc_id for h in hits]
            retrieval_doc_ids[qid] = doc_ids
            retrieval_hits[qid] = hits
            completed += 1

            # Checkpoint (minimal — just doc_ids for resume)
            if checkpoint_fh:
                checkpoint_fh.write(json.dumps({"qid": qid, "doc_ids": doc_ids}, ensure_ascii=False) + "\n")
                checkpoint_fh.flush()

            # Detail log (rich — for analysis)
            relevant = set(qrels.get(qid, []))
            detail_rec = {
                "qid": qid,
                "query": queries[qid],
                "strategy": dbg.get("strategy", args.variant),
                "retrieved": doc_ids,
                "relevant": list(relevant & set(doc_ids)),
                "hit@10": int(bool(relevant & set(doc_ids[:10]))),
                **{k: v for k, v in dbg.items() if k not in ("strategy",) and isinstance(v, (str, int, float, bool))},
            }
            detail_fh.write(json.dumps(detail_rec, ensure_ascii=False) + "\n")
            detail_fh.flush()

        except Exception as e:
            print(f"[ERROR] {e}")
            completed += 1

        if completed % 10 == 0 or completed == len(pending):
            dt = time.perf_counter() - t0
            speed = completed / dt if dt > 0 else 0
            eta = (len(pending) - completed) / speed if speed > 0 else 0
            print(f"[PROGRESS] {completed}/{len(pending)} | {speed:.2f} q/s | ETA {eta/60:.1f} min")

    if checkpoint_fh:
        checkpoint_fh.close()
    detail_fh.close()

    # --- Batch rerank ---
    if args.batch_rerank and args.rerank:
        reranked = batch_rerank_all_queries(queries, retrieval_hits, top_k=args.top_k)
        for qid, hits in reranked.items():
            retrieval_doc_ids[qid] = [h.doc_id for h in hits]

    # --- Metrics ---
    metrics = compute_retrieval_metrics(retrieval_doc_ids, qrels)
    run_seconds = time.perf_counter() - t0

    # Clean summary — only this matters at a glance
    print("\n" + "=" * 60)
    print(f"  DATASET : {args.dataset_id}")
    print(f"  VARIANT : {args.variant}  |  mode={args.search_mode}  fusion={args.fusion}")
    print("=" * 60)
    for k in ["nDCG@1", "nDCG@3", "nDCG@5", "nDCG@10", "Recall@10", "MRR@10"]:
        if k in metrics:
            print(f"  {k:12s}: {metrics[k]:.4f}")
    print("=" * 60)
    print(f"  queries={len(queries)}  runtime={run_seconds:.0f}s")
    print(f"  details → {detail_log_path.name}")

    # --- Save summary to results.jsonl ---
    run_info = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_id": args.dataset_id,
        "variant": args.variant,
        "search_mode": args.search_mode,
        "fusion": args.fusion,
        "embedding": args.embedding,
        "collection": args.collection,
        "queries_count": len(queries),
        "docs_count": len(documents),
        "qrels_count": len(qrels),
        "llm": args.llm if args.variant != "baseline" else None,
        "rerank": args.rerank,
        "runtime_sec": round(run_seconds, 2),
        "detail_log": str(detail_log_path),
        "metrics": metrics,
    }
    run_info = {k: v for k, v in run_info.items() if v is not None}

    RESULTS_FILE.parent.mkdir(exist_ok=True)
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(run_info, ensure_ascii=False) + "\n")
    print(f"\n  summary → {RESULTS_FILE}")

    if logger:
        logger.close()
