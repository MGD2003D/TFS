"""
BEIR NFCorpus benchmark.

Usage:
    python runs/beir_nfcorpus.py
    python runs/beir_nfcorpus.py --reindex
    python runs/beir_nfcorpus.py --variant agentic --include-original
    python runs/beir_nfcorpus.py --variant adaptive --checkpoint logs/nfcorpus_ckpt.jsonl
"""
import asyncio
from _base_beir import make_parser, run_beir

DATASET_ID = "beir/nfcorpus/test"
COLLECTION  = "bench_nfcorpus"

if __name__ == "__main__":
    ap = make_parser(DATASET_ID, COLLECTION)
    args = ap.parse_args()
    asyncio.run(run_beir(args))
