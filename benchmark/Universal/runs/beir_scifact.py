"""
BEIR SciFact benchmark.

Usage:
    python runs/beir_scifact.py                          # baseline hybrid
    python runs/beir_scifact.py --reindex                # reindex first
    python runs/beir_scifact.py --variant adaptive       # adaptive retrieval
    python runs/beir_scifact.py --variant adaptive --checkpoint logs/scifact_ckpt.jsonl
"""
import asyncio
from _base_beir import make_parser, run_beir

DATASET_ID = "beir/scifact/test"
COLLECTION  = "bench_scifact"

if __name__ == "__main__":
    ap = make_parser(DATASET_ID, COLLECTION)
    args = ap.parse_args()
    asyncio.run(run_beir(args))
