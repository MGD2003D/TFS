"""
Multi-hop QA — 2WikiMultihopQA (IRCoT processed_data, fixed passages).

Usage:
    python runs/qa_2wiki.py
    python runs/qa_2wiki.py --llm qwen
    python runs/qa_2wiki.py --classify          # + confusion matrix data
    python runs/qa_2wiki.py --max-samples 200 --workers 8
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _base_qa import make_parser, run_qa_benchmark

DATASET_KEY   = "2wikimultihopqa"
DATASET_LABEL = "2WikiMultihopQA"

if __name__ == "__main__":
    ap = make_parser(DATASET_KEY, DATASET_LABEL)
    args = ap.parse_args()
    asyncio.run(run_qa_benchmark(args, DATASET_KEY, DATASET_LABEL, expected_strategy="decomposition"))
