"""
Shared runner for multi-hop QA benchmarks (IRCoT processed_data).

Passages are fixed (bundled with the dataset) — no vector search needed.
We generate an answer with the configured LLM and compare EM/F1 against the
Adaptive-RAG paper baseline (IRCoT + Flan-T5-XL).
"""
import argparse
import asyncio
import json
import sys
import time
import zipfile
import tarfile
import urllib.request
from pathlib import Path

_UNIVERSAL = Path(__file__).resolve().parents[1]
_PROJECT   = _UNIVERSAL.parents[1]
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_UNIVERSAL))

from core.infra import add_project_root_to_syspath, init_llm_with_key
from core.metrics import compute_em, compute_f1_score

DATA_DIR     = _UNIVERSAL / "data"
LOGS_DIR     = _UNIVERSAL / "logs"
RESULTS_FILE = _UNIVERSAL / "results_qa.jsonl"

PREDICTIONS_TAR = DATA_DIR / "predictions.tar.gz"
PREDICTIONS_DIR = DATA_DIR / "predictions"
IRCOT_ZIP       = DATA_DIR / "processed_data.zip"
IRCOT_DIR       = DATA_DIR / "processed_data"


# =============================================================================
# Data download
# =============================================================================

def ensure_data():
    DATA_DIR.mkdir(exist_ok=True)

    if not PREDICTIONS_TAR.exists():
        print("Downloading predictions.tar.gz ...")
        urllib.request.urlretrieve(
            "https://github.com/starsuzi/Adaptive-RAG/raw/refs/heads/main/predictions.tar.gz",
            PREDICTIONS_TAR,
        )
    if not PREDICTIONS_DIR.exists():
        print("Extracting predictions.tar.gz ...")
        with tarfile.open(PREDICTIONS_TAR, "r:gz") as tar:
            tar.extractall(DATA_DIR)
    print("predictions : OK")

    if not IRCOT_ZIP.exists():
        print("Downloading IRCoT processed_data via gdown ...")
        import subprocess
        subprocess.run(
            ["gdown", "1t2BjJtsejSIUZI54PKObMFG6_wMMG3bC", "-O", str(IRCOT_ZIP)],
            check=True,
        )
    if not IRCOT_DIR.exists():
        print("Extracting processed_data.zip ...")
        with zipfile.ZipFile(IRCOT_ZIP) as z:
            z.extractall(DATA_DIR)
    print("IRCoT data  : OK")


# =============================================================================
# Baseline loading
# =============================================================================

def get_baseline(dataset_key: str, llm_key: str = "ircot_qa_flan_t5_xl") -> dict:
    """Load Adaptive-RAG evaluation_metrics.json for this dataset."""
    if not PREDICTIONS_DIR.exists():
        return {}
    # Search in all subfolders (dev_500, test, etc.)
    for sub in PREDICTIONS_DIR.iterdir():
        if not sub.is_dir():
            continue
        for folder in sub.iterdir():
            if not folder.is_dir():
                continue
            if dataset_key in folder.name and llm_key in folder.name:
                mf = next(folder.glob("evaluation_metrics__*.json"), None)
                if mf:
                    return json.loads(mf.read_text())
    return {}


# =============================================================================
# IRCoT sample loading
# =============================================================================

def load_ircot_samples(dataset_key: str, split: str = "dev_subsampled") -> list:
    candidates = [p for p in IRCOT_DIR.rglob(f"{split}.jsonl") if dataset_key in str(p)]
    if not candidates:
        candidates = list(IRCOT_DIR.rglob(f"{split}.jsonl"))
    if not candidates:
        print(f"  [WARN] {split}.jsonl not found for {dataset_key}")
        return []

    path = candidates[0]
    print(f"  {path.relative_to(IRCOT_DIR)}")
    samples = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            answers = []
            for ao in (row.get("answers_objects") or []):
                answers.extend(s for s in (ao.get("spans") or []) if s)
                if ao.get("number"):
                    answers.append(str(ao["number"]))
            samples.append({
                "qid":      str(row.get("question_id", row.get("id", ""))),
                "question": row.get("question_text", row.get("question", "")),
                "gold":     answers,
                "passages": [
                    {"title": c.get("title", ""), "text": c.get("paragraph_text", "")}
                    for c in (row.get("contexts") or [])
                ],
            })

    avg_p = sum(len(s["passages"]) for s in samples) // max(len(samples), 1)
    print(f"  {len(samples)} samples, ~{avg_p} passages each")
    return samples


# =============================================================================
# Answer generation + optional strategy classification
# =============================================================================

def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks (Qwen3 chain-of-thought)."""
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


async def generate_answer(llm, question: str, passages: list, max_passages: int = 15) -> str:
    ctx = "\n\n".join(
        f"[{i+1}] {p['title']}\n{p['text']}"
        for i, p in enumerate(passages[:max_passages])
    )
    prompt = (
        "Answer the question using ONLY the passages below. "
        "Give a short answer (1–5 words). Do NOT explain.\n\n"
        f"Passages:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"
    )
    raw = await llm.simple_query(prompt)
    return _strip_think(raw).split("\n")[0].strip()


def _parse_strategy(raw) -> str:
    """Normalize extract_aspects result to baseline/decomposition/multihop."""
    if not isinstance(raw, dict):
        return "baseline"
    t = raw.get("type", "simple")
    if t == "sequential":
        return "multihop"
    if t == "parallel":
        return "decomposition"
    return "baseline"


async def run_qa(
    llm,
    samples: list,
    workers: int = 4,
    detail_log_path: Path = None,   # all per-query records go here (append)
    expected_strategy: str = None,  # for confusion matrix
    classify: bool = False,         # also call extract_aspects
) -> list:
    """
    Process samples, write each result to detail_log_path immediately.
    No checkpoint file separate from detail log — detail log IS the record.
    Re-running always appends; use a fresh detail_log_path for a fresh run.
    """
    sem = asyncio.Semaphore(workers)
    log_fh = open(detail_log_path, "a", encoding="utf-8") if detail_log_path else None
    results = []

    async def _one(s):
        async with sem:
            coros = [generate_answer(llm, s["question"], s["passages"])]
            if classify and hasattr(llm, "extract_aspects"):
                coros.append(llm.extract_aspects(s["question"]))

            inner = await asyncio.gather(*coros, return_exceptions=True)

            pred_raw = inner[0]
            pred = "" if isinstance(pred_raw, Exception) else str(pred_raw)

            predicted_strategy = None
            if classify and len(inner) > 1 and not isinstance(inner[1], Exception):
                predicted_strategy = _parse_strategy(inner[1])

            rec = {
                "qid":      s["qid"],
                "question": s["question"],
                "gold":     s["gold"],
                "pred":     pred,
                "em":       compute_em(pred, s["gold"]),
                "f1":       compute_f1_score(pred, s["gold"]),
            }
            if expected_strategy:
                rec["expected_strategy"] = expected_strategy
            if predicted_strategy is not None:
                rec["predicted_strategy"] = predicted_strategy

            if log_fh:
                log_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                log_fh.flush()
            return rec

    tasks = [asyncio.create_task(_one(s)) for s in samples]
    completed = 0
    t0 = time.perf_counter()

    for coro in asyncio.as_completed(tasks):
        try:
            results.append(await coro)
        except Exception as e:
            print(f"[ERROR] {e}")
        completed += 1
        if completed % 20 == 0 or completed == len(samples):
            dt = time.perf_counter() - t0
            speed = completed / dt if dt > 0 else 0
            print(f"[PROGRESS] {completed}/{len(samples)} | {speed:.2f} q/s")

    if log_fh:
        log_fh.close()

    return results


# =============================================================================
# CLI
# =============================================================================

def make_parser(dataset_key: str, dataset_label: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=f"QA benchmark: {dataset_label}")
    ap.add_argument("--max-samples",   type=int, default=100)
    ap.add_argument("--workers",       type=int, default=4)
    ap.add_argument("--llm",           default="gemini:gemini-2.5-flash")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--classify",      action="store_true",
                    help="Also classify query strategy (for confusion matrix)")
    return ap


async def run_qa_benchmark(
    args,
    dataset_key: str,
    dataset_label: str,
    expected_strategy: str = None,
):
    add_project_root_to_syspath()
    LOGS_DIR.mkdir(exist_ok=True)

    if not args.skip_download:
        ensure_data()

    baseline = get_baseline(dataset_key)
    print(f"Baseline (IRCoT+Flan-T5-XL): EM={baseline.get('em', 0):.3f}  F1={baseline.get('f1', 0):.3f}")

    samples = load_ircot_samples(dataset_key)
    if not samples:
        print("[ERROR] No samples found.")
        return

    subset = samples[: args.max_samples]
    print(f"Running on {len(subset)} samples  (classify={args.classify})\n")

    print("=" * 60)
    print("[SETUP] API key")
    print("=" * 60)
    use_env = input("Use API key from .env file? (y/n): ").strip().lower()
    if use_env == "y":
        llm = await init_llm_with_key(llm=args.llm)
    else:
        api_key = input("API Key: ").strip()
        llm = await init_llm_with_key(api_key, llm=args.llm)

    # Fresh timestamped detail log every run — no checkpoint conflicts
    ts = time.strftime("%Y%m%d_%H%M%S")
    llm_slug = args.llm.split(":")[0]  # "gemini" or "qwen"
    classify_tag = "_classify" if args.classify else ""
    detail_log_path = LOGS_DIR / f"qa_{dataset_key}_{llm_slug}{classify_tag}_{ts}.jsonl"
    print(f"[LOG] {detail_log_path.name}\n")

    t0 = time.perf_counter()
    results = await run_qa(
        llm, subset,
        workers=args.workers,
        detail_log_path=detail_log_path,
        expected_strategy=expected_strategy,
        classify=args.classify,
    )
    runtime = time.perf_counter() - t0

    if not results:
        print("[ERROR] No results.")
        return

    avg_em = sum(r["em"] for r in results) / len(results)
    avg_f1 = sum(r["f1"] for r in results) / len(results)
    b_em   = baseline.get("em", 0)
    b_f1   = baseline.get("f1", 0)

    llm_label = args.llm
    print(f"\n{'='*60}")
    print(f"  DATASET : {dataset_label}")
    print(f"  MODEL   : {llm_label}")
    print(f"{'='*60}")
    print(f"  {'':35s}  {'EM':>6}  {'F1':>6}")
    print(f"  {llm_label:35s}  {avg_em:6.3f}  {avg_f1:6.3f}")
    if b_em or b_f1:
        print(f"  {'IRCoT + Flan-T5-XL':35s}  {b_em:6.3f}  {b_f1:6.3f}")
        print(f"  {'Delta':35s}  {avg_em-b_em:+6.3f}  {avg_f1-b_f1:+6.3f}")
    print(f"{'='*60}")
    print(f"  samples={len(results)}  runtime={runtime:.0f}s")
    print(f"  log → {detail_log_path.name}")

    run_info = {
        "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset":      dataset_label,
        "dataset_key":  dataset_key,
        "llm":          llm_label,
        "classify":     args.classify,
        "samples":      len(results),
        "em":           round(avg_em, 4),
        "f1":           round(avg_f1, 4),
        "baseline_em":  round(b_em, 4),
        "baseline_f1":  round(b_f1, 4),
        "delta_em":     round(avg_em - b_em, 4),
        "delta_f1":     round(avg_f1 - b_f1, 4),
        "runtime_sec":  round(runtime, 1),
        "detail_log":   str(detail_log_path),
    }
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(run_info, ensure_ascii=False) + "\n")
    print(f"  summary → {RESULTS_FILE.name}")
