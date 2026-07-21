"""
benchmark.py — Standalone RAG chatbot benchmark.

Usage:
    python benchmark.py --input questions.csv [--output results.csv]

The input CSV must contain a column named 'question'.
Results are written to the output CSV with columns:
    question | answer | response_time_seconds
"""

import argparse
import csv
import os
import sys
import time

from rag import ask_question, get_cached_llm, contextualize_query
from vector_store import get_docs_by_distance
from reranker import rerank
from config import RERANKER_ENABLED, RERANKER_TOP_N, RERANKER_THRESHOLD

# ---------------------------------------------------------------------------
# Forensic debug logging for failed evaluations only.
#
# This does not alter the RAG pipeline, retrieval, prompting, or scoring.
# For questions that fail, it independently replays just the contextualize +
# retrieval stages using the same production functions rag.py itself calls,
# purely to capture intermediate state that ask_question() doesn't expose.
# The final answer is NOT regenerated — the answer already produced by the
# benchmark run above is reused as-is.
# ---------------------------------------------------------------------------
DEBUG_DIR = "evaluation_debug"

# Manual toggle: when True, _is_suspicious() is also consulted to flag
# answers for debug logging even when they aren't outright errors/refusals.
# Placeholder heuristic — customize the check inside _is_suspicious().
FLAG_SUSPICIOUS_ANSWERS = False


def _is_suspicious(answer: str) -> bool:
    return len(answer.strip()) < 20


def _is_failure(answer) -> bool:
    text = str(answer)
    if text.startswith("ERROR:"):
        return True
    if "That's not my specialization." in text:
        return True
    if not text.strip():
        return True
    if FLAG_SUSPICIOUS_ANSWERS and _is_suspicious(text):
        return True
    return False


def _replay_pipeline_for_debug(question: str) -> dict:
    """
    Independently replays the contextualize + retrieval stages (same
    functions rag.py uses) to capture intermediate state for a debug file.
    Does not call the final-answer LLM.
    """
    llm = get_cached_llm()

    t0 = time.perf_counter()
    route, standalone_query = contextualize_query(question, [], llm)
    t_contextualize = time.perf_counter() - t0

    info = {
        "route": route,
        "standalone_query": standalone_query,
        "t_contextualize": t_contextualize,
        "t_retrieval": None,
        "retrieval": [],   # [(rank, distance, doc), ...]
        "final_docs": [],  # [(rank, distance_or_None, doc), ...]
        "context": "",
    }

    if route == "social":
        return info

    t0 = time.perf_counter()
    vector_results = get_docs_by_distance(standalone_query)
    info["t_retrieval"] = time.perf_counter() - t0
    info["retrieval"] = [(i, dist, doc) for i, (doc, dist) in enumerate(vector_results, 1)]

    dist_by_id = {id(doc): dist for doc, dist in vector_results}
    candidates = [doc for doc, _ in vector_results]

    if RERANKER_ENABLED:
        scored = rerank(standalone_query, candidates)
        above_threshold = [(score, doc) for score, doc in scored if score >= RERANKER_THRESHOLD]
        final = [doc for _, doc in above_threshold[:RERANKER_TOP_N]]
    else:
        final = candidates[:RERANKER_TOP_N]

    info["final_docs"] = [(i, dist_by_id.get(id(doc)), doc) for i, doc in enumerate(final, 1)]
    info["context"] = "\n\n---\n\n".join(doc.page_content for doc in final) if final else ""

    return info


def _write_debug_file(idx: int, question: str, answer, elapsed: float) -> None:
    """Write a forensic debug .txt for one failed question. Never raises —
    a logging failure must not interrupt the benchmark run."""
    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
    except Exception as exc:
        print(f"  ⚠  Could not create '{DEBUG_DIR}/': {exc}")
        return

    replay, replay_error = None, None
    try:
        replay = _replay_pipeline_for_debug(question)
    except Exception as exc:
        replay_error = str(exc)

    sep = "-" * 52
    lines = ["QUESTION", "", f"Original Question: {question}"]
    if replay:
        lines.append(f"Standalone Question: {replay['standalone_query']}")
        lines.append(f"Route: {replay['route']}")
    else:
        lines.append("Standalone Question: N/A (debug replay failed)")
        lines.append("Route: N/A (debug replay failed)")
        lines.append(f"Replay error: {replay_error}")

    lines += ["", sep, "", "RETRIEVAL", ""]
    if replay and replay["retrieval"]:
        lines.append("Rank | Distance | Object Name | h1 | h2 | Chunk Length")
        for rank, dist, doc in replay["retrieval"]:
            m = doc.metadata
            lines.append(f"{rank} | {dist:.4f} | {m.get('object_name')} | {m.get('h1')} | "
                         f"{m.get('h2')} | {len(doc.page_content)}")
    elif replay and replay["route"] == "social":
        lines.append("N/A — social route, retrieval skipped.")
    else:
        lines.append("N/A — debug replay failed, see error above.")

    lines += ["", sep, "", "FINAL DOCS", ""]
    if replay and replay["final_docs"]:
        lines.append("Rank | Distance | Object Name | h2")
        for rank, dist, doc in replay["final_docs"]:
            m = doc.metadata
            dist_str = f"{dist:.4f}" if dist is not None else "N/A"
            lines.append(f"{rank} | {dist_str} | {m.get('object_name')} | {m.get('h2')}")
    elif replay and replay["route"] == "social":
        lines.append("N/A — social route, no documents sent.")
    else:
        lines.append("N/A — debug replay failed, see error above.")

    lines += ["", sep, "", "FULL CONTEXT", ""]
    lines.append(replay["context"] if (replay and replay["context"]) else "(empty)")

    lines += ["", sep, "", "FINAL ANSWER", "", str(answer)]

    lines += ["", sep, "", "TIMING", ""]
    lines.append(f"Total time (original benchmark run) : {elapsed:.4f}s")
    if replay and replay["t_retrieval"] is not None:
        lines.append(f"Retrieval time (debug replay)       : {replay['t_retrieval']:.4f}s")
        lines.append(f"Contextualize time (debug replay)   : {replay['t_contextualize']:.4f}s")
    elif replay:
        lines.append("Retrieval time (debug replay)       : N/A (social route)")
        lines.append(f"Contextualize time (debug replay)   : {replay['t_contextualize']:.4f}s")
    else:
        lines.append("Retrieval time (debug replay)       : N/A (replay failed)")
        lines.append("Contextualize time (debug replay)   : N/A (replay failed)")
    lines += [
        "",
        "Note: retrieval/contextualize times above come from a separate,",
        "post-hoc replay of the pipeline for debugging — not the original",
        "run — since ask_question() does not expose internal timing. The",
        "final answer above IS the original answer from this benchmark run",
        "(not regenerated).",
    ]

    try:
        path = os.path.join(DEBUG_DIR, f"Q{idx}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception as exc:
        print(f"  ⚠  Could not write debug file for Q{idx}: {exc}")


def run_benchmark(input_path: str, output_path: str) -> None:
    # ------------------------------------------------------------------ #
    # 1. Load questions
    # ------------------------------------------------------------------ #
    # Try common encodings in order: UTF-8 with BOM, plain UTF-8, Windows-1252
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            with open(input_path, newline="", encoding=encoding) as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                if "question" not in fieldnames:
                    print(f"ERROR: '{input_path}' has no 'question' column.")
                    print(f"  Columns found: {fieldnames}")
                    sys.exit(1)
                questions = [row["question"].strip() for row in reader if row["question"].strip()]
            print(f"Detected encoding: {encoding}")
            break
        except FileNotFoundError:
            print(f"ERROR: Input file not found — '{input_path}'")
            sys.exit(1)
        except UnicodeDecodeError:
            continue
    else:
        print("ERROR: Could not decode the CSV file. Try saving it as UTF-8.")
        sys.exit(1)

    total = len(questions)
    if total == 0:
        print("No questions found in the input file. Exiting.")
        sys.exit(0)

    print(f"\nLoaded {total} question(s) from '{input_path}'.")
    print(f"Results will be saved to '{output_path}'.\n")

    # ------------------------------------------------------------------ #
    # 2. Run each question through the RAG pipeline
    # ------------------------------------------------------------------ #
    rows = []

    for idx, question in enumerate(questions, start=1):
        print(f"Question {idx}/{total}: {question[:80]}{'...' if len(question) > 80 else ''}")

        t_start = time.perf_counter()
        try:
            answer = ask_question(question)
        except Exception as exc:
            answer = f"ERROR: {exc}"
            print(f"  ⚠  Failed — {exc}")
        elapsed = time.perf_counter() - t_start

        print(f"  ✓  {elapsed:.2f}s\n")

        rows.append({
            "question": question,
            "answer": answer,
            "response_time_seconds": round(elapsed, 4),
        })

        if _is_failure(answer):
            print(f"  📝 Writing debug file for Q{idx} (failed evaluation)...")
            _write_debug_file(idx, question, answer, elapsed)

    # ------------------------------------------------------------------ #
    # 3. Write results CSV
    # ------------------------------------------------------------------ #
    fieldnames = ["question", "answer", "response_time_seconds"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ------------------------------------------------------------------ #
    # 4. Summary
    # ------------------------------------------------------------------ #
    times = [r["response_time_seconds"] for r in rows]
    errors = sum(1 for r in rows if str(r["answer"]).startswith("ERROR:"))

    print("=" * 50)
    print("  BENCHMARK COMPLETE")
    print("=" * 50)
    print(f"  Questions   : {total}")
    print(f"  Errors      : {errors}")
    print(f"  Avg time    : {sum(times) / len(times):.2f}s")
    print(f"  Min time    : {min(times):.2f}s")
    print(f"  Max time    : {max(times):.2f}s")
    print(f"  Results     : {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the RAG chatbot.")
    parser.add_argument(
        "--input",
        default="question_expected_answer.csv",
        help="Path to the input CSV file (must have a 'question' column).",
    )
    parser.add_argument(
        "--output",
        default="results.csv",
        help="Path for the output CSV file (default: results.csv).",
    )
    args = parser.parse_args()
    run_benchmark(args.input, args.output)