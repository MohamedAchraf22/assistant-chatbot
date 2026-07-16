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
import sys
import time

from rag import ask_question


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
        default="Dataset(Sheet1).csv",
        help="Path to the input CSV file (must have a 'question' column).",
    )
    parser.add_argument(
        "--output",
        default="results.csv",
        help="Path for the output CSV file (default: results.csv).",
    )
    args = parser.parse_args()
    run_benchmark(args.input, args.output)