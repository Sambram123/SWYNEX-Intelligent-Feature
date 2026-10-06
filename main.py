"""
main.py
-------
Entry point for the AI-Based Product Review Contradiction Detection prototype.

Usage
-----
Run from the project root:

    python main.py

Optional flags:
    --threshold   Cosine similarity threshold (default: 0.60)
    --top-n       Maximum pairs to display (default: 20)
    --save        Save results to examples/example_output.json
    --evaluate    Run Task 3 evaluation on the labelled test dataset

Examples:
    python main.py --threshold 0.55 --top-n 10 --save
    python main.py --evaluate
    python main.py --threshold 0.60 --top-n 5 --save --evaluate
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure stdout uses UTF-8 or safe character replacement on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.load_data import load_reviews
from src.preprocess import preprocess
from src.contradiction import detect_contradictions, format_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI-Based Product Review Contradiction Detection - Task 2 & 3 Prototype"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="Cosine similarity threshold (0-1). Default: 0.60",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Maximum number of contradictory pairs to display. Default: 20",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to examples/example_output.json",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run Task 3 evaluation on the manually labelled test dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\n" + "=" * 70)
    print("  AI-BASED PRODUCT REVIEW CONTRADICTION DETECTION")
    print("  Tasks 2 & 3 - Model Integration + Intelligent Feature")
    print("=" * 70 + "\n")

    # ── Task 3 Evaluation (optional, runs before main pipeline) ─────────────
    if args.evaluate:
        from src.evaluate import run_evaluation
        run_evaluation(threshold=args.threshold)
        print()  # blank line before main pipeline

    # ── Main Pipeline ────────────────────────────────────────────────────────

    # Step 1: Load
    df_raw = load_reviews()

    # Step 2: Preprocess
    df = preprocess(df_raw)

    # Step 3: Detect contradictions (Task 3: with confidence + explanation)
    pairs = detect_contradictions(
        df,
        similarity_threshold=args.threshold,
        top_n=args.top_n,
    )

    # Step 4: Optionally save results BEFORE printing
    #         (so a crash on display doesn't lose the file)
    if args.save:
        output_path = Path("examples") / "example_output.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)
        print(f"[main] Results saved to '{output_path}'.")

    # Step 5: Display results
    print("\n" + format_results(pairs))


if __name__ == "__main__":
    main()
