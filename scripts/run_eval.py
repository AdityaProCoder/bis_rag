"""
Local test harness — NOT part of submission.

Runs inference.py (strict clean output) then injects expected_standards
from public_test_set.json so eval_script.py can score it.

Usage:
    python run_eval.py
    python run_eval.py --input guidelines/public_test_set.json --output data/submission_results.json
"""
import argparse
import json
import subprocess
import sys
import os
from pathlib import Path

DATA_DIR = Path("data")


def load_public_ground_truth(path="guidelines/public_test_set.json"):
    with open(path, "r", encoding="utf-8") as f:
        return {item["id"]: item["expected_standards"] for item in json.load(f)}


def main():
    parser = argparse.ArgumentParser(description="Local eval harness")
    parser.add_argument("--input", default="guidelines/public_test_set.json")
    parser.add_argument("--output", default="data/submission_results.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print("[*] Running inference (strict clean output)...")
    result = subprocess.run(
        [sys.executable, "inference.py",
         "--input", args.input,
         "--output", args.output,
         "--device", args.device],
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"[ERROR] inference.py exited with code {result.returncode}")
        sys.exit(result.returncode)

    # Read clean output
    with open(args.output, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Inject expected_standards from ground truth (local testing only)
    ground_truth = load_public_ground_truth(args.input)
    for r in results:
        r["expected_standards"] = ground_truth.get(r["id"], [])

    scored_output = str(args.output).replace(".json", "_scored.json")
    with open(scored_output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[*] Scoring with eval_script.py...")
    from guidelines.eval_script import evaluate_results
    evaluate_results(scored_output)

    # Clean up scored temp file (keep clean submission output)
    os.remove(scored_output)
    print(f"[*] Clean output saved to: {args.output}")


if __name__ == "__main__":
    main()
