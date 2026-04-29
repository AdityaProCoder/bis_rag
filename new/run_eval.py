"""
BIS Standards Recommendation Engine - Evaluation Script
Evaluates pipeline against public_test_set.json

Metrics:
- Hit Rate @3 (Target: >80%)
- MRR @5 (Target: >0.7)
- Avg Latency (Target: <5 seconds)
"""

import json
import argparse
import sys
from pathlib import Path


def normalize_std(std_string: str) -> str:
    """Normalizes the standard name by removing spaces and converting to lowercase for fair matching."""
    return str(std_string).replace(" ", "").lower()


def evaluate_results(results_file: str):
    try:
        with open(results_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading results file: {e}")
        sys.exit(1)

    total_queries = len(data)
    if total_queries == 0:
        print("No queries found in the result file.")
        return

    hits_at_3 = 0
    mrr_sum_at_5 = 0.0
    total_latency = 0.0

    detailed_results = []

    for item in data:
        qid = item.get("id", "UNKNOWN")
        query = item.get("query", "")
        expected = set(normalize_std(std) for std in item.get("expected_standards", []))
        retrieved = [normalize_std(std) for std in item.get("retrieved_standards", [])]
        latency = item.get("latency_seconds", 0.0)

        total_latency += latency

        # Hit Rate @3
        top_3_retrieved = retrieved[:3]
        hit = any(std in expected for std in top_3_retrieved)
        if hit:
            hits_at_3 += 1

        # MRR @5
        top_5_retrieved = retrieved[:5]
        mrr = 0.0
        for rank, std in enumerate(top_5_retrieved, start=1):
            if std in expected:
                mrr = 1.0 / rank
                break
        mrr_sum_at_5 += mrr

        # Detailed result
        detailed_results.append({
            "id": qid,
            "query": query[:80] + "..." if len(query) > 80 else query,
            "expected": list(expected),
            "retrieved": retrieved[:3],
            "hit": hit,
            "mrr": mrr,
            "latency": latency
        })

    # Calculate Final Metrics
    hit_rate_3 = (hits_at_3 / total_queries) * 100
    mrr_5 = mrr_sum_at_5 / total_queries
    avg_latency = total_latency / total_queries

    print("=" * 50)
    print("   BIS HACKATHON EVALUATION RESULTS (HARDENED)")
    print("=" * 50)
    print(f"Total Queries Evaluated : {total_queries}")
    print(f"Hit Rate @3             : {hit_rate_3:.2f}% \t(Target: >80%)")
    print(f"MRR @5                  : {mrr_5:.4f} \t(Target: >0.7)")
    print(f"Avg Latency             : {avg_latency:.2f} sec \t(Target: <5 seconds)")
    print("=" * 50)
    
    # Detailed breakdown
    print("\nDetailed Results:")
    print("-" * 50)
    for r in detailed_results:
        status = "HIT" if r["hit"] else "MISS"
        print(f"{r['id']}: {status} | MRR: {r['mrr']:.2f} | Latency: {r['latency']:.2f}s")
        print(f"  Expected: {r['expected']}")
        print(f"  Retrieved: {r['retrieved']}")
        print()

    return {
        "hit_rate_3": hit_rate_3,
        "mrr_5": mrr_5,
        "avg_latency": avg_latency,
        "total_queries": total_queries,
        "hits_at_3": hits_at_3
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate RAG Pipeline Results for BIS Hackathon"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to the participant's output JSON file",
    )
    args = parser.parse_args()

    evaluate_results(args.results)
