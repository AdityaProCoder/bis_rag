"""
Evaluation script for the baseline pipeline.
Computes Hit@3, MRR@5, and latency on public_test_set.json.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time

from old.utils import load_indexes, norm_full
from old.run_pipeline import run_pipeline


def compute_hit_at_k(retrieved, expected, k=3):
    """Check if expected standard is in top-k results."""
    for r in retrieved[:k]:
        if norm_full(r) == norm_full(expected):
            return 1
    return 0


def compute_mrr(retrieved, expected, k=5):
    """Compute Mean Reciprocal Rank for top-k results."""
    for rank, r in enumerate(retrieved[:k], 1):
        if norm_full(r) == norm_full(expected):
            return 1.0 / rank
    return 0.0


def evaluate():
    """Run evaluation on public_test_set.json."""
    print("=" * 60)
    print("BASELINE PIPELINE EVALUATION")
    print("=" * 60)
    
    load_indexes()
    
    test_set_path = Path(__file__).parent.parent / "guidelines" / "public_test_set.json"
    with open(test_set_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    
    print(f"\n[*] Loaded {len(queries)} queries from public_test_set.json")
    
    results = []
    total_latency = 0.0
    hit3_count = 0
    mrr_sum = 0.0
    failures = []
    
    for i, item in enumerate(queries):
        qid = item.get("id", f"Q_{i}")
        query = item.get("query", "").strip()
        expected_list = item.get("expected_standards", [])
        expected = expected_list[0] if expected_list else ""
        
        result = run_pipeline(query)
        
        retrieved = result["retrieved"]
        latency = result["latency_seconds"]
        
        hit3 = compute_hit_at_k(retrieved, expected, k=3)
        mrr = compute_mrr(retrieved, expected, k=5)
        
        hit3_count += hit3
        mrr_sum += mrr
        total_latency += latency
        
        results.append({
            "id": qid,
            "query": query,
            "expected": expected,
            "retrieved": retrieved,
            "hit@3": hit3,
            "mrr": mrr,
            "latency": latency
        })
        
        if not hit3:
            failures.append({
                "id": qid,
                "query": query,
                "expected": expected,
                "retrieved": retrieved
            })
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(queries)} queries...")
    
    n = len(results)
    hit3_pct = (hit3_count / n) * 100
    mrr_avg = mrr_sum / n
    avg_latency = total_latency / n
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Hit@3:    {hit3_pct:.1f}%")
    print(f"MRR@5:    {mrr_avg:.4f}")
    print(f"Latency:  {avg_latency:.3f}s per query")
    print(f"Failures: {len(failures)}/{n}")
    
    print("\n" + "=" * 60)
    print("FAILURES")
    print("=" * 60)
    for f in failures:
        print(f"  {f['id']}: Expected={f['expected']}, Got={f['retrieved'][:3]}")
    
    output = {
        "hit3_pct": hit3_pct,
        "mrr_avg": mrr_avg,
        "avg_latency": avg_latency,
        "total_failures": len(failures),
        "results": results,
        "failures": failures
    }
    
    output_path = Path(__file__).parent.parent / "data" / "old_pipeline_eval.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n[*] Results saved to {output_path}")
    
    return output


if __name__ == "__main__":
    evaluate()
