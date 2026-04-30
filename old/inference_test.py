"""
Hackathon Inference Script (test version)
Run with: python old/inference_test.py --input guidelines/public_test_set.json --output old/public_results.json
Then evaluate: python old/eval_script.py --results old/public_results.json
"""
import json
import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))
import precision_eval_v2 as pe


def process_item(item):
    qid = item.get("id", "UNKNOWN")
    query = item.get("query", "").strip()
    expected = item.get("expected_standards", [])
    
    result = pe.run_pipeline(query)
    retrieved = result["retrieved"]
    latency = result["latency_seconds"]
    
    return {
        "id": qid,
        "query": query,
        "expected_standards": expected,
        "retrieved_standards": retrieved,
        "latency_seconds": latency
    }


def main():
    parser = argparse.ArgumentParser(description="BIS Hackathon Inference")
    parser.add_argument("--input", required=True, help="Input JSON with queries")
    parser.add_argument("--output", required=True, help="Output JSON for eval_script.py")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    
    pe.load_indexes()
    
    with open(args.input, "r", encoding="utf-8") as f:
        queries = json.load(f)
    
    print(f"[*] Loaded {len(queries)} queries, {args.workers} workers")
    
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_item, q): q for q in queries}
        done = 0
        for f in as_completed(futures):
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(queries)}")
            results.append(f.result())
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"[*] Saved to {args.output}")


if __name__ == "__main__":
    main()
