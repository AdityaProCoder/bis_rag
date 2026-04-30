import argparse
import json
import re
from collections import Counter


def norm(s):
    return re.sub(r"\s+", "", str(s).lower())


def loose_key(s):
    t = str(s).lower()
    t = re.sub(r":\s*\d{4}", "", t)
    t = re.sub(r"\(\s*part[^)]*\)", "", t)
    t = re.sub(r"[^a-z0-9]", "", t)
    return t


def evaluate(rows):
    hit3 = 0
    mrr = 0.0
    misses = []
    buckets = Counter()

    for row in rows:
        exp = [norm(e) for e in row.get("expected_standards", [])]
        exp_set = set(exp)
        ret = [norm(r) for r in row.get("retrieved_standards", [])[:5]]

        if any(r in exp_set for r in ret[:3]):
            hit3 += 1

        rr = 0.0
        for i, r in enumerate(ret, 1):
            if r in exp_set:
                rr = 1.0 / i
                break
        mrr += rr

        if rr == 0.0:
            misses.append(row)
            exp_raw = row.get("expected_standards", [""])[0]
            top = row.get("retrieved_standards", [])[:5]
            if any(loose_key(r) == loose_key(exp_raw) for r in top):
                buckets["format_or_part_variant"] += 1
            elif re.search(r"part", exp_raw, re.IGNORECASE) and all(not re.search(r"part", r, re.IGNORECASE) for r in top):
                buckets["missing_part_family"] += 1
            elif re.search(r"\bis\s*\d+", row.get("query", ""), re.IGNORECASE):
                buckets["near_id_or_number_confusion"] += 1
            else:
                buckets["semantic_miss"] += 1

    n = max(1, len(rows))
    return {
        "queries": len(rows),
        "hit_at_3": round((hit3 / n) * 100.0, 2),
        "mrr_at_5": round(mrr / n, 4),
        "zero_rr_count": len(misses),
        "miss_buckets": dict(buckets),
        "sample_misses": [
            {
                "id": x.get("id"),
                "query": x.get("query", "")[:180],
                "expected": x.get("expected_standards", []),
                "retrieved_top5": x.get("retrieved_standards", [])[:5],
            }
            for x in misses[:15]
        ],
    }


def main():
    p = argparse.ArgumentParser(description="Failure bucketing for BIS inference results")
    p.add_argument("--results", required=True)
    p.add_argument("--output", default="")
    args = p.parse_args()

    rows = json.load(open(args.results, "r", encoding="utf-8"))
    report = evaluate(rows)
    print(json.dumps(report, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
