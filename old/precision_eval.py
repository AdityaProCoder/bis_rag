"""
Precision Engine: Hybrid Scoring + LLM Arbitration

Step 1: Composite scoring (keyword, bigram, type, part, near-id penalty)
Step 2: LLM as final arbiter
Step 3: Hybrid merge
Step 4: Adaptive expansion
"""
import json
import time
import re
import pickle
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import urllib.request
from collections import defaultdict

# =============================================================================
# CONFIGURATION
# =============================================================================
LM_BASE_URL = "http://127.0.0.1:1234"
LM_API_KEY = "lmstudio"
DEFAULT_MODEL = "google/gemma-4-e2b"
DATA_DIR = Path(__file__).parent.parent / "data"
EMBED_DEVICE = "cpu"

TOP_DENSE = 20
TOP_BM25 = 20
FUSION_K = 10
RRF_K = 10
PARAPHRASE_TOP_DENSE = 50
RERANK_K = 5
OUTPUT_K = 3
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 32

# Material types for type matching
MATERIAL_TYPES = [
    'portland', 'slag', 'pozzolana', 'masonry', 'white', 'super sulphated',
    'ordinary', 'rapid', 'low heat', 'hydrophobic', 'sulphate resisting',
    'puzzolana', 'fly ash', 'aggregate', 'cement', 'concrete', 'brick',
    'lime', 'gypsum', 'plaster', 'mortar', 'timber', 'wood', 'steel',
    'aluminium', 'copper', 'zinc', 'glass', 'plastic', 'pvc', 'cpvc',
    'bitumen', 'tar', 'asphalt', 'felt', 'sheet', 'pipe', 'fiting', 'valve'
]

# =============================================================================
# UTILITIES
# =============================================================================
_index_store = {}

def standard_key(std):
    explicit = std.get("_key")
    if explicit:
        return explicit
    base = re.sub(r"\s+", " ", str(std.get("id", "")).strip().upper())
    part = std.get("part")
    if part:
        part_clean = re.sub(r"\s+", " ", str(part).strip().upper())
        base = f"{base} (Part {part_clean})"
    return base

def norm_id(s):
    base = str(s).split(":")[0].strip()
    return re.sub(r"\s+", " ", base).upper()

def norm_full(s):
    base = str(s).split(":")[0].strip()
    part_m = re.search(r"\(PART[^)]+\)", str(s))
    part = part_m.group(0) if part_m else ""
    return re.sub(r"\s+", " ", f"{base} {part}").strip().upper()

def _loose_standard_key(value: str) -> str:
    text = str(value).upper().replace("PART PART", "PART")
    text = re.sub(r":\s*\d{4}", "", text)
    text = re.sub(r"\s*\(?\s*PART[^)]*\)?", "", text)
    return re.sub(r"[^A-Z0-9]+", "", text)

def apply_year_mapping(retrieved_list, expected_list):
    if not expected_list:
        return retrieved_list
    exp_base = _loose_standard_key(expected_list[0])
    exp_full = expected_list[0]
    return [exp_full if _loose_standard_key(s) == exp_base else s for s in retrieved_list]

def load_indexes():
    global _index_store
    if _index_store:
        return _index_store
    
    faiss_idx = faiss.read_index(str(DATA_DIR / "faiss_index.bin"))
    with open(DATA_DIR / "bm25_index.pkl", "rb") as f:
        bm25_data = pickle.load(f)
    with open(DATA_DIR / "whitelist.txt", "r", encoding="utf-8") as f:
        whitelist = {l.strip(): True for l in f if l.strip()}
    
    cfg = json.load(open(DATA_DIR / "embedding_config.json"))
    model = SentenceTransformer(cfg["model_name"], device=EMBED_DEVICE)
    
    _index_store = {
        "faiss": faiss_idx,
        "bm25": bm25_data["bm25"],
        "standards": bm25_data["standards"],
        "whitelist": whitelist,
        "model": model,
    }
    return _index_store

def g(key):
    return _index_store[key]

# =============================================================================
# RETRIEVAL
# =============================================================================
def retrieve_dense(query, k=20):
    model = g("model")
    idx = g("faiss")
    standards = g("standards")
    qe = np.array(model.encode([query], normalize_embeddings=True), dtype=np.float32)
    D, I = idx.search(qe, k)
    return [(standard_key(standards[i]), float(D[0][j])) for j, i in enumerate(I[0]) if 0 <= i < len(standards)]

def retrieve_bm25(query, k=20):
    bm = g("bm25")
    standards = g("standards")
    scores = bm.get_scores(query.lower().split())
    top_idx = np.argsort(scores)[::-1][:k]
    return [(standard_key(standards[i]), float(scores[i])) for i in top_idx if i < len(standards) and scores[i] > 0]

# =============================================================================
# FUSION
# =============================================================================
def fuse_results(dense_results, bm25_results):
    score_map = {}
    for rank, (doc_id, _) in enumerate(dense_results, 1):
        score_map[doc_id] = score_map.get(doc_id, 0) + 1.0 / (RRF_K + rank)
    for rank, (doc_id, _) in enumerate(bm25_results, 1):
        score_map[doc_id] = score_map.get(doc_id, 0) + 1.0 / (RRF_K + rank)
    return sorted(score_map.items(), key=lambda x: x[1], reverse=True)

# =============================================================================
# PARAPHRASE
# =============================================================================
def generate_paraphrase(query):
    payload = {
        "model": DEFAULT_MODEL,
        "max_tokens": 48,
        "temperature": 0.4,
        "system": "You are a technical assistant for BIS standards lookup.",
        "messages": [{"role": "user", "content": f"Rewrite the query using precise BIS technical terminology. Return ONE short sentence only.\n\nQuery: {query}\n\nRewritten:"}]
    }
    req = urllib.request.Request(
        f"{LM_BASE_URL}/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-api-key": LM_API_KEY},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            response = json.loads(resp.read().decode("utf-8"))
            text = response.get("content", [{}])[0].get("text", "").strip()
            return text if text else ""
    except Exception:
        return ""

def paraphrase_retrieval(query, k=PARAPHRASE_TOP_DENSE):
    para_query = generate_paraphrase(query)
    if not para_query:
        return []
    return retrieve_dense(para_query, k=k)

# =============================================================================
# STEP 1: COMPOSITE SCORING
# =============================================================================
def get_ngrams(text, n=2):
    """Extract n-grams from text."""
    words = text.lower().split()
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)] if len(words) >= n else []

def extract_keywords(text):
    """Extract significant keywords from text (length > 3, not stopwords)."""
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'for', 'and', 'or', 
                 'with', 'from', 'our', 'we', 'i', 'need', 'want', 'looking', 'what',
                 'which', 'that', 'this', 'their', 'have', 'has', 'had', 'using'}
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    return set(w for w in words if w not in stopwords)

def check_part_alignment(query, candidate):
    """
    Check if query and candidate have aligned Part expectations.
    Returns 1.0 if aligned, 0.5 if partial, 0.0 if mismatched.
    """
    # Extract Part mentions from query
    query_part = re.search(r'\(?(part)\s*(\d+)\)?', query, re.IGNORECASE)
    if not query_part:
        return 0.0  # No specific part in query
    
    query_part_num = query_part.group(2)
    cand_part = candidate.get('part', '')
    
    if not cand_part:
        return 0.3  # Query specifies part but candidate doesn't have one
    
    return 1.0 if str(cand_part).strip() == query_part_num else 0.0

def near_id_penalty(query, candidate):
    """
    Penalize candidates with similar base IDs to expected but wrong.
    e.g., IS 736 vs IS 739 (both aluminium, but different types)
    """
    # Extract numbers from query that might indicate specific standard
    query_nums = re.findall(r'\bIS\s*(\d+)\b', query, re.IGNORECASE)
    if not query_nums:
        return 0.0
    
    cand_id = str(candidate.get('id', ''))
    cand_num = re.sub(r'[^0-9]', '', cand_id)
    
    if not cand_num:
        return 0.0
    
    # Check for near-matches (same prefix, different suffix)
    for qn in query_nums:
        if len(qn) >= 3:
            # Same first 3 digits but different last digit(s)
            if qn[:3] == cand_num[:3] and qn != cand_num:
                return 1.0  # Penalize
    
    return 0.0

def composite_score(query, candidate, standards_db):
    """
    Compute composite relevance score for a candidate.
    Higher = more likely to be the correct standard.
    """
    cid = standard_key(candidate) if isinstance(candidate, dict) else candidate
    if cid in standards_db:
        cand = standards_db[cid]
    else:
        cand = candidate if isinstance(candidate, dict) else {}
    
    score = 0.0
    query_lower = query.lower()
    title = cand.get('title', '').lower()
    content = cand.get('content', '')[:500].lower()
    full_text = f"{title} {content}"
    
    # 1. Keyword Overlap (weight = 3)
    query_kw = extract_keywords(query)
    content_kw = extract_keywords(full_text)
    keyword_matches = len(query_kw & content_kw)
    score += keyword_matches * 3.0
    
    # 2. Bigram Overlap (weight = 5)
    query_bigrams = set(get_ngrams(query_lower, 2))
    content_bigrams = set(get_ngrams(full_text, 2))
    bigram_matches = len(query_bigrams & content_bigrams)
    score += bigram_matches * 5.0
    
    # 3. Title Exact Match Bonus (+10)
    # Check if significant query terms appear in title
    title_terms = set(re.findall(r'\b[a-z]{4,}\b', title))
    title_match_count = len(query_kw & title_terms)
    score += title_match_count * 10.0
    
    # 4. Material Type Match Bonus (+8)
    for mat in MATERIAL_TYPES:
        if mat in query_lower and mat in title:
            score += 8.0
            break
    
    # 5. Part Alignment Bonus (+6)
    part_score = check_part_alignment(query, cand)
    score += part_score * 6.0
    
    # 6. Near-ID Penalty (-4)
    near_penalty = near_id_penalty(query, cand)
    score -= near_penalty * 4.0
    
    # 7. Content Presence Bonus (+2)
    # If query has "standard/specification" and content mentions these
    if any(term in query_lower for term in ['standard', 'specification', 'requirements']):
        if any(term in full_text for term in ['standard', 'specification']):
            score += 2.0
    
    return score

def rank_candidates_composite(query, candidates, standards_db):
    """
    Rank candidates by composite score.
    Returns list of (cid, score) sorted by descending score.
    """
    scored = []
    for cid in candidates:
        if cid in standards_db:
            score = composite_score(query, standards_db[cid], standards_db)
        else:
            score = 0.0
        scored.append((cid, score))
    return sorted(scored, key=lambda x: x[1], reverse=True)

# =============================================================================
# STEP 2: LLM ARBITRATION
# =============================================================================
def parse_numbers(response, candidate_ids):
    if not response:
        return None
    response_upper = response.upper()
    found = []
    for cid in candidate_ids:
        if cid.upper() in response_upper:
            found.append(cid)
    for cid in candidate_ids:
        if len(found) >= RERANK_K:
            break
        if cid not in found:
            found.append(cid)
    return found[:RERANK_K]

def llm_rerank(query, candidates, standards_db):
    """
    LLM reranking with enhanced prompt for discrimination.
    """
    id_to_std = {standard_key(s): s for s in standards_db.values()}
    
    candidate_display = []
    for cid in candidates:
        if cid not in id_to_std:
            continue
        s = id_to_std[cid]
        title = s.get("title", "")[:80]
        sid = s.get("id", "").strip()
        part = s.get("part", "")
        display_id = f"IS {sid}" + (f" (Part {part})" if part else "")
        candidate_display.append((cid, title, display_id))
    
    if len(candidate_display) < 2:
        return candidates[:OUTPUT_K]
    
    # Extract keywords for prompt
    keywords = extract_keywords(query)
    keyword_str = ', '.join(sorted(keywords)[:8])
    
    candidates_list = "\n".join([f"{i+1}. [{did}] {title}" for i, (cid, title, did) in enumerate(candidate_display)])
    
    prompt = f"""Query: {query}

Keywords: {keyword_str}

Pre-ranked candidates:
{candidates_list}

TASK:
- Choose the EXACT specification matching the query's material type and application
- If standards have similar names, select the one with best keyword/type alignment
- For products with parts (Part 1, Part 2, etc.), choose the part matching the query
- Output ONLY numbers, most relevant first: "3, 1, 5"
"""
    
    payload = {
        "model": DEFAULT_MODEL,
        "prompt": prompt,
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": LLM_TEMPERATURE,
        "stream": False
    }
    req = urllib.request.Request(
        f"{LM_BASE_URL}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-api-key": LM_API_KEY},
        method="POST",
    )
    
    llm_output = ""
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            response = json.loads(resp.read().decode("utf-8"))
            llm_output = response.get("choices", [{}])[0].get("text", "").strip()
    except Exception:
        pass
    
    candidate_ids = [cid for cid, _, _ in candidate_display]
    reranked = parse_numbers(llm_output, candidate_ids)
    
    if reranked is None:
        reranked = candidates[:RERANK_K]
    
    return reranked

# =============================================================================
# STEP 3: HYBRID MERGE
# =============================================================================
def hybrid_merge(composite_ranked, llm_ranked, top_k=3):
    """
    Merge composite and LLM rankings.
    Prefer composite when there's strong keyword/type alignment.
    """
    result = []
    seen = set()
    
    # First: Add items where both agree (high confidence)
    for i, cid in enumerate(llm_ranked[:top_k]):
        if cid not in seen:
            # Check if composite also ranks this high
            comp_pos = next((j for j, (c, _) in enumerate(composite_ranked) if c == cid), 999)
            llm_pos = i
            
            # If LLM agrees with composite OR composite strongly agrees (within 1 pos)
            if comp_pos <= llm_pos + 1:
                result.append(cid)
                seen.add(cid)
    
    # Fill remaining from composite (prioritize keyword match)
    for cid, score in composite_ranked:
        if cid not in seen and len(result) < top_k:
            result.append(cid)
            seen.add(cid)
    
    return result

# =============================================================================
# PIPELINE
# =============================================================================
def run_pipeline(query):
    start_time = time.perf_counter()
    
    load_indexes()
    standards = g("standards")
    standards_db = {standard_key(s): s for s in standards}
    whitelist = g("whitelist")
    
    # STEP 1: Dual retrieval
    dense = retrieve_dense(query, k=TOP_DENSE)
    bm25 = retrieve_bm25(query, k=TOP_BM25)
    fused = fuse_results(dense, bm25)
    
    # STEP 2: Paraphrase expansion
    p_results = paraphrase_retrieval(query)
    if p_results:
        merged_scores = {}
        for rank, (cid, score) in enumerate(fused[:FUSION_K], 1):
            merged_scores[cid] = merged_scores.get(cid, 0) + 1.0 / (RRF_K + rank)
        for rank, (cid, score) in enumerate(p_results, 1):
            merged_scores[cid] = merged_scores.get(cid, 0) + 1.0 / (RRF_K + rank)
        fused = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)
    
    candidates = [cid for cid, _ in fused[:FUSION_K] if cid in whitelist]
    
    # STEP 3: Composite scoring
    composite_ranked = rank_candidates_composite(query, candidates, standards_db)
    composite_top5 = [cid for cid, _ in composite_ranked[:5]]
    
    # STEP 4: LLM arbitration on top 5
    llm_ranked = llm_rerank(query, composite_top5, standards_db)
    
    # STEP 5: Hybrid merge
    final_ranked = hybrid_merge(composite_ranked, llm_ranked, OUTPUT_K)
    
    # Validate and return
    validated = [c for c in final_ranked if c in whitelist]
    for c, _ in fused:
        if len(validated) >= OUTPUT_K:
            break
        if c not in validated and c in whitelist:
            validated.append(c)
    
    final_results = validated[:OUTPUT_K]
    latency = time.perf_counter() - start_time
    
    return {
        "retrieved": final_results,
        "latency_seconds": round(latency, 3),
    }

# =============================================================================
# EVALUATION
# =============================================================================
def compute_hit_at_k(retrieved, expected, k=3):
    mapped_retrieved = apply_year_mapping(retrieved, [expected])
    for r in mapped_retrieved[:k]:
        if norm_full(r) == norm_full(expected):
            return 1
    return 0

def compute_mrr(retrieved, expected, k=5):
    mapped_retrieved = apply_year_mapping(retrieved, [expected])
    for rank, r in enumerate(mapped_retrieved[:k], 1):
        if norm_full(r) == norm_full(expected):
            return 1.0 / rank
    return 0.0

def evaluate(test_set_path):
    print("=" * 70)
    print(f"PRECISION ENGINE EVALUATION: {Path(test_set_path).name}")
    print("=" * 70)
    
    load_indexes()
    
    with open(test_set_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    
    print(f"\n[*] Loaded {len(queries)} queries")
    
    results = []
    total_latency = 0.0
    hit3_count = 0
    mrr_sum = 0.0
    failures = []
    by_section = defaultdict(lambda: {"hits": 0, "total": 0, "mrr": 0.0})
    
    for i, item in enumerate(queries):
        qid = item.get("id", f"Q_{i}")
        query = item.get("query", "").strip()
        expected_list = item.get("expected_standards", [])
        expected = expected_list[0] if expected_list else ""
        section = item.get("section", 0)
        section_name = item.get("section_name", "unknown")
        
        result = run_pipeline(query)
        retrieved = result["retrieved"]
        latency = result["latency_seconds"]
        
        hit3 = compute_hit_at_k(retrieved, expected, k=3)
        mrr = compute_mrr(retrieved, expected, k=5)
        
        hit3_count += hit3
        mrr_sum += mrr
        total_latency += latency
        
        by_section[section]["total"] += 1
        by_section[section]["hits"] += hit3
        by_section[section]["mrr"] += mrr
        by_section[section]["section_name"] = section_name
        
        results.append({
            "id": qid, "section": section, "query": query,
            "expected": expected, "retrieved": retrieved,
            "hit@3": hit3, "mrr": mrr, "latency": latency
        })
        
        if not hit3:
            failures.append({
                "id": qid, "section": section, "section_name": section_name,
                "query": query, "expected": expected, "retrieved": retrieved
            })
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(queries)} queries...")
    
    n = len(results)
    hit3_pct = (hit3_count / n) * 100
    mrr_avg = mrr_sum / n
    avg_latency = total_latency / n
    
    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print(f"Hit@3:    {hit3_pct:.1f}%")
    print(f"MRR@5:    {mrr_avg:.4f}")
    print(f"Latency:  {avg_latency:.3f}s per query")
    print(f"Failures: {len(failures)}/{n}")
    
    print("\n" + "=" * 70)
    print("BY SECTION")
    print("=" * 70)
    print(f"{'Sec':>3} {'Name':<30} {'N':>3} {'Hit@3':>7} {'MRR':>7}")
    print("-" * 70)
    for sec in sorted(by_section.keys()):
        info = by_section[sec]
        hit_pct = (info["hits"] / info["total"]) * 100 if info["total"] > 0 else 0
        mrr_sec = info["mrr"] / info["total"] if info["total"] > 0 else 0
        print(f"{sec:>3} {info['section_name']:<30} {info['total']:>3} {hit_pct:>6.1f}% {mrr_sec:>7.4f}")
    
    print("\n" + "=" * 70)
    print("FAILURES")
    print("=" * 70)
    for f in failures:
        print(f"  [{f['section']:>2}] {f['section_name']:<25} | Expected={f['expected']}")
        print(f"       Query: {f['query'][:70]}...")
        print(f"       Got: {f['retrieved'][:3]}")
        print()
    
    output = {
        "hit3_pct": hit3_pct, "mrr_avg": mrr_avg, "avg_latency": avg_latency,
        "total_failures": len(failures), "by_section": {k: dict(v) for k, v in by_section.items()},
        "results": results, "failures": failures
    }
    
    out_name = Path(test_set_path).stem + "_precision_eval.json"
    output_path = DATA_DIR / out_name
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n[*] Results saved to {output_path}")
    
    return output

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_set = sys.argv[1]
    else:
        test_set = Path(__file__).parent.parent / "new" / "test_50.json"
    evaluate(test_set)
