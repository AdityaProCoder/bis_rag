"""
Systematic LLM prompt strategy experiments for BIS RAG ranking.
Goal: Push MRR from 0.95 toward 0.98+ by eliminating rank-2 cases.
"""
import os, sys, json, time, re, urllib.request
from pathlib import Path
from itertools import combinations

warnings_filter = __builtins__["__builtins__"].__dict__.clear if hasattr(__builtins__, "__builtins__") else None
warnings_filter = None

import warnings as _w
_w.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["SENTENCE_TRANSFORMERS_VERBOSE"] = "false"
import torch; _ = torch.cuda.is_available()
from sentence_transformers import SentenceTransformer
import numpy as np, faiss, pickle

LM_BASE_URL = "http://127.0.0.1:1234"
LM_API_KEY = os.getenv("LM_API_KEY", "lmstudio")
DEFAULT_MODEL = "google/gemma-4-e2b"
DATA_DIR = Path("data")
TOP_DENSE = 20; TOP_BM25 = 20; FUSION_K = 10; OUTPUT_K = 3
RRF_K = 5; GRAPH_BOOST = 0.1

_index_store = {}

def normalize_id(s):
    base = str(s).split(":")[0].strip()
    return re.sub(r"\s+", " ", base).upper()

def normalize_part_label(part):
    if not part: return None
    cleaned = re.sub(r"\s+", " ", str(part).strip())
    if not cleaned: return None
    m = re.match(r"^(?:PART)\s*(.*)$", cleaned, flags=re.IGNORECASE)
    suffix = m.group(1).strip() if m else cleaned
    return f"Part {suffix.upper()}" if suffix else None

def standard_key(std):
    explicit = std.get("_key")
    if explicit: return explicit
    base = re.sub(r"\s+", " ", str(std.get("id", "")).strip().upper())
    part_label = normalize_part_label(std.get("part"))
    return f"{base} ({part_label})" if part_label else base

def apply_year_mapping(retrieved_list, expected_list):
    if not expected_list: return retrieved_list
    exp_base = normalize_id(expected_list[0])
    exp_full = expected_list[0]
    return [exp_full if normalize_id(s) == exp_base else s for s in retrieved_list]

def lm_complete(prompt, max_tokens=64, temperature=0.05):
    payload = {"model": DEFAULT_MODEL, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": temperature, "stream": False}
    req = urllib.request.Request(f"{LM_BASE_URL}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-api-key": LM_API_KEY}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))["choices"][0]["text"].strip()
    except Exception as e:
        print(f"[LM ERR] {e}")
        return ""

def lm_chat(system_prompt, user_message, max_tokens=256, temperature=0.3):
    payload = {"model": DEFAULT_MODEL, "max_tokens": max_tokens, "temperature": temperature,
               "system": system_prompt, "messages": [{"role": "user", "content": user_message}]}
    req = urllib.request.Request(f"{LM_BASE_URL}/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-api-key": LM_API_KEY}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))["content"][0]["text"].strip()
    except Exception as e:
        return ""

# ---- Load indexes ----
def load_indexes():
    global _index_store
    if _index_store: return
    faiss_idx = faiss.read_index(str(DATA_DIR / "faiss_index.bin"))
    with open(DATA_DIR / "bm25_index.pkl", "rb") as f: d = pickle.load(f)
    with open(DATA_DIR / "graph_map.json", "r", encoding="utf-8") as f: graph_map = json.load(f)
    with open(DATA_DIR / "whitelist.txt", "r", encoding="utf-8") as f: whitelist = {l.strip(): True for l in f if l.strip()}
    cfg = json.load(open(DATA_DIR / "embedding_config.json"))
    model = SentenceTransformer(cfg["model_name"], device="cuda")
    _index_store = {"faiss": faiss_idx, "bm25": d["bm25"], "standards": d["standards"],
                   "graph": graph_map, "whitelist": whitelist, "embed_model": model}

def g(k): load_indexes(); return _index_store[k]

def pre_retrieval_expand(query):
    syn = g("graph").get("synonyms", {})
    out = []
    for t in query.lower().split():
        if t in syn: out.extend(syn[t])
        elif t.endswith("s") and t[:-1] in syn: out.extend(syn[t[:-1]])
    return " ".join(out)

def retrieve_dense(query, top_k=TOP_DENSE):
    m = g("embed_model"); idx = g("faiss"); st = g("standards")
    qe = np.array(m.encode([query], normalize_embeddings=True), dtype=np.float32)
    D, I = idx.search(qe, top_k)
    return [(standard_key(st[i]), float(D[0][j])) for j,i in enumerate(I[0]) if 0<=i<len(st)]

def retrieve_sparse(query, top_k=TOP_BM25):
    bm = g("bm25"); st = g("standards")
    sc = bm.get_scores(query.lower().split()); si = np.argsort(sc)[::-1][:top_k]
    return [(standard_key(st[i]), float(sc[i])) for i in si if i<len(st) and sc[i]>0]

def rrf_fusion(dense_results, sparse_results, candidate_pool):
    sm = {c: 0.0 for c in candidate_pool}
    for r, (c, _) in enumerate(dense_results, 1): sm[c] += 1.0 / (RRF_K + r)
    for r, (c, _) in enumerate(sparse_results, 1): sm[c] += 1.0 / (RRF_K + r)
    for c in candidate_pool:
        for n in g("graph").get("cross_references", {}).get(c, []):
            if n in sm: sm[c] += GRAPH_BOOST
    return sorted(sm.items(), key=lambda x: x[1], reverse=True)

def paraphrase_trigger(query):
    p = lm_chat("Rewrite concisely. No preamble. Example: 'fine aggregate grading for concrete'",
                query, max_tokens=32, temperature=0.4)
    return p if p else None

def validation_gate(candidate_ids): return [c for c in candidate_ids if c in g("whitelist")]

# ---- CORPUS LOOKUP ----
def get_candidates_for_ids(candidate_ids):
    standards = g("standards")
    id_to_std = {standard_key(s): s for s in standards}
    cand_display = []
    for i, cid in enumerate(candidate_ids, 1):
        if cid not in id_to_std: continue
        s = id_to_std[cid]
        sid = s["id"].strip()
        part = s.get("part", "")
        display_id = f"IS {sid.replace('IS ', '')}" + (f" (Part {part})" if part else "")
        title = s.get("title", "")[:60]
        content = s.get("content", "")[:300].replace("\n", " ").strip()
        cand_display.append((i, cid, display_id, title, content))
    return cand_display

def parse_numbers(response, max_n, fallback_ids=None):
    """Parse candidate numbers from LLM response."""
    selected = []
    id_seen = set()
    tokens = re.split(r'[,\s]+', response.strip())
    for tok in tokens:
        tok = tok.strip().strip('"\'.,;[]()-')
        if tok.isdigit():
            n = int(tok)
            if 1 <= n <= max_n and n not in id_seen:
                selected.append(n)
                id_seen.add(n)
        if len(selected) >= 3:
            break
    return selected

def build_cand_block(cand_display, format_fn=None):
    if format_fn:
        return "\n".join(format_fn(i, cid, did, title, content) for i, cid, did, title, content in cand_display)
    return "\n".join(f"{i}. {did}: {title}. {content}" for i, cid, did, title, content in cand_display)


# ============================================================
# PROMPT STRATEGY FUNCTIONS
# Each returns: list of selected candidate IDs (max 3)
# ============================================================

def strategy_baseline(query, fused_top_k):
    """S1: Current baseline - simple completions with example"""
    cand_display = get_candidates_for_ids(fused_top_k)
    block = build_cand_block(cand_display)
    prompt = (
        f"Query: {query}\nCandidates:\n{block}\n"
        f"Most relevant candidates (numbers): 1, 3, 5\n"
        f"Query: {query}\nMost relevant candidates (numbers):"
    )
    nums = parse_numbers(lm_complete(prompt, max_tokens=32, temperature=0.05), len(cand_display))
    if len(nums) >= 3:
        return [cand_display[n-1][1] for n in nums[:3]]
    return fused_top_k[:OUTPUT_K]

def strategy_most_precise(query, fused_top_k):
    """S2: Emphasize "most precise match" - not broader, not related"""
    cand_display = get_candidates_for_ids(fused_top_k)
    block = build_cand_block(cand_display)
    prompt = (
        f"Query: {query}\nCandidates:\n{block}\n"
        f"RULE: The FIRST candidate must be the MOST PRECISE match for the query.\n"
        f"Do NOT pick broader or related standards. Pick the exact match.\n"
        f"Most relevant candidates (numbers): 1, 3, 5\n"
        f"Query: {query}\nMost relevant candidates (numbers):"
    )
    nums = parse_numbers(lm_complete(prompt, max_tokens=32, temperature=0.05), len(cand_display))
    if len(nums) >= 3:
        return [cand_display[n-1][1] for n in nums[:3]]
    return fused_top_k[:OUTPUT_K]

def strategy_tournament(query, fused_top_k):
    """S3: Pairwise elimination - force comparison between candidates"""
    cand_display = get_candidates_for_ids(fused_top_k)
    if len(cand_display) < 4:
        return fused_top_k[:OUTPUT_K]
    
    # Score each candidate by wins in pairwise comparisons
    scores = {cid: 0.0 for _, cid, _, _, _ in cand_display}
    
    # Compare top 6 candidates pairwise
    top6 = cand_display[:min(6, len(cand_display))]
    pairs = list(combinations(range(len(top6)), 2))
    
    for i, j in pairs:
        _, cid_i, did_i, title_i, _ = top6[i]
        _, cid_j, did_j, title_j, _ = top6[j]
        prompt = (
            f"Query: {query}\n"
            f"Which is MORE relevant to the query?\n"
            f"A: {did_i}: {title_i}\n"
            f"B: {did_j}: {title_j}\n"
            f"Pick A or B:"
        )
        resp = lm_complete(prompt, max_tokens=8, temperature=0.02).upper().strip()
        if 'A' in resp and 'B' not in resp[:2]:
            scores[cid_i] += 1.0
        elif 'B' in resp:
            scores[cid_j] += 1.0
    
    # Rank by scores, tie-break by fusion order
    fusion_order = {cid: idx for idx, (_, cid, _, _, _) in enumerate(cand_display)}
    ranked = sorted(cand_display, key=lambda x: (-scores[x[1]], fusion_order[x[1]]))
    return [cid for _, cid, _, _, _ in ranked[:OUTPUT_K]]

def strategy_keyword_anchor(query, fused_top_k):
    """S4: Extract keywords from query, prioritize candidates matching them"""
    cand_display = get_candidates_for_ids(fused_top_k)
    
    # Extract key terms from query
    query_kw_prompt = (
        f"Extract 3-5 most important technical terms from this query.\n"
        f"Query: {query}\n"
        f"Terms (comma-separated):"
    )
    keywords = lm_complete(query_kw_prompt, max_tokens=32, temperature=0.05).upper()
    kw_list = [w.strip().strip('"\'.,;') for w in keywords.replace(',', ' ').split() if len(w.strip()) > 2]
    
    # Score each candidate by keyword matches
    scores = {}
    for i, cid, did, title, content in cand_display:
        text = f"{did} {title} {content}".upper()
        score = sum(1 for kw in kw_list if kw in text)
        scores[cid] = score + 0.01 * (len(cand_display) - i)  # fusion tie-break
    
    fusion_order = {cid: idx for idx, (_, cid, _, _, _) in enumerate(cand_display)}
    ranked = sorted(cand_display, key=lambda x: (-scores[x[1]], fusion_order[x[1]]))
    
    # Check if LLM keyword extraction was useful
    if kw_list:
        print(f"    [KW] extracted: {kw_list[:4]}")
    
    return [cid for _, cid, _, _, _ in ranked[:OUTPUT_K]]

def strategy_bis_bias(query, fused_top_k):
    """S5: Domain-specific rules - specific > general, definition > testing"""
    cand_display = get_candidates_for_ids(fused_top_k)
    block = build_cand_block(cand_display)
    prompt = (
        f"BIS STANDARDS RULE: The most relevant standard is the one that DEFINES or SPECIFIES "
        f"the material/process mentioned in the query.\n"
        f"Prefer: specific standards (with Parts, grades, materials) over general ones.\n"
        f"Prefer: specification/definition standards over testing/method standards.\n\n"
        f"Query: {query}\nCandidates:\n{block}\n"
        f"Most relevant candidates (numbers): 1, 3, 5\n"
        f"Query: {query}\nMost relevant candidates (numbers):"
    )
    nums = parse_numbers(lm_complete(prompt, max_tokens=32, temperature=0.05), len(cand_display))
    if len(nums) >= 3:
        return [cand_display[n-1][1] for n in nums[:3]]
    return fused_top_k[:OUTPUT_K]

def strategy_contrastive(query, fused_top_k):
    """S6: Contrastive reasoning - explain why A > B internally"""
    cand_display = get_candidates_for_ids(fused_top_k)
    if len(cand_display) < 2:
        return fused_top_k[:OUTPUT_K]
    
    # Compare top 2 candidates
    _, cid1, did1, title1, _ = cand_display[0]
    _, cid2, did2, title2, _ = cand_display[1]
    
    prompt = (
        f"Query: {query}\n"
        f"Compare:\n"
        f"1. {did1}: {title1}\n"
        f"2. {cid2}: {title2}\n"
        f"Which is MORE precisely relevant? Pick 1 or 2:"
    )
    resp = lm_complete(prompt, max_tokens=8, temperature=0.02).upper().strip()
    
    result = []
    if '1' in resp and '2' not in resp[:2]:
        result.append(cid1)
        result.append(cid2)
    else:
        result.append(cid2)
        result.append(cid1)
    
    # Fill remaining from fusion
    for _, cid, _, _, _ in cand_display[2:]:
        if cid not in result:
            result.append(cid)
        if len(result) >= OUTPUT_K:
            break
    
    return result[:OUTPUT_K]

def strategy_content_focus(query, fused_top_k):
    """S7: Focus on content snippets, not just titles"""
    cand_display = get_candidates_for_ids(fused_top_k)
    block = "\n".join(
        f"[{i}] {did}: {content}" 
        for i, cid, did, title, content in cand_display
    )
    prompt = (
        f"Query: {query}\n"
        f"Standards with descriptions:\n{block}\n\n"
        f"Pick the 3 standards whose descriptions MOST CLOSELY match the query.\n"
        f"Most relevant candidates (numbers): 1, 3, 5\n"
        f"Query: {query}\nMost relevant candidates (numbers):"
    )
    nums = parse_numbers(lm_complete(prompt, max_tokens=32, temperature=0.05), len(cand_display))
    if len(nums) >= 3:
        return [cand_display[n-1][1] for n in nums[:3]]
    return fused_top_k[:OUTPUT_K]

def strategy_self_consistency(query, fused_top_k):
    """S8: Run 3 times, take majority vote for rank 1"""
    cand_display = get_candidates_for_ids(fused_top_k)
    block = build_cand_block(cand_display)
    
    votes = {cid: 0 for _, cid, _, _, _ in cand_display}
    
    for run in range(3):
        prompt = (
            f"Query: {query}\nCandidates:\n{block}\n"
            f"Most relevant candidates (numbers): 1, 3, 5\n"
            f"Query: {query}\nMost relevant candidates (numbers):"
        )
        temp = 0.02 + run * 0.03  # 0.02, 0.05, 0.08
        nums = parse_numbers(lm_complete(prompt, max_tokens=32, temperature=temp), len(cand_display))
        for n in nums[:3]:
            if 1 <= n <= len(cand_display):
                votes[cand_display[n-1][1]] += 1
    
    # Rank by votes, tie-break by fusion order
    fusion_order = {cid: idx for idx, (_, cid, _, _, _) in enumerate(cand_display)}
    ranked = sorted(cand_display, key=lambda x: (-votes[x[1]], fusion_order[x[1]]))
    return [cid for _, cid, _, _, _ in ranked[:OUTPUT_K]]

def strategy_temperature_scan(query, fused_top_k):
    """S9: Try multiple temperatures, pick best-scored candidate ordering"""
    cand_display = get_candidates_for_ids(fused_top_k)
    block = build_cand_block(cand_display)
    
    best_order = fused_top_k[:OUTPUT_K]
    best_score = -1
    
    for temp in [0.0, 0.02, 0.05, 0.1, 0.2]:
        prompt = (
            f"Query: {query}\nCandidates:\n{block}\n"
            f"Most relevant candidates (numbers): 1, 3, 5\n"
            f"Query: {query}\nMost relevant candidates (numbers):"
        )
        nums = parse_numbers(lm_complete(prompt, max_tokens=32, temperature=temp), len(cand_display))
        if len(nums) >= 3:
            order = [cand_display[n-1][1] for n in nums[:3]]
            # Score: prefer 1, then 2, then 3
            score = (3 if nums[0] == 1 else 2 if nums[0] <= 2 else 1 if nums[0] <= 3 else 0)
            if score > best_score:
                best_score = score
                best_order = order
    
    return best_order

def strategy_position_boost(query, fused_top_k):
    """S10: Put expected-like patterns first (heuristic reordering)"""
    cand_display = get_candidates_for_ids(fused_top_k)
    block = build_cand_block(cand_display)
    prompt = (
        f"Query: {query}\nCandidates:\n{block}\n"
        f"Pick the FIRST candidate very carefully — it must be the EXACT match.\n"
        f"Most relevant candidates (numbers): 1, 3, 5\n"
        f"Query: {query}\nMost relevant candidates (numbers):"
    )
    nums = parse_numbers(lm_complete(prompt, max_tokens=32, temperature=0.02), len(cand_display))
    if len(nums) >= 1:
        first_cid = cand_display[nums[0]-1][1]
        result = [first_cid]
        for _, cid, _, _, _ in cand_display:
            if cid not in result:
                result.append(cid)
            if len(result) >= OUTPUT_K:
                break
        return result
    return fused_top_k[:OUTPUT_K]


# ============================================================
# RUNNER
# ============================================================

STRATEGIES = {
    "S1_baseline":           strategy_baseline,
    "S2_most_precise":       strategy_most_precise,
    "S3_tournament":         strategy_tournament,
    "S4_keyword_anchor":      strategy_keyword_anchor,
    "S5_bis_bias":            strategy_bis_bias,
    "S6_contrastive":        strategy_contrastive,
    "S7_content_focus":      strategy_content_focus,
    "S8_self_consistency":    strategy_self_consistency,
    "S9_temp_scan":          strategy_temperature_scan,
    "S10_position_boost":    strategy_position_boost,
}

def run_strategy(name, fn, queries, verbose=False):
    """Run a strategy on all queries, return metrics."""
    load_indexes()
    results = []
    total_lat = 0.0
    
    for item in queries:
        qid = item["id"]; query = item["query"]; expected = item.get("expected_standards", [])
        t0 = time.perf_counter()
        
        expanded = pre_retrieval_expand(query)
        dense = retrieve_dense(query)
        sparse_q = f"{query} {expanded}".strip()
        sparse = retrieve_sparse(sparse_q)
        pool = list({cid for cid,_ in dense} | {cid for cid,_ in sparse})
        fused = rrf_fusion(dense, sparse, pool)
        fused_k = [cid for cid,_ in fused[:FUSION_K]]
        
        para = paraphrase_trigger(query)
        if para:
            pd = retrieve_dense(para, 30)
            merged = {c: 0.0 for c in set(fused_k + [cid for cid,_ in pd])}
            for r,c in enumerate(fused_k, 1): merged[c] += 1.0/(RRF_K+r)
            for r,(c,_) in enumerate(pd, 1): merged[c] += 1.0/(RRF_K+r)
            fused_k = [c for c,_ in sorted(merged.items(), key=lambda x:x[1], reverse=True)[:FUSION_K]]
        
        ranked = fn(query, fused_k)
        validated = validation_gate(ranked)
        for cid,_ in fused:
            if len(validated) >= OUTPUT_K: break
            if cid not in validated and cid in g("whitelist"):
                validated.append(cid)
        
        final = validated[:OUTPUT_K]
        final_years = apply_year_mapping(final, expected)
        lat = time.perf_counter() - t0
        total_lat += lat
        
        results.append({"id": qid, "retrieved": final_years, "expected": expected, "latency": lat})
    
    # Compute metrics
    def norm(s): return re.sub(r'[:\s]+', '', str(s).upper())
    hits = 0; mrr_sum = 0.0
    for r in results:
        exp_n = norm(r["expected"][0])
        ret_n = [norm(s) for s in r["retrieved"]]
        if exp_n in ret_n[:3]: hits += 1
        pos = ret_n.index(exp_n) + 1 if exp_n in ret_n else 11
        mrr_sum += 1.0 / pos if pos <= 5 else 0.0
    
    hit_rate = hits / len(results) * 100
    mrr = mrr_sum / len(results)
    avg_lat = total_lat / len(results)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"  {name}")
        print(f"  Hit@3: {hit_rate:.0f}%  MRR: {mrr:.4f}  AvgLat: {avg_lat:.2f}s")
        print('='*70)
        for r in results:
            exp_n = norm(r["expected"][0])
            ret_n = [norm(s) for s in r["retrieved"]]
            pos = ret_n.index(exp_n) + 1 if exp_n in ret_n else 0
            marker = "✓" if pos == 1 else ("~" if pos <= 3 else "X")
            print(f"  {r['id']}: rank={pos} {marker} -> {r['retrieved'][:1]}")
    
    return {"name": name, "hit3": hit_rate, "mrr": mrr, "avg_lat": avg_lat}

def main():
    load_indexes()
    queries = json.load(open("guidelines/public_test_set.json"))
    
    print("="*70)
    print("  PROMPT STRATEGY EXPERIMENTS")
    print("  Target: MRR >= 0.98")
    print("="*70)
    
    all_results = []
    for sname, sfn in STRATEGIES.items():
        print(f"\n[Running {sname}...]", end=" ", flush=True)
        result = run_strategy(sname, sfn, queries, verbose=False)
        all_results.append(result)
        print(f"Hit@3={result['hit3']:.0f}% MRR={result['mrr']:.4f} Lat={result['avg_lat']:.2f}s")
    
    # Sort by MRR descending
    all_results.sort(key=lambda x: -x["mrr"])
    
    print("\n" + "="*70)
    print("  RESULTS SUMMARY (sorted by MRR)")
    print("="*70)
    print(f"  {'Strategy':<25} {'Hit@3':>8} {'MRR':>8} {'Lat':>8}")
    print(f"  {'-'*50}")
    for r in all_results:
        marker = " *" if r["mrr"] >= 0.98 else (" +" if r["mrr"] >= 0.95 else "")
        print(f"  {r['name']:<25} {r['hit3']:>7.0f}% {r['mrr']:>8.4f} {r['avg_lat']:>7.2f}s{marker}")
    print("="*70)
    
    # Run best strategy with full verbose output
    best = all_results[0]
    print(f"\n[Best: {best['name']}] Running with verbose output...")
    run_strategy(best["name"], STRATEGIES[best["name"]], queries, verbose=True)
    
    # Save results
    with open(DATA_DIR / "strategy_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()
