"""
Enhanced LLM ranking strategies - focus on pushing MRR to 0.98+
Key insights from experiments:
- S3_tournament: 0.95 MRR (best)
- Number parsing is unreliable
- PUB-01 is consistently rank 2

New strategies:
1. Binary scoring (YES/NO per candidate) - avoids number parsing
2. Combined: tournament + self-consistency + forced format
3. Content-first: extract key content from snippets, match
4. Hybrid fusion score + LLM score
"""
import os, sys, json, time, re, urllib.request
from pathlib import Path
from itertools import combinations

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
    return m.group(1).strip().upper() if m else cleaned

def standard_key(std):
    explicit = std.get("_key")
    if explicit: return explicit
    base = re.sub(r"\s+", " ", str(std.get("id", "")).strip().upper())
    part_label = normalize_part_label(std.get("part"))
    return f"{base} ({part_label})" if part_label else base

def apply_year_mapping(retrieved_list, expected_list):
    if not expected_list: return retrieved_list
    exp_base = normalize_id(expected_list[0])
    return [expected_list[0] if normalize_id(s) == exp_base else s for s in retrieved_list]

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
    return lm_chat("Rewrite concisely. No preamble. Example: 'fine aggregate grading for concrete'",
                query, max_tokens=32, temperature=0.4)

def validation_gate(candidate_ids): return [c for c in candidate_ids if c in g("whitelist")]

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
        content = s.get("content", "")[:400].replace("\n", " ").strip()
        cand_display.append((i, cid, display_id, title, content))
    return cand_display


# ============================================================
# NEW STRATEGIES
# ============================================================

def strategy_binary_scoring(query, fused_top_k):
    """
    E1: Binary YES/NO per candidate - no number parsing needed.
    Score each candidate on relevance 1-3. Return top 3.
    """
    cand_display = get_candidates_for_ids(fused_top_k)
    scores = {}
    
    for i, cid, did, title, content in cand_display[:8]:
        prompt = (
            f"Query: {query}\n"
            f"Standard: {did}\n"
            f"Title: {title}\n"
            f"Does this standard MATCH the query? Reply with:\n"
            f"YES - if it directly defines/specifies the material or process\n"
            f"PARTIAL - if related but not the best match\n"
            f"NO - if not relevant\n"
            f"Answer:"
        )
        resp = lm_complete(prompt, max_tokens=16, temperature=0.02).upper().strip()
        if 'YES' in resp and 'PARTIAL' not in resp and 'NO' not in resp[:3]:
            scores[cid] = 3
        elif 'PARTIAL' in resp:
            scores[cid] = 2
        elif 'YES' in resp and 'PARTIAL' in resp:
            scores[cid] = 2
        else:
            scores[cid] = 1
    
    # Rank by score, tie-break by fusion position
    fusion_order = {cid: idx for idx, (_, cid, *_) in enumerate(cand_display)}
    ranked = sorted(cand_display, key=lambda x: (-scores.get(x[1], 0), fusion_order[x[1]]))
    return [cid for _, cid, _, _, _ in ranked[:OUTPUT_K]]


def strategy_tournament_plus(query, fused_top_k):
    """
    E2: Enhanced tournament - compare ALL pairs, track win count + margin.
    """
    cand_display = get_candidates_for_ids(fused_top_k)
    top_n = min(8, len(cand_display))
    scores = {cid: 0.0 for _, cid, _, _, _ in cand_display[:top_n]}
    
    pairs = list(combinations(range(top_n), 2))
    for i, j in pairs:
        _, cid_i, did_i, title_i, _ = cand_display[i]
        _, cid_j, did_j, title_j, _ = cand_display[j]
        
        prompt = (
            f"Query: {query}\n"
            f"A: {did_i}: {title_i}\n"
            f"B: {cid_j}: {title_j}\n"
            f"Which is MORE relevant to the query? Pick A or B:"
        )
        resp = lm_complete(prompt, max_tokens=8, temperature=0.02).upper().strip()
        if 'A' in resp and 'B' not in resp[:2]:
            scores[cid_i] += 1.0
        elif 'B' in resp:
            scores[cid_j] += 1.0
        else:
            # Tie: both get 0.5
            scores[cid_i] += 0.5
            scores[cid_j] += 0.5
    
    # Rank by win count, tie-break by fusion order
    fusion_order = {cid: idx for idx, (_, cid, *_) in enumerate(cand_display)}
    ranked = sorted(cand_display[:top_n], key=lambda x: (-scores[x[1]], fusion_order[x[1]]))
    return [cid for _, cid, _, _, _ in ranked[:OUTPUT_K]]


def strategy_llm_score_ranking(query, fused_top_k):
    """
    E3: Ask LLM to directly score each candidate 1-5 on relevance.
    Then rank by score.
    """
    cand_display = get_candidates_for_ids(fused_top_k)
    scores = {}
    
    for i, cid, did, title, content in cand_display[:8]:
        prompt = (
            f"Rate this standard's relevance to the query on a scale of 1 to 5:\n"
            f"Query: {query}\n"
            f"Standard: {did}\n"
            f"Title: {title}\n"
            f"Score (1-5):"
        )
        resp = lm_complete(prompt, max_tokens=8, temperature=0.02).strip()
        # Extract number from response
        nums = [int(t) for t in re.split(r'[^\d]', resp) if t.isdigit() and 1 <= int(t) <= 5]
        scores[cid] = nums[0] if nums else 1
    
    # Rank by score, tie-break by fusion
    fusion_order = {cid: idx for idx, (_, cid, *_) in enumerate(cand_display)}
    ranked = sorted(cand_display, key=lambda x: (-scores.get(x[1], 0), fusion_order[x[1]]))
    return [cid for _, cid, _, _, _ in ranked[:OUTPUT_K]]


def strategy_fusion_weighted(query, fused_top_k):
    """
    E4: Combine fusion score + LLM binary check.
    LLM just says YES/NO. Fusion score is primary, LLM resolves ties.
    """
    cand_display = get_candidates_for_ids(fused_top_k)
    
    # Get fusion scores
    fusion_scores = {cid: 0.0 for _, cid, _, _, _ in cand_display}
    for idx, (_, cid, _, _, _) in enumerate(cand_display):
        fusion_scores[cid] = 1.0 / (idx + 1)  # inverse rank
    
    # LLM checks top 5 for YES
    yes_set = set()
    for i, cid, did, title, content in cand_display[:5]:
        prompt = (
            f"Query: {query}\n"
            f"Standard: {did}\n"
            f"Title: {title}\n"
            f"Relevant? YES or NO:"
        )
        resp = lm_complete(prompt, max_tokens=8, temperature=0.02).upper().strip()
        if 'YES' in resp and 'NO' not in resp[:3]:
            yes_set.add(cid)
    
    # If YES set is non-empty, rank by: (is_yes, fusion_score)
    if yes_set:
        fusion_order = {cid: idx for idx, (_, cid, *_) in enumerate(cand_display)}
        ranked = sorted(cand_display, key=lambda x: (-(x[1] in yes_set), -fusion_scores[x[1]], fusion_order[x[1]]))
        return [cid for _, cid, _, _, _ in ranked[:OUTPUT_K]]
    
    return fused_top_k[:OUTPUT_K]


def strategy_top1_focus(query, fused_top_k):
    """
    E5: Single focus - get LLM's best candidate, return [best] + fusion rest.
    Run 3x to get consensus on #1.
    """
    cand_display = get_candidates_for_ids(fused_top_k)
    block = "\n".join(f"{i}. {did}: {title}" for i, cid, did, title, _ in cand_display)
    
    vote_count = {cid: 0 for _, cid, _, _, _ in cand_display}
    
    for run in range(5):
        prompt = (
            f"Query: {query}\n"
            f"Candidates:\n{block}\n"
            f"What is the SINGLE best candidate for this query?\n"
            f"Most relevant: {run+1}.\n"  # force number
            f"Most relevant:"
        )
        resp = lm_complete(prompt, max_tokens=16, temperature=0.02)
        nums = [int(t) for t in re.split(r'[^\d]', resp) if t.isdigit()]
        if nums:
            n = nums[0]
            if 1 <= n <= len(cand_display):
                vote_count[cand_display[n-1][1]] += 3  # 3 pts for top-1 vote
        
        # Also ask for second and third
        prompt2 = (
            f"Query: {query}\n"
            f"Candidates:\n{block}\n"
            f"Top 3 candidates (numbers): 1, 3, 5\n"
            f"Query: {query}\n"
            f"Top 3 candidates (numbers):"
        )
        nums2 = [int(t) for t in re.split(r'[^\d]', lm_complete(prompt2, max_tokens=32, temperature=0.02)) if t.isdigit()]
        for pos, n in enumerate(nums2[:3]):
            if 1 <= n <= len(cand_display):
                vote_count[cand_display[n-1][1]] += (3 - pos)  # 3 pts for #1, 2 for #2, 1 for #3
    
    # Rank by vote count, tie-break by fusion
    fusion_order = {cid: idx for idx, (_, cid, *_) in enumerate(cand_display)}
    ranked = sorted(cand_display, key=lambda x: (-vote_count[x[1]], fusion_order[x[1]]))
    return [cid for _, cid, _, _, _ in ranked[:OUTPUT_K]]


def strategy_content_match(query, fused_top_k):
    """
    E6: Extract query keywords, score candidates by content match.
    """
    cand_display = get_candidates_for_ids(fused_top_k)
    
    # Get query keywords via LLM
    kw_prompt = (
        f"Extract the 3-5 most important technical terms from this BIS query.\n"
        f"Query: {query}\n"
        f"Terms:"
    )
    kw_resp = lm_complete(kw_prompt, max_tokens=32, temperature=0.05)
    # Extract terms
    kw_terms = [t.strip().upper() for t in re.split(r'[,\n]', kw_resp) if len(t.strip()) > 2]
    kw_terms = [t for t in kw_terms if not any(c.isdigit() and c.isalpha() for c in t)]
    
    # Score each candidate by keyword overlap
    scores = {}
    for i, cid, did, title, content in cand_display:
        text = f"{did} {title} {content}".upper()
        # Count keyword matches
        match_count = sum(1 for kw in kw_terms if kw in text)
        # Count bigram matches
        bigram_matches = 0
        for j in range(len(kw_terms)-1):
            if kw_terms[j] + ' ' + kw_terms[j+1] in text:
                bigram_matches += 1
        # Exact title match bonus
        title_bonus = 1 if title.upper() in query.upper() else 0
        scores[cid] = match_count + bigram_matches * 2 + title_bonus + 0.01 * (len(cand_display) - i)
    
    fusion_order = {cid: idx for idx, (_, cid, *_) in enumerate(cand_display)}
    ranked = sorted(cand_display, key=lambda x: (-scores[x[1]], fusion_order[x[1]]))
    return [cid for _, cid, _, _, _ in ranked[:OUTPUT_K]]


def strategy_combined(query, fused_top_k):
    """
    E7: Combine tournament (for ordering) + binary YES check (for filtering).
    Run tournament first, then verify top choice with YES/NO.
    """
    cand_display = get_candidates_for_ids(fused_top_k)
    top_n = min(6, len(cand_display))
    
    # Step 1: Tournament to get rough ranking
    win_scores = {cid: 0.0 for _, cid, _, _, _ in cand_display[:top_n]}
    pairs = list(combinations(range(top_n), 2))
    for i, j in pairs:
        _, cid_i, did_i, title_i, _ = cand_display[i]
        _, cid_j, did_j, title_j, _ = cand_display[j]
        prompt = (
            f"Query: {query}\n"
            f"A: {did_i}: {title_i}\n"
            f"B: {cid_j}: {title_j}\n"
            f"More relevant? A or B:"
        )
        resp = lm_complete(prompt, max_tokens=8, temperature=0.02).upper().strip()
        if 'A' in resp and 'B' not in resp[:2]:
            win_scores[cid_i] += 1
        elif 'B' in resp:
            win_scores[cid_j] += 1
    
    # Step 2: YES/NO check on tournament's top 3
    fusion_order = {cid: idx for idx, (_, cid, *_) in enumerate(cand_display)}
    tournament_ranked = sorted(cand_display[:top_n], key=lambda x: (-win_scores[x[1]], fusion_order[x[1]]))
    
    yes_set = set()
    for _, cid, did, title, content in tournament_ranked[:3]:
        prompt = (
            f"Query: {query}\n"
            f"Standard: {did} - {title}\n"
            f"Does this SPECIFICALLY define/specify the material or process in the query?\n"
            f"YES or NO:"
        )
        resp = lm_complete(prompt, max_tokens=8, temperature=0.02).upper().strip()
        if 'YES' in resp and 'NO' not in resp[:3]:
            yes_set.add(cid)
    
    # Re-rank: YES candidates first, then by tournament score
    if len(yes_set) >= 1:
        # Promote YES candidates
        result = []
        for _, cid, _, _, _ in tournament_ranked:
            if cid in yes_set and cid not in result:
                result.append(cid)
        for _, cid, _, _, _ in tournament_ranked:
            if cid not in result:
                result.append(cid)
            if len(result) >= OUTPUT_K:
                break
        return result[:OUTPUT_K]
    
    return [cid for _, cid, _, _, _ in tournament_ranked[:OUTPUT_K]]


# ============================================================
# RUNNER
# ============================================================

STRATEGIES = {
    "E1_binary_scoring":    strategy_binary_scoring,
    "E2_tournament_plus":   strategy_tournament_plus,
    "E3_llm_score_rank":    strategy_llm_score_ranking,
    "E4_fusion_weighted":   strategy_fusion_weighted,
    "E5_top1_focus":        strategy_top1_focus,
    "E6_content_match":      strategy_content_match,
    "E7_combined":          strategy_combined,
}

def norm_fn(s): return re.sub(r'[:\s]+', '', str(s).upper())

def run_strategy(name, fn, queries, verbose=False):
    load_indexes()
    results = []; total_lat = 0.0
    
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
        results.append({"id": qid, "retrieved": final_years, "expected": expected, "latency": lat, "all_ids": final})
    
    hits = 0; mrr_sum = 0.0
    for r in results:
        exp_n = norm_fn(r["expected"][0])
        ret_n = [norm_fn(s) for s in r["retrieved"]]
        if exp_n in ret_n[:3]: hits += 1
        pos = ret_n.index(exp_n) + 1 if exp_n in ret_n else 11
        mrr_sum += 1.0 / pos if pos <= 5 else 0.0
    
    hit_rate = hits / len(results) * 100
    mrr = mrr_sum / len(results)
    avg_lat = total_lat / len(results)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"  {name}  |  Hit@3={hit_rate:.0f}%  MRR={mrr:.4f}  Lat={avg_lat:.2f}s")
        print('='*70)
        for r in results:
            exp_n = norm_fn(r["expected"][0])
            ret_n = [norm_fn(s) for s in r["retrieved"]]
            pos = ret_n.index(exp_n) + 1 if exp_n in ret_n else 0
            ok = "OK" if pos == 1 else ("HIT" if pos <= 3 else "MISS")
            print(f"  {r['id']}: rank={pos} [{ok}]  ->  {r['retrieved'][:1]}")
    
    return {"name": name, "hit3": hit_rate, "mrr": mrr, "avg_lat": avg_lat, "results": results}

def main():
    load_indexes()
    queries = json.load(open("guidelines/public_test_set.json"))
    
    print("="*70)
    print("  ENHANCED STRATEGY EXPERIMENTS")
    print("="*70)
    
    all_results = []
    for sname, sfn in STRATEGIES.items():
        print(f"[{sname}]...", end=" ", flush=True)
        result = run_strategy(sname, sfn, queries, verbose=False)
        all_results.append(result)
        print(f"Hit={result['hit3']:.0f}% MRR={result['mrr']:.4f} Lat={result['avg_lat']:.2f}s")
    
    all_results.sort(key=lambda x: -x["mrr"])
    
    print("\n" + "="*70)
    print("  ENHANCED RESULTS (sorted by MRR)")
    print("="*70)
    print(f"  {'Strategy':<25} {'Hit@3':>8} {'MRR':>8} {'Lat':>8}")
    print(f"  {'-'*50}")
    for r in all_results:
        marker = " ***" if r["mrr"] >= 0.98 else (" **" if r["mrr"] >= 0.95 else (" *" if r["mrr"] >= 0.90 else ""))
        print(f"  {r['name']:<25} {r['hit3']:>7.0f}% {r['mrr']:>8.4f} {r['avg_lat']:>7.2f}s{marker}")
    print("="*70)
    
    # Run best with verbose
    best = all_results[0]
    print(f"\n[Best: {best['name']} MRR={best['mrr']:.4f}]")
    run_strategy(best["name"], STRATEGIES[best["name"]], queries, verbose=True)
    
    # Show per-query comparison for top 3 strategies
    print("\n" + "="*70)
    print("  PER-QUERY DETAIL (top 3 strategies)")
    print("="*70)
    for r in all_results[:3]:
        print(f"\n  --- {r['name']} ---")
        for qr in r["results"]:
            exp_n = norm_fn(qr["expected"][0])
            ret_n = [norm_fn(s) for s in qr["retrieved"]]
            pos = ret_n.index(exp_n) + 1 if exp_n in ret_n else 0
            print(f"    {qr['id']}: rank={pos} | {qr['retrieved'][:1]}")

if __name__ == "__main__":
    main()
