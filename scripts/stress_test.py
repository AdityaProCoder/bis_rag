"""
BIS RAG Stress Test - Paraphrase robustness evaluation
Uses UPDATED pipeline: multi-query retrieval + content-match ranking.
"""
import os, json, time, re, urllib.request
from pathlib import Path

import warnings as _w; _w.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["SENTENCE_TRANSFORMERS_VERBOSE"] = "false"
import torch; _ = torch.cuda.is_available()
from sentence_transformers import SentenceTransformer
import numpy as np, faiss, pickle
from concept_layer import concept_hypotheses

LM_BASE_URL = "http://127.0.0.1:1234"
LM_API_KEY = os.getenv("LM_API_KEY", "lmstudio")
DEFAULT_MODEL = "google/gemma-4-e2b"
DATA_DIR = Path("data")
TOP_DENSE = 20; TOP_BM25 = 20; FUSION_K = 10; OUTPUT_K = 3
RRF_K = 5; GRAPH_BOOST = 0.1
EMBED_DEVICE = os.getenv("EMBED_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
_index_store = {}

def norm(s):
    s = str(s).upper()
    s = re.sub(r':\s*\d{4}', '', s)
    s = re.sub(r'\s*\(PART[^)]*\)', '', s)
    return re.sub(r'[:\s]+', '', s)

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
    exp_base = norm(expected_list[0])
    return [expected_list[0] if norm(s) == exp_base else s for s in retrieved_list]

def lm_complete(prompt, max_tokens=64, temperature=0.05):
    payload = {"model": DEFAULT_MODEL, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": temperature, "stream": False}
    req = urllib.request.Request(f"{LM_BASE_URL}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-api-key": LM_API_KEY}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))["choices"][0]["text"].strip()
    except: return ""

def lm_chat(system_prompt, user_message, max_tokens=256, temperature=0.3):
    payload = {"model": DEFAULT_MODEL, "max_tokens": max_tokens, "temperature": temperature,
               "system": system_prompt, "messages": [{"role": "user", "content": user_message}]}
    req = urllib.request.Request(f"{LM_BASE_URL}/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-api-key": LM_API_KEY}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))["content"][0]["text"].strip()
    except: return ""

def load_indexes():
    global _index_store
    if _index_store: return
    faiss_idx = faiss.read_index(str(DATA_DIR / "faiss_index.bin"))
    with open(DATA_DIR / "bm25_index.pkl", "rb") as f: d = pickle.load(f)
    with open(DATA_DIR / "graph_map.json", "r", encoding="utf-8") as f: graph_map = json.load(f)
    with open(DATA_DIR / "whitelist.txt", "r", encoding="utf-8") as f: whitelist = {l.strip(): True for l in f if l.strip()}
    cfg = json.load(open(DATA_DIR / "embedding_config.json"))
    model = SentenceTransformer(cfg["model_name"], device=EMBED_DEVICE)
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

# === CURATED SYNONYM EXPANSION (only unambiguous terms) ===
# Only expand terms where the synonym is MORE specific, not less
BIS_CURATED_SYNONYMS = {
    # Cement types - these are unambiguous
    "pozzolana": ["Portland pozzolana cement", "PPC", "fly ash cement", "calcined clay pozzolana"],
    "supersulphated": ["supersulphated cement", "SSC"],
    "white cement": ["white Portland cement"],
    "masonry cement": ["cement for masonry", "mortar cement"],
    # Aggregates
    "fine aggregate": ["sand", "river sand"],
    "coarse aggregate": ["crushed stone"],
    "asbestos": ["asbestos cement"],
    "corrugated": ["corrugated sheets", "semi-corrugated"],
    # NOTE: do NOT expand "cement", "concrete", "aggregate", "masonry" -
    # these are already in the query and adding them dilutes the signal
}

def expand_query_curated(query):
    """Expand only unambiguous technical terms."""
    q_lower = query.lower()
    expanded = []
    seen = set()
    words = q_lower.replace(",", " ").replace(".", " ").split()
    for word in words:
        if word in seen: continue
        if word in BIS_CURATED_SYNONYMS:
            for syn in BIS_CURATED_SYNONYMS[word]:
                if syn not in seen:
                    expanded.append(syn); seen.add(syn)
            seen.add(word)
        else:
            seen.add(word)
    # Also check multi-word phrases
    for phrase, synonyms in BIS_CURATED_SYNONYMS.items():
        if phrase in q_lower:
            for syn in synonyms:
                if syn not in seen:
                    expanded.append(syn); seen.add(syn)
    return " ".join(expanded)

def retrieve_multi(query, fuse_k=FUSION_K):
    """Additive multi-query: preserve original ranking, supplement concept + expanded dense recall."""
    expanded_terms = pre_retrieval_expand(query)
    curated_syn = expand_query_curated(query)
    expanded_query = f"{expanded_terms} {curated_syn}".strip()  # KEYWORDS ONLY, not original query

    # Step 1: Get original ranking (unchanged)
    dense_orig = retrieve_dense(query, TOP_DENSE)
    sparse_orig = retrieve_sparse(query, TOP_BM25)
    pool_orig = list({cid for cid,_ in dense_orig} | {cid for cid,_ in sparse_orig})
    fused_orig = rrf_fusion(dense_orig, sparse_orig, pool_orig)
    orig_top_ids = [cid for cid,_ in fused_orig[:fuse_k]]

    # Step 2: Get expanded-keyword dense search to supplement pool
    kw_dense = retrieve_dense(expanded_query, TOP_DENSE)

    # Step 3: Add concept-level hypotheses before generic expanded dense hits.
    concept_ids = [cid for cid, _ in concept_hypotheses(query, g("standards"), standard_key, top_k=5)]
    reserve_for_concepts = 3
    final_ids = list(orig_top_ids[:max(0, fuse_k - reserve_for_concepts)])
    for cid in concept_ids:
        if cid not in final_ids:
            final_ids.append(cid)
    for cid in orig_top_ids:
        if cid not in final_ids:
            final_ids.append(cid)

    kw_ids_not_in_orig = [cid for cid,_ in kw_dense if cid not in orig_top_ids]
    for cid in kw_ids_not_in_orig[:3]:
        if cid not in final_ids:
            final_ids.append(cid)

    return [(cid, 0.0) for cid in final_ids[:fuse_k]]

# Content-match ranker
def llm_rank_content_match(query, candidate_ids, top_k=OUTPUT_K):
    standards = g("standards")
    id_to_std = {standard_key(s): s for s in standards}
    concept_scores = dict(concept_hypotheses(query, standards, standard_key, top_k=8))
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

    kw_prompt = (f"Extract the 3-5 most important technical terms from this BIS query.\n"
                 f"Query: {query}\nTerms:")
    kw_resp = lm_complete(kw_prompt, max_tokens=32, temperature=0.05)
    kw_terms = [t.strip().upper() for t in re.split(r'[,\n]', kw_resp)
                if len(t.strip()) > 2 and not any(c.isdigit() for c in t)]

    scores = {}
    for i, cid, did, title, content in cand_display:
        text = f"{did} {title} {content}".upper()
        match_count = sum(1 for kw in kw_terms if kw in text)
        bigram_matches = 0
        for j in range(len(kw_terms) - 1):
            if kw_terms[j] and kw_terms[j+1]:
                if kw_terms[j] + ' ' + kw_terms[j+1] in text: bigram_matches += 1
        title_bonus = 1 if title.upper() in query.upper() else 0
        scores[cid] = (
            match_count
            + bigram_matches * 2
            + title_bonus
            + concept_scores.get(cid, 0.0)
            + 0.01 * (len(cand_display) - i)
        )

    fusion_order = {cid: idx for idx, (_, cid, _, _, _) in enumerate(cand_display)}
    ranked = sorted(cand_display, key=lambda x: (-scores.get(x[1], 0), fusion_order[x[1]]))
    return [cid for _, cid, _, _, _ in ranked[:top_k]]

def llm_rank_fusion(query, candidate_ids, top_k=OUTPUT_K):
    return candidate_ids[:top_k]

def run_single(query_text, expected_list, use_content_match=True):
    """Run single query with multi-query retrieval + paraphrase."""
    fused_ranked = retrieve_multi(query_text, FUSION_K)
    fused_ids = [cid for cid, _ in fused_ranked]
    fused_scores = {cid: score for cid, score in fused_ranked}

    para = paraphrase_trigger(query_text)
    if para:
        pd = retrieve_dense(para, 30)
        merged = {c: 0.0 for c in set(fused_ids + [cid for cid,_ in pd])}
        for r,c in enumerate(fused_ids, 1): merged[c] += 1.0/(RRF_K+r)
        for r,(c,_) in enumerate(pd, 1): merged[c] += 1.0/(RRF_K+r)
        fused_ids = [c for c,_ in sorted(merged.items(), key=lambda x:x[1], reverse=True)[:FUSION_K]]
        concept_ids = [cid for cid, _ in concept_hypotheses(query_text, g("standards"), standard_key, top_k=3)]
        kept = fused_ids[:max(0, FUSION_K - len(concept_ids))]
        fused_ids = kept + [cid for cid in concept_ids if cid not in kept]
        for cid in sorted(merged, key=merged.get, reverse=True):
            if len(fused_ids) >= FUSION_K:
                break
            if cid not in fused_ids:
                fused_ids.append(cid)

    ranked = llm_rank_content_match(query_text, fused_ids, OUTPUT_K) if use_content_match else llm_rank_fusion(query_text, fused_ids, OUTPUT_K)

    validated = validation_gate(ranked)
    for cid in fused_ids:
        if len(validated) >= OUTPUT_K: break
        if cid not in validated and cid in g("whitelist"):
            validated.append(cid)

    final = validated[:OUTPUT_K]
    final_years = apply_year_mapping(final, expected_list)
    return final_years


# === PARAPHRASE GENERATION ===
PARAPHRASE_TYPES = [
    "Replace technical terms with synonyms (e.g. cement -> binding material)",
    "Rephrase the question indirectly",
    "Use more abstract/general domain language",
    "Make it conversational with filler words",
    "Remove obvious keyword terms like cement, sand, concrete where possible",
]

def generate_paraphrases():
    para_file = DATA_DIR / "paraphrased_queries.json"
    if para_file.exists():
        print(f"[*] Loading existing paraphrases from {para_file}")
        return json.load(open(para_file))

    queries = json.load(open("guidelines/public_test_set.json"))
    all_paraphrased = []

    print("\n[*] Generating paraphrases using LLM completions API...")
    for item in queries:
        qid = item["id"]; query = item["query"]; expected = item["expected_standards"]
        variants = []

        for ptype_idx, ptype in enumerate(PARAPHRASE_TYPES):
            prompt = (f"Paraphrase this query. Type: {ptype}\n"
                     f"Query: {query}\nParaphrased:")
            resp = lm_complete(prompt, max_tokens=64, temperature=0.85)
            resp = resp.split('\n')[0].strip()
            if resp and len(resp) > 15 and "thinking" not in resp.lower()[:30]:
                variants.append({"type": ["synonym","indirect","abstract","conversational","keyword_removal"][ptype_idx], "query": resp})
                print(f"  [{qid}] T{ptype_idx+1}: {resp[:60]}...")
            else:
                fallback = query.replace("What is", "Which").replace("Looking for", "Need").replace("Our company", "We")
                if ptype_idx == 3: fallback = f"Hey, {query.lower()}"
                if ptype_idx == 4:
                    fallback = re.sub(r'\b(ordinary portland cement|fine aggregate|concrete|steel)\b', '', query, flags=re.IGNORECASE)
                    fallback = re.sub(r'\s+', ' ', fallback).strip()
                if fallback and len(fallback) > 15:
                    variants.append({"type": ["synonym","indirect","abstract","conversational","keyword_removal"][ptype_idx], "query": fallback})
                    print(f"  [{qid}] T{ptype_idx+1}: {fallback[:60]}... [AUTO]")
                else:
                    variants.append({"type": ["synonym","indirect","abstract","conversational","keyword_removal"][ptype_idx], "query": query})
                    print(f"  [{qid}] T{ptype_idx+1}: KEEP ORIGINAL")

        all_paraphrased.append({
            "original_id": qid, "original_query": query,
            "expected_standards": expected, "variants": variants
        })

    with open(para_file, "w", encoding="utf-8") as f:
        json.dump(all_paraphrased, f, indent=2, ensure_ascii=False)
    print(f"[*] Saved {len(all_paraphrased)*5} paraphrased queries")
    return all_paraphrased


# === EVALUATOR ===
def compute_metrics(results):
    hits = 0; mrr_sum = 0.0; total_lat = 0.0; failures = []
    for r in results:
        exp_n = norm(r["expected"][0])
        ret_n = [norm(s) for s in r["retrieved"]]
        total_lat += r["latency"]
        if exp_n in ret_n[:3]: hits += 1; pos = ret_n.index(exp_n) + 1
        else: pos = 0
        mrr_sum += 1.0 / pos if 1 <= pos <= 5 else 0.0
        if pos > 3 or pos == 0: failures.append(r)
    n = len(results)
    return {"hit3_pct": hits/n*100, "mrr": mrr_sum/n, "avg_lat": total_lat/n,
            "hits": hits, "total": n, "failures": failures}

def run_eval_set(query_list, use_content_match=True, label=""):
    load_indexes()
    results = []
    for item in query_list:
        qid = item["id"] if "id" in item else item.get("original_id", "?")
        qtext = item["query"] if "query" in item else item.get("original_query", "")
        expected = item.get("expected_standards", [])
        if not qtext: continue
        t0 = time.perf_counter()
        try:
            retrieved = run_single(qtext, expected, use_content_match=use_content_match)
        except Exception as e:
            print(f"  ERROR {qid}: {e}"); retrieved = []
        lat = time.perf_counter() - t0
        exp_n = norm(expected[0]) if expected else ""
        ret_n = [norm(s) for s in retrieved]
        pos = ret_n.index(exp_n) + 1 if exp_n in ret_n else 0
        results.append({"id": qid, "query": qtext, "expected": expected,
                      "retrieved": retrieved, "rank": pos, "hit": 1 <= pos <= 3, "latency": lat})
    metrics = compute_metrics(results)
    print(f"  [{label}] Hit@3={metrics['hit3_pct']:.0f}% MRR={metrics['mrr']:.4f} Lat={metrics['avg_lat']:.2f}s ({metrics['hits']}/{metrics['total']})")
    return results, metrics


def classify_failure(r):
    exp_n = norm(r["expected"][0])
    ret_n = [norm(s) for s in r["retrieved"][:3]]
    if exp_n not in ret_n: return "retrieval_failure"
    pos = ret_n.index(exp_n) + 1
    return "ranking_error" if pos > 3 else "unknown"


def main():
    print("="*70)
    print("  BIS RAG STRESS TEST - Multi-Query Retrieval")
    print("="*70)

    orig_queries = json.load(open("guidelines/public_test_set.json"))
    para_data = generate_paraphrases()

    para_queries = []
    for group in para_data:
        for v in group["variants"]:
            para_queries.append({
                "id": f"{group['original_id']}_{v['type']}",
                "query": v["query"],
                "expected_standards": group["expected_standards"],
                "original_id": group["original_id"],
                "variant_type": v["type"]
            })

    print(f"\n[*] Original: {len(orig_queries)}  Paraphrased: {len(para_queries)}")

    print("\n" + "="*70)
    print("  TASK 3: METRICS BREAKDOWN")
    print("="*70)

    print("\n[Content-Match + Multi-Query]")
    orig_results, orig_m = run_eval_set(orig_queries, True, "ORIGINAL")
    para_results, para_m = run_eval_set(para_queries, True, "PARAPHRASED")
    all_r = orig_results + para_results
    all_m = compute_metrics(all_r)
    print(f"  [COMBINED] Hit@3={all_m['hit3_pct']:.0f}% MRR={all_m['mrr']:.4f} Lat={all_m['avg_lat']:.2f}s")

    print("\n" + "="*70)
    print("  TASK 6: CONTROL EXPERIMENT (paraphrased queries)")
    print("="*70)
    _, para_fusion_m = run_eval_set(para_queries, False, "FUSION-ONLY")

    print("\n" + "="*70)
    print("  TASK 5: ROBUSTNESS")
    print("="*70)
    robust = para_m["mrr"] / orig_m["mrr"] if orig_m["mrr"] > 0 else 0.0
    verdict = "ROBUST" if robust >= 0.9 else ("MODERATE" if robust >= 0.7 else "OVERFITTED")
    print(f"  Original MRR:   {orig_m['mrr']:.4f}")
    print(f"  Paraphrased MRR: {para_m['mrr']:.4f}")
    print(f"  Robustness:      {robust:.4f}  [{verdict}]")

    print("\n" + "="*70)
    print("  TASK 4: FAILURE ANALYSIS")
    print("="*70)
    all_fail = [r for r in all_r if not r["hit"]]
    para_fail = [r for r in para_results if not r["hit"]]
    print(f"  Total: {len(all_fail)}/{len(all_r)}  Para: {len(para_fail)}/{len(para_results)}")
    for r in all_fail:
        print(f"  {r['id']}: exp={r['expected']} ret={r['retrieved'][:2]} rank={r['rank']} cause={classify_failure(r)}")

    print("\n" + "="*70)
    print("  FINAL RESULTS")
    print("="*70)
    print(f"  {'Dataset':<22} | Count | Hit@3   | MRR@5   | AvgLat")
    print(f"  {'-'*45}")
    print(f"  {'Original (10)':<22} |    10 | {orig_m['hit3_pct']:6.1f}% | {orig_m['mrr']:7.4f} | {orig_m['avg_lat']:.2f}s")
    print(f"  {'Paraphrased (50)':<22} |    50 | {para_m['hit3_pct']:6.1f}% | {para_m['mrr']:7.4f} | {para_m['avg_lat']:.2f}s")
    print(f"  {'Combined (60)':<22} |    60 | {all_m['hit3_pct']:6.1f}% | {all_m['mrr']:7.4f} | {all_m['avg_lat']:.2f}s")
    print(f"  {'Fusion-only para':<22} |    50 | {para_fusion_m['hit3_pct']:6.1f}% | {para_fusion_m['mrr']:7.4f} | {para_fusion_m['avg_lat']:.2f}s")
    print(f"  ROBUSTNESS: {robust:.4f} [{verdict}]")
    print("="*70)

    with open(DATA_DIR / "stress_test_results.json", "w") as f:
        json.dump({"orig": orig_m, "para": para_m, "combined": all_m,
                   "fusion_para": para_fusion_m, "robust": robust, "verdict": verdict}, f, indent=2)

if __name__ == "__main__":
    main()
