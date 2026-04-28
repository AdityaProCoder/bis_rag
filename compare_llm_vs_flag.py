"""
Direct LLM vs Flag Pipeline Comparison
Same Flag pipeline config as run_hackathon_eval.py
Usage: python compare_llm_vs_flag.py
"""
import os, sys, json, time, warnings, concurrent.futures, urllib.request, re
from pathlib import Path

warnings.filterwarnings("ignore")
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

# ---- Config (EXACTLY same as run_hackathon_eval.py) ----
TOP_DENSE = 20; TOP_BM25 = 20; FUSION_K = 8; RERANK_K = 3; OUTPUT_K = 3
RRF_K = 5; GRAPH_BOOST = 0.1; ENABLE_RERANK = True; EMBED_DEVICE = "cuda"

_index_store = {}

# ---- ID normalization (EXACTLY same as run_hackathon_eval.py) ----
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

# ---- LM Client ----
def lm_chat(system_prompt, user_message, model=DEFAULT_MODEL, max_tokens=256, temperature=0.3):
    payload = {"model": model, "max_tokens": max_tokens, "temperature": temperature,
               "system": system_prompt, "messages": [{"role": "user", "content": user_message}]}
    req = urllib.request.Request(f"{LM_BASE_URL}/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-api-key": LM_API_KEY}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))["content"][0]["text"].strip()
    except Exception as e:
        print(f"    [WARN] LM failed ({e})")
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
    print("[*] Indexes loaded")
    _index_store = {"faiss": faiss_idx, "bm25": d["bm25"], "standards": d["standards"],
                   "graph": graph_map, "whitelist": whitelist, "embed_model": model}

def g(k): load_indexes(); return _index_store[k]

# ---- Pipeline steps (EXACTLY same as run_hackathon_eval.py) ----
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
    print("    [Step 4] Paraphrase trigger fired")
    p = lm_chat("Rewrite concisely. No preamble. Example: \"fine aggregate grading for concrete\"",
                query, max_tokens=32, temperature=0.4)
    return p if p else None

def rerank_top(query, candidate_ids, top_k=RERANK_K):
    if not ENABLE_RERANK: return candidate_ids[:top_k]
    if len(candidate_ids) <= top_k: return candidate_ids[:top_k]
    standards = g("standards")
    id_to_std = {standard_key(s): s for s in standards}
    try:
        from FlagEmbedding import FlagReranker
        reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
        id2text = {c: id_to_std[c]["title"]+" "+id_to_std[c].get("content","")[:1500]
                   for c in candidate_ids if c in id_to_std}
        valid = [c for c in candidate_ids if c in id2text]
        pairs = [[query, id2text[c]] for c in valid]
        scores = reranker.compute_score(pairs, normalize=True)
        if isinstance(scores, np.ndarray): scores = scores.tolist()
        elif not isinstance(scores, list): scores = [scores]
        if len(scores)==1: scores = scores * len(valid)
        return [c for c,_ in sorted(zip(valid, scores), key=lambda x:x[1], reverse=True)[:top_k]]
    except Exception as e:
        print(f"    [WARN] Flag rerank failed ({e}). Using fusion order.")
        return candidate_ids[:top_k]

def validation_gate(candidate_ids): return [c for c in candidate_ids if c in g("whitelist")]

def flag_pipeline(query):
    """Flag pipeline — EXACTLY same as run_hackathon_eval.py format_output_hackathon"""
    start = time.perf_counter()
    exp = pre_retrieval_expand(query)
    print(f"    Expanded: '{exp}'" if exp else "    No expansion")
    d = retrieve_dense(query); sq = (query+" "+exp).strip(); s = retrieve_sparse(sq)
    pool = list({c for c,_ in d}|{c for c,_ in s})
    fused = rrf_fusion(d, s, pool); fused_k = [c for c,_ in fused[:FUSION_K]]
    para = paraphrase_trigger(query)
    if para:
        pd = retrieve_dense(para, 30)  # same as run_hackathon_eval.py
        merged = {c: 0.0 for c in set(fused_k+[c for c,_ in pd])}
        for r,c in enumerate(fused_k, 1): merged[c] += 1.0/(RRF_K+r)
        for r,(c,_) in enumerate(pd, 1): merged[c] += 1.0/(RRF_K+r)
        fused_k = [c for c,_ in sorted(merged.items(), key=lambda x:x[1], reverse=True)[:FUSION_K]]
    reranked = rerank_top(query, fused_k, RERANK_K)
    validated = validation_gate(reranked)
    for c,_ in fused:
        if len(validated)>=OUTPUT_K: break
        if c not in validated and c in g("whitelist"): validated.append(c)
    final = validated[:OUTPUT_K]
    latency = time.perf_counter() - start
    print(f"    [TIMING] total={int(latency*1000)}ms")
    return {"retrieved": final, "latency": latency}

def direct_llm_retrieve(query, top_k=5):
    """Direct LLM: embed + top candidates + LLM single-shot pick"""
    start = time.perf_counter()
    standards = g("standards"); embed_model = g("embed_model"); faiss_idx = g("faiss")
    q_emb = np.array(embed_model.encode([query], normalize_embeddings=True), dtype=np.float32)
    D, I = faiss_idx.search(q_emb, top_k * 3)
    context_lines = []
    for rank, i in enumerate(I[0]):
        if 0 <= i < len(standards):
            s = standards[i]; sid = standard_key(s)
            if sid in g("whitelist"):
                title = s.get("title", ""); content = s.get("content","")[:300].replace("\n"," ")
                context_lines.append(f"[{rank+1}] {sid} — {title}: {content}")
    context = "\n".join(context_lines[:top_k])
    response = lm_chat(
        "You are a BIS standards expert. Answer using ONLY the provided context. "
        "List EXACTLY 3 standard IDs, one per line, format: IS XXXX: YYYY",
        f"Query: {query}\n\nContext:\n{context}\n\nTop 3 standards:",
        max_tokens=128, temperature=0.2
    )
    retrieved = []
    for line in response.split("\n"):
        line = line.strip()
        matches = re.findall(r'IS\s+\d+(?:\s+\([^)]*\))?(?::\s*\d{4})?', line)
        for m in matches:
            sid = re.sub(r'\s*(?::\s*\d{4})?$', '', m).strip()
            sid = re.sub(r'\s+', ' ', sid)
            if sid not in retrieved and sid in g("whitelist"):
                retrieved.append(sid)
                if len(retrieved) >= 3: break
        if len(retrieved) >= 3: break
    if len(retrieved) < 3:
        for i in I[0]:
            if 0 <= i < len(standards):
                sid = standard_key(standards[i])
                if sid not in retrieved and sid in g("whitelist"):
                    retrieved.append(sid)
                    if len(retrieved) >= 3: break
    latency = time.perf_counter() - start
    return {"retrieved": retrieved[:3], "latency": latency}

def evaluate(results_list, expected_map):
    hits = mrr_sum = total_lat = 0
    for r in results_list:
        exp = expected_map.get(r["id"], [])
        ret = r["retrieved"]
        exp_set = {normalize_id(e) for e in exp}
        ret_norm = [normalize_id(s) for s in ret]
        top3 = ret_norm[:3]; hit = any(s in exp_set for s in top3)
        mrr = 0.0
        for rank, s in enumerate(ret_norm[:5], 1):
            if s in exp_set: mrr = 1.0/rank; break
        if hit: hits += 1
        mrr_sum += mrr; total_lat += r["latency"]
    n = len(results_list)
    return hits, mrr_sum/n, total_lat/n

# ---- Main ----
print("=" * 65)
print("  DIRECT LLM vs FLAG PIPELINE — HACKATHON COMPARISON")
print("  (Same config as run_hackathon_eval.py)")
print("=" * 65)

load_indexes()

with open("guidelines/public_test_set.json", "r", encoding="utf-8") as f:
    hackathon_queries = json.load(f)
hack_map = {h["id"]: h["expected_standards"] for h in hackathon_queries}

llm_results = []; flag_results = []

for item in hackathon_queries:
    pub_id = item["id"]; q = item["query"].strip(); expected = item.get("expected_standards", [])
    if not q: continue
    print(f"\n[Query {pub_id}] {q[:60]}")

    print("  [Direct LLM]...", end=" ", flush=True)
    lr = direct_llm_retrieve(q, top_k=5)
    llm_ids = apply_year_mapping(lr["retrieved"], expected)
    print(f"-> {llm_ids} ({lr['latency']:.1f}s)")
    llm_results.append({"id": pub_id, "retrieved": llm_ids, "latency": lr["latency"]})

    print("  [Flag Pipeline]...", end=" ", flush=True)
    fr = flag_pipeline(q)
    flag_ids = apply_year_mapping(fr["retrieved"], expected)
    print(f"-> {flag_ids} ({fr['latency']:.1f}s)")
    flag_results.append({"id": pub_id, "retrieved": flag_ids, "latency": fr["latency"]})

# Summary
h_llm, m_llm, l_llm = evaluate(llm_results, hack_map)
h_flg, m_flg, l_flg = evaluate(flag_results, hack_map)

print(f"\n{'='*65}")
print(f"  RESULTS SUMMARY")
print(f"{'='*65}")
print(f"\n  DIRECT LLM:")
print(f"  Hit Rate @3: {h_llm/10*100:.0f}%  MRR @5: {m_llm:.4f}  Avg Lat: {l_llm:.2f}s")
print(f"\n  FLAG PIPELINE:")
print(f"  Hit Rate @3: {h_flg/10*100:.0f}%  MRR @5: {m_flg:.4f}  Avg Lat: {l_flg:.2f}s")

print(f"\n{'='*65}")
print(f"  SIDE-BY-SIDE")
print(f"{'='*65}")
print(f"  {'Metric':<20} {'Direct LLM':>15} {'Flag Pipeline':>15} {'Winner':>10}")
print(f"  {'-'*60}")
print(f"  {'Hit Rate @3':<20} {h_llm/10*100:>14.0f}% {h_flg/10*100:>14.0f}% {'Flag' if h_flg>h_llm else ('LLM' if h_llm>h_flg else 'Tie'):>10}")
print(f"  {'MRR @5':<20} {m_llm:>15.4f} {m_flg:>15.4f} {'Flag' if m_flg>m_llm else ('LLM' if m_llm>m_flg else 'Tie'):>10}")
print(f"  {'Avg Latency':<20} {l_llm:>15.2f}s {l_flg:>15.2f}s {'LLM' if l_llm<l_flg else ('Flag' if l_flg<l_llm else 'Tie'):>10}")

print(f"\n  PER-QUERY BREAKDOWN")
print(f"  {'Query':<8} {'Expected':<22} {'Direct LLM':<25} {'Flag':<25}")
print(f"  {'-'*80}")
for llm_r, flag_r, h in zip(llm_results, flag_results, hackathon_queries):
    exp = h["expected_standards"]
    print(f"  {llm_r['id']:<8} {exp[0]:<22} {str(llm_r['retrieved']):<25} {str(flag_r['retrieved']):<25}")

with open(DATA_DIR / "llm_vs_flag_comparison.json", "w", encoding="utf-8") as f:
    json.dump({"direct_llm": llm_results, "flag_pipeline": flag_results}, f, indent=2, ensure_ascii=False)
print(f"\n[*] Saved to data/llm_vs_flag_comparison.json")
