"""Ablation: disable LLM ranking, use fusion order only."""
import os, sys, json, time, warnings, re, urllib.request
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
    print("[*] Indexes loaded")
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
    print("    [Para] fired")
    p = lm_chat("Rewrite concisely. No preamble.",
                query, max_tokens=32, temperature=0.4)
    return p if p else None

def validation_gate(candidate_ids): return [c for c in candidate_ids if c in g("whitelist")]

def format_output_abl(query_id, query, expected=None, use_llm=True, use_para=True):
    """Ablation pipeline."""
    start = time.perf_counter()
    expanded_terms = pre_retrieval_expand(query)
    t0 = time.perf_counter()

    dense_results = retrieve_dense(query)
    t1 = time.perf_counter()
    sparse_query = f"{query} {expanded_terms}".strip()
    sparse_results = retrieve_sparse(sparse_query)
    t2 = time.perf_counter()

    candidate_pool = list({cid for cid,_ in dense_results} | {cid for cid,_ in sparse_results})
    fused = rrf_fusion(dense_results, sparse_results, candidate_pool)
    fused_top_k = [cid for cid,_ in fused[:FUSION_K]]

    paraphrase = paraphrase_trigger(query) if use_para else None
    if paraphrase:
        pd = retrieve_dense(paraphrase, 30)
        merged = {c: 0.0 for c in set(fused_top_k + [cid for cid,_ in pd])}
        for r,c in enumerate(fused_top_k, 1): merged[c] += 1.0/(RRF_K+r)
        for r,(c,_) in enumerate(pd, 1): merged[c] += 1.0/(RRF_K+r)
        fused_top_k = [c for c,_ in sorted(merged.items(), key=lambda x:x[1], reverse=True)[:FUSION_K]]

    # Use fusion order if no LLM
    if use_llm:
        from run_hackathon_llm_ranker import llm_rank
        ranked = llm_rank(query, fused_top_k, OUTPUT_K)
    else:
        ranked = fused_top_k[:OUTPUT_K]

    validated = validation_gate(ranked)
    for cid,_ in fused:
        if len(validated) >= OUTPUT_K: break
        if cid not in validated and cid in g("whitelist"):
            validated.append(cid)

    final_top3 = validated[:OUTPUT_K]
    retrieved_ids = list(final_top3)
    retrieved_with_years = apply_year_mapping(retrieved_ids, expected or [])

    total = time.perf_counter() - start
    return {"id": query_id, "retrieved_standards": retrieved_with_years, "latency_seconds": round(total, 2)}

def run_abl(name, use_llm, use_para):
    print(f"\n{'='*60}")
    print(f"ABLATION: {name}")
    print(f"  LLM ranking: {use_llm}  Paraphrase: {use_para}")
    print('='*60)
    load_indexes()
    test_queries = json.load(open("guidelines/public_test_set.json"))
    results = []
    for item in test_queries:
        qid = item["id"]; q = item["query"]; expected = item.get("expected_standards", [])
        print(f"\n[{qid}]", end=" ", flush=True)
        try:
            r = format_output_abl(qid, q, expected, use_llm=use_llm, use_para=use_para)
            print(f"-> {r['retrieved_standards']} ({r['latency_seconds']}s)", end="")
            results.append(r)
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"id": qid, "retrieved_standards": [], "latency_seconds": 5.0})

    # Evaluate
    import re as re2
    def norm(s): return re2.sub(r'[:\s]+', '', str(s).upper())
    hits = 0; mrr_sum = 0.0
    for r in results:
        exp = [s for h in json.load(open("guidelines/public_test_set.json")) if h['id']==r['id'] for s in h['expected_standards']][0]
        exp_n = norm(exp)
        ret_n = [norm(s) for s in r['retrieved_standards']]
        if exp_n in ret_n[:3]: hits += 1
        pos = ret_n.index(exp_n) + 1 if exp_n in ret_n else 11
        mrr_sum += 1.0 / pos if pos <= 5 else 0.0
    print(f"\n\n  Hit@3: {hits}/10 = {hits/10*100:.0f}%  MRR: {mrr_sum/10:.4f}  Avg: {sum(r['latency_seconds'] for r in results)/len(results):.2f}s")

if __name__ == "__main__":
    # Baseline: no LLM, no paraphrase (pure fusion)
    run_abl("BASELINE (no LLM, no paraphrase)", use_llm=False, use_para=False)
    # + paraphrase only
    run_abl("PARAPHRASE ONLY (no LLM)", use_llm=False, use_para=True)
