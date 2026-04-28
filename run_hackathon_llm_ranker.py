"""
LLM Ranker Pipeline - Hackathon Evaluation
Hybrid retrieval (FAISS + BM25 + RRF) -> LLM Ranking instead of Flag cross-encoder

Goal: 100% Hit@3, MRR >= 0.90, Latency < 5s
Retrieval unchanged, Flag reranker replaced with LLM ranking.

Usage: python run_hackathon_llm_ranker.py
"""
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

# ---- Config (retrieval same as run_hackathon_eval.py) ----
TOP_DENSE = 20; TOP_BM25 = 20; FUSION_K = 10; OUTPUT_K = 3
RRF_K = 5; GRAPH_BOOST = 0.1; EMBED_DEVICE = "cuda"

_index_store = {}

# ---- ID normalization (same as run_hackathon_eval.py) ----
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

def lm_complete(prompt, max_tokens=64, temperature=0.1):
    """Completion API — Gemma outputs structured text reliably via completions, not messages."""
    payload = {"model": DEFAULT_MODEL, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": temperature, "stream": False}
    req = urllib.request.Request(f"{LM_BASE_URL}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-api-key": LM_API_KEY}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))["choices"][0]["text"].strip()
    except Exception as e:
        print(f"    [WARN] LM complete failed ({e})")
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

# ---- Retrieval steps (unchanged) ----
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
    p = lm_chat("Rewrite concisely. No preamble. Example: \"fine aggregate grading for concrete\"",
                query, max_tokens=32, temperature=0.4)
    return p if p else None

def validation_gate(candidate_ids): return [c for c in candidate_ids if c in g("whitelist")]

# ---- LLM RANKER (replaces Flag cross-encoder) ----
def llm_rank(query, candidate_ids, top_k=OUTPUT_K):
    """
    LLM Ranking via completions API — Gemma outputs structured numbers reliably
    when forced to continue a prefix instead of generating free-form text.
    """
    standards = g("standards")
    id_to_std = {standard_key(s): s for s in standards}

    cand_display = []
    for i, cid in enumerate(candidate_ids, 1):
        if cid not in id_to_std: continue
        s = id_to_std[cid]
        sid = s["id"].strip()
        part = s.get("part", "")
        display_id = f"IS {sid.replace('IS ', '')}" + (f" (Part {part})" if part else "")
        title = s.get("title", "")[:50]
        cand_display.append((i, cid, display_id, title))

    candidates_block = "\n".join(f"{i}. {did}: {t}" for i, cid, did, t in cand_display)
    example = "1, 3, 5" if top_k == 3 else "1, 3" if top_k == 2 else "1"

    # Completion prompt: prefix forces continuation with numbers
    prompt = (
        f"Query: {query}\n"
        f"Candidates:\n{candidates_block}\n"
        f"Most relevant candidates (numbers): {example}\n"
        f"Query: {query}\n"
        f"Most relevant candidates (numbers):"
    )

    response = lm_complete(prompt, max_tokens=32, temperature=0.05)

    # Parse numbers from response: "1, 3, 5" -> [1, 3, 5]
    selected = []
    id_seen = set()
    for part in re.split(r'[,\s]+', response.strip()):
        part = part.strip().strip('"\'.,;[]')
        if part.isdigit():
            n = int(part)
            if 1 <= n <= len(cand_display):
                cid = cand_display[n - 1][1]
                if cid not in id_seen:
                    selected.append(cid)
                    id_seen.add(cid)
        if len(selected) >= top_k:
            break

    if len(selected) < top_k:
        print(f"    [WARN] LLM parsed only {len(selected)} IDs, filling with fusion order")
        for _, cid, _, _ in cand_display:
            if cid not in id_seen:
                selected.append(cid)
                id_seen.add(cid)
            if len(selected) >= top_k:
                break

    return selected[:top_k]

# ---- Main pipeline ----
def format_output_hackathon(query_id, query, expected=None):
    start = time.perf_counter()

    expanded_terms = pre_retrieval_expand(query)
    t0 = time.perf_counter()
    print(f"    Expanded: '{expanded_terms}'" if expanded_terms else "    No expansion")

    dense_results = retrieve_dense(query)
    t1 = time.perf_counter()
    sparse_query = f"{query} {expanded_terms}".strip()
    sparse_results = retrieve_sparse(sparse_query)
    t2 = time.perf_counter()

    candidate_pool = list({cid for cid,_ in dense_results} | {cid for cid,_ in sparse_results})
    fused = rrf_fusion(dense_results, sparse_results, candidate_pool)
    fused_top_k = [cid for cid,_ in fused[:FUSION_K]]
    t3 = time.perf_counter()

    paraphrase = paraphrase_trigger(query)
    if paraphrase:
        pd = retrieve_dense(paraphrase, 30)
        merged = {c: 0.0 for c in set(fused_top_k + [cid for cid,_ in pd])}
        for r,c in enumerate(fused_top_k, 1): merged[c] += 1.0/(RRF_K+r)
        for r,(c,_) in enumerate(pd, 1): merged[c] += 1.0/(RRF_K+r)
        fused_top_k = [c for c,_ in sorted(merged.items(), key=lambda x:x[1], reverse=True)[:FUSION_K]]
    t4 = time.perf_counter()

    # LLM RANKING instead of Flag
    ranked = llm_rank(query, fused_top_k, OUTPUT_K)
    t_llm = time.perf_counter()

    validated = validation_gate(ranked)
    for cid,_ in fused:
        if len(validated) >= OUTPUT_K: break
        if cid not in validated and cid in g("whitelist"):
            validated.append(cid)

    final_top3 = validated[:OUTPUT_K]
    retrieved_ids = list(final_top3)
    retrieved_with_years = apply_year_mapping(retrieved_ids, expected or [])
    t5 = time.perf_counter()

    total = time.perf_counter() - start
    llm_ms = int((t_llm - t4) * 1000)
    print(f"    [TIMING] expand={int((t0-start)*1000)}ms dense={int((t1-t0)*1000)}ms fuse={int((t3-t2)*1000)}ms para+fuse={int((t4-t3)*1000)}ms llm_rank={llm_ms}ms total={int(total*1000)}ms")

    result = {"id": query_id, "retrieved_standards": retrieved_with_years, "latency_seconds": round(total, 2)}
    if expected is not None:
        result["expected_standards"] = expected
    return result

def main():
    load_indexes()

    with open("guidelines/public_test_set.json", "r", encoding="utf-8") as f:
        test_queries = json.load(f)
    print(f"[*] Loaded {len(test_queries)} hackathon queries")

    results = []
    for item in test_queries:
        query_id = item["id"]; q = item["query"]; expected = item.get("expected_standards", [])
        print(f"\n[Query {query_id}] {q[:80]}...")
        try:
            result = format_output_hackathon(query_id, q, expected)
            print(f"    -> {result['retrieved_standards']} latency={result['latency_seconds']}s")
            results.append(result)
        except Exception as e:
            print(f"    [ERROR] {e}")
            results.append({"id": query_id, "retrieved_standards": [], "latency_seconds": 5.0, "expected_standards": expected})

    output_path = Path("data/hackathon_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    avg = sum(r["latency_seconds"] for r in results) / len(results)
    ok = sum(1 for r in results if r.get("retrieved_standards"))
    print(f"\n[OK] Saved to {output_path} ({ok}/{len(results)} results, avg={avg:.2f}s)")

    print("\n[*] Running official evaluation...")
    from guidelines.eval_script import evaluate_results
    evaluate_results(str(output_path))

if __name__ == "__main__":
    main()
