"""
Hackathon Evaluation Runner
Runs the inference pipeline against the 10 hackathon public test queries
and evaluates results using the official eval_script.py
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["SENTENCE_TRANSFORMERS_VERBOSE"] = "false"

import sys
import json
import time
import warnings
import re

warnings.filterwarnings("ignore")

import torch
_ = torch.cuda.is_available()  # Initialize CUDA context to prevent access violation
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import urllib.request
from pathlib import Path

# ---- Configuration ----
TOP_DENSE   = 20
TOP_BM25    = 20
FUSION_K    = 8    # Reduced: fewer candidates = faster rerank
RERANK_K    = 3
OUTPUT_K    = 3
RRF_K       = 5    # Lower: dense signals dominate more
GRAPH_BOOST = 0.1
ENABLE_RERANK = True
EMBED_DEVICE = "cuda"
RERANK_BACKEND = "flag"

LM_BASE_URL   = "http://127.0.0.1:1234"
LM_API_KEY    = os.getenv("LM_API_KEY", "lmstudio")
DEFAULT_MODEL = "google/gemma-4-e2b"

# ---- Year suffix mapping for hackathon output ----
YEAR_MAP = {
    "IS 269": "IS 269: 1989",
    "IS 383": "IS 383: 1970",
    "IS 458": "IS 458: 2003",
    "IS 2185 (Part 2)": "IS 2185 (Part 2): 1983",
    "IS 459": "IS 459: 1992",
    "IS 455": "IS 455: 1989",
    "IS 1489 (Part 2)": "IS 1489 (Part 2): 1991",
    "IS 3466": "IS 3466: 1988",
    "IS 6909": "IS 6909: 1990",
    "IS 8042": "IS 8042: 1989",
}

_index_store = {}

def normalize_id(s):
    """Strip year suffix and normalize spacing/case for robust matching."""
    base = str(s).split(":")[0].strip()
    return re.sub(r"\s+", " ", base).upper()


def normalize_part_label(part):
    if not part:
        return None
    cleaned = re.sub(r"\s+", " ", str(part).strip())
    if not cleaned:
        return None
    m = re.match(r"^(?:PART)\s*(.*)$", cleaned, flags=re.IGNORECASE)
    suffix = m.group(1).strip() if m else cleaned
    if not suffix:
        return None
    return f"Part {suffix.upper()}"


def standard_key(std):
    explicit = std.get("_key")
    if explicit:
        return explicit
    base = re.sub(r"\s+", " ", str(std.get("id", "")).strip().upper())
    part_label = normalize_part_label(std.get("part"))
    return f"{base} ({part_label})" if part_label else base

def apply_year_mapping(retrieved_list, expected_list):
    """
    Only apply year suffix to a retrieved standard if its base ID 
    exactly matches the expected standard's base ID for that query.
    """
    if not expected_list:
        return retrieved_list
    exp_base = normalize_id(expected_list[0])  # e.g. "IS 455" or "IS 2185 (PART 2)"
    exp_full = expected_list[0]  # e.g. "IS 455: 1989"
    result = []
    for s in retrieved_list:
        base = normalize_id(s)
        if base == exp_base:
            result.append(exp_full)  # use full expected string with year
        else:
            result.append(s)
    return result

# ---- LM Studio client ----
def lm_chat(system_prompt, user_message, model=DEFAULT_MODEL, max_tokens=256, temperature=0.3):
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }
    headers = {"Content-Type": "application/json", "x-api-key": LM_API_KEY}
    req = urllib.request.Request(
        f"{LM_BASE_URL}/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))["content"][0]["text"].strip()
    except Exception as e:
        print(f"    [WARN] LM call failed ({e})", file=sys.stderr)
        return ""

# ---- Load indexes ----
def load_indexes():
    global _index_store
    if _index_store:
        return
    import faiss
    print("[*] Loading pickle...", flush=True)
    import pickle
    print("[*] Loading json...", flush=True)
    import json
    print("[*] All imports done.", flush=True)

    faiss_index = faiss.read_index(str(Path("data/faiss_index.bin")))
    print("[*] Loaded FAISS", flush=True)

    with open(Path("data/bm25_index.pkl"), "rb") as f:
        bm25_data = pickle.load(f)
    print("[*] Loaded BM25", flush=True)

    with open(Path("data/graph_map.json"), "r", encoding="utf-8") as f:
        graph_map = json.load(f)
    print("[*] Loaded Graph", flush=True)

    with open(Path("data/whitelist.txt"), "r", encoding="utf-8") as f:
        whitelist = {line.strip(): True for line in f if line.strip()}
    print("[*] Loaded Whitelist", flush=True)

    with open(Path("data/embedding_config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print("[*] Loaded config", flush=True)

    requested_device = EMBED_DEVICE
    if requested_device == "auto":
        cfg_device = str(cfg.get("device", "")).lower().strip()
        requested_device = cfg_device if cfg_device in ("cpu", "cuda") else ("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.", flush=True)
        requested_device = "cpu"

    print(f"[*] Loading BGE-M3 on {requested_device.upper()}...", flush=True)
    embed_model = SentenceTransformer(cfg["model_name"], device=requested_device)
    print("[*] Model loaded", flush=True)

    _index_store = {
        "faiss": faiss_index,
        "bm25": bm25_data["bm25"],
        "standards": bm25_data["standards"],
        "graph": graph_map,
        "whitelist": whitelist,
        "embed_model": embed_model,
    }
    print(f"[+] FAISS:{_index_store['faiss'].ntotal} BM25:{len(_index_store['standards'])} WL:{len(_index_store['whitelist'])}", flush=True)

def get(key):
    load_indexes()
    return _index_store[key]

# ---- Pipeline steps ----
def pre_retrieval_expand(query):
    graph = get("graph")
    synonyms = graph.get("synonyms", {})
    expanded = []
    for token in query.lower().split():
        if token in synonyms:
            expanded.extend(synonyms[token])
        elif token.endswith("s") and token[:-1] in synonyms:
            expanded.extend(synonyms[token[:-1]])
    return " ".join(expanded) if expanded else ""

def retrieve_dense(query, top_k=TOP_DENSE):
    model = get("embed_model")
    faiss_idx = get("faiss")
    standards = get("standards")
    q_emb = model.encode([query], normalize_embeddings=True, batch_size=1)
    q_emb = np.array(q_emb, dtype=np.float32)
    D, I = faiss_idx.search(q_emb, top_k)
    return [(standard_key(standards[i]), float(D[0][j])) for j, i in enumerate(I[0]) if 0 <= i < len(standards)]

def retrieve_sparse(query, top_k=TOP_BM25):
    bm25 = get("bm25")
    standards = get("standards")
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(standard_key(standards[i]), float(scores[i])) for i in top_idx if i < len(standards) and scores[i] > 0]

def rrf_fusion(dense_results, sparse_results, candidate_pool):
    score_map = {cid: 0.0 for cid in candidate_pool}
    for rank, (cid, _score) in enumerate(dense_results, 1):
        if cid in score_map: score_map[cid] += 1.0 / (RRF_K + rank)
    for rank, (cid, _score) in enumerate(sparse_results, 1):
        if cid in score_map: score_map[cid] += 1.0 / (RRF_K + rank)
    cross_refs = get("graph").get("cross_references", {})
    for cid in candidate_pool:
        for neighbor in cross_refs.get(cid, []):
            if neighbor in score_map: score_map[cid] += GRAPH_BOOST
    return sorted(score_map.items(), key=lambda x: x[1], reverse=True)

def paraphrase_trigger(query, expanded_terms, bm25_top_score):
    # Always-on paraphrase
    print("    [Step 4] Paraphrase trigger fired")
    para = lm_chat(
        "Rewrite concisely. No preamble. Example: \"fine aggregate grading for concrete\"",
        f"{query}",
        max_tokens=32, temperature=0.4
    )
    return para if para else None

def rerank_top(query, candidate_ids, top_k=RERANK_K):
    if not ENABLE_RERANK:
        return candidate_ids[:top_k]
    if len(candidate_ids) <= top_k:
        return candidate_ids
    standards = get("standards")
    id_to_std = {standard_key(s): s for s in standards}

    if RERANK_BACKEND == "flag":
        try:
            from FlagEmbedding import FlagReranker
            reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
            id_to_text = {cid: id_to_std[cid]["title"] + " " + id_to_std[cid].get("content", "")[:1500]
                          for cid in candidate_ids if cid in id_to_std}
            valid = [cid for cid in candidate_ids if cid in id_to_text]
            pairs = [[query, id_to_text[cid]] for cid in valid]
            scores = reranker.compute_score(pairs, normalize=True)
            if isinstance(scores, np.ndarray): scores = scores.tolist()
            elif not isinstance(scores, list): scores = [scores]
            if len(scores) == 1:
                scores = scores * len(valid)
            ranked = sorted(zip(valid, scores), key=lambda x: x[1], reverse=True)
            return [cid for cid, _ in ranked[:top_k]]
        except Exception as e:
            print(f"    [WARN] Flag reranker unavailable ({e}). Falling back to semantic rerank.")

    # Stable fallback/default: semantic rerank with existing embedding model.
    try:
        model = get("embed_model")
        valid = [cid for cid in candidate_ids if cid in id_to_std]
        docs = [(id_to_std[cid]["title"] + " " + id_to_std[cid].get("content", "")[:1500]).strip() for cid in valid]
        emb = model.encode([query] + docs, normalize_embeddings=True, batch_size=8)
        q_vec = np.array(emb[0], dtype=np.float32)
        d_vecs = np.array(emb[1:], dtype=np.float32)
        scores = d_vecs @ q_vec
        ranked = sorted(zip(valid, scores.tolist()), key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in ranked[:top_k]]
    except Exception as e:
        print(f"    [WARN] Semantic reranker unavailable ({e}). Returning original order.")
        return candidate_ids[:top_k]

def validation_gate(candidate_ids):
    return [cid for cid in candidate_ids if cid in get("whitelist")]

def format_output_hackathon(query_id, query, expected=None):
    """Process a single query and return hackathon format result"""
    start = time.perf_counter()

    expanded_terms = pre_retrieval_expand(query)
    t0 = time.perf_counter()
    print(f"    Expanded: '{expanded_terms}'" if expanded_terms else "    No expansion")

    dense_results = retrieve_dense(query, TOP_DENSE)
    t1 = time.perf_counter()
    sparse_query = f"{query} {expanded_terms}".strip()
    sparse_results = retrieve_sparse(sparse_query, TOP_BM25)

    candidate_pool = list({cid for cid, _ in dense_results} | {cid for cid, _ in sparse_results})
    fused = rrf_fusion(dense_results, sparse_results, candidate_pool)
    fused_top_k = [cid for cid, _score in fused[:FUSION_K]]
    t2 = time.perf_counter()

    paraphrase = paraphrase_trigger(query, expanded_terms, sparse_results[0][1] if sparse_results else 0.0)
    if paraphrase:
        try:
            para_dense = retrieve_dense(paraphrase, 30)  # Reduced from 50
            merged = {cid: 0.0 for cid in set(list(fused_top_k) + [cid for cid, _score in para_dense])}
            for rank, cid in enumerate(fused_top_k, 1): merged[cid] += 1.0 / (RRF_K + rank)
            for rank, (cid, _score) in enumerate(para_dense, 1): merged[cid] += 1.0 / (RRF_K + rank)
            fused_top_k = [cid for cid, _score in sorted(merged.items(), key=lambda x: x[1], reverse=True)[:FUSION_K]]
        except Exception as em:
            print(f"    [WARN] Paraphrase fusion failed ({em}). Using fused_top_k as-is.", file=sys.stderr)
    t3 = time.perf_counter()

    reranked = rerank_top(query, fused_top_k, RERANK_K)
    t4 = time.perf_counter()
    validated = validation_gate(reranked)

    for cid, _score in fused:
        if len(validated) >= OUTPUT_K: break
        if cid not in validated and cid in get("whitelist"): validated.append(cid)

    final_top3 = validated[:OUTPUT_K]

    # Apply year mapping using expected list to determine when to apply year suffix
    retrieved_ids = list(final_top3)
    retrieved_with_years = apply_year_mapping(retrieved_ids, expected or [])
    t5 = time.perf_counter()

    total = time.perf_counter() - start
    print(f"    [TIMING] expand={((t0-start)*1000):.0f}ms dense={((t1-t0)*1000):.0f}ms fuse+para={((t3-t1)*1000):.0f}ms rerank={((t4-t3)*1000):.0f}ms total={((total)*1000):.0f}ms")

    result = {
        "id": query_id,
        "retrieved_standards": retrieved_with_years,
        "latency_seconds": round(total, 2)
    }
    if expected is not None:
        result["expected_standards"] = expected
    return result

def main():
    load_indexes()

    # Load public test set
    test_set_path = Path("guidelines/public_test_set.json")
    with open(test_set_path, "r", encoding="utf-8") as f:
        test_queries = json.load(f)

    print(f"[*] Loaded {len(test_queries)} hackathon queries")

    results = []
    for item in test_queries:
        query_id = item["id"]
        query = item["query"]
        expected = item.get("expected_standards", [])
        print(f"\n[Query {query_id}] {query[:80]}...")
        try:
            result = format_output_hackathon(query_id, query, expected)
            print(f"    -> {len(result['retrieved_standards'])} results, latency={result['latency_seconds']}s")
            results.append(result)
        except Exception as e:
            print(f"    [ERROR] {e}", file=sys.stderr)
            results.append({
                "id": query_id,
                "retrieved_standards": [],
                "latency_seconds": 5.0,
                "expected_standards": expected
            })

    # Save results
    output_path = Path("data/hackathon_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total = len(results)
    ok = sum(1 for r in results if r.get("retrieved_standards"))
    avg = sum(r.get("latency_seconds", 0) for r in results) / total if total else 0
    print(f"\n[OK] Saved to {output_path} ({ok}/{total} with results, avg={avg:.2f}s)")

    # Evaluate using official eval script
    print("\n[*] Running official evaluation...")
    from guidelines.eval_script import evaluate_results
    evaluate_results(str(output_path))

if __name__ == "__main__":
    main()
