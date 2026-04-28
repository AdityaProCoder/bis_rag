"""
BIS Standards RAG Inference Pipeline - Semantic Reranker Version
Same as inference_clean.py but uses semantic (dot-product) reranker instead of Flag.
Best config without Flag: RRF_K=60, conditional paraphrase, semantic rerank.
CLI: python inference_clean_semantic.py --output data/results_semantic.json
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["SENTENCE_TRANSFORMERS_VERBOSE"] = "false"

import sys
import json
import time
import argparse
import warnings
import concurrent.futures
import re
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
_ = torch.cuda.is_available()
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import urllib.request

LM_BASE_URL   = "http://127.0.0.1:1234"
LM_API_KEY    = os.getenv("LM_API_KEY", "lmstudio")
DEFAULT_MODEL = "google/gemma-4-e2b"

TOP_DENSE   = 20
TOP_BM25    = 20
FUSION_K    = 10
RERANK_K    = 5
OUTPUT_K    = 3
RRF_K       = 60   # Balanced: dense and sparse contribute more evenly
GRAPH_BOOST = 0.1
ENABLE_RERANK = True
EMBED_DEVICE = "cuda"

_index_store = {}


def normalize_is_id(text):
    return re.sub(r"\s+", " ", str(text).strip().upper())


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
    base = normalize_is_id(std.get("id", ""))
    part_label = normalize_part_label(std.get("part"))
    return f"{base} ({part_label})" if part_label else base

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
    # Conditional: only fire if no synonym expansion or no BM25 signal
    if expanded_terms.strip() and bm25_top_score > 0:
        return None
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
    # Semantic rerank: dot-product with BGE-M3 (no Flag dependency)
    try:
        model = get("embed_model")
        valid = [cid for cid in candidate_ids if cid in id_to_std]
        docs = [(id_to_std[cid]["title"] + " " + id_to_std[cid].get("content", "")[:512]).strip() for cid in valid]
        emb = model.encode([query] + docs, normalize_embeddings=True, batch_size=8)
        q_vec = np.array(emb[0], dtype=np.float32)
        d_vecs = np.array(emb[1:], dtype=np.float32)
        scores = (d_vecs @ q_vec).tolist()
        ranked = sorted(zip(valid, scores), key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in ranked[:top_k]]
    except Exception as e:
        print(f"    [WARN] Semantic reranker failed ({e}). Returning original order.")
        return candidate_ids[:top_k]

def validation_gate(candidate_ids):
    return [cid for cid in candidate_ids if cid in get("whitelist")]

def format_output(query):
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
            para_dense = retrieve_dense(paraphrase, 20)  # OLD config: 20
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

    standards = get("standards")
    id_to_std = {standard_key(s): s for s in standards}

    def get_rationale(sid):
        rationale = lm_chat(
            "You are a BIS standards technical writer. Write one brief sentence (under 20 words) explaining why this BIS standard applies. Example: \"IS 383 covers fine aggregate grading for use in concrete production.\" Do not repeat the prompt.",
            f"Query: {query} | Standard: {sid} ({id_to_std.get(sid, {}).get('title','')})",
            max_tokens=64, temperature=0.3
        )
        return {"id": sid, "rationale": rationale if rationale else f"Matches: {query[:40]}"}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        results_list = list(executor.map(get_rationale, final_top3))
    retrieved_standards = list(results_list)
    t5 = time.perf_counter()

    total = time.perf_counter() - start
    print(f"    [TIMING] expand={((t0-start)*1000):.0f}ms dense={((t1-t0)*1000):.0f}ms fuse+para={((t3-t1)*1000):.0f}ms rerank={((t4-t3)*1000):.0f}ms llm={((t5-t4)*1000):.0f}ms total={((total)*1000):.0f}ms")

    return {"id": query[:80], "retrieved_standards": retrieved_standards, "latency_seconds": round(total, 2)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(Path("data/sample_queries.json")))
    parser.add_argument("--output", default=str(Path("data/results_semantic.json")))
    parser.add_argument("--model", default="google/gemma-4-e2b")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    parser.add_argument("--disable-rerank", action="store_true")
    args = parser.parse_args()
    if args.model:
        global DEFAULT_MODEL
        DEFAULT_MODEL = args.model
    global ENABLE_RERANK
    ENABLE_RERANK = not bool(args.disable_rerank)
    global EMBED_DEVICE
    EMBED_DEVICE = args.device

    load_indexes()

    with open(args.input, "r", encoding="utf-8") as f:
        raw = json.load(f)
    queries = raw if isinstance(raw, list) else raw.get("queries", [raw.get("query", "")])

    results = []
    for i, q in enumerate(queries, 1):
        q = q.strip()
        if not q: continue
        print(f"\n[Query {i}/{len(queries)}] {q[:80]}")
        try:
            result = format_output(q)
            print(f"    -> {len(result['retrieved_standards'])} results, latency={result['latency_seconds']}s")
            results.append(result)
        except Exception as e:
            print(f"    [ERROR] {e}", file=sys.stderr)
            results.append({"id": q[:80], "retrieved_standards": [], "latency_seconds": 5.0, "error": str(e)})

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    total = len(results)
    ok = sum(1 for r in results if r.get("retrieved_standards"))
    avg = sum(r.get("latency_seconds", 0) for r in results) / total if total else 0
    print(f"\n[OK] Saved to {args.output} ({ok}/{total} with results, avg={avg:.2f}s)")

if __name__ == "__main__":
    main()
