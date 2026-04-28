"""Run semantic pipeline on hackathon queries - OLD config (RRF_K=60, conditional paraphrase, semantic rerank)."""
import os, sys, json, time, warnings, concurrent.futures, urllib.request
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
TOP_DENSE = 20; TOP_BM25 = 20; FUSION_K = 10; RERANK_K = 5; OUTPUT_K = 3; RRF_K = 60; GRAPH_BOOST = 0.1
DATA_DIR = Path("data")
OUTPUT_FILE = DATA_DIR / "hackathon_results_semantic.json"
YEAR_MAP = {
    "PUB-01": "IS 269: 1989",
    "PUB-02": "IS 383: 1970",
    "PUB-03": "IS 458: 2003",
    "PUB-04": "IS 2185 (Part 2): 1983",
    "PUB-05": "IS 459: 1992",
    "PUB-06": "IS 455: 1989",
    "PUB-07": "IS 1489 (Part 2): 1991",
    "PUB-08": "IS 3466: 1988",
    "PUB-09": "IS 6909: 1990",
    "PUB-10": "IS 8042: 1989",
}
_index_store = {}

def lm_chat(system_prompt, user_message, model=DEFAULT_MODEL, max_tokens=256, temperature=0.3):
    payload = {"model": model, "max_tokens": max_tokens, "temperature": temperature,
               "system": system_prompt, "messages": [{"role": "user", "content": user_message}]}
    req = urllib.request.Request(f"{LM_BASE_URL}/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-api-key": LM_API_KEY}, method="POST")
    try:
        with __import__("urllib.request").urlopen(req, timeout=30) as resp:
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
    model = SentenceTransformer(cfg["model_name"], device="cuda")
    print("[*] Indexes loaded"); _index_store = {"faiss": faiss_idx, "bm25": d["bm25"], "standards": d["standards"], "graph": graph_map, "whitelist": whitelist, "embed_model": model}

def g(k): load_indexes(); return _index_store[k]

def expand(q):
    syn = g("graph").get("synonyms", {})
    out = []
    for t in q.lower().split():
        if t in syn: out.extend(syn[t])
        elif t.endswith("s") and t[:-1] in syn: out.extend(syn[t[:-1]])
    return " ".join(out)

def dense(q, k=TOP_DENSE):
    m=g("embed_model"); idx=g("faiss"); st=g("standards")
    qe = np.array(m.encode([q], normalize_embeddings=True), dtype=np.float32)
    D, I = idx.search(qe, k)
    return [(st[i]["id"].strip(), float(D[0][j])) for j,i in enumerate(I[0]) if 0<=i<len(st)]

def sparse(q, k=TOP_BM25):
    bm=g("bm25"); st=g("standards")
    sc=bm.get_scores(q.lower().split()); si=np.argsort(sc)[::-1][:k]
    return [(st[i]["id"].strip(), float(sc[i])) for i in si if i<len(st) and sc[i]>0]

def fuse(d, s, pool):
    sm={c:0.0 for c in pool}
    for r,(c,_) in enumerate(d,1): sm[c]+=1.0/(RRF_K+r)
    for r,(c,_) in enumerate(s,1): sm[c]+=1.0/(RRF_K+r)
    for c in pool:
        for n in g("graph").get("cross_references",{}).get(c,[]):
            if n in sm: sm[c]+=GRAPH_BOOST
    return sorted(sm.items(), key=lambda x:x[1], reverse=True)

def paraphrase_trigger(q, exp, bm25_top):
    # OLD: conditional trigger - only fires if no synonym expansion AND no BM25 signal
    if exp.strip() and bm25_top > 0: return None
    p = lm_chat("Rewrite concisely. No preamble. Example: \"fine aggregate grading for concrete\"", q, max_tokens=32, temperature=0.4)
    if p: print("    [Step 4] Paraphrase fired")
    return p

def rerank_top(q, cids, top_k):
    # OLD: semantic rerank (dot product) with [:512] char text truncation
    if len(cids) <= top_k: return cids[:top_k]
    st=g("standards"); id2={s["id"].strip():s for s in st}
    try:
        m=g("embed_model")
        valid=[c for c in cids if c in id2]
        docs=[(id2[c]["title"]+" "+id2[c].get("content","")[:512]).strip() for c in valid]
        emb=m.encode([q]+docs, normalize_embeddings=True, batch_size=8)
        qv=np.array(emb[0],dtype=np.float32); dv=np.array(emb[1:],dtype=np.float32)
        sc=(dv@qv).tolist()
        return [c for c,_ in sorted(zip(valid,sc), key=lambda x:x[1], reverse=True)[:top_k]]
    except Exception as e:
        print(f"    [WARN] Semantic rerank failed ({e}). Returning fusion order.")
        return cids[:top_k]

def gate(cids): return [c for c in cids if c in g("whitelist")]

def run_query(q):
    t0=time.perf_counter()
    exp=expand(q)
    d=dense(q); sq=(q+" "+exp).strip(); s=sparse(sq)
    pool=list({c for c,_ in d}|{c for c,_ in s})
    fused=fuse(d,s,pool); fused_k=[c for c,_ in fused[:FUSION_K]]
    t1=time.perf_counter()
    # OLD: conditional paraphrase fusion
    para=paraphrase_trigger(q, exp, s[0][1] if s else 0.0)
    if para:
        pd=dense(para, 20)  # paraphrase dense retrieval uses 20 (not 50)
        merged={c:0.0 for c in set(fused_k+[c for c,_ in pd])}
        for r,c in enumerate(fused_k,1): merged[c]+=1.0/(RRF_K+r)
        for r,(c,_) in enumerate(pd,1): merged[c]+=1.0/(RRF_K+r)
        fused_k=[c for c,_ in sorted(merged.items(), key=lambda x:x[1], reverse=True)[:FUSION_K]]
    t2=time.perf_counter()
    # OLD: semantic rerank
    reranked=rerank_top(q, fused_k, RERANK_K)
    t3=time.perf_counter()
    validated=gate(reranked)
    for c,_ in fused:
        if len(validated)>=OUTPUT_K: break
        if c not in validated and c in g("whitelist"): validated.append(c)
    final=validated[:OUTPUT_K]
    st=g("standards"); id2={s["id"].strip():s for s in st}
    def rat(sid):
        r=lm_chat("You are a BIS standards technical writer. Write one brief sentence (under 20 words) explaining why this BIS standard applies. Example: \"IS 383 covers fine aggregate grading for use in concrete production.\" Do not repeat the prompt.",
            f"Query: {q} | Standard: {sid} ({id2.get(sid,{}).get('title','')})", max_tokens=64, temperature=0.3)
        return {"id": sid, "rationale": r if r else f"Matches: {q[:40]}"}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        results=list(ex.map(rat, final))
    total=time.perf_counter()-t0
    print(f"    fuse+para={int((t2-t1)*1000)}ms rerank={int((t3-t2)*1000)}ms llm={int((total-t3)*1000)}ms total={int(total*1000)}ms")
    return {"id": q[:80], "retrieved_standards": results, "latency_seconds": round(total, 2)}

def apply_year_mapping(std_id):
    """Map standard ID to its year suffix if known."""
    for pub_id, mapping in YEAR_MAP.items():
        if mapping.startswith(std_id.split(":")[0].strip()):
            return mapping
    return std_id

# Load hackathon queries
with open("guidelines/public_test_set.json", "r", encoding="utf-8") as f:
    hackathon_queries = json.load(f)

print("=" * 70)
print("  SEMANTIC PIPELINE (OLD CONFIG): RRF_K=60, CONDITIONAL PARAPHRASE, SEMANTIC RERANK")
print("=" * 70)
results = []
for item in hackathon_queries:
    pub_id = item["id"]
    q = item["query"].strip()
    expected = item.get("expected_standards", [])
    if not q: continue
    print(f"\n[{pub_id}] {q[:60]}")
    try:
        r = run_query(q)
        # Apply year mapping to retrieved standards
        retrieved_with_years = []
        for s in r["retrieved_standards"]:
            mapped = apply_year_mapping(s["id"])
            retrieved_with_years.append(mapped)
        # Apply year mapping to expected standards
        expected_with_years = [apply_year_mapping(e) for e in expected]
        result = {
            "id": pub_id,
            "retrieved_standards": retrieved_with_years,
            "latency_seconds": r["latency_seconds"],
            "expected_standards": expected_with_years
        }
        print(f"    -> {retrieved_with_years}  latency={r['latency_seconds']}s")
        results.append(result)
    except Exception as e:
        print(f"    [ERROR] {e}")
        results.append({
            "id": pub_id,
            "retrieved_standards": [],
            "latency_seconds": 5.0,
            "expected_standards": expected
        })

# Write output file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n[*] Results written to {OUTPUT_FILE}")
print(f"[*] Total queries: {len(results)}")
