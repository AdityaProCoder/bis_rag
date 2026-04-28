"""Diagnostic: check if correct standards appear in top-20/top-50 for each failure."""
import os, json, re, urllib.request
from pathlib import Path
import warnings as _w; _w.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["SENTENCE_TRANSFORMERS_VERBOSE"] = "false"
import torch; _ = torch.cuda.is_available()
from sentence_transformers import SentenceTransformer
import numpy as np, faiss, pickle

LM_BASE_URL = "http://127.0.0.1:1234"
LM_API_KEY = os.getenv("LM_API_KEY", "lmstudio")
DATA_DIR = Path("data")
TOP_DENSE = 20; TOP_BM25 = 20; RRF_K = 5; GRAPH_BOOST = 0.1
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

def lm_complete(prompt, max_tokens=64, temperature=0.05):
    payload = {"model": "google/gemma-4-e2b", "prompt": prompt, "max_tokens": max_tokens,
               "temperature": temperature, "stream": False}
    req = urllib.request.Request(f"{LM_BASE_URL}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-api-key": LM_API_KEY}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))["choices"][0]["text"].strip()
    except: return ""

def lm_chat(system_prompt, user_message, max_tokens=256, temperature=0.3):
    payload = {"model": "google/gemma-4-e2b", "max_tokens": max_tokens, "temperature": temperature,
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
    cfg = json.load(open(DATA_DIR / "embedding_config.json"))
    model = SentenceTransformer(cfg["model_name"], device="cuda")
    _index_store = {"faiss": faiss_idx, "bm25": d["bm25"], "standards": d["standards"],
                   "graph": graph_map, "embed_model": model}

def g(k): load_indexes(); return _index_store[k]

def pre_retrieval_expand(query):
    syn = g("graph").get("synonyms", {})
    out = []
    for t in query.lower().split():
        if t in syn: out.extend(syn[t])
        elif t.endswith("s") and t[:-1] in syn: out.extend(syn[t[:-1]])
    return " ".join(out)

def retrieve_dense(query, top_k=50):
    m = g("embed_model"); idx = g("faiss"); st = g("standards")
    qe = np.array(m.encode([query], normalize_embeddings=True), dtype=np.float32)
    D, I = idx.search(qe, top_k)
    return [(standard_key(st[i]), float(D[0][j]), j) for j,i in enumerate(I[0]) if 0<=i<len(st)]

def retrieve_sparse(query, top_k=50):
    bm = g("bm25"); st = g("standards")
    sc = bm.get_scores(query.lower().split()); si = np.argsort(sc)[::-1][:top_k]
    return [(standard_key(st[i]), float(sc[i]), j) for j,i in enumerate(si) if i<len(st) and sc[i]>0]

def rrf_fusion(dense_results, sparse_results, candidate_pool, k_vals=None):
    if k_vals is None: k_vals = {c: RRF_K for c in candidate_pool}
    sm = {c: 0.0 for c in candidate_pool}
    for r, (c, _, _) in enumerate(dense_results, 1): sm[c] += 1.0 / (k_vals.get(c, RRF_K) + r)
    for r, (c, _, _) in enumerate(sparse_results, 1): sm[c] += 1.0 / (k_vals.get(c, RRF_K) + r)
    return sorted(sm.items(), key=lambda x: x[1], reverse=True)

# Static synonym dictionary for BIS domain
BIS_SYNONYMS = {
    # Cement types
    "cement": ["hydraulic cement", "OPC", "ordinary portland cement", "binding material"],
    "portland cement": ["OPC", "ordinary portland cement", "Portland cement"],
    "ordinary portland cement": ["OPC", "33 grade", "43 grade", "53 grade"],
    "slag cement": ["Portland slag cement", "PSC", "blast furnace slag"],
    "pozzolana": ["pozzolanic cement", "Portland pozzolana cement", "PPC", "fly ash", "calcined clay"],
    "white cement": ["white Portland cement", "architectural cement"],
    "supersulphated": ["supersulphated cement", "SSC"],
    "masonry cement": ["cement for masonry", " mortar cement"],
    # Aggregates
    "aggregate": ["fine aggregate", "coarse aggregate", "crushed stone"],
    "fine aggregate": ["sand", "river sand", "crushed sand", "M sand"],
    "coarse aggregate": ["crushed stone", "aggregates", "stone aggregate"],
    "sand": ["fine aggregate", "river sand", "crushed sand"],
    # Concrete
    "concrete": ["mass concrete", "structural concrete", "precast concrete", " RCC"],
    "precast": ["precast concrete", "pre-cast"],
    "masonry": ["brick masonry", "stone masonry", "hollow blocks"],
    "blocks": ["concrete blocks", "masonry blocks", "hollow blocks"],
    # Materials
    "steel": ["reinforcement", "TMT bars", "deformed bars", "mild steel"],
    "pipes": ["concrete pipes", "pressure pipes", " Hume pipes"],
    "sheets": ["corrugated sheets", "roofing sheets", "cladding"],
    "asbestos": ["asbestos cement", "AC sheets"],
    # Standards
    "specification": ["IS code", "BIS standard", "Indian Standard"],
    "standard": ["IS code", "BIS standard", "Indian Standard", "IS standard"],
    # Processes
    "manufacture": ["manufacturing", "production", "making"],
    "testing": ["test methods", "testing procedures"],
    "composition": ["chemical composition", "makeup"],
}

def expand_query_synonyms(query):
    """Expand query with BIS domain synonyms."""
    q_lower = query.lower()
    expanded = []
    seen = set()
    words = q_lower.replace(",", " ").replace(".", " ").split()
    for word in words:
        if word in seen: continue
        if word in BIS_SYNONYMS:
            for syn in BIS_SYNONYMS[word]:
                if syn not in seen:
                    expanded.append(syn)
                    seen.add(syn)
            seen.add(word)
        else:
            seen.add(word)
    return " ".join(expanded)

def llm_extract_keywords(query):
    """Extract key technical terms from query via LLM."""
    prompt = (f"Extract 5-8 most important technical terms from this BIS query.\n"
               f"Query: {query}\nTerms:")
    resp = lm_complete(prompt, max_tokens=48, temperature=0.05)
    terms = [t.strip().upper() for t in re.split(r'[,\n]', resp) if len(t.strip()) > 2]
    terms = [t for t in terms if not any(c.isdigit() for c in t)]
    return terms

def run_diagnostic(query, expected_norm, label=""):
    """Run full diagnostic for a query."""
    print(f"\n{'='*70}")
    print(f"  {label}: {query[:60]}")
    print(f"  Expected: {expected_norm}")

    # Standard expansion
    expanded = pre_retrieval_expand(query)
    synonyms = expand_query_synonyms(query)
    llm_kw = llm_extract_keywords(query)

    combined_query = f"{query} {expanded} {synonyms} {' '.join(llm_kw)}".strip()

    print(f"  Graph expand: '{expanded}'")
    print(f"  Synonyms: '{synonyms[:80]}'")
    print(f"  LLM kw: {llm_kw[:5]}")
    print(f"  Combined: '{combined_query[:80]}'")

    # Retrieve with different query versions
    dense_orig = retrieve_dense(query, 50)
    dense_combined = retrieve_dense(combined_query, 50)
    sparse_orig = retrieve_sparse(query, 50)
    sparse_combined = retrieve_sparse(combined_query, 50)

    # Check fusion at different pool sizes
    pool_orig = list({cid for cid,_,_ in dense_orig} | {cid for cid,_,_ in sparse_orig})
    fused_orig = rrf_fusion(dense_orig, sparse_orig, pool_orig)
    fused_orig_top20 = [cid for cid,_ in fused_orig[:20]]
    fused_orig_top10 = [cid for cid,_ in fused_orig[:10]]

    pool_combined = list({cid for cid,_,_ in dense_combined} | {cid for cid,_,_ in sparse_combined})
    fused_combined = rrf_fusion(dense_combined, sparse_combined, pool_combined)
    fused_combined_top20 = [cid for cid,_ in fused_combined[:20]]
    fused_combined_top10 = [cid for cid,_ in fused_combined[:10]]

    # Check positions
    def get_pos(lst, target):
        normed = [norm(c) for c in lst]
        t = norm(target)
        if t in normed: return normed.index(t) + 1
        return 0

    pos_orig_10 = get_pos(fused_orig_top10, expected_norm)
    pos_orig_20 = get_pos(fused_orig_top20, expected_norm)
    pos_combined_10 = get_pos(fused_combined_top10, expected_norm)
    pos_combined_20 = get_pos(fused_combined_top20, expected_norm)

    print(f"\n  Positions (norm={expected_norm}):")
    print(f"    Original query  top10: {pos_orig_10} | top20: {pos_orig_20}")
    print(f"    Combined query  top10: {pos_combined_10} | top20: {pos_combined_20}")

    # Show top 10 for combined
    print(f"    Combined top10: {[cid for cid,_ in fused_combined[:10]]}")

    status = "OK" if pos_combined_10 <= 3 else ("TOP20" if pos_combined_20 <= 20 else "FAIL")
    print(f"  Status: {status}")
    return pos_orig_10, pos_orig_20, pos_combined_10, pos_combined_20

def main():
    load_indexes()

    # Load paraphrased queries and failures
    para_data = json.load(open(DATA_DIR / "paraphrased_queries.json"))
    stress_results = json.load(open(DATA_DIR / "stress_test_results.json"))

    # Find failures from last run
    para_results = stress_results["paraphrased"]["queries"]
    failures = [r for r in para_results if not r["hit"]]

    print("="*70)
    print("  RETRIEVAL FAILURE DIAGNOSTIC")
    print("="*70)

    all_pos = {"orig10": [], "orig20": [], "combined10": [], "combined20": []}

    for r in failures:
        orig_id = r["id"].rsplit("_", 1)[0]
        expected = r["expected"][0]
        expected_norm = norm(expected)

        # Find original query
        orig_query = None
        for group in para_data:
            if group["original_id"] == orig_id:
                orig_query = group["original_query"]
                break

        p1, p2, p3, p4 = run_diagnostic(r["query"], expected_norm, label=r["id"])
        all_pos["orig10"].append(p1)
        all_pos["orig20"].append(p2)
        all_pos["combined10"].append(p3)
        all_pos["combined20"].append(p4)

    print("\n" + "="*70)
    print("  AGGREGATE DIAGNOSTIC")
    print("="*70)
    for k, vals in all_pos.items():
        in10 = sum(1 for v in vals if 1 <= v <= 10)
        in20 = sum(1 for v in vals if 1 <= v <= 20)
        hit = sum(1 for v in vals if v > 0)
        print(f"  {k}: in_top10={in10}/{len(vals)} in_top20={in20}/{len(vals)} found={hit}/{len(vals)}")

if __name__ == "__main__":
    main()
