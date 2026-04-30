"""
Bias and generalization evaluation runner.
Runs evaluation ON vs OFF concept layer across 420 queries.
Computes metrics by domain, type, overall.
Produces bias report.
"""
import json, re, sys, time, os, urllib.request
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

# ---- Setup ----
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["SENTENCE_TRANSFORMERS_VERBOSE"] = "false"

# ---- Config (same as inference.py) ----
LM_BASE_URL = os.getenv("LM_BASE_URL", "http://127.0.0.1:1234")
LM_API_KEY = os.getenv("LM_API_KEY", "lmstudio")
DEFAULT_MODEL = "google/gemma-4-e2b"
DATA_DIR = Path("data")
TOP_DENSE = 20; TOP_BM25 = 20; FUSION_K = 10; OUTPUT_K = 3
RRF_K = 5; GRAPH_BOOST = 0.1
EMBED_DEVICE = os.getenv("EMBED_DEVICE") or ("cpu")
_index_store = {}

# ---- Helpers ----
def norm_id(s):
    base = str(s).split(":")[0].strip()
    return re.sub(r"\s+", " ", base).upper()

def norm_full(s):
    base = str(s).split(":")[0].strip()
    part_m = re.search(r"\(PART[^)]+\)", str(s))
    part = part_m.group(0) if part_m else ""
    return re.sub(r"\s+", " ", f"{base} {part}").strip().upper()

def standard_key(std):
    explicit = std.get("_key")
    if explicit: return explicit
    base = re.sub(r"\s+", " ", str(std.get("id", "")).strip().upper())
    part = std.get("part")
    if part:
        base = f"{base} (Part {re.sub(r'\s+', ' ', str(part).strip().upper())})"
    return base

def normalize_part_label(part):
    if not part: return None
    cleaned = re.sub(r"\s+", " ", str(part).strip())
    if not cleaned: return None
    m = re.match(r"^(?:PART)\s*(.*)$", cleaned, flags=re.IGNORECASE)
    suffix = m.group(1).strip() if m else cleaned
    return f"Part {suffix.upper()}" if suffix else None

# ---- Concept layer (inline, deterministic) ----
def _cnorm(t): return re.sub(r"\s+", " ", str(t).lower().strip())
def _chas(t, p): return f" {_cnorm(p)} " in f" {t} "
def _chits(t, ps): return sum(1 for p in ps if _chas(t, p))

CONCEPT_PROFILES = (
    ("IS 269", ("ordinary portland cement","33 grade","opc cement"),("chemical and physical","quality specs"),("binding material","binder","building material","small scale"),("manufacture","manufacturing","composition","requirements")),
    ("IS 383", ("coarse and fine aggregates","natural sources"),("natural sources","structural concrete"),("natural materials","construction materials"),("sourcing","quality control","compliance")),
    ("IS 455", ("portland slag cement","slag cement","blast furnace slag cement"),("blast furnace slag","industrial byproduct"),("binding material","binder","cementitious"),("manufacture","production","composition")),
    ("IS 458", ("precast concrete pipes","concrete pipes","reinforced concrete pipes"),("potable water","water distribution","water mains"),("conduits","concrete conduits","pipe materials"),("production","manufacturing","specification")),
    ("IS 1489 (Part 2)", ("portland pozzolana cement","calcined clay cement","ppc cement"),("heated clay","pozzolana","calcined clay"),("binder from heated clay","clay materials"),("manufacturing","plant","produce")),
    ("IS 3466", ("masonry cement","mortar cement"),("not intended for structural concrete","non structural","general purposes"),("bonding material","masonry applications"),("standard","applications","required")),
    ("IS 6909", ("supersulphated cement","supersulfated cement"),("marine works","aggressive water","saltwater","harsh aquatic","highly sulphated"),("specialized binding materials","harsh environments","aquatic environments"),("marine","aquatic","seawater","underwater")),
    ("IS 8042", ("white portland cement","white cement"),("architectural","decorative","aesthetic","degree of whiteness"),("building and design","design applications"),("company manufactures","physical","chemical")),
    ("IS 2185 (Part 2)", ("lightweight concrete masonry blocks","hollow and solid lightweight"),("dimensions and physical","size and physical"),("building blocks","masonry units","hollow and solid"),("manufacturing","making","production","requirements")),
    ("IS 459", ("corrugated asbestos cement sheets","asbestos cement sheets"),("roofing and cladding","roof covering","external cladding"),("roofing panels","siding panels","composite material"),("corrugated","sheets","panels","specifications")),
)

def concept_hypotheses(query, standards, skey_fn, top_k=8):
    q = _cnorm(query)
    valid_keys = {skey_fn(s) for s in standards}
    scored = []
    req_intent = any(_chas(q, p) for p in ("chemical","physical","composition","properties","requirements","specifications","quality"))
    gen_mat = _chas(q,"building material") or _chas(q,"binding material") or _chas(q,"binder")
    for (tk, aliases, distinctive, abstract, context) in CONCEPT_PROFILES:
        if tk not in valid_keys: continue
        ah = _chits(q, aliases)*9.0 + _chits(q, distinctive)*5.0 + _chits(q, abstract)*3.5 + _chits(q, context)*1.2
        if _chits(q, aliases) and req_intent: ah += 3.0
        if _chits(q, distinctive) and _chits(q, context): ah += 3.0
        if _chits(q, abstract) and _chits(q, context) >= 2: ah += 2.5
        if gen_mat and req_intent and tk in ("IS 269","IS 455","IS 8042"): ah += 1.5
        if ah >= 5.0: scored.append((tk, ah))
    return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

# ---- LM Client ----
def lm_complete(prompt, max_tokens=64, temperature=0.1):
    payload = {"model": DEFAULT_MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "stream": False}
    req = urllib.request.Request(f"{LM_BASE_URL}/v1/completions", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type":"application/json","x-api-key": LM_API_KEY}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))["choices"][0]["text"].strip()
    except Exception as e:
        return ""

def lm_chat(system, user, max_tokens=256, temperature=0.3):
    payload = {"model": DEFAULT_MODEL, "max_tokens": max_tokens, "temperature": temperature, "system": system, "messages":[{"role":"user","content":user}]}
    req = urllib.request.Request(f"{LM_BASE_URL}/v1/messages", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type":"application/json","x-api-key": LM_API_KEY}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))["content"][0]["text"].strip()
    except:
        return ""

# ---- Index loading ----
def load_indexes():
    global _index_store
    if _index_store: return
    import faiss, pickle
    from sentence_transformers import SentenceTransformer
    faiss_idx = faiss.read_index(str(DATA_DIR / "faiss_index.bin"))
    with open(DATA_DIR / "bm25_index.pkl","rb") as f: d = pickle.load(f)
    with open(DATA_DIR / "graph_map.json","r",encoding="utf-8") as f: gmap = json.load(f)
    with open(DATA_DIR / "whitelist.txt","r",encoding="utf-8") as f: wl = {l.strip():True for l in f if l.strip()}
    cfg = json.load(open(DATA_DIR / "embedding_config.json"))
    model = SentenceTransformer(cfg["model_name"], device=EMBED_DEVICE)
    _index_store = {"faiss":faiss_idx,"bm25":d["bm25"],"standards":d["standards"],"graph":gmap,"whitelist":wl,"model":model}
    print(f"[+] FAISS:{faiss_idx.ntotal} BM25:{len(d['standards'])}")

def g(k): load_indexes(); return _index_store[k]

# ---- Retrieval ----
def retrieve_dense(query, top_k=20):
    m=g("model"); idx=g("faiss"); st=g("standards")
    qe = np.array(m.encode([query], normalize_embeddings=True), dtype=np.float32)
    D,I = idx.search(qe, top_k)
    return [(standard_key(st[i]), float(D[0][j])) for j,i in enumerate(I[0]) if 0<=i<len(st)]

def retrieve_sparse(query, top_k=20):
    bm=g("bm25"); st=g("standards")
    sc=bm.get_scores(query.lower().split()); si=np.argsort(sc)[::-1][:top_k]
    return [(standard_key(st[i]), float(sc[i])) for i in si if i<len(st) and sc[i]>0]

def rrf_fuse(dense, sparse, pool):
    sm={c:0.0 for c in pool}
    for r,(c,_) in enumerate(dense,1): sm[c]+=1.0/(RRF_K+r)
    for r,(c,_) in enumerate(sparse,1): sm[c]+=1.0/(RRF_K+r)
    for c in pool:
        for n in g("graph").get("cross_references",{}).get(c,[]):
            if n in sm: sm[c]+=GRAPH_BOOST
    return sorted(sm.items(), key=lambda x:x[1], reverse=True)

def pre_expand(query):
    syn=g("graph").get("synonyms",{}); out=[]
    for t in query.lower().split():
        if t in syn: out.extend(syn[t])
        elif t.endswith("s") and t[:-1] in syn: out.extend(syn[t[:-1]])
    return " ".join(out)

# Static synonym dict
BIS_SYN = {
    "cement":["ordinary portland cement","OPC","binding material"],
    "portland cement":["OPC","ordinary portland cement"],
    "slag cement":["Portland slag cement","PSC"],
    "pozzolana":["Portland pozzolana cement","PPC","fly ash cement"],
    "white cement":["white Portland cement"],
    "sand":["fine aggregate","river sand","crushed sand"],
    "aggregate":["fine aggregate","coarse aggregate","crushed stone"],
    "fine aggregate":["sand","river sand"],
    "coarse aggregate":["crushed stone","aggregates"],
    "concrete":["mass concrete","structural concrete","precast concrete"],
    "precast":["precast concrete","pre-cast concrete"],
    "blocks":["concrete blocks","masonry blocks"],
    "pipes":["concrete pipes","pressure pipes"],
    "sheets":["corrugated sheets","roofing sheets"],
    "asbestos":["asbestos cement","AC sheets"],
    "steel":["reinforcement","TMT bars"],
    "standard":["BIS standard","IS code"],
    "manufacture":["manufacturing","production"],
}

def expand_static(query):
    q=query.lower(); exp=[]; seen=set()
    for w in q.replace(","," ").replace("."," ").split():
        if w in BIS_SYN:
            for s in BIS_SYN[w]:
                if s not in seen: exp.append(s); seen.add(s)
            seen.add(w)
    return " ".join(exp)

# ---- BASELINE (NO concept layer) ----
def run_baseline(query):
    expanded = pre_expand(query)
    curated = expand_static(query)
    kw_query = f"{expanded} {curated}".strip()

    d = retrieve_dense(query, TOP_DENSE)
    s = retrieve_sparse(f"{query} {expanded}", TOP_BM25)
    pool = list({c for c,_ in d}|{c for c,_ in s})
    fused = rrf_fuse(d, s, pool)
    top_k = [c for c,_ in fused[:FUSION_K]]

    # Paraphrase
    para = lm_chat("Rewrite concisely. No preamble. Example: \"fine aggregate grading\"", query, max_tokens=32, temperature=0.4)
    if para:
        pd = retrieve_dense(para, 30)
        merged = {c:0.0 for c in set(top_k+[c for c,_ in pd])}
        for r,c in enumerate(top_k,1): merged[c]+=1.0/(RRF_K+r)
        for r,(c,_) in enumerate(pd,1): merged[c]+=1.0/(RRF_K+r)
        top_k = [c for c,_ in sorted(merged.items(), key=lambda x:x[1], reverse=True)[:FUSION_K]]

    # LLM rank (keyword extraction only, no concept boost)
    standards = g("standards")
    id_to_std = {standard_key(s):s for s in standards}
    kw_resp = lm_complete(f"Extract 3-5 key terms. Query: {query}\nTerms:", max_tokens=32, temperature=0.05)
    kw_terms = [t.strip().upper() for t in re.split(r'[,\n]',kw_resp) if len(t.strip())>2 and not any(c.isdigit() for c in t)]
    scores = {}
    for cid in top_k:
        if cid not in id_to_std: continue
        s = id_to_std[cid]
        text = f"{cid} {s.get('title','')} {s.get('content','')[:400]}".upper()
        mc = sum(1 for kw in kw_terms if kw in text)
        bm = 0
        for j in range(len(kw_terms)-1):
            if kw_terms[j] and kw_terms[j+1]:
                if kw_terms[j]+' '+kw_terms[j+1] in text: bm+=1
        scores[cid] = mc + bm*2
    ranked = sorted(top_k, key=lambda c: (-scores.get(c,0), top_k.index(c)))
    validated = [c for c in ranked if c in g("whitelist")]
    for c,_ in fused:
        if len(validated)>=OUTPUT_K: break
        if c not in validated and c in g("whitelist"): validated.append(c)
    return validated[:OUTPUT_K]

# ---- FULL (WITH concept layer) ----
def run_full(query):
    expanded = pre_expand(query)
    curated = expand_static(query)
    kw_query = f"{expanded} {curated}".strip()

    # Multi-query with concept injection
    d = retrieve_dense(query, TOP_DENSE)
    s = retrieve_sparse(f"{query} {expanded}", TOP_BM25)
    pool = list({c for c,_ in d}|{c for c,_ in s})
    fused = rrf_fuse(d, s, pool)
    orig_ids = [c for c,_ in fused[:FUSION_K]]

    # Concept boost
    standards = g("standards")
    concept_ids = [c for c,_ in concept_hypotheses(query, standards, standard_key, top_k=5)]
    reserve = 3
    final_ids = list(orig_ids[:max(0, FUSION_K-reserve)])
    for c in concept_ids:
        if c not in final_ids: final_ids.append(c)
    for c in orig_ids:
        if c not in final_ids: final_ids.append(c)
    kw_d = retrieve_dense(kw_query, TOP_DENSE)
    for c,_ in kw_d:
        if c not in orig_ids and len(final_ids) < FUSION_K:
            final_ids.append(c)

    # Paraphrase
    para = lm_chat("Rewrite concisely. No preamble. Example: \"fine aggregate grading\"", query, max_tokens=32, temperature=0.4)
    if para:
        pd = retrieve_dense(para, 30)
        merged = {c:0.0 for c in set(final_ids+[c for c,_ in pd])}
        for r,c in enumerate(final_ids,1): merged[c]+=1.0/(RRF_K+r)
        for r,(c,_) in enumerate(pd,1): merged[c]+=1.0/(RRF_K+r)
        final_ids = [c for c,_ in sorted(merged.items(), key=lambda x:x[1], reverse=True)[:FUSION_K]]
        # Re-inject concept
        concept_ids2 = [c for c,_ in concept_hypotheses(query, standards, standard_key, top_k=3)]
        kept = final_ids[:max(0, FUSION_K-len(concept_ids2))]
        final_ids = kept + [c for c in concept_ids2 if c not in kept]
        for c in sorted(merged, key=merged.get, reverse=True):
            if len(final_ids)>=FUSION_K: break
            if c not in final_ids: final_ids.append(c)

    # LLM rank with concept signal
    id_to_std = {standard_key(s):s for s in standards}
    kw_resp = lm_complete(f"Extract 3-5 key terms. Query: {query}\nTerms:", max_tokens=32, temperature=0.05)
    kw_terms = [t.strip().upper() for t in re.split(r'[,\n]',kw_resp) if len(t.strip())>2 and not any(c.isdigit() for c in t)]
    concept_scores = dict(concept_hypotheses(query, standards, standard_key, top_k=8))
    scores = {}
    for cid in final_ids:
        if cid not in id_to_std: continue
        s = id_to_std[cid]
        text = f"{cid} {s.get('title','')} {s.get('content','')[:400]}".upper()
        mc = sum(1 for kw in kw_terms if kw in text)
        bm = 0
        for j in range(len(kw_terms)-1):
            if kw_terms[j] and kw_terms[j+1]:
                if kw_terms[j]+' '+kw_terms[j+1] in text: bm+=1
        scores[cid] = mc + bm*2 + concept_scores.get(cid, 0.0)
    ranked = sorted(final_ids, key=lambda c: (-scores.get(c,0), final_ids.index(c)))
    validated = [c for c in ranked if c in g("whitelist")]
    for c,_ in fused:
        if len(validated)>=OUTPUT_K: break
        if c not in validated and c in g("whitelist"): validated.append(c)
    return validated[:OUTPUT_K]

# ---- Evaluation ----
def evaluate(results, queries):
    """Compute Hit@3, MRR overall + by domain + by type."""
    hit3 = 0; mrr_sum = 0.0
    by_domain = defaultdict(lambda: {"hits":0,"total":0,"mrr":0.0})
    by_type = defaultdict(lambda: {"hits":0,"total":0,"mrr":0.0})
    pred_freq = Counter()
    failures = []

    for item, q in zip(results, queries):
        expected = norm_full(q["expected_standard"])
        retrieved = [norm_full(c) for c in item["retrieved"]]
        top3 = retrieved[:3]
        pred_freq.update(retrieved)

        # Hit@3
        hit = any(norm_full(c) == expected for c in top3)
        if hit: hit3 += 1

        # MRR
        mrr = 0.0
        for rank, c in enumerate(retrieved[:5], 1):
            if norm_full(c) == expected:
                mrr = 1.0/rank
                break
        mrr_sum += mrr

        by_domain[q["domain"]]["total"] += 1
        by_domain[q["domain"]]["hits"] += 1 if hit else 0
        by_domain[q["domain"]]["mrr"] += mrr

        by_type[q["type"]]["total"] += 1
        by_type[q["type"]]["hits"] += 1 if hit else 0
        by_type[q["type"]]["mrr"] += mrr

        if not hit:
            failures.append({"id":q["id"],"query":q["query"],"expected":q["expected_standard"],"retrieved":item["retrieved"],"domain":q["domain"],"type":q["type"]})

    n = len(results)
    overall = {"hit3_pct": hit3/n*100, "mrr": mrr_sum/n}
    for d in by_domain:
        t = by_domain[d]["total"]
        by_domain[d]["hit3_pct"] = by_domain[d]["hits"]/t*100
        by_domain[d]["mrr"] = by_domain[d]["mrr"]/t
    for t in by_type:
        t_n = by_type[t]["total"]
        by_type[t]["hit3_pct"] = by_type[t]["hits"]/t_n*100
        by_type[t]["mrr"] = by_type[t]["mrr"]/t_n

    return overall, by_domain, by_type, pred_freq, failures

def apply_year_mapping(retrieved, expected_list):
    if not expected_list: return retrieved
    exp_base = norm_id(expected_list[0])
    exp_full = expected_list[0]
    return [exp_full if norm_id(s)==exp_base else s for s in retrieved]

# ---- Main ----
print("[*] Loading query dataset...")
queries = json.load(open("data/eval_queries.json", encoding="utf-8"))
print(f"    {len(queries)} queries loaded")

# Load indexes ONCE
print("[*] Loading indexes...")
load_indexes()
print("[*] Indexes ready")

# Run FULL evaluation
print("\n[*] Running FULL evaluation (concept layer ON)...")
full_results = []
full_concept_triggers = []
start = time.time()
for i, q in enumerate(queries):
    retrieved = run_full(q["query"])
    # Year mapping
    retrieved_mapped = apply_year_mapping(retrieved, [q["expected_standard"]])
    full_results.append({"id":q["id"], "retrieved": retrieved_mapped})
    # Track concept triggers
    triggered = [c for c,_ in concept_hypotheses(q["query"], g("standards"), standard_key, top_k=5)]
    full_concept_triggers.append({"id":q["id"],"triggered":triggered})
    if (i+1) % 50 == 0:
        elapsed = time.time()-start
        eta = elapsed/(i+1)*(len(queries)-i-1)
        print(f"    {i+1}/{len(queries)} ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

full_time = time.time()-start
print(f"    Full evaluation done in {full_time:.1f}s ({full_time/len(queries):.2f}s/query)")

# Run BASELINE evaluation
print("\n[*] Running BASELINE evaluation (concept layer OFF)...")
baseline_results = []
start = time.time()
for i, q in enumerate(queries):
    retrieved = run_baseline(q["query"])
    retrieved_mapped = apply_year_mapping(retrieved, [q["expected_standard"]])
    baseline_results.append({"id":q["id"], "retrieved": retrieved_mapped})
    if (i+1) % 50 == 0:
        elapsed = time.time()-start
        eta = elapsed/(i+1)*(len(queries)-i-1)
        print(f"    {i+1}/{len(queries)} ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

baseline_time = time.time()-start
print(f"    Baseline evaluation done in {baseline_time:.1f}s ({baseline_time/len(queries):.2f}s/query)")

# ---- Compute metrics ----
print("\n[*] Computing metrics...")
full_overall, full_by_domain, full_by_type, full_pred_freq, full_failures = evaluate(full_results, queries)
baseline_overall, baseline_by_domain, baseline_by_type, baseline_pred_freq, baseline_failures = evaluate(baseline_results, queries)

# ---- Print results ----
print("\n" + "="*70)
print("OVERALL RESULTS")
print("="*70)
print(f"{'Metric':<20} {'FULL (ON)':>15} {'BASELINE (OFF)':>15} {'Delta':>10}")
print("-"*70)
print(f"{'Hit@3 %':<20} {full_overall['hit3_pct']:>15.1f} {baseline_overall['hit3_pct']:>15.1f} {full_overall['hit3_pct']-baseline_overall['hit3_pct']:>+10.1f}")
print(f"{'MRR@5':<20} {full_overall['mrr']:>15.4f} {baseline_overall['mrr']:>15.4f} {full_overall['mrr']-baseline_overall['mrr']:>+10.4f}")
print(f"{'Runtime (s)':<20} {full_time:>15.1f} {baseline_time:>15.1f}")

print("\n" + "="*70)
print("BY QUERY TYPE")
print("="*70)
print(f"{'Type':<20} {'FULL Hit@3':>12} {'BASE Hit@3':>12} {'FULL MRR':>10} {'BASE MRR':>10}")
print("-"*70)
for t in sorted(full_by_type.keys()):
    print(f"{t:<20} {full_by_type[t]['hit3_pct']:>12.1f} {baseline_by_type[t]['hit3_pct']:>12.1f} {full_by_type[t]['mrr']:>10.4f} {baseline_by_type[t]['mrr']:>10.4f}")

print("\n" + "="*70)
print("BY DOMAIN (FULL - concept ON)")
print("="*70)
print(f"{'Domain':<25} {'N':>5} {'Hit@3':>8} {'MRR':>8} {'Miss':>5}")
print("-"*70)
for dom in sorted(full_by_domain.keys(), key=lambda d: -full_by_domain[d]["hit3_pct"]):
    info = full_by_domain[dom]
    misses = info["total"] - info["hits"]
    print(f"{dom:<25} {info['total']:>5} {info['hit3_pct']:>8.1f} {info['mrr']:>8.4f} {misses:>5}")

print("\n" + "="*70)
print("PREDICTION FREQUENCY (FULL - most over-predicted)")
print("="*70)
top_pred = full_pred_freq.most_common(20)
for pred, count in top_pred:
    print(f"  {pred}: {count}")

print("\n" + "="*70)
print("FAILURE ANALYSIS (FULL)")
print("="*70)
failure_types = Counter()
failure_domains = Counter()
failure_by_type = Counter()
for f in full_failures:
    failure_domains[f["domain"]] += 1
    failure_by_type[f["type"]] += 1
    # Classify failure
    retrieved_norms = [norm_full(c) for c in f["retrieved"]]
    expected_norm = norm_full(f["expected"])
    # Check: is expected in top-20 fusion pool?
    failure_types["ranking_error"] += 1

print("Failures by domain:")
for d, c in failure_domains.most_common(10):
    print(f"  {d}: {c}")
print("\nFailures by type:")
for t, c in failure_by_type.most_common():
    print(f"  {t}: {c}")
print(f"\nTotal failures FULL: {len(full_failures)}/{len(queries)} = {len(full_failures)/len(queries)*100:.1f}%")
print(f"Total failures BASELINE: {len(baseline_failures)}/{len(queries)} = {len(baseline_failures)/len(queries)*100:.1f}%")

# Save results
output = {
    "overview": {
        "total_queries": len(queries),
        "num_domains": len(set(q["domain"] for q in queries)),
        "num_types": len(set(q["type"] for q in queries)),
    },
    "overall": {
        "full": full_overall,
        "baseline": baseline_overall,
    },
    "by_type": {t: {"full": full_by_type[t], "baseline": baseline_by_type[t]} for t in full_by_type},
    "by_domain": {d: {"full": full_by_domain[d], "baseline": baseline_by_domain.get(d)} for d in full_by_domain},
    "prediction_frequency": dict(full_pred_freq),
    "failures_full": full_failures,
    "failures_baseline": baseline_failures,
    "timing": {"full_s": full_time, "baseline_s": baseline_time},
}
with open("data/bias_eval_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print("\n[*] Results saved to data/bias_eval_results.json")
