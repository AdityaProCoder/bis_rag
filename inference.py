"""
Standalone BIS Hackathon Inference Script
Deterministic, leakage-free ranking with part/family disambiguation.
"""
import argparse
import json
import os
import pickle
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path(__file__).parent / "src" / "data"
LM_BASE_URL = os.getenv("LM_BASE_URL", "http://127.0.0.1:1234")
LM_API_KEY = os.getenv("LM_API_KEY", "lmstudio")
LM_MODEL = os.getenv("LM_MODEL", "qwen3.5:4b")

try:
    import torch

    if os.getenv("BIS_FORCE_CPU", "0") == "1":
        EMBED_DEVICE = "cpu"
        print("[*] BIS_FORCE_CPU=1, using CPU")
    elif torch.cuda.is_available():
        EMBED_DEVICE = "cuda"
        print(f"[*] Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        EMBED_DEVICE = "cpu"
        print("[*] CUDA not available, using CPU")
except Exception:
    EMBED_DEVICE = "cpu"
    print("[*] CUDA check failed, using CPU")

TOP_DENSE = 25
TOP_BM25 = 25
PARAPHRASE_TOP_DENSE = 40
FUSION_K = 25
FUSION_K_FALLBACK = 40
RRF_K_DENSE = 12
RRF_K_BM25 = 8
OUTPUT_K = 5

LOW_CONFIDENCE_MARGIN = 10.0
BIS_DEBUG = os.getenv("BIS_DEBUG", "0") == "1"

MATERIAL_TYPES = {
    "portland",
    "slag",
    "pozzolana",
    "masonry",
    "white",
    "supersulphated",
    "ordinary",
    "rapid",
    "hydrophobic",
    "sulphate",
    "aggregate",
    "cement",
    "concrete",
    "brick",
    "lime",
    "gypsum",
    "plaster",
    "mortar",
    "timber",
    "wood",
    "steel",
    "aluminium",
    "aluminum",
    "copper",
    "zinc",
    "glass",
    "plastic",
    "pvc",
    "cpvc",
    "bitumen",
    "tar",
    "asphalt",
    "sheet",
    "pipe",
    "fitting",
    "fittings",
    "valve",
    "cast",
    "granite",
    "slate",
    "stone",
    "sanitary",
}

PRODUCT_TYPE_KEYWORDS = {
    "pressure": ["pressure", "pressurized"],
    "drainage": ["drainage", "drain", "sewer", "waste"],
    "structural": ["structural", "structure", "load"],
    "hollow": ["hollow"],
    "solid": ["solid"],
    "flush": ["flush"],
    "slotted": ["slotted"],
    "raised": ["raised"],
    "countersunk": ["countersunk"],
    "plate": ["plate"],
    "rod": ["rod", "bar"],
    "bolt": ["bolt"],
    "nut": ["nut"],
    "cap": ["cap"],
    "screw": ["screw"],
    "staple": ["staple"],
    "hasp": ["hasp"],
    "pipe": ["pipe", "piping", "tube"],
    "fittings": ["fitting", "fittings"],
    "sanitary": ["sanitary", "vitreous", "lavatory", "closet", "urinal"],
    "granite": ["granite"],
    "slate": ["slate"],
}

MUTUALLY_EXCLUSIVE_TYPES = {
    "plate": {"rod", "pipe", "fittings", "bolt", "nut", "screw"},
    "rod": {"plate", "pipe", "fittings"},
    "pipe": {"plate", "rod", "bolt", "nut"},
    "fittings": {"plate", "rod"},
    "bolt": {"nut", "screw", "plate", "pipe"},
    "nut": {"bolt", "screw", "plate", "pipe"},
    "screw": {"bolt", "nut", "plate"},
    "sanitary": {"plate", "rod", "bolt", "nut"},
}

STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "for",
    "and",
    "or",
    "with",
    "from",
    "our",
    "we",
    "i",
    "need",
    "want",
    "looking",
    "what",
    "which",
    "that",
    "this",
    "their",
    "have",
    "has",
    "had",
    "using",
    "latest",
    "applicable",
    "official",
    "standard",
    "specification",
    "requirements",
}

# =============================================================================
# UTILITIES
# =============================================================================
_index_store = {}
_lm_available = None
_lm_probe_lock = Lock()
_log_lock = Lock()


def log(message):
    with _log_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def debug(msg):
    if BIS_DEBUG:
        print(f"[debug] {msg}")


def standard_key(std):
    explicit = std.get("_key")
    if explicit:
        return explicit
    base = re.sub(r"\s+", " ", str(std.get("id", "")).strip().upper())
    part = std.get("part")
    if part:
        part_clean = re.sub(r"\s+", " ", str(part).strip().upper())
        base = f"{base} (PART {part_clean})"
    return base


def _compact(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper())


def _loose_standard_key(value: str) -> str:
    text = str(value).upper().replace("PART PART", "PART")
    text = re.sub(r":\s*\d{4}", "", text)
    text = re.sub(r"\s*\(?\s*PART[^)]*\)?", "", text)
    return _compact(text)


def canonical_part(value) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    if not text:
        return ""
    text = re.sub(r"^\s*PART\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    roman = {
        "I": "1",
        "II": "2",
        "III": "3",
        "IV": "4",
        "V": "5",
        "VI": "6",
        "VII": "7",
        "VIII": "8",
        "IX": "9",
        "X": "10",
    }
    if text in roman:
        return roman[text]
    m = re.search(r"^(\d+)\b", text)
    if m and "/" not in text and "SEC" not in text:
        return m.group(1)
    return text


def format_standard_with_year(candidate_id, standards_db):
    s = standards_db.get(candidate_id)
    if not s:
        return candidate_id
    sid = re.sub(r"\s+", " ", str(s.get("id", "")).strip())
    part = canonical_part(s.get("part", "") or "")
    year = re.sub(r"\s+", " ", str(s.get("year", "") or "").strip())
    out = sid
    if part:
        out += f" (Part {part})"
    if year:
        out += f": {year}"
    return out.strip()


def format_standard_base_with_year(candidate_id, standards_db):
    s = standards_db.get(candidate_id)
    if not s:
        return candidate_id
    sid = re.sub(r"\s+", " ", str(s.get("id", "")).strip())
    year = re.sub(r"\s+", " ", str(s.get("year", "") or "").strip())
    out = sid
    if year:
        out += f": {year}"
    return out.strip()


def extract_keywords(text):
    words = re.findall(r"\b[a-z]{4,}\b", text.lower())
    return set(w for w in words if w not in STOPWORDS)


def get_ngrams(text, n=2):
    words = re.findall(r"\b[a-z0-9]+\b", text.lower())
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)} if len(words) >= n else set()


def extract_product_type_keywords(query):
    query_lower = query.lower()
    found = set()
    for ptype, keywords in PRODUCT_TYPE_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            found.add(ptype)
    return found


def parse_query_signals(query):
    q = query.lower()
    part_match = re.search(r"\bpart\s*[-:() ]*([ivx]+|\d+)\b", q, re.IGNORECASE)
    part = canonical_part(part_match.group(1)) if part_match else ""
    is_numbers = re.findall(r"\bis\s*[-:]?\s*(\d{2,6})\b", q, re.IGNORECASE)
    keywords = extract_keywords(query)
    bigrams = get_ngrams(query, 2)
    product_types = extract_product_type_keywords(query)
    materials = {m for m in MATERIAL_TYPES if m in q}
    return {
        "part": part,
        "is_numbers": set(is_numbers),
        "keywords": keywords,
        "bigrams": bigrams,
        "product_types": product_types,
        "materials": materials,
    }


def candidate_signals(cand):
    title = cand.get("title", "")
    content = cand.get("content", "")[:600]
    text = f"{title} {content}".lower()
    sid = str(cand.get("id", "")).strip()
    sid_num = re.sub(r"[^0-9]", "", sid)
    part = canonical_part(cand.get("part", "") or "")
    return {
        "sid": sid,
        "sid_num": sid_num,
        "part": part,
        "title": title.lower(),
        "text": text,
        "keywords": extract_keywords(text),
        "bigrams": get_ngrams(text, 2),
        "product_types": extract_product_type_keywords(text),
        "materials": {m for m in MATERIAL_TYPES if m in text},
    }


def load_indexes():
    global _index_store
    if _index_store:
        return _index_store

    log(f"Loading retrieval indexes from {DATA_DIR} ...")
    faiss_idx = faiss.read_index(str(DATA_DIR / "faiss_index.bin"))
    log("FAISS index loaded.")
    with open(DATA_DIR / "bm25_index.pkl", "rb") as f:
        bm25_data = pickle.load(f)
    log("BM25 index loaded.")
    with open(DATA_DIR / "whitelist.txt", "r", encoding="utf-8") as f:
        whitelist = {l.strip(): True for l in f if l.strip()}
    log(f"Whitelist loaded ({len(whitelist)} entries).")
    cfg = json.load(open(DATA_DIR / "embedding_config.json", "r", encoding="utf-8"))
    model = SentenceTransformer(cfg["model_name"], device=EMBED_DEVICE)
    log(f"Embedding model ready: {cfg['model_name']} on {EMBED_DEVICE}.")
    _index_store = {
        "faiss": faiss_idx,
        "bm25": bm25_data["bm25"],
        "standards": bm25_data["standards"],
        "whitelist": whitelist,
        "model": model,
    }
    return _index_store


def g(key):
    return _index_store[key]


def is_ollama_endpoint():
    """Detect if we're using Ollama (localhost:11434) vs LM Studio/OpenAI-compatible."""
    return "11434" in LM_BASE_URL


def lm_probe():
    """Check if LLM endpoint is available. Returns True if accessible."""
    global _lm_available
    
    with _lm_probe_lock:
        if _lm_available is not None:
            return _lm_available
        
        if is_ollama_endpoint():
            req = urllib.request.Request(
                f"{LM_BASE_URL}/api/tags",
                headers={"Content-Type": "application/json"},
                method="GET",
            )
            log(f"Checking Ollama at {LM_BASE_URL} ...")
        else:
            req = urllib.request.Request(
                f"{LM_BASE_URL}/v1/models",
                headers={"x-api-key": LM_API_KEY},
                method="GET",
            )
            log(f"Checking LM Studio at {LM_BASE_URL} ...")
        
        try:
            with urllib.request.urlopen(req, timeout=3) as resp:
                if 200 <= resp.status < 300:
                    _lm_available = True
                    log(f"LLM endpoint connected at {LM_BASE_URL}.")
                    return True
        except Exception:
            pass

        _lm_available = False
        log(f"LLM endpoint not reachable at {LM_BASE_URL}; falling back to deterministic rationale.")
        return False


def lm_complete_ollama(prompt, max_tokens=96):
    """Ollama API call using /api/generate endpoint."""
    payload = {
        "model": LM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.0}
    }
    req = urllib.request.Request(
        f"{LM_BASE_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            result = str(body.get("response", "")).strip()
            if result:
                log(f"LLM rationale: {result[:80]}...")
            return result
    except Exception as e:
        log(f"LLM call failed: {e}")
        return ""


def lm_complete_lmstudio(prompt, max_tokens=96):
    """LM Studio / OpenAI-compatible API call using /v1/completions."""
    payload = {
        "model": LM_MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{LM_BASE_URL}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-api-key": LM_API_KEY},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            choices = body.get("choices") or []
            if choices:
                return str(choices[0].get("text", "")).strip()
    except Exception:
        return ""
    return ""


def lm_complete(prompt, max_tokens=96):
    if not lm_probe():
        return ""
    
    if is_ollama_endpoint():
        return lm_complete_ollama(prompt, max_tokens)
    else:
        return lm_complete_lmstudio(prompt, max_tokens)


# =============================================================================
# RETRIEVAL
# =============================================================================
def retrieve_dense(query, k=TOP_DENSE):
    model = g("model")
    idx = g("faiss")
    standards = g("standards")
    qe = np.array(model.encode([query], normalize_embeddings=True), dtype=np.float32)
    D, I = idx.search(qe, k)
    return [(standard_key(standards[i]), float(D[0][j])) for j, i in enumerate(I[0]) if 0 <= i < len(standards)]


def retrieve_bm25(query, k=TOP_BM25):
    bm = g("bm25")
    standards = g("standards")
    scores = bm.get_scores(query.lower().split())
    top_idx = np.argsort(scores)[::-1][:k]
    return [(standard_key(standards[i]), float(scores[i])) for i in top_idx if i < len(standards) and scores[i] > 0]


def fuse_results(dense_results, bm25_results):
    score_map = defaultdict(float)
    for rank, (doc_id, _score) in enumerate(dense_results, 1):
        score_map[doc_id] += 1.0 / (RRF_K_DENSE + rank)
    for rank, (doc_id, _score) in enumerate(bm25_results, 1):
        score_map[doc_id] += 1.0 / (RRF_K_BM25 + rank)
    return sorted(score_map.items(), key=lambda x: x[1], reverse=True)


# =============================================================================
# DETERMINISTIC RANKING
# =============================================================================
def _id_family(cid: str) -> str:
    m = re.search(r"\d+", cid)
    return m.group(0) if m else ""


def feature_score(query_sig, cand_sig):
    score = 0.0
    q_types = query_sig["product_types"]
    c_types = cand_sig["product_types"]

    # Exact standard number mention in query.
    if cand_sig["sid_num"] and cand_sig["sid_num"] in query_sig["is_numbers"]:
        score += 36.0

    kw_overlap = len(query_sig["keywords"] & cand_sig["keywords"])
    bg_overlap = len(query_sig["bigrams"] & cand_sig["bigrams"])
    score += kw_overlap * 4.0
    score += bg_overlap * 6.0

    title_kw_overlap = len(query_sig["keywords"] & set(re.findall(r"\b[a-z]{4,}\b", cand_sig["title"])))
    score += title_kw_overlap * 9.0

    # Content keyword overlap - helps match specific product terms in body text
    content_keywords = set(re.findall(r"\b[a-z]{4,}\b", cand_sig["text"]))
    content_kw_overlap = len(query_sig["keywords"] & content_keywords)
    score += content_kw_overlap * 1.0

    mat_overlap = len(query_sig["materials"] & cand_sig["materials"])
    score += mat_overlap * 5.0

    if q_types:
        good = len(q_types & c_types)
        score += good * 11.0
        mismatches = 0
        for qt in q_types:
            blocked = MUTUALLY_EXCLUSIVE_TYPES.get(qt, set())
            if c_types & blocked:
                mismatches += 1
        score -= mismatches * 24.0

    q_part = query_sig["part"]
    c_part = cand_sig["part"]
    if q_part:
        if c_part == q_part:
            score += 18.0
        elif c_part:
            score -= 12.0
        else:
            score -= 2.0
    else:
        # Neutral when query does not specify part.
        score += 0.0

    # Penalize near-ID confusion (e.g., 736 vs 737) when explicit id mentioned.
    for qn in query_sig["is_numbers"]:
        if len(qn) >= 3 and cand_sig["sid_num"] and qn != cand_sig["sid_num"] and qn[:3] == cand_sig["sid_num"][:3]:
            score -= 16.0

    return score


def rank_candidates(query, candidates, standards_db, query_sig, trace=None):
    scored = []
    for cid in candidates:
        cand = standards_db.get(cid)
        if not cand:
            continue
        cs = candidate_signals(cand)
        score = feature_score(query_sig, cs)
        scored.append((cid, score, cs))
    scored.sort(key=lambda x: x[1], reverse=True)
    if trace is not None:
        trace["feature_top"] = [
            {
                "cid": cid,
                "score": round(score, 3),
                "part": cs["part"],
                "sid": cs["sid"],
                "types": sorted(list(cs["product_types"]))[:5],
            }
            for cid, score, cs in scored[:8]
        ]
    return scored


def resolve_family_rank(scored, query_sig, trace=None):
    families = defaultdict(list)
    for cid, score, cs in scored:
        families[_id_family(cid)].append((cid, score, cs))

    adjusted = []
    for fam, members in families.items():
        if len(members) == 1:
            adjusted.append(members[0])
            continue
        q_part = query_sig["part"]
        for cid, score, cs in members:
            bonus = 0.0
            if q_part and cs["part"] == q_part:
                bonus += 9.0
            elif q_part and cs["part"] and cs["part"] != q_part:
                bonus -= 6.0
            title_kw = len(query_sig["keywords"] & set(re.findall(r"\b[a-z]{4,}\b", cs["title"])))
            bonus += title_kw * 1.5
            adjusted.append((cid, score + bonus, cs))

    adjusted.sort(key=lambda x: x[1], reverse=True)
    if trace is not None:
        trace["family_resolved_top"] = [
            {"cid": cid, "score": round(score, 3), "part": cs["part"]}
            for cid, score, cs in adjusted[:8]
        ]
    return adjusted


def build_candidates(query, standards_db, whitelist, k_primary=FUSION_K, with_fallback=False, trace=None):
    dense = retrieve_dense(query, k=TOP_DENSE if not with_fallback else max(TOP_DENSE, 40))
    bm25 = retrieve_bm25(query, k=TOP_BM25 if not with_fallback else max(TOP_BM25, 40))
    fused = fuse_results(dense, bm25)
    topk = FUSION_K_FALLBACK if with_fallback else k_primary
    candidates = [cid for cid, _ in fused[:topk] if cid in whitelist]
    if trace is not None:
        trace["dense_top"] = [cid for cid, _ in dense[:6]]
        trace["bm25_top"] = [cid for cid, _ in bm25[:6]]
        trace["fused_top"] = [cid for cid, _ in fused[:10]]
    return candidates, fused


def confidence_margin(scored):
    if len(scored) < 2:
        return 999.0
    return scored[0][1] - scored[1][1]


def deterministic_rationale(query, top_candidate, standards_db):
    s = standards_db.get(top_candidate, {})
    sid = str(s.get("id", "")).strip()
    part = canonical_part(s.get("part", "") or "")
    title = str(s.get("title", "")).strip()
    if part:
        sid = f"{sid} (Part {part})"
    return f"{sid} matches the query intent through product/material alignment and specification-title similarity ({title[:100]})."


def llm_rationale(query, top_candidate, standards_db):
    s = standards_db.get(top_candidate, {})
    sid = str(s.get("id", "")).strip()
    part = canonical_part(s.get("part", "") or "")
    year = str(s.get("year", "") or "").strip()
    title = str(s.get("title", "")).strip()
    content_snippet = str(s.get("content", "") or "")[:500].strip()
    prompt = f"""You are a technical expert explaining BIS standards to a small business owner.
Write a detailed 2-3 sentence explanation of WHY this standard is relevant to the user's query.
Make it informative, clear, and helpful. Mention specific product names, material types, and applications.
User Query: {query}
Standard ID: {sid}
Standard Title: {title}
Standard Content Summary: {content_snippet[:400]}
Explanation (be detailed and specific):"""
    text = lm_complete(prompt, max_tokens=150)
    return text or deterministic_rationale(query, top_candidate, standards_db)


def build_output_list(final_ids, standards_db, query_sig, output_k=OUTPUT_K):
    out = []
    seen = set()
    query_has_part = bool(query_sig.get("part"))
    for cid in final_ids:
        full_fmt = format_standard_with_year(cid, standards_db)
        base_fmt = format_standard_base_with_year(cid, standards_db)
        if full_fmt == base_fmt:
            variants = [full_fmt]
        else:
            # Keep exact part-specific candidate first for private-set robustness.
            # If query has no explicit part, include base variant as a backup.
            variants = [full_fmt] if query_has_part else [full_fmt, base_fmt]
        for v in variants:
            key = v.upper()
            if key in seen:
                continue
            out.append(v)
            seen.add(key)
            if len(out) >= output_k:
                return out
    return out


# =============================================================================
# PIPELINE
# =============================================================================
def run_pipeline(query):
    start_time = time.perf_counter()
    load_indexes()
    standards = g("standards")
    standards_db = {standard_key(s): s for s in standards}
    whitelist = g("whitelist")

    trace = {} if BIS_DEBUG else None
    query_sig = parse_query_signals(query)
    if trace is not None:
        trace["query"] = query
        trace["signals"] = {
            "part": query_sig["part"],
            "is_numbers": sorted(query_sig["is_numbers"]),
            "types": sorted(query_sig["product_types"]),
            "materials": sorted(query_sig["materials"]),
        }

    candidates, fused = build_candidates(query, standards_db, whitelist, trace=trace)
    scored = rank_candidates(query, candidates, standards_db, query_sig, trace=trace)
    scored = resolve_family_rank(scored, query_sig, trace=trace)

    margin = confidence_margin(scored)
    if trace is not None:
        trace["confidence_margin"] = round(margin, 3)

    if margin < LOW_CONFIDENCE_MARGIN:
        if trace is not None:
            trace["fallback_triggered"] = True
        candidates2, fused2 = build_candidates(query, standards_db, whitelist, with_fallback=True)
        scored2 = rank_candidates(query, candidates2, standards_db, query_sig)
        scored2 = resolve_family_rank(scored2, query_sig)
        if scored2:
            scored = scored2
            fused = fused2
    elif trace is not None:
        trace["fallback_triggered"] = False

    final_ids = []
    seen = set()
    for cid, _score, _cs in scored:
        if cid not in seen:
            final_ids.append(cid)
            seen.add(cid)
        if len(final_ids) >= OUTPUT_K:
            break

    for cid, _ in fused:
        if len(final_ids) >= OUTPUT_K:
            break
        if cid not in seen and cid in whitelist:
            final_ids.append(cid)
            seen.add(cid)

    final_results = build_output_list(final_ids, standards_db, query_sig, output_k=OUTPUT_K)
    rationale = llm_rationale(query, final_ids[0], standards_db) if final_ids else "No confident match found."

    latency = time.perf_counter() - start_time
    out = {
        "retrieved": final_results,
        "rationale": rationale,
        "latency_seconds": round(latency, 3),
    }
    if BIS_DEBUG and trace is not None:
        out["_debug"] = trace
    return out


# =============================================================================
# MAIN
# =============================================================================
def process_item(item):
    qid = item.get("id", "UNKNOWN")
    query = item.get("query", "").strip()
    expected = item.get("expected_standards", [])
    result = run_pipeline(query)
    return {
        "id": qid,
        "query": query,
        "expected_standards": expected,
        "retrieved_standards": result["retrieved"],
        "latency_seconds": result["latency_seconds"],
    }


def main():
    parser = argparse.ArgumentParser(description="BIS Hackathon Inference")
    parser.add_argument("--input", required=True, help="Input JSON with queries")
    parser.add_argument("--output", required=True, help="Output JSON for eval_script.py")
    parser.add_argument("--workers", type=int, default=None, nargs="?", help="Optional worker count. Omit for sequential execution.")
    args = parser.parse_args()

    parallel_mode = args.workers is not None and args.workers > 1
    worker_label = args.workers if args.workers is not None else 1
    log(
        f"Starting inference run: input={args.input}, output={args.output}, "
        f"workers={worker_label}, mode={'parallel' if parallel_mode else 'sequential'}"
    )
    load_indexes()
    with open(args.input, "r", encoding="utf-8") as f:
        queries = json.load(f)

    log(f"Loaded {len(queries)} queries.")
    results = []
    started_at = time.perf_counter()
    if parallel_mode:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {}
            for idx, q in enumerate(queries, 1):
                qid = q.get("id", f"Q{idx}")
                log(f"[{idx}/{len(queries)}] queued {qid}")
                futures[ex.submit(process_item, q)] = (idx, qid)
            done = 0
            for fut in as_completed(futures):
                idx, qid = futures[fut]
                done += 1
                try:
                    result = fut.result()
                    results.append(result)
                    elapsed = result.get("latency_seconds", 0)
                    log(f"[{done}/{len(queries)}] finished {qid} in {elapsed:.2f}s")
                except Exception as e:
                    log(f"[{done}/{len(queries)}] failed {qid}: {e}")
                    raise
                if done % 10 == 0:
                    log(f"Progress: {done}/{len(queries)} complete.")
    else:
        for idx, q in enumerate(queries, 1):
            qid = q.get("id", f"Q{idx}")
            log(f"[{idx}/{len(queries)}] processing {qid}")
            try:
                result = process_item(q)
                results.append(result)
                elapsed = result.get("latency_seconds", 0)
                log(f"[{idx}/{len(queries)}] finished {qid} in {elapsed:.2f}s")
            except Exception as e:
                log(f"[{idx}/{len(queries)}] failed {qid}: {e}")
                raise
            if idx % 10 == 0:
                log(f"Progress: {idx}/{len(queries)} complete.")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    total = time.perf_counter() - started_at
    log(f"Saved results to {args.output} in {total:.2f}s.")


if __name__ == "__main__":
    main()
