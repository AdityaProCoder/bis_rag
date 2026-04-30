"""
Data-driven evaluation dataset generation and bias analysis.
Derives domains from the corpus itself, generates 300-400 queries,
and runs full vs baseline evaluation.
"""
import json, re, os, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

# ---- STEP 1: Load ----
print("[1] Loading dataset...")
data = json.load(open("data/sp21_standards.json", encoding="utf-8"))
print(f"    Total standards: {len(data)}")

# ---- STEP 2: Domain clustering (data-driven) ----
print("[2] Auto-deriving domain clusters...")

# Each domain has 1+ discriminating keyword(s)
DOMAIN_SIGNATURES = {
    "cement": ["cement", "slag", "pozzolana", "portland", "supersulphated", "alumina"],
    "concrete": ["concrete", "mortar", "grout", "aggregate", "sand", "stone", "lightweight"],
    "concrete_products": ["blocks", "block", "brick", "bricks", "tiles", "tile", "slab", "slabs", "kerb", "coping"],
    "pipes_fittings": ["pipe", "pipes", "fitting", "fittings", "conduit", "sewer", "drain", "gully"],
    "precast": ["precast", "pre-cast"],
    "roofing_sheets": ["roofing", "sheet", "sheets", "corrugated", "cladding"],
    "asbestos_cement": ["asbestos", "asbestic"],
    "steel_reinforcement": ["steel", "reinforcement", "rebar", "bars", "wire", "rope"],
    "aluminium": ["aluminium", "aluminum"],
    "pvc_polymers": ["pvc", "polyvinyl", "polymer", "plastic pipe"],
    "water_supply": ["water", "potable", "drinking water"],
    "clay_products": ["clay brick", "earthenware", "terracotta"],
    "vitreous_ceramic": ["vitreous", "ceramic", "sanitary ware", "enamel"],
    "bitumen": ["bitumen", "asphalt", "tar"],
    "structural": ["structural", "prestressed", "prestress"],
    "test_methods": ["methods of test", "testing", "test method"],
}

DOMAIN_KEYWORD_MAP = {}
for domain, kws in DOMAIN_SIGNATURES.items():
    for kw in kws:
        DOMAIN_KEYWORD_MAP[kw] = domain

def get_domain(title):
    title_lower = title.lower()
    scores = defaultdict(int)
    # Check multi-word phrases first
    for kw, domain in DOMAIN_KEYWORD_MAP.items():
        if " " in kw:
            if kw in title_lower:
                scores[domain] += 3
        else:
            words = title_lower.split()
            for w in words:
                if w == kw:
                    scores[domain] += 1
    if not scores:
        return "other"
    return max(scores, key=scores.get)

domain_of = {}
for s in data:
    dom = get_domain(s["title"])
    domain_of[s["id"]] = dom
    if s.get("part"):
        domain_of[f"{s['id']} (Part {s['part']})"] = dom

domain_counts = Counter(domain_of.values())
# Only use domains with 5+ standards
major_domains = sorted([d for d, c in domain_counts.items() if c >= 5], key=lambda d: -domain_counts[d])
print(f"    Major domains ({len(major_domains)}):")
for dom in major_domains:
    print(f"      {dom}: {domain_counts[dom]}")

# ---- STEP 3: Key term extraction ----
print("[3] Extracting key terms...")

STOPWORDS = {"the", "a", "an", "of", "and", "for", "to", "in", "with", "part", "is", "w", "shall", "such"}

def extract_terms(s, n=8):
    text = (s["title"] + " " + s.get("content", "")[:500]).lower()
    words = re.findall(r"[a-z]{3,}", text)
    freq = Counter(w for w in words if w not in STOPWORDS)
    return [w for w, c in freq.most_common(n)]

for s in data:
    s["_terms"] = extract_terms(s)

# ---- STEP 4: Query generation ----
print("[4] Generating evaluation queries...")

SYNONYMS = {
    "cement": ["hydraulic binder", "binding material", "cementitious material"],
    "concrete": ["construction material", "building material"],
    "sand": ["fine aggregate", "river sand"],
    "aggregate": ["coarse material", "stone aggregate"],
    "blocks": ["masonry units", "construction blocks"],
    "steel": ["reinforcement bars", "TMT bars"],
    "mortar": ["masonry binder", "bonding material"],
    "precast": ["factory made", "shop fabricated"],
    "corrugated": ["wavy", "ridged"],
    "asbestos": ["mineral fiber", "AC sheets"],
    "vitreous": ["glass-like", "ceramic"],
    "aluminium": ["aluminum", "light metal"],
    "pvc": ["polyvinyl chloride", "plastic pipe"],
    "water": ["potable", "drinking"],
    "roofing": ["roof covering", "roof cladding"],
}

def paraphrase_synonyms(text):
    result = text.lower()
    for word, syns in SYNONYMS.items():
        if word in result:
            result = result.replace(word, syns[0])
    return result

def make_direct(s):
    title = s["title"]
    words = re.findall(r"[A-Z][A-Z0-9 ]+", title)
    if words:
        core = " ".join(words[:3])
    else:
        core = title[:60]
    return f"What standard covers {core.lower()}?"

def make_paraphrase(s):
    direct_q = make_direct(s)
    para = paraphrase_synonyms(direct_q)
    if para == direct_q:
        para = direct_q.replace("standard", "BIS code").replace("covers", "applies to")
    return para

def make_abstract(s):
    dom = domain_of.get(s["id"], "construction")
    dom_clean = dom.replace("_", " ")
    templates = [
        f"What BIS standard applies to materials used for {dom_clean}?",
        f"Which Indian Standard governs {dom_clean} in construction?",
        f"I need the BIS code for {dom_clean} applications.",
        f"Show me the standard for {dom_clean}.",
    ]
    return templates[hash(s["id"]) % len(templates)]

def make_keyword_removed(s):
    terms = s["_terms"]
    if len(terms) < 2:
        return make_direct(s)
    removed = terms[0]
    remaining = " ".join(terms[1:4])
    return f"What is the BIS standard for {remaining}?"

def make_adversarial(s, all_data):
    dom = domain_of.get(s["id"], "other")
    other_stds = [x for x in all_data if domain_of.get(x["id"]) == dom and x["id"] != s["id"]]
    if not other_stds:
        return make_direct(s)
    other = other_stds[hash(s["id"] + "x") % len(other_stds)]
    our_term = s["_terms"][0] if s["_terms"] else "material"
    their_term = other["_terms"][0] if other["_terms"] else "properties"
    return f"BIS standard for {our_term} combined with {their_term}?"

# Target: ~60 standards × 5 types = ~300 queries, well distributed
MAX_PER_DOMAIN = 7  # max standards sampled per domain
queries = []
used_by_domain = defaultdict(int)

for dom in major_domains:
    dom_stds = [s for s in data if domain_of.get(s["id"]) == dom]
    n = min(MAX_PER_DOMAIN, len(dom_stds))
    np.random.seed(hash(dom) % 2**32)
    sampled = list(np.random.choice(dom_stds, n, replace=False)) if len(dom_stds) >= n else dom_stds

    for s in sampled:
        sid = s["id"]
        key = f"{sid} (Part {s['part']})" if s.get("part") else sid
        d = domain_of.get(sid, "other")

        q_defs = [
            ("direct", make_direct(s)),
            ("paraphrase", make_paraphrase(s)),
            ("abstract", make_abstract(s)),
            ("keyword_removed", make_keyword_removed(s)),
            ("adversarial", make_adversarial(s, data)),
        ]
        for qtype, qtext in q_defs:
            queries.append({
                "id": f"EVAL-{len(queries)+1:04d}",
                "query": qtext,
                "expected_standard": key,
                "domain": d,
                "type": qtype,
            })

print(f"    Generated {len(queries)} queries")
print(f"    Type distribution: {dict(Counter(q['type'] for q in queries))}")
print(f"    Domain distribution: {dict(Counter(q['domain'] for q in queries))}")

# Save
out_path = Path("data/eval_queries.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(queries, f, indent=2, ensure_ascii=False)
print(f"    Saved to {out_path}")
print(f"\n[5] Query dataset ready.")
