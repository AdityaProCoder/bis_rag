"""
BIS Standards Recommendation Engine - Index Builder
Step [0]: Build FAISS dense index, BM25 sparse index, metadata store,
          graph_map.json (synonym + cross-reference edges), and whitelist.txt.
Run once before inference.py.
"""

import os
import json
import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
import faiss

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
STANDARDS_FILE = DATA_DIR / "sp21_standards.json"
FAISS_INDEX_FILE = DATA_DIR / "faiss_index.bin"
BM25_INDEX_FILE = DATA_DIR / "bm25_index.pkl"
METADATA_FILE = DATA_DIR / "metadata_store.json"
GRAPH_MAP_FILE = DATA_DIR / "graph_map.json"
WHITELIST_FILE = DATA_DIR / "whitelist.txt"

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_BATCH_SIZE = 64
EMBEDDING_DIM = 1024  # bge-m3 dense output dimension

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_is_id(text: str) -> str:
    """Normalize IS ID for consistent matching (uppercase, strip whitespace)."""
    return re.sub(r"\s+", " ", text.strip().upper())


def normalize_part_label(part: str | None) -> str | None:
    """Normalize part labels to canonical form, e.g. 'Part 2', 'Part I', 'Part 3/SEC 1'."""
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


def make_standard_key(std: dict) -> str:
    """
    Build a stable standard identifier that preserves part-level granularity.
    Examples: 'IS 383', 'IS 2185 (Part 2)'.
    """
    base = normalize_is_id(std.get("id", ""))
    part_label = normalize_part_label(std.get("part"))
    return f"{base} ({part_label})" if part_label else base


def extract_is_references(content: str) -> list[str]:
    """
    Mine IS-to-IS cross-references from content.
    E.g. 'refer to IS : 2386 (Part V) 1963' -> IS 2386
    """
    pattern = r"(?:IS\s*[:\s]?\s*|IS\s+)(\d+(?:\s*\(Part\s+\w+\))?)"
    matches = re.findall(pattern, content, re.IGNORECASE)
    normalized = [normalize_is_id(f"IS {m}") for m in matches if m.strip()]
    return list(set(normalized))


def build_synonym_graph() -> dict[str, list[str]]:
    """
    Manual synonym dictionary for colloquial -> technical terms.
    Zero LLM - purely dictionary lookup at query time.
    Covers singular and plural forms.
    """
    return {
        "sand": ["fine aggregate", "aggregates for concrete", "natural sand"],
        "cement": ["portland cement", "ordinary portland cement", "slag cement"],
        "concrete": ["mass concrete", "structural concrete", "masonry"],
        "aggregate": ["coarse aggregate", "fine aggregate", "crushed stone"],
        "brick": ["clay brick", "burnt brick", "masonry units"],
        "bricks": ["clay brick", "burnt brick", "masonry units"],
        "steel": ["reinforcement", "tmt bar", "mild steel"],
        "timber": ["wood", "seasoned wood", "hardwood"],
        "fly ash": ["pozzolana", "pulverised fuel ash"],
        "mortar": ["sand masonry mortar", "cement mortar"],
        "aggregate crushing": ["aggregate crushing value"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[0] Loading standards...")
    with open(STANDARDS_FILE, "r", encoding="utf-8") as f:
        standards_raw = json.load(f)
    print(f"    Loaded {len(standards_raw)} standards")

    standards = []
    for std in standards_raw:
        row = dict(std)
        row["_key"] = make_standard_key(row)
        standards.append(row)

    # --- A. Whitelist (all valid IDs) ---
    whitelist = [s["_key"] for s in standards]
    # Deduplicate while preserving order
    seen = set()
    unique_whitelist = []
    for wid in whitelist:
        if wid not in seen:
            seen.add(wid)
            unique_whitelist.append(wid)
    whitelist = unique_whitelist

    with open(WHITELIST_FILE, "w", encoding="utf-8") as f:
        for wid in whitelist:
            f.write(wid + "\n")
    print(f"    [A] Whitelist: {len(whitelist)} entries -> {WHITELIST_FILE}")

    # --- B. Metadata store (IS number -> index, title) ---
    metadata = {}
    for idx, std in enumerate(standards):
        sid = std["_key"]
        metadata[sid] = {"index": idx, "title": std["title"]}
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"    [B] Metadata store: {len(metadata)} entries -> {METADATA_FILE}")

    # --- C. Graph map (synonym graph + mined cross-references) ---
    manual_synonyms = build_synonym_graph()
    by_base_id: dict[str, list[str]] = {}
    for std in standards:
        base = normalize_is_id(std.get("id", ""))
        by_base_id.setdefault(base, []).append(std["_key"])

    # Mine cross-references
    cross_refs: dict[str, set[str]] = {wid: set() for wid in whitelist}
    for std in standards:
        sid = std["_key"]
        refs = extract_is_references(std.get("content", ""))
        for ref in refs:
            if ref in metadata:  # direct key hit (exact part/base)
                cross_refs[sid].add(ref)
            elif ref in by_base_id:  # generic ref -> link to all parts/variants
                for ref_sid in by_base_id[ref]:
                    cross_refs[sid].add(ref_sid)

    # Convert sets to sorted lists for JSON serialization
    graph_map = {
        "synonyms": manual_synonyms,
        "cross_references": {k: sorted(v) for k, v in cross_refs.items() if v},
    }
    with open(GRAPH_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(graph_map, f, indent=2, ensure_ascii=False)
    print(f"    [C] Graph map: {len(graph_map['cross_references'])} nodes with cross-refs -> {GRAPH_MAP_FILE}")

    # --- D. BM25 index ---
    print("[D] Building BM25 index...")
    tokenized_corpus = []
    for std in standards:
        combined = f"{std.get('title', '')} {std.get('content', '')}"
        tokens = combined.lower().split()
        tokenized_corpus.append(tokens)

    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_INDEX_FILE, "wb") as f:
        pickle.dump({"bm25": bm25, "standards": standards}, f)
    print(f"    BM25 index built and saved -> {BM25_INDEX_FILE}")

    # --- E. FAISS dense index (BGE-M3) ---
    print("[E] Loading BGE-M3 embedding model...")
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    Device: {device}")

    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    # For BGE-M3, we use the "tokens" pool (dense_dim = 1024 for bge-m3)
    # We embed title + content fields
    texts_for_embedding = []
    for std in standards:
        title = std.get("title", "")
        content = std.get("content", "")
        # Truncate content to first 512 tokens to keep embedding batch fast
        content_snippet = " ".join(content.split()[:512])
        texts_for_embedding.append(f"{title}\n{content_snippet}")

    print(f"    Embedding {len(texts_for_embedding)} standards in batches of {EMBEDDING_BATCH_SIZE}...")
    embeddings = model.encode(
        texts_for_embedding,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # inner product = cosine similarity for normalized vectors
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    # Build FAISS IndexFlatIP (Inner Product) — works with normalize_embeddings=True
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"    FAISS index built: {index.ntotal} vectors, dim={dimension}")

    faiss.write_index(index, str(FAISS_INDEX_FILE))
    print(f"    [E] FAISS index saved -> {FAISS_INDEX_FILE}")

    print("\n[[OK]] Index build complete.")
    print(f"    Outputs: {WHITELIST_FILE}, {METADATA_FILE}, {GRAPH_MAP_FILE}, {BM25_INDEX_FILE}, {FAISS_INDEX_FILE}")

    # Save embedding model name for inference
    embed_config = {"model_name": EMBEDDING_MODEL, "dimension": dimension, "device": device}
    with open(DATA_DIR / "embedding_config.json", "w", encoding="utf-8") as f:
        json.dump(embed_config, f, indent=2)
    print(f"    Embedding config -> {DATA_DIR / 'embedding_config.json'}")


if __name__ == "__main__":
    main()
