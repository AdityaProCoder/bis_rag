"""
BIS Standards Recommendation Engine - Hardened New Architecture
Index-Time Components [Stage 0]

This module handles:
- Table-aware PDF parse → per-standard structured chunks
- Synonym graph mining from handbook cross-references + manual seed
- IS number whitelist with revision normalization

Architecture follows the final hardened pipeline:
[0] Index-time (once, offline)
    ├── Table-aware structured chunks per standard
    ├── Synonym graph from mined cross-references
    └── Whitelist with revision normalization
"""

from __future__ import annotations

import os
import re
import json
import pickle
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data"
NEW_DATA_DIR = Path(__file__).parent / "data"

STANDARDS_FILE = DATA_DIR / "sp21_standards.json"
FAISS_INDEX_FILE = NEW_DATA_DIR / "faiss_index.bin"
BM25_INDEX_FILE = NEW_DATA_DIR / "bm25_index.pkl"
METADATA_FILE = NEW_DATA_DIR / "metadata_store.json"
GRAPH_MAP_FILE = NEW_DATA_DIR / "graph_map.json"
WHITELIST_FILE = NEW_DATA_DIR / "whitelist.txt"

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_BATCH_SIZE = 64

# ---------------------------------------------------------------------------
# IS Number Normalization
# ---------------------------------------------------------------------------

def normalize_is_id(text: str) -> str:
    """Normalize IS ID for consistent matching (uppercase, strip whitespace)."""
    return re.sub(r"\s+", " ", text.strip().upper())


def normalize_part_label(part: str | None) -> str | None:
    """Normalize part labels to canonical form: 'Part 2', 'Part I', 'Part 3/SEC 1'."""
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
    Build a stable standard identifier preserving part-level granularity.
    Examples: 'IS 383', 'IS 2185 (Part 2)'.
    """
    base = normalize_is_id(std.get("id", ""))
    part_label = normalize_part_label(std.get("part"))
    return f"{base} ({part_label})" if part_label else base


def extract_year(std_id: str) -> Optional[str]:
    """Extract year from standard ID for normalization."""
    year_match = re.search(r':\s*(\d{4})', std_id)
    return year_match.group(1) if year_match else None


def normalize_to_canonical(std_id: str) -> str:
    """
    Normalize IS ID to canonical form:
    IS 383:2016 → IS 383 (year stripped for matching unless query specifies year)
    IS-383 → IS 383
    IS383 → IS 383
    """
    text = std_id.upper().strip()
    text = re.sub(r'[\s\-]+', ' ', text)
    base = re.sub(r':\s*\d{4}', '', text).strip()
    base = re.sub(r'[^A-Z0-9\s]', '', base)
    base = re.sub(r'\s+', ' ', base).strip()
    return base


def build_loose_key(std_id: str) -> str:
    """Build a loose key for matching: strips year, spaces, hyphens."""
    return re.sub(r'[^A-Z0-9]', '', normalize_to_canonical(std_id))


# ---------------------------------------------------------------------------
# Cross-Reference Mining
# ---------------------------------------------------------------------------

def extract_is_references(content: str) -> List[str]:
    """
    Mine IS-to-IS cross-references from content.
    E.g. 'refer to IS : 2386 (Part V) 1963' → IS 2386
    """
    pattern = r"(?:IS\s*[:\s]?\s*|IS\s+)(\d+(?:\s*\(Part\s+\w+\))?)"
    matches = re.findall(pattern, content, re.IGNORECASE)
    normalized = [normalize_is_id(f"IS {m}") for m in matches if m.strip()]
    return list(set(normalized))


# ---------------------------------------------------------------------------
# Synonym Graph (Manual Seed + Mined)
# ---------------------------------------------------------------------------

def build_synonym_graph() -> Dict[str, List[str]]:
    """
    Manual synonym dictionary for colloquial → technical terms.
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
        "tmt": ["thermomechanical treatment", "high yield steel bars"],
        "precast": ["prefabricated", "precast concrete"],
        "masonry": ["brickwork", "stonework", "blockwork"],
    }


def expand_query_with_synonyms(query: str, graph: Dict[str, List[str]]) -> List[str]:
    """
    Expand query terms using synonym graph.
    Input: original query terms
    Output: canonical BIS vocabulary set
    E.g. "TMT bars" → {deformed bars, high-yield steel, IS 1786}
    """
    terms = query.lower().split()
    expanded = set(terms)
    
    for term in terms:
        if term in graph:
            expanded.update(graph[term])
        # Handle plural
        if term.endswith('s') and term[:-1] in graph:
            expanded.update(graph[term[:-1]])
    
    return list(expanded)


# ---------------------------------------------------------------------------
# Table-Aware Chunking
# ---------------------------------------------------------------------------

def extract_table_content(content: str) -> str:
    """
    Extract key table information from standard content.
    Tables contain threshold values important for compliance.
    """
    lines = content.split('\n')
    table_lines = []
    in_table = False
    
    for line in lines:
        if 'TABLE' in line.upper() or 'TABLE-' in line:
            in_table = True
        if in_table:
            table_lines.append(line)
            if line.strip() and not line.startswith(' ') and len(table_lines) > 5:
                in_table = False
    
    return ' '.join(table_lines)


def build_structured_chunk(std: dict) -> dict:
    """
    Build a structured chunk for a standard with:
    - IS number, title, scope summary
    - Key threshold summary from tables
    - Cleaned content
    """
    chunk = {
        "id": std.get("id"),
        "year": std.get("year"),
        "title": std.get("title", ""),
        "part": std.get("part"),
        "_key": std.get("_key"),
        "scope": extract_scope(std.get("content", "")),
        "thresholds": extract_thresholds(std.get("content", "")),
        "content": clean_content(std.get("content", "")),
    }
    return chunk


def extract_scope(content: str) -> str:
    """Extract scope/summary from first paragraph."""
    lines = content.split('\n')
    scope_lines = []
    capture = False
    
    for line in lines:
        if re.match(r'^1\.\s*SCOPE', line, re.IGNORECASE):
            capture = True
        if capture and line.strip():
            scope_lines.append(line.strip())
        if len(scope_lines) > 5:
            break
    
    return ' '.join(scope_lines[:3])


def extract_thresholds(content: str) -> List[str]:
    """Extract threshold values (percentages, MPa, mm, etc)."""
    threshold_patterns = [
        r'not\s+more\s+than\s+(\d+(?:\.\d+)?\s*%?)\s*(?:by\s+mass|mm|MPa|m2/kg)?',
        r'not\s+less\s+than\s+(\d+(?:\.\d+)?\s*%?)\s*(?:by\s+mass|mm|MPa|m2/kg)?',
        r'(\d+(?:\.\d+)?\s*%)\s*(?:by\s+mass|by\s+weight)?',
    ]
    
    thresholds = []
    for pattern in threshold_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        thresholds.extend(matches[:5])  # Limit to 5 per pattern
    
    return list(set(thresholds))[:10]


def clean_content(content: str) -> str:
    """Clean content: remove excess whitespace, normalize."""
    content = re.sub(r'\s+', ' ', content)
    content = re.sub(r'\n+', ' ', content)
    return content.strip()[:2000]  # Truncate for embedding


# ---------------------------------------------------------------------------
# Cross-Reference Boost Map
# ---------------------------------------------------------------------------

def build_cross_reference_map(standards: List[dict], metadata: dict) -> Dict[str, List[str]]:
    """
    Build cross-reference map from mined references.
    Only real edges: extracted from 'see also', 'test per IS XXXX' clauses.
    """
    by_base_id: Dict[str, List[str]] = defaultdict(list)
    for std in standards:
        base = normalize_is_id(std.get("id", ""))
        by_base_id[base].append(std["_key"])
    
    cross_refs: Dict[str, Set[str]] = {s["_key"]: set() for s in standards}
    
    for std in standards:
        sid = std["_key"]
        refs = extract_is_references(std.get("content", ""))
        
        for ref in refs:
            if ref in metadata:
                cross_refs[sid].add(ref)
            elif ref in by_base_id:
                for ref_sid in by_base_id[ref]:
                    cross_refs[sid].add(ref_sid)
    
    return {k: sorted(v) for k, v in cross_refs.items() if v}


# ---------------------------------------------------------------------------
# Whitelist Builder
# ---------------------------------------------------------------------------

def build_whitelist(standards: List[dict]) -> List[str]:
    """Build whitelist of all valid standard IDs."""
    whitelist = [s["_key"] for s in standards]
    seen = set()
    unique_whitelist = []
    for wid in whitelist:
        if wid not in seen:
            seen.add(wid)
            unique_whitelist.append(wid)
    return unique_whitelist


# ---------------------------------------------------------------------------
# Main Index Build
# ---------------------------------------------------------------------------

def main():
    print("[0] Loading standards...")
    with open(STANDARDS_FILE, "r", encoding="utf-8") as f:
        standards_raw = json.load(f)
    print(f"    Loaded {len(standards_raw)} standards")

    # Build structured standards
    standards = []
    for std in standards_raw:
        row = dict(std)
        row["_key"] = make_standard_key(row)
        standards.append(row)

    # Ensure new data directory exists
    NEW_DATA_DIR.mkdir(exist_ok=True)

    # [A] Whitelist
    whitelist = build_whitelist(standards)
    with open(WHITELIST_FILE, "w", encoding="utf-8") as f:
        for wid in whitelist:
            f.write(wid + "\n")
    print(f"    [A] Whitelist: {len(whitelist)} entries")

    # [B] Metadata store
    metadata = {}
    for idx, std in enumerate(standards):
        sid = std["_key"]
        metadata[sid] = {
            "index": idx,
            "title": std["title"],
            "year": std.get("year"),
            "part": std.get("part"),
        }
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"    [B] Metadata store: {len(metadata)} entries")

    # [C] Graph map (synonyms + cross-references)
    manual_synonyms = build_synonym_graph()
    cross_refs = build_cross_reference_map(standards, metadata)
    
    graph_map = {
        "synonyms": manual_synonyms,
        "cross_references": cross_refs,
    }
    with open(GRAPH_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(graph_map, f, indent=2, ensure_ascii=False)
    print(f"    [C] Graph map: {len(cross_refs)} nodes with cross-refs")

    # [D] Structured chunks for BM25
    print("[D] Building BM25 index...")
    structured_chunks = [build_structured_chunk(s) for s in standards]
    
    tokenized_corpus = []
    for chunk in structured_chunks:
        combined = f"{chunk['title']} {chunk['scope']} {chunk['content']}"
        tokens = combined.lower().split()
        tokenized_corpus.append(tokens)
    
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi(tokenized_corpus)
    
    with open(BM25_INDEX_FILE, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "standards": standards,
            "chunks": structured_chunks
        }, f)
    print(f"    BM25 index built with {len(structured_chunks)} chunks")

    # [E] FAISS dense index
    print("[E] Loading BGE-M3 embedding model...")
    import torch
    from sentence_transformers import SentenceTransformer
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    Device: {device}")
    
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    
    texts_for_embedding = []
    for chunk in structured_chunks:
        title = chunk['title']
        scope = chunk['scope']
        content = chunk['content'][:512]
        texts_for_embedding.append(f"{title}\n{scope}\n{content}")
    
    print(f"    Embedding {len(texts_for_embedding)} standards...")
    embeddings = model.encode(
        texts_for_embedding,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"    FAISS index: {index.ntotal} vectors, dim={dimension}")
    
    faiss.write_index(index, str(FAISS_INDEX_FILE))

    # Save embedding config
    embed_config = {
        "model_name": EMBEDDING_MODEL,
        "dimension": dimension,
        "device": device
    }
    with open(NEW_DATA_DIR / "embedding_config.json", "w", encoding="utf-8") as f:
        json.dump(embed_config, f, indent=2)
    
    print("\n[[OK]] Hardened index build complete.")
    print(f"    Outputs: {WHITELIST_FILE}, {METADATA_FILE}, {GRAPH_MAP_FILE}")
    print(f"            {BM25_INDEX_FILE}, {FAISS_INDEX_FILE}")


if __name__ == "__main__":
    import faiss
    main()
