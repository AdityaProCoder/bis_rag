"""
Utility functions for the baseline pipeline.
"""
import re
import json
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

_DATA_DIR = Path(__file__).parent.parent / "data"
_EMBED_DEVICE = "cpu"

_index_store = {}


def standard_key(std):
    explicit = std.get("_key")
    if explicit:
        return explicit
    base = re.sub(r"\s+", " ", str(std.get("id", "")).strip().upper())
    part = std.get("part")
    if part:
        part_clean = re.sub(r"\s+", " ", str(part).strip().upper())
        base = f"{base} (Part {part_clean})"
    return base


def norm_id(s):
    base = str(s).split(":")[0].strip()
    return re.sub(r"\s+", " ", base).upper()


def norm_full(s):
    base = str(s).split(":")[0].strip()
    part_m = re.search(r"\(PART[^)]+\)", str(s))
    part = part_m.group(0) if part_m else ""
    return re.sub(r"\s+", " ", f"{base} {part}").strip().upper()


def load_indexes():
    global _index_store
    if _index_store:
        return _index_store
    
    faiss_idx = faiss.read_index(str(_DATA_DIR / "faiss_index.bin"))
    with open(_DATA_DIR / "bm25_index.pkl", "rb") as f:
        bm25_data = pickle.load(f)
    with open(_DATA_DIR / "whitelist.txt", "r", encoding="utf-8") as f:
        whitelist = {l.strip(): True for l in f if l.strip()}
    
    cfg = json.load(open(_DATA_DIR / "embedding_config.json"))
    model = SentenceTransformer(cfg["model_name"], device=_EMBED_DEVICE)
    
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
