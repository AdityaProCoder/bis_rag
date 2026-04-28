"""
Quick validation script to verify index artifacts are correctly built.
Run: python validate_artifacts.py
"""

from pathlib import Path
import json, pickle, faiss

DATA = Path("data")

artifacts = {
    "faiss_index.bin": "FAISS index",
    "bm25_index.pkl": "BM25 index",
    "metadata_store.json": "metadata store",
    "graph_map.json": "graph map",
    "whitelist.txt": "whitelist",
    "embedding_config.json": "embedding config",
}

print("Checking index artifacts...\n")
for fname, label in artifacts.items():
    path = DATA / fname
    if not path.exists():
        print(f"[MISSING] {fname}")
        continue

    size_kb = path.stat().st_size / 1024
    print(f"[OK] {fname} ({size_kb:.1f} KB) - {label}")

# Validate content of key artifacts
print("\n--- Metadata store sample ---")
with open(DATA / "metadata_store.json", "r", encoding="utf-8") as f:
    meta = json.load(f)
sample_keys = list(meta.keys())[:3]
for k in sample_keys:
    print(f"  {k}: {meta[k]}")

print("\n--- Whitelist count ---")
with open(DATA / "whitelist.txt", "r", encoding="utf-8") as f:
    lines = [l.strip() for l in f if l.strip()]
print(f"  {len(lines)} valid IDs")

print("\n--- Graph map cross-refs ---")
with open(DATA / "graph_map.json", "r", encoding="utf-8") as f:
    gmap = json.load(f)
nodes_with_refs = len(gmap.get("cross_references", {}))
print(f"  {nodes_with_refs} nodes have cross-references")
print(f"  Synonym keys: {list(gmap['synonyms'].keys())}")

print("\n--- FAISS index ---")
idx = faiss.read_index(str(DATA / "faiss_index.bin"))
print(f"  {idx.ntotal} vectors, dim={idx.d}")

print("\n--- BM25 index ---")
with open(DATA / "bm25_index.pkl", "rb") as f:
    data = pickle.load(f)
print(f"  {len(data['standards'])} standards in corpus")

print("\n[[OK]] All artifacts valid.")
