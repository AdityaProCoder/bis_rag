"""
Retrieval primitives: Dense (FAISS) and Sparse (BM25).
NO section filtering.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from old.utils import g, standard_key


def retrieve_dense(query, k=20):
    """Dense retrieval using FAISS."""
    model = g("model")
    idx = g("faiss")
    standards = g("standards")
    qe = np.array(model.encode([query], normalize_embeddings=True), dtype=np.float32)
    D, I = idx.search(qe, k)
    return [(standard_key(standards[i]), float(D[0][j])) for j, i in enumerate(I[0]) if 0 <= i < len(standards)]


def retrieve_bm25(query, k=20):
    """BM25 sparse retrieval."""
    bm = g("bm25")
    standards = g("standards")
    scores = bm.get_scores(query.lower().split())
    top_idx = np.argsort(scores)[::-1][:k]
    return [(standard_key(standards[i]), float(scores[i])) for i in top_idx if i < len(standards) and scores[i] > 0]
