"""
RRF Fusion for combining dense and sparse retrieval results.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

RRF_K = 10


def fuse_results(dense_results, bm25_results):
    """
    Fuse results using Reciprocal Rank Fusion.
    Returns sorted list of (doc_id, score) up to FUSION_K candidates.
    """
    score_map = {}
    
    for rank, (doc_id, _) in enumerate(dense_results, 1):
        score_map[doc_id] = score_map.get(doc_id, 0) + 1.0 / (RRF_K + rank)
    
    for rank, (doc_id, _) in enumerate(bm25_results, 1):
        score_map[doc_id] = score_map.get(doc_id, 0) + 1.0 / (RRF_K + rank)
    
    return sorted(score_map.items(), key=lambda x: x[1], reverse=True)
