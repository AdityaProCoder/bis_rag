"""
Main pipeline: Wide Retrieval → Candidate Expansion → LLM Reranking

Exact flow from yesterday's winning pipeline:
1. Dense (FAISS) + BM25 retrieval
2. RRF fusion
3. Paraphrase expansion (always on)
4. Merge candidates
5. LLM reranking via /v1/completions
6. Return top OUTPUT_K results

NO section filtering. NO concept layer. NO hierarchical routing.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from old.retrieval import retrieve_dense, retrieve_bm25
from old.fusion import fuse_results
from old.paraphrase import paraphrase_retrieval
from old.reranker import llm_rerank
from old.utils import load_indexes, g

TOP_DENSE = 20
TOP_BM25 = 20
FUSION_K = 10
RERANK_K = 3
OUTPUT_K = 3
PARAPHRASE_ENABLED = True
RRF_K = 10


def run_pipeline(query):
    """
    Execute the full retrieval pipeline for a single query.
    
    Returns:
        dict with keys: retrieved (list), latency_seconds (float), 
                        top10_candidates (list), llm_output (str)
    """
    start_time = time.perf_counter()
    
    load_indexes()
    
    # STEP 1: Dual retrieval
    dense = retrieve_dense(query, k=TOP_DENSE)
    bm25 = retrieve_bm25(query, k=TOP_BM25)
    
    # STEP 2: RRF fusion
    fused = fuse_results(dense, bm25)
    fused_top = [cid for cid, _ in fused[:FUSION_K]]
    
    # STEP 3 & 4: Paraphrase and paraphrase retrieval
    if PARAPHRASE_ENABLED:
        p_results = paraphrase_retrieval(query)
        if p_results:
            # STEP 5: Merge candidates
            merged_scores = {}
            for rank, (cid, score) in enumerate(fused[:FUSION_K], 1):
                merged_scores[cid] = merged_scores.get(cid, 0) + 1.0 / (RRF_K + rank)
            for rank, (cid, score) in enumerate(p_results, 1):
                merged_scores[cid] = merged_scores.get(cid, 0) + 1.0 / (RRF_K + rank)
            fused = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)
    
    candidates = [cid for cid, _ in fused[:FUSION_K]]
    top10_for_log = candidates[:10]
    
    # STEP 6: LLM rerank
    reranked = llm_rerank(query, candidates)
    
    # STEP 7 & 8: Validate and return
    validated = [c for c in reranked if c in g("whitelist")]
    for c, _ in fused:
        if len(validated) >= OUTPUT_K:
            break
        if c not in validated and c in g("whitelist"):
            validated.append(c)
    
    final_results = validated[:OUTPUT_K]
    latency = time.perf_counter() - start_time
    
    return {
        "retrieved": final_results,
        "latency_seconds": round(latency, 3),
        "top10_candidates": top10_for_log,
        "llm_output": ""
    }
