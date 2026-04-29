# BIS Standards Recommendation Engine - Hardened New Architecture

## Overview

This is a re-implementation of the BIS Standards Recommendation Engine using the **hardened 7-stage pipeline** architecture. The new implementation addresses known weaknesses identified in the previous approach:

1. **Dual-channel retrieval** instead of merged-query-as-semantic-soup
2. **Conditional paraphrase** instead of always-on paraphrase
3. **Cross-encoder reranking** instead of LLM reranking
4. **Mined cross-reference edges** instead of guessed edges
5. **Proper IS number validation gate** with normalization

## Architecture

```
[0] Index-time (once, offline)
    ├── Table-aware PDF parse → per-standard structured chunks
    ├── Build synonym graph from handbook cross-references + manual seed
    └── Build IS number whitelist with revision normalization

[1] Query-time: Pre-retrieval
    └── Graph synonym lookup (in-memory, <10ms, zero LLM)

[2] Query-time: Dual retrieval (parallel, not sequential)
    ├── Channel A: Dense retrieval (query only, NOT synonym soup)
    └── Channel B: Sparse retrieval (BM25 on query + graph-expanded terms)

[3] Fusion
    └── RRF merge of Channel A + Channel B → top-15 candidates
        Graph cross-reference boost applied here

[4] Paraphrase trigger (conditional, not always-on)
    └── IF top-15 confidence spread is low OR top-1 score below threshold
        THEN: generate 1 paraphrase, run dense retrieval, RRF-merge into pool
        ELSE: skip (saves latency on clear-cut queries)

[5] Rerank
    └── ms-marco-MiniLM-L-6-v2 cross-encoder (fast, CPU-runnable)
        Scores each candidate against ORIGINAL query only

[6] IS number validation gate
    └── Check each IS number against whitelist
        Normalize formatting variants (IS383, IS-383, IS 383:2016 → same key)
        Drop any candidate whose IS number fails validation

[7] Output formatter
    └── Strict schema compliance, top 3 returned
```

## Files

```
new/
├── __init__.py              # Package init
├── build_index.py           # Index-time components [Stage 0]
├── inference.py             # Query-time pipeline [Stages 1-7]
├── run_eval.py              # Evaluation script
├── data/                    # Generated index files
│   ├── faiss_index.bin      # Dense FAISS index
│   ├── bm25_index.pkl      # BM25 sparse index
│   ├── metadata_store.json  # IS number → index/title mapping
│   ├── graph_map.json       # Synonyms + cross-references
│   ├── whitelist.txt        # Valid IS numbers
│   └── embedding_config.json # Embedding model config
└── results.json             # Test results on public_test_set
```

## Usage

### Build Index (once)

```bash
python new/build_index.py
```

### Run Inference

```bash
python new/inference.py --input guidelines/public_test_set.json --output new/results.json
```

### Evaluate Results

```bash
python new/run_eval.py --results new/results.json
```

## Results on Public Test Set

```
==================================================
   BIS HACKATHON EVALUATION RESULTS (HARDENED)
==================================================
Total Queries Evaluated : 10
Hit Rate @3             : 90.00%  (Target: >80%)
MRR @5                  : 0.7500  (Target: >0.7)
Avg Latency             : 1.50 sec (Target: <5 seconds)
==================================================
```

**Note:** The 90% Hit Rate @3 and 0.75 MRR @5 meet the hackathon targets (>80% and >0.7 respectively).

## Key Decisions

| Question | Decision | Reason |
|---|---|---|
| Merged query or dual channel? | Dual channel + RRF | Dense on semantics, sparse on keywords — each channel stays clean |
| Paraphrase always or conditional? | Conditional on confidence spread | Saves latency on easy queries, fires only when retrieval is uncertain |
| Which reranker? | MiniLM-L-6 cross-encoder | Fast on CPU, deterministic, no API dependency |
| Graph edges: how to build? | Mine from handbook only (cross-refs, "test per", "see also") | No guessed edges, ever |
| Whitelist: revision handling? | Normalize to IS number + year, treat all revisions as same canonical key unless query specifies year | Safe default |
| Paraphrase drift risk? | Constrained prompt + engineering vocabulary restriction in system prompt | Limits the blast radius |

## Environment

- Python 3.12+
- CUDA for embedding (CPU fallback available)
- Required packages: numpy, faiss, sentence-transformers, rank_bm25

## Notes

- The `new/` folder does not delete or modify any existing files
- The original `inference.py` and `build_index.py` remain untouched
- This is a parallel implementation following the hardened pipeline specification
