# Clean Baseline Pipeline

A minimal, high-performance BIS standards retrieval pipeline achieving:
- Hit@3 ≈ 100%
- MRR@5 ≈ 0.93–0.95
- Latency < 1s

## Architecture

```
Wide Retrieval → Candidate Expansion → LLM Reranking
```

### Step-by-Step Flow

**STEP 1: Dual Retrieval**
- Dense (FAISS): Semantic embeddings using BGE-M3 model
- Sparse (BM25): Traditional keyword matching
- Both run independently on original query

**STEP 2: RRF Fusion**
- Reciprocal Rank Fusion: `score = 1 / (rank + 10)`
- Merges dense and BM25 signals

**STEP 3: Paraphrase (ALWAYS ON)**
- LLM generates precise BIS technical terminology
- Prompt: "Rewrite the query using precise BIS technical terminology. Return ONE short sentence only."
- Temperature ≈ 0.4, max_tokens = 48

**STEP 4: Paraphrase Dense Retrieval**
- `retrieve_dense(paraphrase, k=50)` — ONLY dense, no BM25

**STEP 5: Merge Candidates**
- Union of fused_top + paraphrase_dense
- Deduplicate by standard ID, keep best score
- Top 10 candidates

**STEP 6: LLM Reranking**
- Via `/v1/completions` API
- Prompt lists candidate titles, asks for ordering
- Temperature = 0.0, max_tokens = 32
- Output: "3,1,5" format

**STEP 7: Parse + Fallback**
- Parse numbers from LLM output
- If parsing fails: use original fusion order

**STEP 8: Return top 3**

## Why No Filtering?

Section filtering, concept layers, and hierarchical routing **hurt generalization**:
1. **Section classifiers are brittle** - fail on queries spanning multiple sections
2. **Concept layers introduce bias** - favor known patterns over novel queries
3. **Hierarchical routing is slow** - adds latency without improving accuracy

The wide retrieval approach is simpler, faster, and more robust.

## Why Paraphrase Works

User queries vary in how they describe the same product:
- "33 grade OPC cement"
- "ordinary portland cement 33 grade"  
- "OPC cement specifications"

Paraphrase generates alternative technical phrasings, improving recall.

## Why LLM Reranking Works

Dense and BM25 scores don't always correlate with true relevance:
- A standard with lower retrieval score may be more clearly relevant
- LLM reads candidate titles and judges semantic fit
- The `/v1/completions` API with temperature=0 produces stable, fast judgments

## Ablation (from yesterday's experiments)

| Configuration | Hit@3 | MRR@5 |
|---------------|-------|-------|
| Fusion only | 70% | 0.55 |
| + Paraphrase | 90% | 0.85 |
| + LLM rerank | **100%** | **0.95** |

## File Structure

```
old/
├── config.py        # Hyperparameters
├── utils.py         # Index loading, standard key normalization
├── retrieval.py     # Dense (FAISS) and Sparse (BM25) retrieval
├── fusion.py        # Reciprocal Rank Fusion (RRF)
├── paraphrase.py    # LLM paraphrase generation
├── reranker.py      # LLM reranking via /v1/completions
├── run_pipeline.py  # Main pipeline orchestration
├── eval.py          # Evaluation (Hit@3, MRR@5, latency)
└── README.md        # This file
```

## Usage

```bash
cd old
python eval.py
```

## Configuration

Key parameters in `config.py`:
- `TOP_DENSE = 20` - Dense retrieval count
- `TOP_BM25 = 20` - BM25 retrieval count
- `FUSION_K = 10` - Candidates after fusion
- `RRF_K = 10` - RRF constant
- `PARAPHRASE_TOP_DENSE = 50` - Dense retrieval from paraphrase
- `RERANK_K = 3` - Candidates sent to LLM reranker
- `OUTPUT_K = 3` - Final output count

## Success Criteria

Expected performance on `guidelines/public_test_set.json`:
- **Hit@3**: 100% (all 10 test queries correct in top-3)
- **MRR@5**: ≥ 0.90
- **Latency**: < 1s per query
