# BIS Standards Discovery

A production-ready RAG pipeline with an interactive dashboard that maps natural-language queries about Indian BIS construction standards to the correct IS codes. Achieves **MRR=0.920** with deterministic ranking and ~1.6 second latency.

---

## Project Overview

Given a query like:
> *"What is the Indian Standard covering the manufacture, chemical, and physical requirements for Portland slag cement?"*

It returns five recommended BIS standards ranked by relevance, plus an AI-generated rationale:
```json
{
  "retrieved": ["IS 455: 1989", "IS 269: 1989", "IS 1489 (Part 1): 1991", "IS 8043: 1991", "IS 1489 (Part 2): 1991"],
  "rationale": "IS 455 is the standard for Portland slag cement, covering its manufacture and physical requirements for use in marine environments.",
  "latency_seconds": 1.6
}
```

### Two Search Modes

**Option 1 - AI Agent Search:**
Direct natural-language query input with AI-powered ranking and rationale generation.

**Option 2 - Guided Discovery:**
Click-through category → keyword → standards workflow for structured browsing.

### Hackathon Performance

| Metric | Result | Target |
|--------|--------|--------|
| Hit Rate @3 | **90%** | >80% |
| MRR @5 | **0.920** | >0.7 |
| Avg Latency | **~1.6s** | <5s |

---

## Architecture

```mermaid
flowchart TD
    Q[User Query] --> R[Retrieval]
    R --> D[Dense FAISS<br/>Embedding top-25]
    R --> B[BM25 Sparse<br/>top-25]
    D --> F[RRF Fusion]
    B --> F
    F --> S[Feature Scoring]
    S --> FR[Family Resolution]
    FR --> C{Confidence<br/>Margin >10?}
    C -->|No| FB[Fallback<br/>Pool 30]
    FB --> S
    C -->|OK| O[Output Top-5]
    O --> L[LLM Rationale<br/>Ollama]
```

### Pipeline Stages

```
Query: "Portland slag cement chemical requirements"
│
├─[1] PARSE QUERY SIGNALS
│   ├─ Extract keywords, bigrams, product types
│   ├─ Detect "part" mentions, IS numbers
│   └─ Identify material types (Portland, slag, cement)
│
├─[2] MULTI-QUERY RETRIEVAL
│   ├─ Dense (FAISS BGE-M3, top-25)
│   ├─ Sparse (BM25, top-25)
│   └─ RRF fusion → candidate pool
│
├─[3] FEATURE SCORING
│   ├─ IS number exact match: +36
│   ├─ Keyword/bigram overlap: weighted scoring
│   ├─ Product type matching: +11 per match
│   ├─ Mutual exclusivity penalties: -24 per mismatch
│   └─ Part alignment bonus/penalty: +18/-12
│
├─[4] FAMILY RESOLUTION
│   ├─ Group candidates by IS number family
│   ├─ Boost correct part variant when query specifies part
│   └─ Penalize wrong part variants
│
├─[5] CONFIDENCE CHECK
│   └─ If margin < 10 → fallback to larger candidate pool
│
└─[6] OUTPUT
    ├─ Format standards with year (e.g., "IS 455: 1989")
    ├─ Generate LLM rationale via Ollama
    └─ Return top-5 results
```

---

## File Structure

```
bis_rag/
├── app.py                    # FastAPI dashboard
├── inference.py              # ⭐ Submission entry point
├── eval_script.py            # Official evaluator
├── requirements.txt         # Dependencies
├── uv.lock                   # Locked versions
├── README.md
├── src/
│   ├── bis_parser.py         # PDF → sp21_standards.json
│   ├── build_index.py        # Build FAISS + BM25 indexes
│   └── data/
│       ├── faiss_index.bin        # Dense vector index
│       ├── bm25_index.pkl         # BM25 sparse index
│       ├── whitelist.txt          # Approved IS codes (576 entries)
│       ├── embedding_config.json  # Embedding model config
│       ├── metadata_store.json    # IS code metadata
│       ├── section_profiles.json # Category profiles
│       ├── sp21_standards.json   # Source corpus
│       ├── standard_to_section.json
│       └── graph_map.json
├── static/
│   ├── css/style.css
│   ├── js/script.js
│   └── favicon.ico
└── templates/
    └── index.html
```

---

## Quick Start

### 1. Install Dependencies

```bash
uv pip install -r requirements.txt
```

### 2. Start Ollama (Optional - for LLM rationale)

```bash
ollama serve
ollama pull qwen3.5:4b
```

### 3. Run Dashboard

```bash
python app.py
# Open http://localhost:8000
```

### 4. Run Inference (Submission Mode)

```bash
python inference.py \
  --input guidelines/public_test_set.json \
  --output results.json
```

### 5. Evaluate

```bash
python eval_script.py --results results.json
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LM_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `LM_API_KEY` | `ollama` | API key |
| `LM_MODEL` | `qwen3.5:4b` | Model name (recommended: qwen3.5:4b) |
| `BIS_FORCE_CPU` | `0` | Set to `1` to force CPU |

### CPU/CUDA Behavior

```
BIS_FORCE_CPU=1  →  Force CPU (overrides CUDA)
         ↓
torch.cuda.is_available()?
         ↓
    Yes → Use CUDA for embeddings
         ↓
     No → CPU fallback
```

---

## Feature Scoring Weights

| Feature | Weight | Purpose |
|---------|--------|---------|
| IS number exact match | +36 | Direct mention in query |
| Keyword overlap | +4/bigram | Semantic matching |
| Bigram overlap | +6/bigram | Phrase matching |
| Product type match | +11/match | Material classification |
| Mutual exclusivity | -24/mismatch | Prevents wrong family |
| Part alignment | +18/-12 | Correct Part variant |
| Near-ID penalty | -16 | e.g., 736 vs 737 |

---

## Rebuild Pipeline (If Corpus Changes)

If you modify `src/data/sp21_standards.json`, rebuild the indexes:

```bash
# Parse PDF → sp21_standards.json
python src/bis_parser.py --input SP21.pdf --output src/data/sp21_standards.json

# Build FAISS + BM25 indexes
python src/build_index.py
```

---

## Reproducibility

- Deterministic ranking (no random operations)
- LLM rationale optional (falls back to template if Ollama unavailable)
- CPU-only reproducible on any machine
- CUDA auto-detected if available

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.6.0 | CUDA detection, tensor ops |
| sentence-transformers | 2.7.0 | BGE-M3 embeddings |
| faiss-cpu | 1.13.2 | Dense vector search |
| numpy | >=1.26.0 | Numerical operations |
| fastapi | 0.136.1 | Web framework |
| uvicorn | 0.46.0 | ASGI server |
| pydantic | 2.13.3 | Data validation |
| rank-bm25 | 0.2.2 | BM25 sparse retrieval |
| pypdf | 6.10.2 | PDF parsing |

---

## Submission Checklist

```bash
# 1. Run inference
python inference.py --input guidelines/public_test_set.json --output results.json

# 2. Evaluate
python eval_script.py --results results.json

# 3. Verify output format
# Should have: id, retrieved_standards, latency_seconds
# Target: Hit@3 >80%, MRR >0.7, Latency <5s
```
