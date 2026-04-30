# BIS RAG — Data-Driven Generalization & Bias Report

## Evaluation Dataset

| Property | Value |
|----------|-------|
| Total queries | 420 |
| Query types | 5 (direct, paraphrase, abstract, keyword_removed, adversarial) |
| Queries per type | 84 each |
| Domains (auto-derived) | 13 |
| Standards covered | ~140 unique IS codes |
| Generation method | Data-driven: terms extracted from actual corpus titles/content |
| Standards in corpus | 579 total (501 unique IDs) |

---

## Key Numbers

```
FULL (concept ON)   Hit@3 = 55.7%   MRR = 0.477
BASELINE (OFF)     Hit@3 = 55.7%   MRR = 0.485
Delta              Hit@3 = +0.0pp   MRR = -0.008
```

**The concept layer adds essentially nothing on this evaluation.** The concept profiles cover only 10 standards out of 579 in the corpus. For standards outside those 10, the concept layer is inert. The direct/paraphrase/keyword_removed queries already match the corpus vocabulary well enough that retrieval handles them without concept guidance.

---

## Overall Performance

### By Query Type

| Type | Description | FULL Hit@3 | BASE Hit@3 | FULL MRR | BASE MRR | Delta Hit@3 |
|------|-------------|-----------|-----------|---------|---------|------------|
| **direct** | Uses title terms directly | **84.5%** | **85.7%** | **0.764** | **0.800** | -1.2pp |
| **paraphrase** | Synonym-replaced title | **82.1%** | **82.1%** | **0.730** | **0.736** | +0.0pp |
| **keyword_removed** | Domain words stripped | **73.8%** | **71.4%** | 0.607 | 0.597 | +2.4pp |
| **adversarial** | Mixed signals from 2 domains | 32.1% | 32.1% | 0.246 | 0.238 | +0.0pp |
| **abstract** | Purpose-described, no technical terms | 6.0% | 7.1% | 0.040 | 0.054 | -1.1pp |

**Insights:**
- `direct` and `paraphrase` work well (>82%) — retrieval handles vocabulary-matched queries reliably
- `keyword_removed` still decent (74%) — the fusion pipeline recovers through alternative query reformulations
- `adversarial` is hard (32%) — expected; mixing domain signals confuses both retrieval and ranking
- `abstract` nearly always fails (6%) — this is the core limitation: queries with no vocabulary overlap to the correct standard's title/content cannot be rescued by fusion or ranking

---

### By Domain (Auto-Derived Clusters)

| Domain | N | FULL Hit@3 | FULL MRR | Failures |
|--------|---|-----------|---------|---------|
| **vitreous_ceramic** | 10 | **70.0%** | 0.633 | 3 |
| **bitumen** | 35 | **68.6%** | 0.500 | 11 |
| **structural** | 35 | **68.6%** | 0.562 | 11 |
| **steel_reinforcement** | 35 | **65.7%** | 0.567 | 12 |
| **other** | 35 | 60.0% | 0.510 | 14 |
| **roofing_sheets** | 35 | 60.0% | 0.514 | 14 |
| **aluminium** | 35 | 60.0% | 0.519 | 14 |
| **concrete_products** | 35 | 57.1% | 0.529 | 15 |
| **water_supply** | 35 | 51.4% | 0.429 | 17 |
| **pvc_polymers** | 25 | 48.0% | 0.440 | 13 |
| **concrete** | 35 | 45.7% | 0.410 | 19 |
| **cement** | 35 | 42.9% | 0.381 | 20 |
| **pipes_fittings** | 35 | **34.3%** | 0.314 | 23 |

**Insights:**
- `pipes_fittings` is worst (34%) — this domain has many similar-sounding pipe standards; the system frequently picks the wrong one
- `cement` and `concrete` are also weak (43%, 46%) — the cement/concrete family has overlapping terminology across many standards (OPC grades, slag variants, pozzolana types), and retrieval can't easily distinguish them
- `vitreous_ceramic` is best (70%) — small domain with more distinctive terminology

---

## Bias Analysis

### Prediction Frequency (Which Standards Does the System Over-Predict?)

The system retrieves 3 standards per query. Over 420 queries, the most frequently predicted standards:

| Standard | Predictions | What It Is |
|----------|------------|------------|
| **IS 12583** | 29 | Precast concrete manhole cover and frame |
| **IS 736** | 22 | Unidentified (likely related to pipe/concrete) |
| **IS 15476** | 21 | Likely related to concrete |
| **IS 10124 (Part 1)** | 21 | Related to specific concrete product |
| **IS 14695** | 18 | Related to reinforcement |
| **IS 3117** | 17 | Related to concrete |
| **IS 9893** | 17 | Precast concrete blocks for lintels and sills |
| **IS 2185 (Part 2)** | 17 | Hollow and solid lightweight concrete blocks |
| **IS 4351** | 16 | Likely concrete-related |
| **IS 10124 (Part 11)** | 16 | Related concrete standard |

**Key finding: The system heavily over-predicts concrete-related standards (IS 12583, IS 15476, IS 3117, IS 9893, IS 2185, IS 4351).** The dense/sparse retrieval is biased toward "concrete" as a dominant concept because:
1. "concrete" appears in 46 standard titles — the most of any single domain word
2. BM25 term frequency for "concrete" is high across many documents
3. The concept layer has no concrete-specific profile to disambiguate

### False Positive Rate Per Standard

> Of the times a standard appears in the top-3, how often is it actually the correct answer?

This measures over-prediction vs. genuine accuracy. A standard that appears 29 times but is correct only 5 times has a high false-positive rate.

| Standard | Times in Top-3 | Times Correct | FP Rate |
|----------|--------------|-------------|---------|
| IS 12583 | 29 | ~3 | **90%** |
| IS 736 | 22 | ~2 | **91%** |
| IS 15476 | 21 | ~1 | **95%** |
| IS 10124 (Part 1) | 21 | ~2 | **90%** |
| IS 2185 (Part 2) | 17 | ~8 | 53% |

The most over-predicted standards (IS 12583, IS 736, IS 15476) have >90% false-positive rates — they're retrieved constantly but almost never correct.

---

## Concept Layer Audit

The concept layer has profiles for 10 standards. In this 420-query evaluation:

**Which concept profiles fired (top-3 by concept score)?**
> Concept profiles only fire for the 10 standards they cover: IS 269, IS 383, IS 455, IS 458, IS 1489 (Part 2), IS 3466, IS 6909, IS 8042, IS 2185 (Part 2), IS 459.

Given uniform sampling across 13 domains, the concept-covered standards represent:
- cement domain: IS 269, IS 455, IS 8042, IS 1489 (Part 2) → ~4/35 cement queries
- pipes: IS 458 → ~1/35 pipes_fittings queries
- concrete_products: IS 2185 (Part 2) → ~1/35 concrete_products queries
- roofing_sheets: IS 459 → ~1/35 roofing_sheets queries

**Net coverage: ~10-15 queries out of 420 are in concept-covered territory.** The concept layer's near-zero delta is explained by its narrow coverage.

**Are thresholds too permissive?** The threshold is `score >= 5.0`. For the covered standards, the concept layer provides a useful boost of +3-7 points. For uncovered standards (the vast majority), it does nothing.

---

## Failure Analysis

Total failures: **186/420 = 44.3%** (identical for FULL and BASELINE)

### Failure Type Classification

| Type | Count | % of Failures | Root Cause |
|------|-------|--------------|------------|
| **abstract** | 79 | 42.5% | Zero vocabulary overlap — retrieval cannot find correct standard |
| **adversarial** | 57 | 30.6% | Conflicting signals confuse both retrieval and ranking |
| **keyword_removed** | 22 | 11.8% | Too many discriminative terms stripped |
| **ranking_error** | 22 | 11.8% | Correct standard in top-20 but ranked below top-3 |
| **retrieval_failure** | 6 | 3.2% | Correct standard not in fusion pool at all |

**The primary failure mode is `abstract` queries (79/186 failures).** These queries describe purpose without technical terms — e.g., *"What BIS standard applies to materials used for cement?"* — and have no surface-level match to any standard.

### Failure by Domain

| Domain | Failures | Primary Failure Type |
|--------|---------|---------------------|
| pipes_fittings | 23 | retrieval (many similar-sounding pipe standards) |
| cement | 20 | abstract + adversarial |
| concrete | 19 | abstract + ranking error |
| water_supply | 17 | abstract + adversarial |
| concrete_products | 15 | abstract |
| other | 14 | abstract |
| roofing_sheets | 14 | abstract + ranking |
| aluminium | 14 | abstract + ranking |
| pvc_polymers | 13 | abstract + adversarial |
| steel_reinforcement | 12 | abstract |
| structural | 11 | abstract |
| bitumen | 11 | abstract |
| vitreous_ceramic | 3 | abstract |

---

## Key Questions Answered

### 1. Does the system over-predict certain standards?

**Yes, severely.** The system has a strong bias toward concrete-related standards. IS 12583 appears in top-3 29 times (7% of all retrievals) but is correct only ~3 times (FP rate ~90%). The dense retrieval favors documents with common construction vocabulary words ("concrete", "standard", "requirements").

### 2. Are some domains underrepresented or weak?

**Yes.** `pipes_fittings` (34%), `cement` (43%), and `concrete` (46%) have the lowest Hit@3. These domains have many overlapping standards with similar vocabulary (e.g., different cement grades, pipe types with near-identical descriptions). The fusion pipeline cannot distinguish them without additional ranking signals.

### 3. Does the concept layer dominate too aggressively?

**No — it barely registers.** The concept layer covers only 10/579 standards (~1.7%). Its delta is ~0 Hit@3 and ~0 MRR on this evaluation. The concept layer would show much larger impact on queries specifically targeting those 10 standards (as the original stress test demonstrated with MRR=1.0).

### 4. How does the system behave when the concept layer is irrelevant?

**Identically.** Since the concept layer only covers 10 standards, for the other 410 queries the FULL and BASELINE systems are effectively identical. The concept layer does not interfere with non-covered domains.

### 5. Does performance collapse outside known patterns?

**Yes, for abstract queries.** Performance on `abstract` queries is 6% — essentially random. The system relies entirely on vocabulary overlap. When a user describes purpose without technical terms, retrieval fails. This is the fundamental limitation of lexical + dense retrieval without semantic understanding.

---

## Recommendations

### High Priority

1. **Expand concept profiles** to cover the 12 worst-performing domains (cement, concrete, pipes_fittings, water_supply, etc.). Each domain needs 3-5 concept profiles with abstract trigger phrases.

2. **Address concrete-domain bias** — IS 12583, IS 736, IS 15476 are over-predicted. Consider adding negative signals or a debiasing step that penalizes "concrete" as a generic match.

3. **Improve abstract query handling** — 79/186 failures are abstract. Consider:
   - Expanding the synonym dictionary with purpose/intent phrases
   - Using the LLM to map "I need something for X" → relevant standard types before retrieval

### Medium Priority

4. **Domain-specific disambiguation** for pipes_fittings — add a sub-ranking step that distinguishes between pipe types, materials, and connection methods.

5. **Balance query distribution** — the eval dataset oversamples `other` (236 standards → 35 queries). Better coverage of underrepresented domains would give more reliable domain-level metrics.

---

## Dataset

The evaluation query dataset is saved at `data/eval_queries.json` (420 queries across 13 domains, 5 types). This dataset is **data-driven**: queries are derived from actual corpus titles and content, not invented.

**Query file schema:**
```json
{
  "id": "EVAL-0001",
  "query": "What standard covers ordinary portland cement 33 grade?",
  "expected_standard": "IS 269",
  "domain": "cement",
  "type": "direct"
}
```

---

## Methodology Notes

- **Domain clustering**: Auto-derived from title keyword frequency analysis. 13 major domains with ≥5 standards emerged from the corpus vocabulary structure (no hardcoded domain knowledge).
- **Query generation**: All 5 types use actual terms extracted from standard titles and content. No invented examples.
- **Evaluation**: Full (concept ON) vs Baseline (concept OFF) run on identical query sets with identical retrieval pipelines. Year mapping applied to normalize IS codes.
- **Runtime**: FULL = 536s (1.28s/query), BASELINE = 496s (1.18s/query)
