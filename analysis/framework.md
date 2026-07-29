# Framework for MOSAIC

This document contains an analysis to create a unified architectural blueprint designed to achieve the maximum possible **Precision, Recall, and F-score** on OAEI benchmarks. This is created directly from top-performing alignment systems sourced from UG students in Computer Science (IN3067) and MSc students (INM713) in Data Science and Software Engineering with Cloud Computing at City St George's, University of London.

# Systems to Analyse

AirAlign, OAS, OmniAlign, Fluffy, MatchCraft, PairMap, CityAligner, AdvancedAlign, MMOAS, MLOA, BlendMap, NTMAlign, Lochys-alignment  

## Key Systems to Utilise

Lochys-alignment, MatchCraft, PairMap, CityAligner, AdvancedAlign, MMOAS, MLOA

## System Techniques Framework Matrix

These are the gathered techniques considered before implementation:

| Category | Techniques | Target OAEI Impact | System Inspirations |
| :--- | :--- | :--- | :--- |
| **1. Lexical** | Lowercase normalization, String-tokenization, Lemmatization, ISUB similarity metric, Levenshtein/RapidFuzz edit distances, Jaro-Winkler prefixes, Character N-Gram Cosine, and Substring boosting. | **Peak Precision** (Safely locks down obvious spelling and string variant matches instantly). | Fluffy, WildAlig, PairMap, AlignKG, Lochys-alignment, VAlign |
| **2. Structural** | Immediate parent checking, Ancestor path confirmation, Hierarchy propagation, Graph Locality Boosting, and Property Neighborhood Jaccard evaluation. | **Eliminates False Positives** (Confirms contextual domain matching using the semantic graph layout). | PairMap, AirAlign, OAS, OmniAlign, NTMAlign |
| **3. Advanced** | Sparse TF-IDF Vectorization, Trigram word patterns, WordNet dictionary sync, Dense Transformers (`all-MiniLM-L6-v2` / `paraphrase-MiniLM-L6-v2`), domain-specific `SapBERT`, and LLM Oracles as tie-breakers. | **Peak Recall** (Bridges the gap on complex synonyms, phrasing variations, and technical jargon). | AdvancedAlign, PairMap, MMOAS, MLOA, CityAligner, MatchCraft |
| **4. Scalability** | Token-based Inverted Index Blocking, TF-IDF bucket partitioning, Target vector caching, Candidate limit pruning, Top-K lexical truncation, and Aho-Corasick data structures. | **System Stability & Speed** (Prevents system timeouts or memory crashes on large Bio-ML/KG datasets). | MatchCraft, PairMap, AdvancedAlign, MMOAS, MMA-Match |
| **5. Other** | Multi-processing parallel batches, Weighted Combiner logic, Greedy 1-to-1 extraction filtering, and `rdfs:subClassOf` Subsumption containment sorting. | **F-Score Optimization** (Cleans up data bindings, handles multi-threading, and routes complex relationships properly). | PairMap, WildAlig, MatchCraft, OAS, NTMAlign, MySystem |

## Pipeline Execution Overview

To implement these techniques efficiently without performance bottlenecks, route data linearly across the engineered layers:

1. **Stage 1: Scalability Filter (Category 4)** -> Apply token-based inverted index blocking to narrow down candidate pairs.
2. **Stage 2: Lexical Layer (Category 1)** -> Compute rapid exact mappings and character-level metrics (ISUB) on the remainder.
3. **Stage 3: Advanced Semantic Layer (Category 3)** -> Generate dense transformer cosine similarities for synonym resolution.
4. **Stage 4: Structural Boosting (Category 2)** -> Apply hierarchy and neighborhood context score validation bonuses.
5. **Stage 5: Final Resolution (Category 5)** -> Filter mappings using a greedy 1-to-1 matching constraint for maximum final F-score.

This creates a filter and refine paradigm.

# Implementation

MOSAIC implements the five-stage pipeline above as a modular, local, deterministic matching engine with no reliance on external generative LLMs at inference time.

**Domain Detection & Adaptive Encoder Selection.** Entity URIs, metadata labels, and token distributions are inspected before indexing to select an embedding encoder:
- **Small scale (< 3,000 entities):** `BAAI/bge-m3` (64-token sequence limit) for fine-grained representations.
- **Biomedical domain:** `SapBERT` (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`) for medical terminology.
- **General scale:** `all-MiniLM-L12-v2` for fast, lightweight vectors.

**Vector Retrieval & Candidate Reduction.** Entities are embedded into normalised vector spaces, cached locally as memory-mapped float16 arrays, and searched via FAISS:
- **Standard schemas (< 60,000 entities):** `IndexIVFFlat` with adaptive candidate pool size (k).
- **Large-scale schemas (≥ 60,000 entities):** `IndexIVFPQ` (m = 16 sub-quantizers, 8-bit codes) to bound memory.

**Hybrid Scoring Matrix.** Candidate pairs (s, t) are scored by combining semantic cosine similarity, lexical distance, and structural Jaccard overlap:

S(s, t) = (0.60 · S_sem) + (0.40 · S_str) + S_struct

where the lexical component blends character n-grams, Levenshtein distance, and ISUB similarity:

S_str = (0.30 · S_ngram) + (0.35 · S_lev) + (0.35 · S_isub)

**Precision Guardrails & Filtering.** Four controls suppress false positives:
1. **Length Ratio Guard** – filters pairs whose label length ratio exceeds 2.5.
2. **Hubness Discounting** – progressive penalty (0.03) on targets retrieved across excessive query instances.
3. **Reciprocal Symmetry Check** – discards forward matches whose reverse-search score diverges by more than 0.20.
4. **Cardinality Optimization** – pre-locks 1-to-1 exact label matches and enforces overall 1-to-1 constraints via greedy priority queue.

**Core execution flow:**

```python
from mosaic import MOSAIC, MOSAICConfig, OAEITrackRunner, MedicalDomainDetector

# 1. Initialize configuration and engine
config = MOSAICConfig(
    semantic_weight=0.60,
    string_weight=0.40,
    use_fp16_compute=True
)
matcher = MOSAIC(config=config, device="cuda")

# 2. Domain detection and model assignment
is_med, _ = MedicalDomainDetector.evaluate_is_medical(
    "anatomy", "human-mouse", src_entities, tgt_entities
)
matcher.apply_domain_model(
    is_medical=is_med,
    total_entities=len(src_entities) + len(tgt_entities)
)

# 3. Dense search and precision filtering
alignments = matcher.align_optimized(src_entities, tgt_entities)

# 4. Export results and compute metrics
runner = OAEITrackRunner(matcher=matcher)
rdf_triples = matcher.convert_to_rdf_triples(alignments)
metrics = runner.calculate_metrics(rdf_triples, reference_alignments)
```

The implementation lives in `amdCode.py`, optimised for AMD support but CUDA-compatible (usable on NVIDIA hardware).

## Development of Choices

The general-purpose backbone went through several iterations before settling: `paraphrase-multilingual-MiniLM-L12-v2` → `LaBSE` → `all-MiniLM-L6-v2` → the final domain/scale-adaptive split (`bge-m3` / `SapBERT` / `all-MiniLM-L12-v2`), each swap trading off multilingual coverage, memory, speed, and recall in turn. Early versions used a fast word-token index for blocking, combined character/substring/word text metrics, WordNet synonym checks plus SentenceBERT embeddings, and a greedy 1-to-1 filter, with triple comparisons made order-independent.

Semantic similarity moved from `sentence_transformers.util.cos_sim` to a single-pass FAISS nearest-neighbour search (`faiss.IndexFlatIP`), later replaced by adaptive `IndexIVFFlat`/`IndexIVFPQ` selection with size-scaled k and nprobe values. Unused WordNet synset lookups were removed once profiling showed they were computed but never read downstream, and similarity scoring was batched per chunk instead of per entity for speed.

Memory efficiency drove several changes: full graph-extraction caching was replaced with streaming entity extraction generators, embedding caches were capped and disk-backed, and dynamic INT8 CPU quantization was dropped in favour of a lighter default model — pushing the system toward GPU-dependent execution for peak performance, with AMD tested on Linux and CUDA support added for NVIDIA. A CPU fallback path remains but performs poorly on very large ontologies.

The precision guardrails evolved from separate fixes: a mutual-match verification coefficient was tightened from 0.88 to 0.91 to filter weak reverse matches (the basis of the reciprocal symmetry check), a candidate noise filter discarded low-scoring semantic matches with poor string similarity (ngram + Levenshtein average < 0.15), and length-imbalance ratios plus short-token dampening were added later to catch mismatched label lengths. An early, cruder size-difference penalty (dividing reported ontology size by 4 when one side was much smaller) was generalised into the current length-ratio guard.

Parsing evolved from basic graph traversal to `pyoxigraph`/`pyhornedowl`-based extraction with O(1) dictionary lookups, RDF/TTL/TSV/CSV reference format support, and Wiki/DBKwik namespace stripping — improving both speed and format coverage.

Large-ontology recall remained the most persistent open problem throughout development, consistent with the comparatively low F1 scores on Bio-ML and Knowledge Graph in the final results below.

# Evaluation

MOSAIC was evaluated across 39 matching tasks spanning 6 OAEI tracks, using GPU-accelerated local hardware.

| Track | Precision | Recall | F1-Score | Time (s) |
| :--- | :--- | :--- | :--- | :--- |
| Anatomy | 0.8574 | 0.8015 | 0.8285 | 7.51 |
| Bio-ML | 0.5557 | 0.3766 | 0.4220 | 184.86 |
| Circular Economy | 0.3976 | 0.9171 | 0.5344 | 4.97 |
| Conference | 0.6076 | 0.5838 | 0.5813 | 2.45 |
| Digital Humanities | 0.6832 | 0.5569 | 0.5938 | 2.29 |
| Knowledge Graph | 0.3634 | 0.7376 | 0.4832 | 114.05 |
| **Overall Avg** | **0.5815** | **0.6054** | **0.5626** | **52.69** |

Representative task results: human-mouse (Anatomy, F1 0.828); ncit-doid, snomed-fma, snomed-ncit (Bio-ML, F1 0.531/0.248/0.488); CEON-BiOnto, CEON-MATONTO (Circular Economy, F1 0.684/0.385); defc-pactols1, ironagedanube-pactols3 (Digital Humanities, F1 0.737/0.727); memoryalpha-stexpanded, starwars-swtor (Knowledge Graph, F1 0.531/0.505).

## Evaluation of Choices

- **Specialised domains perform best:** Anatomy hits the highest F1 (0.8285) in 7.51s — the SapBERT backbone plus exact string pre-pass pays off directly.
- **Recall trade-off on small ontologies:** Circular Economy reaches high recall (0.9171) but low precision (0.3976), showing the semantic-weight bias toward recall isn't fully checked by the guardrails on sparse, small ontologies.
- **Scalability holds, speed doesn't scale as well:** Knowledge Graph and Bio-ML complete without memory overflow, but runtime rises sharply (114–185s avg) versus the sub-3s Conference/Digital Humanities tracks — blocking prevents crashes but not slowdown at scale.
- **Local-only design costs recall on hard cases:** the lower F1 on Bio-ML (0.4220) and Knowledge Graph (0.4832) suggests complex synonym resolution is the weakest link without an LLM tie-breaker.
- **Runtime target met on simpler tracks:** Conference (21 tasks) and Digital Humanities (7 tasks) both average under 2.5s/task, confirming the filter-and-refine ordering works well at moderate scale.

# Final Credits of Contribution

This section contains the references of work used from the final code.

| Referenced System | Repository | Attributed Architectural Features & Techniques |
| :--- | :--- | :--- |
| **AdvancedAlign** (Primary Author: Faiz Mirza) | [Semantic-Phone-Aligner](https://github.com/FaizMirza321/Semantic-Phone-Aligner) | Primary foundational baseline; establishes lexical normalisation, dictionary lookup pre-pass, token blocking via inverted indexes, and sentence transformer embeddings. |
| **PairMap** | [PairMap](https://github.com/Myo-Shwe-Sin-Ei/PairMap) | Domain-adaptive backbone selection using SapBERT (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`). |
| **AlignKG** | [swtkgcw2](https://github.com/OlegPaska/swtkgcw2) | Composite multi-metric lexical similarity matrix blending ISUB, Levenshtein, and n-gram representations. |
| **MatchCraft** | [ontology-alignment-230044396](https://github.com/miavo090821/ontology-alignment-230044396) | Scale-aware inverted index blocking for candidate retrieval space reduction across small and large ontologies. |
| **AirAlign** | [AirAlign](https://github.com/anushka0110/AirAlign) | Structural Jaccard distance modeling across contextual parent classes and concept neighborhoods. |
| **Lochys-alignment** | [Loch-align](https://github.com/Lochy2000/Loch-align) | Exact label pre-locking pass and 1-to-1 cardinality greedy priority queue optimization. |
| **MLOA** | [INM713-Coursework-Part-2-Ontology-Alignment](https://github.com/Matthew1819Lau/INM713-Coursework-Part-2-Ontology-Alignment) | Hierarchical structural filtering guardrails to suppress false positive candidate mappings. |
| **BlendMap** | [ontology-alignment](https://github.com/sercan00/ontology-alignment) | Expanded transformer sequence length representations via MiniLM-L12 model backbones. |