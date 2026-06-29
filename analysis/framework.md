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



## Development of Choices




# Evaluation

## Evaluation of Choices




# Final Credits of Contribution

This section contains the references of work used from the final code.
