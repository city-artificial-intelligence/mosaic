import csv, gc, re, time, warnings, logging
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, List, Tuple, Set, Dict, Optional
from collections import defaultdict, Counter
import numpy as np
import torch
import faiss

from sentence_transformers import SentenceTransformer
from rdflib import Graph, RDF, RDFS, URIRef
from rdflib.namespace import OWL, SKOS

# Optional dependency detection
try:
    import pyoxigraph
    HAS_PYOXIGRAPH = True
except ImportError:
    HAS_PYOXIGRAPH = False

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

# Suppress verbose warnings and RDFLib log messages
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
logging.getLogger("rdflib").setLevel(logging.ERROR)

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR / "../tracks"
RESULTS_DIR = SCRIPT_DIR / "results"
CACHE_DIR = SCRIPT_DIR / ".embedding_cache"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_CORE_NAMESPACES = (str(RDF), str(RDFS), str(OWL), str(SKOS))


class MedicalDomainDetector:
    """Classifies ontologies into medical or general domain using heuristics, URIs, and vocabulary."""
    MEDICAL_URI_PATTERNS = [
        re.compile(r'purl\.obolibrary\.org/obo/(DOID|HP|MP|MONDO|CHEBI|NCIT|UBERON|CL|GO|PR|SYMP|VO|MA|FMA)_', re.I),
        re.compile(r'snomed\.info/id/', re.I),
        re.compile(r'nlm\.nih\.gov/mesh/', re.I),
        re.compile(r'bioportal\.bioontology\.org/ontologies/', re.I),
        re.compile(r'identifiers\.org/(doid|hp|chebi|mesh|ncit)', re.I)
    ]
    MEDICAL_VOCAB_ROOTS = {
        "disease", "syndrome", "disorder", "symptom", "phenotype", "carcinoma", "tumor", 
        "neoplasm", "lesion", "infection", "pathology", "anatomy", "tissue", "organ", 
        "cell", "protein", "receptor", "peptide", "amino", "molecule", "chemical", 
        "compound", "pharmaceutical", "drug", "therapy", "clinical", "patient", "mutation"
    }

    @classmethod
    def evaluate_is_medical(cls, track_name: str, task_name: str, 
                           src_entities: Dict, tgt_entities: Dict) -> Tuple[bool, str]:
        """Evaluates whether track/task metadata or entity attributes belong to a medical domain."""
        combined_id = f"{track_name} {task_name}".lower()
        if any(kw in combined_id for kw in ["anatomy", "disease", "phenotype", "human-mouse", "pharmaceutical", "med"]):
            return True, "Task/Track Name Heuristics"
            
        all_uris = list(src_entities.keys()) + list(tgt_entities.keys())
        if not all_uris:
            return False, "No entities found"
            
        sample_uris = all_uris[:1000]
        matched_uris = sum(1 for uri in sample_uris if any(p.search(uri) for p in cls.MEDICAL_URI_PATTERNS))
        uri_ratio = matched_uris / len(sample_uris)
        if uri_ratio >= 0.05:
            return True, f"URI Namespace Match ({uri_ratio:.1%} matches)"

        all_labels = [meta["label"] for meta in list(src_entities.values())[:500]] + \
                     [meta["label"] for meta in list(tgt_entities.values())[:500]]
        
        medical_tokens = sum(len(set(lbl.lower().split()).intersection(cls.MEDICAL_VOCAB_ROOTS)) for lbl in all_labels)
        total_tokens = sum(len(set(lbl.lower().split())) for lbl in all_labels)
            
        if total_tokens > 0 and (medical_tokens / total_tokens) >= 0.03:
            return True, f"Biomedical Token Density ({(medical_tokens / total_tokens):.1%} density)"

        return False, "General Domain Fallback"


@dataclass
class AlignmentCandidate:
    """Data container for an alignment pair with individual and combined similarity metrics."""
    source: str
    target: str
    etype: str
    semantic_score: float
    ngram_score: float = 0.0
    levenshtein_score: float = 0.0
    isub_score: float = 0.0
    structural_bonus: float = 0.0
    
    @property
    def combined_score(self) -> float:
        """Calculates weighted similarity score combined with structural adjustments."""
        string_score = (self.ngram_score * 0.25) + (self.levenshtein_score * 0.35) + (self.isub_score * 0.40)
        base = (self.semantic_score * 0.60) + (string_score * 0.40)
        return max(0.0, min(1.0, base + self.structural_bonus))


class EmbeddingDiskCache:
    """Manages memory-mapped disk storage for persistent label embeddings."""
    def __init__(self, cache_dir: Path, embedding_dim: int):
        self.cache_dir, self.embedding_dim = cache_dir, embedding_dim
        self.label_to_idx, self.embeddings_mmap = {}, None
        self.current_count, self.max_embeddings = 0, 3_000_000
        self._init_mmap()
    
    def _init_mmap(self):
        """Initializes or loads existing memory-mapped numpy matrix and label index."""
        mmap_path, index_path = self.cache_dir / "embeddings.npy", self.cache_dir / "labels.txt"
        if mmap_path.exists() and index_path.exists():
            try:
                self.embeddings_mmap = np.load(str(mmap_path), mmap_mode='r+')
                with open(index_path, 'r', encoding='utf-8') as f:
                    self.label_to_idx = {line.strip(): idx for idx, line in enumerate(f)}
                self.current_count = len(self.label_to_idx)
                return
            except Exception: pass
        
        self.embeddings_mmap = np.memmap(str(mmap_path), dtype='float32', mode='w+', shape=(self.max_embeddings, self.embedding_dim))
        self.current_count, self.label_to_idx = 0, {}
    
    def add_batch(self, labels: List[str], embeddings: np.ndarray):
        """Appends a new batch of text labels and their corresponding embeddings to cache."""
        if not labels: return
        start_idx = self.current_count
        end_idx = min(start_idx + len(labels), self.max_embeddings)
        labels, embeddings = labels[:end_idx - start_idx], embeddings[:end_idx - start_idx]
        if not labels: return
            
        self.embeddings_mmap[start_idx:end_idx] = embeddings
        self.embeddings_mmap.flush()
        for i, label in enumerate(labels):
            self.label_to_idx[label] = start_idx + i
        self.current_count = end_idx
    
    def save_index(self):
        """Saves current text-label-to-row indexing metadata to disk."""
        sorted_labels = [None] * self.current_count
        for label, idx in self.label_to_idx.items():
            if idx < self.current_count: sorted_labels[idx] = label
        with open(self.cache_dir / "labels.txt", 'w', encoding='utf-8') as f:
            f.writelines(f"{lbl}\n" for lbl in sorted_labels if lbl is not None)


class MOSAIC:
    """Core ontology matching system utilizing language embeddings, string metrics, and FAISS vector indices."""
    _CAMEL_RE = re.compile(r'([a-z])([A-Z])')
    _PCT_RE = re.compile(r'%[0-9A-Fa-f]{2}')
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
    SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

    def __init__(self, model_name=None, thresholds=None, device="cuda"):
        self.thresholds = thresholds or {
            OWL.Class: 0.82, SKOS.Concept: 0.85, OWL.ObjectProperty: 0.80,
            OWL.DatatypeProperty: 0.80, OWL.NamedIndividual: 0.78
        }
        self.default_thres, self.default_model_name = 0.80, model_name or self.DEFAULT_MODEL
        self.current_model_name, self.is_medical_domain, self.device = None, False, device
        self.model, self.embedding_dim, self.ontology_size, self.is_mega_scale = None, None, 0, False
        self.embedding_cache = None
        self._ngram_cache, self._lev_cache, self._isub_cache = {}, {}, {}

    def apply_domain_model(self, is_medical: bool):
        """Loads domain-specific transformer models (SapBERT vs MiniLM) dynamically."""
        self.is_medical_domain = is_medical
        target_model = self.SAPBERT_MODEL if self.is_medical_domain else self.default_model_name
        
        if self.current_model_name != target_model:
            print(f" [INFO] Switching Backbone Model to: {target_model}")
            self.model = SentenceTransformer(target_model, device=self.device)
            try: self.model.max_seq_length = 128
            except Exception: pass
            
            self.current_model_name = target_model
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            model_cache_dir = CACHE_DIR / ("sapbert" if self.is_medical_domain else "minilm")
            model_cache_dir.mkdir(parents=True, exist_ok=True)
            self.embedding_cache = EmbeddingDiskCache(model_cache_dir, embedding_dim=self.embedding_dim)

    def init_model(self):
        """Lazy loader initializing transformer neural network standard backbone."""
        if self.model is None: self.apply_domain_model(is_medical=False)

    def normalise_label(self, text: str) -> str:
        """Cleans, strips prefixes, un-camels, and normalizes label strings."""
        if not text: return ""
        text = str(text)
        for prefix in ["Category:", "Template:", "File:", "Property:", "Category_talk:", "User:"]:
            if text.startswith(prefix): text = text[len(prefix):]
        text = self._CAMEL_RE.sub(r'\1 \2', text).replace('_', ' ').replace('-', ' ').lower().strip()
        return " ".join(text.split()[:20])

    def load_ontology(self, path: Path) -> Optional[Graph]:
        """Parses an ontology file path into an RDFLib graph instance."""
        g = Graph()
        formats = ["turtle", "xml"] if path.suffix == ".ttl" else ["xml", "turtle"]
        for fmt in formats:
            try: g.parse(str(path), format=fmt); return g
            except Exception: continue
        try: g.parse(str(path)); return g
        except Exception: return None

    def extract_entities_streaming(self, path: Path, graph: Optional[Graph] = None) -> Iterator[Tuple[URIRef, str, URIRef, Set[str]]]:
        """Streams URIs, normalized labels, entity types, and parent hierarchies from ontology files."""
        if HAS_PYOXIGRAPH and path.exists():
            lbl_map, type_map, parent_map = defaultdict(list), defaultdict(list), defaultdict(set)
            ox_format = pyoxigraph.RdfFormat.TURTLE if path.suffix.lower() == ".ttl" else pyoxigraph.RdfFormat.RDF_XML
            try:
                for triple in pyoxigraph.parse(path=str(path), format=ox_format, lenient=True):
                    s, p, o = triple.subject, triple.predicate, triple.object
                    if not isinstance(s, pyoxigraph.NamedNode): continue
                    s_uri, p_val = URIRef(s.value), p.value
                    
                    if p_val in (str(RDFS.label), str(SKOS.prefLabel), str(SKOS.altLabel)) and isinstance(o, pyoxigraph.Literal):
                        lbl_map[s_uri].append((URIRef(p_val), o))
                    elif p_val == str(RDF.type) and isinstance(o, pyoxigraph.NamedNode):
                        type_map[s_uri].append(URIRef(o.value))
                    elif p_val in (str(RDFS.subClassOf), "http://www.w3.org/2004/02/skos/core#broader") and isinstance(o, pyoxigraph.NamedNode):
                        parent_map[s_uri].add(str(o.value))
            except Exception:
                if graph is None: graph = self.load_ontology(path)
                yield from self._extract_entities_rdflib(graph)
                return
            yield from self._process_extracted_maps(lbl_map, type_map, parent_map)
        else:
            if graph is None: graph = self.load_ontology(path)
            yield from self._extract_entities_rdflib(graph)

    def _extract_entities_rdflib(self, graph: Graph) -> Iterator[Tuple[URIRef, str, URIRef, Set[str]]]:
        """RDFLib parsing fallback logic for entity extraction."""
        if graph is None: return
        lbl_map, type_map, parent_map = defaultdict(list), defaultdict(list), defaultdict(set)
        for s, p, o in graph:
            if p in (RDFS.label, SKOS.prefLabel, SKOS.altLabel): lbl_map[s].append((p, o))
            elif p == RDF.type: type_map[s].append(o)
            elif p in (RDFS.subClassOf, URIRef("http://www.w3.org/2004/02/skos/core#broader")) and isinstance(o, URIRef): 
                parent_map[s].add(str(o))
        yield from self._process_extracted_maps(lbl_map, type_map, parent_map)

    def _process_extracted_maps(self, lbl_map, type_map, parent_map) -> Iterator[Tuple[URIRef, str, URIRef, Set[str]]]:
        """Constructs and categorizes extracted URIs by ontology node types."""
        skos_concepts, owl_classes, props, all_subjects = set(), set(), set(), set()
        for s, types in type_map.items():
            if any(t == SKOS.Concept for t in types): skos_concepts.add(s)
            elif any(t == OWL.Class for t in types): owl_classes.add(s)
            elif any(t in (OWL.ObjectProperty, OWL.DatatypeProperty) for t in types): props.add(s)
            all_subjects.add(s)
        
        is_valid = lambda uri: "oboInOwl" not in str(uri) and not str(uri).startswith(_CORE_NAMESPACES)
        target_langs, preds_order = ['en', 'de', 'fr'], [SKOS.prefLabel, RDFS.label, SKOS.altLabel]
        
        def get_fast_label(uri) -> str:
            triples = lbl_map.get(uri)
            if triples:
                for lang in target_langs:
                    for pred in preds_order:
                        for p, obj in triples:
                            if p == pred and getattr(obj, 'language', None) == lang:
                                return self.normalise_label(str(getattr(obj, 'value', obj)))
                for pred in [RDFS.label, SKOS.prefLabel]:
                    for p, obj in triples:
                        if p == pred:
                            return self.normalise_label(str(getattr(obj, 'value', obj)))
            return self.normalise_label(self._PCT_RE.sub(' ', str(uri).split('/')[-1].split('#')[-1]))

        for group, etype in [(owl_classes, OWL.Class), (skos_concepts, SKOS.Concept), (props, OWL.ObjectProperty)]:
            for uri in filter(is_valid, group):
                lbl = get_fast_label(uri)
                if lbl: yield uri, lbl, etype, parent_map.get(uri, set())
                
        for uri in filter(is_valid, all_subjects):
            if uri not in owl_classes and uri not in skos_concepts and uri not in props:
                lbl = get_fast_label(uri)
                if lbl: yield uri, lbl, OWL.NamedIndividual, parent_map.get(uri, set())

    def get_embeddings_adaptive(self, labels: List[str]) -> np.ndarray:
        """Retrieves vectors from memory cache or generates new vectors via transformer model."""
        self.init_model()
        cached_indices, to_embed, to_embed_indices = [], [], []
        cache_dict = self.embedding_cache.label_to_idx
        
        for i, label in enumerate(labels):
            idx = cache_dict.get(label)
            if idx is not None: cached_indices.append((i, idx))
            else:
                to_embed.append(label)
                to_embed_indices.append(i)
                
        result = np.zeros((len(labels), self.embedding_dim), dtype='float32')
        if cached_indices:
            cached_indices.sort(key=lambda x: x[1])
            orig_indices, cache_idxs = zip(*cached_indices)
            result[list(orig_indices)] = self.embedding_cache.embeddings_mmap[list(cache_idxs)]
        
        if to_embed:
            with torch.inference_mode():
                new_embs = self.model.encode(to_embed, convert_to_tensor=False, show_progress_bar=False, batch_size=512)
            if len(new_embs) > 0:
                self.embedding_cache.add_batch(to_embed, new_embs)
                result[to_embed_indices] = new_embs
        return result

    def isub_similarity(self, s1: str, s2: str) -> float:
        """Computes ISUB string metric distance designed for ontology matching."""
        key = (s1, s2)
        if key in self._isub_cache: return self._isub_cache[key]
        if s1 == s2: return 1.0
        if not s1 or not s2: return 0.0
        
        l1, l2, s1_w, s2_w, common_len = len(s1), len(s2), s1, s2, 0
        while True:
            best_sub = ""
            for i in range(len(s1_w)):
                for j in range(i + 1, len(s1_w) + 1):
                    sub = s1_w[i:j]
                    if sub in s2_w and len(sub) > len(best_sub): best_sub = sub
            if not best_sub or len(best_sub) < 2: break
            common_len += len(best_sub)
            s1_w, s2_w = s1_w.replace(best_sub, "", 1), s2_w.replace(best_sub, "", 1)
            
        if common_len == 0: res = 0.0
        else:
            s_comm = (2.0 * common_len) / (l1 + l2)
            p_unmatched = (len(s1_w) * len(s2_w)) / (l1 * l2) if (l1 * l2) > 0 else 0.0
            res = max(0.0, min(1.0, s_comm - (p_unmatched * 0.3)))
            
        self._isub_cache[key] = res
        return res

    def ngram_similarity(self, s1: str, s2: str, n: int = 2) -> float:
        """Calculates character n-gram similarity score between strings."""
        key = (s1, s2, n)
        if key in self._ngram_cache: return self._ngram_cache[key]
        s1, s2 = s1.lower(), s2.lower()
        if s1 == s2: res = 1.0
        else:
            ng1 = set(s1[i:i+n] for i in range(max(0, len(s1)-n+1)))
            ng2 = set(s2[i:i+n] for i in range(max(0, len(s2)-n+1)))
            union = len(ng1.union(ng2))
            res = len(ng1.intersection(ng2)) / union if ng1 and ng2 and union > 0 else 0.0
        self._ngram_cache[key] = res
        return res

    def levenshtein_similarity(self, s1: str, s2: str) -> float:
        """Calculates Levenshtein edit distance similarity ratio between two strings."""
        key = (s1, s2)
        if key in self._lev_cache: return self._lev_cache[key]
        if s1 == s2: res = 1.0
        elif HAS_RAPIDFUZZ: res = fuzz.ratio(s1, s2) / 100.0
        else:
            if not s1 or not s2 or abs(len(s1) - len(s2)) / max(len(s1), len(s2), 1) > 0.4: res = 0.0
            else:
                if len(s1) < len(s2): s1, s2 = s2, s1
                prev = list(range(len(s2) + 1))
                for i, c1 in enumerate(s1):
                    curr = [i + 1]
                    for j, c2 in enumerate(s2):
                        curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
                    prev = curr
                res = (max(len(s1), len(s2)) - prev[-1]) / max(len(s1), len(s2))
        self._lev_cache[key] = res
        return res

    def evaluate_structural_similarity(self, src_uri: str, tgt_uri: str, 
                                       src_entities: Dict, tgt_entities: Dict) -> float:
        """Evaluates parent node relationships to calculate a structural alignment bonus."""
        sp = src_entities.get(src_uri, {}).get("parents", set())
        tp = tgt_entities.get(tgt_uri, {}).get("parents", set())
        return 0.08 if sp and tp and sp.intersection(tp) else 0.0

    def _get_adaptive_k(self) -> int:
        """Determines top-k nearest neighbors dynamically according to ontology size."""
        s = self.ontology_size
        return 1 if s < 80000 else 2 if s < 100000 else 5 if s < 200000 else 18

    def _get_threshold_scale(self) -> float:
        """Calculates dynamic scale factor based on domain and ontology scale."""
        base = 1.11 if self.is_medical_domain else 1.00
        s = self.ontology_size
        scale = 1.10 if s < 1000 else 1.1 if s < 12500 else 1.15 if s < 50000 else 1.1 if s < 100000 else 0.92
        return scale * base

    def semantic_match_chunked(self, src_labels: List[str], src_uris: List[str],
                               tgt_labels: List[str], tgt_uris: List[str],
                               etype: URIRef, src_entities: Dict, tgt_entities: Dict,
                               relaxation_factor: float = 1.0) -> Iterator[AlignmentCandidate]:
        """Runs vector search using FAISS and yields filtered alignment candidates."""
        raw_cutoff = self.thresholds.get(etype, self.default_thres) * relaxation_factor
        src_embs, tgt_embs = self.get_embeddings_adaptive(src_labels), self.get_embeddings_adaptive(tgt_labels)
        dim, k = src_embs.shape[1], min(self._get_adaptive_k(), len(tgt_labels))
        if k <= 0: return

        def build_faiss_index(embs, num_labels):
            if num_labels < 15000: idx = faiss.IndexFlatIP(dim)
            else:
                nlist = max(16, min(16384, int(4 * np.sqrt(num_labels))))
                if num_labels < (nlist * 39): idx = faiss.IndexFlatIP(dim)
                else:
                    idx = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, nlist, faiss.METRIC_INNER_PRODUCT)
                    idx.train(embs)
                    idx.nprobe = max(8, min(nlist, nlist // 16))
            idx.add(embs)
            return idx

        index_fwd, index_rev = build_faiss_index(tgt_embs, len(tgt_labels)), build_faiss_index(src_embs, len(src_labels))
        scores_fwd, indices_fwd = index_fwd.search(src_embs, k=k)
        target_selection_freq = Counter(indices_fwd.flatten().tolist())

        tgt_indices_list = list(set(indices_fwd.flatten().tolist()))
        rev_scores_full, rev_indices_full = index_rev.search(tgt_embs[tgt_indices_list], k=min(self._get_adaptive_k(), len(src_labels)))

        rev_map = {(int(rev_indices_full[b_idx][s_rank]), int(tgt_idx)): float(rev_scores_full[b_idx][s_rank])
                   for b_idx, tgt_idx in enumerate(tgt_indices_list) 
                   for s_rank in range(len(rev_indices_full[b_idx]))}

        src_token_sets = [set(s.split()) for s in src_labels]
        tgt_token_sets = [set(t.split()) for t in tgt_labels]

        for src_idx in range(len(src_labels)):
            src_uri, src_lbl, s_toks = src_uris[src_idx], src_labels[src_idx], src_token_sets[src_idx]

            for kth in range(k):
                score_fwd, tgt_idx = float(scores_fwd[src_idx][kth]), int(indices_fwd[src_idx][kth])

                if self.is_mega_scale and target_selection_freq.get(tgt_idx, 1) > 3:
                    score_fwd *= max(0.70, 1.0 - (0.02 * target_selection_freq[tgt_idx]))

                if score_fwd < raw_cutoff * 0.70 or rev_map.get((src_idx, tgt_idx), 0.0) < (raw_cutoff * 0.70):
                    continue

                tgt_lbl, t_toks, target_uri = tgt_labels[tgt_idx], tgt_token_sets[tgt_idx], str(tgt_uris[tgt_idx])

                if min(len(s_toks), len(t_toks)) > 1 and not s_toks.intersection(t_toks):
                    continue

                isub, lev = self.isub_similarity(src_lbl, tgt_lbl), self.levenshtein_similarity(src_lbl, tgt_lbl)
                if score_fwd < 0.93 and lev < 0.50 and isub < 0.50: continue

                yield AlignmentCandidate(
                    source=str(src_uri), target=target_uri, etype=str(etype),
                    semantic_score=score_fwd, ngram_score=self.ngram_similarity(src_lbl, tgt_lbl),
                    levenshtein_score=lev, isub_score=isub,
                    structural_bonus=self.evaluate_structural_similarity(str(src_uri), target_uri, src_entities, tgt_entities)
                )

        del index_fwd, index_rev
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def align_optimized(self, src_entities: Dict, tgt_entities: Dict) -> Set[Tuple[str, str, str]]:
        """Executes exact matching followed by multi-stage semantic vector matching."""
        final_alignments, claimed_src, claimed_tgt = set(), set(), set()
        self.ontology_size = (len(src_entities) + len(tgt_entities)) / 2
        self.is_mega_scale = (self.ontology_size >= 130000)

        print(f" [INFO] {'Mega-scale Optimization Mode ACTIVE' if self.is_mega_scale else 'Standard matching logic active'} (size: {self.ontology_size:.0f} nodes)")

        tgt_lookup = defaultdict(list)
        for uri, meta in tgt_entities.items(): tgt_lookup[meta["label"]].append((uri, meta["type"]))

        for s_uri, s_meta in src_entities.items():
            if s_meta["label"] in tgt_lookup:
                for t_uri, t_type in tgt_lookup[s_meta["label"]]:
                    if s_meta["type"] == t_type and t_uri not in claimed_tgt:
                        claimed_src.add(s_uri); claimed_tgt.add(t_uri)
                        final_alignments.add((str(s_uri), str(t_uri), str(s_meta["type"])))
                        break
        del tgt_lookup
        gc.collect()

        types = [OWL.Class, SKOS.Concept, OWL.ObjectProperty, OWL.DatatypeProperty, OWL.NamedIndividual]
        final_alignments, claimed_src, claimed_tgt = self._match_stage(src_entities, tgt_entities, types, claimed_src, claimed_tgt, final_alignments, 1.0)

        if not self.is_mega_scale:
            final_alignments, claimed_src, claimed_tgt = self._match_stage(src_entities, tgt_entities, types, claimed_src, claimed_tgt, final_alignments, 0.92)

        return final_alignments

    def _match_stage(self, src_entities, tgt_entities, distinct_types, claimed_src, claimed_tgt, alignments, relaxation_factor=1.0):
        """Runs a candidate generation and selection phase filtered by dynamic thresholds."""
        filtered_src = {k: v for k, v in src_entities.items() if k not in claimed_src}
        filtered_tgt = {k: v for k, v in tgt_entities.items() if k not in claimed_tgt}
        if not filtered_src or not filtered_tgt: return alignments, claimed_src, claimed_tgt

        scale_factor = self._get_threshold_scale()
        stage_candidates = []

        for etype in distinct_types:
            src_sub = [(u, m["label"]) for u, m in filtered_src.items() if m["type"] == etype]
            tgt_sub = [(u, m["label"]) for u, m in filtered_tgt.items() if m["type"] == etype]
            if not src_sub or not tgt_sub: continue

            src_uris, src_labels = zip(*src_sub)
            tgt_uris, tgt_labels = zip(*tgt_sub)
            required_cutoff = self.thresholds.get(etype, self.default_thres) * scale_factor * relaxation_factor

            for c in self.semantic_match_chunked(list(src_labels), list(src_uris), list(tgt_labels), list(tgt_uris), etype, src_entities, tgt_entities, relaxation_factor):
                if c.combined_score >= required_cutoff: stage_candidates.append(c)

        stage_candidates.sort(key=lambda x: x.combined_score, reverse=True)
        for c in stage_candidates:
            if c.source not in claimed_src and c.target not in claimed_tgt:
                claimed_src.add(c.source); claimed_tgt.add(c.target)
                alignments.add((c.source, c.target, c.etype))
        return alignments, claimed_src, claimed_tgt

    def convert_to_rdf_triples(self, alignments: Set[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        """Translates internal alignments to formal OWL/RDF equivalence triples."""
        return {
            (src, "http://www.w3.org/2002/07/owl#equivalentClass" if "Class" in etype else
                  "http://www.w3.org/2002/07/owl#equivalentProperty" if "Property" in etype else
                  "http://www.w3.org/2002/07/owl#sameAs", tgt)
            for src, tgt, etype in alignments
        }


class OAEITrackRunner:
    """Orchestrates ontology benchmark execution, alignment evaluation, and metric logging."""
    def __init__(self, matcher: MOSAIC, results_dir: Path = None):
        self.matcher, self.results_dir, self.log = matcher, results_dir or RESULTS_DIR, []

    def load_reference_alignments(self, path: Path) -> Set[Tuple[str, str, str]]:
        """Loads and normalizes baseline reference alignment triples from file."""
        ref_set = set()
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        match = re.match(r'<([^>]+)>\s+<([^>]+)>\s+<([^>]+)>\s*\.\s*$', line)
                        if match:
                            s, p, o = match.groups()
                            nodes = tuple(sorted([s, o]))
                            ref_set.add((nodes[0], p, nodes[1]))
        except Exception as e: print(f" [ERROR] Could not read reference: {e}")
        return ref_set

    def serialize_alignments_to_ttl(self, alignments: Set[Tuple[str, str, str]], path: Path) -> bool:
        """Serializes resulting alignment set into a Turtle formatted (.ttl) graph file."""
        g = Graph()
        for src, pred, tgt in alignments:
            try: g.add((URIRef(src), URIRef(pred), URIRef(tgt)))
            except Exception: pass
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            g.serialize(destination=str(path), format="turtle")
            return True
        except Exception: return False

    def calculate_metrics(self, sys_align, ref_align) -> Tuple[float, float, float, int, int]:
        """Calculates Precision, Recall, and F1-Score metrics against reference benchmark."""
        if not ref_align: return 0.0, 0.0, 0.0, 0, 0
        sys_canon = {(tuple(sorted([str(s), str(o)]))[0], str(p), tuple(sorted([str(s), str(o)]))[1]) for s, p, o in sys_align}
        tp = len(sys_canon.intersection(ref_align))
        p = tp / len(sys_canon) if sys_canon else 0.0
        r = tp / len(ref_align) if ref_align else 0.0
        return round(p, 4), round(r, 4), round((2 * p * r) / (p + r) if (p + r) > 0 else 0.0, 4), tp, len(ref_align)

    def find_ontology_file(self, folder: Path, name: str) -> Optional[Path]:
        """Finds matching ontology file matching targeted extensions."""
        for ext in [".owl", ".rdf", ".ttl", ".xml"]:
            if (folder / f"{name}{ext}").exists(): return folder / f"{name}{ext}"
        return None

    def run_all_tracks(self, base_dir: str, csv_out: str = "report_mega_scale.csv"):
        """Iterates through tracks, runs domain adaptation, calculates alignments, and logs results."""
        base_path = Path(base_dir)
        if not base_path.exists(): return

        for track in sorted(base_path.iterdir()):
            if not track.is_dir(): continue
            for tf in sorted(track.glob("*.ttl")):
                parts = tf.stem.split("-")
                if len(parts) != 2:
                    if "human-mouse" in tf.stem: parts = ["human", "mouse"]
                    else: continue

                src_p, tgt_p = self.find_ontology_file(track / "ontologies", parts[0]), self.find_ontology_file(track / "ontologies", parts[1])
                if not src_p or not tgt_p: continue

                print(f"\n [TASK] Loading reference alignments: {tf.stem}")
                ref_align = self.load_reference_alignments(tf)
                t_ext0 = time.time()

                src_entities = {str(u): {"label": l, "type": t, "parents": par} for u, l, t, par in self.matcher.extract_entities_streaming(src_p)}
                tgt_entities = {str(u): {"label": l, "type": t, "parents": par} for u, l, t, par in self.matcher.extract_entities_streaming(tgt_p)}
                print(f" [TASK] Extracted {len(src_entities)} source and {len(tgt_entities)} target entities in {round(time.time() - t_ext0, 2)}s")

                is_med, reason = MedicalDomainDetector.evaluate_is_medical(track.name, tf.stem, src_entities, tgt_entities)
                print(f" [DOMAIN DETECTOR] Result: {'MEDICAL' if is_med else 'GENERAL'} | Reason: {reason}")

                self.matcher.apply_domain_model(is_medical=is_med)
                gc.collect()

                t0 = time.time()
                alignments = self.matcher.align_optimized(src_entities, tgt_entities)
                dt = round(time.time() - t0, 2)

                out_ttl = self.results_dir / f"mosaic_{track.name}_{tf.stem}.ttl"
                rdf_triples = self.matcher.convert_to_rdf_triples(alignments)
                self.serialize_alignments_to_ttl(rdf_triples, out_ttl)

                p, r, f1, tp, total = self.calculate_metrics(rdf_triples, ref_align)
                print(f"         Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f} | Correct: {tp}/{total} | Time: {dt}s")

                self.log.append({
                    "Track": track.name, "Task": tf.stem, "Precision": p, "Recall": r, "F1-Score": f1,
                    "Time (s)": dt, "Alignments": len(alignments), "Correct": tp, "Reference": total, "Type": "Task"
                })

        self.results_to_csv(csv_out)
        if self.matcher.embedding_cache: self.matcher.embedding_cache.save_index()

    def results_to_csv(self, filename: str):
        """Writes execution performance metrics to CSV file."""
        fields = ["Track", "Task", "Precision", "Recall", "F1-Score", "Time (s)", "Alignments", "Correct", "Reference", "Type"]
        with open(self.results_dir / filename, mode="w", newline="", encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=fields).writerows(self.log)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" [INFO] Running on execution device: {device}")

    m = MOSAIC(model_name="sentence-transformers/all-MiniLM-L12-v2", device=device)
    runner = OAEITrackRunner(matcher=m, results_dir=RESULTS_DIR)
    runner.run_all_tracks(str(BASE_DIR), csv_out="mosaic_report.csv")