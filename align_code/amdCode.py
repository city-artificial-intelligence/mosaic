import csv
import gc
import os
import re
import time
import warnings
import psutil
import numpy as np
import torch
import faiss
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterator, List, Tuple, Set, Dict, Optional
from collections import defaultdict, Counter
import logging

from sentence_transformers import SentenceTransformer
from rdflib import Graph, RDF, RDFS, URIRef
from rdflib.namespace import OWL, SKOS


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


@dataclass
class AlignmentCandidate:
    source: str
    target: str
    etype: str
    semantic_score: float
    ngram_score: float = 0.0
    levenshtein_score: float = 0.0
    
    @property
    def combined_score(self) -> float:
        # Computes a weighted score combining semantic (55%) and string similarities (45%)
        string_score = (self.ngram_score * 0.5) + (self.levenshtein_score * 0.5)
        return (self.semantic_score * 0.55) + (string_score * 0.45)


class EmbeddingDiskCache:
    def __init__(self, cache_dir: Path, embedding_dim: int):
        # Initializes the embedding disk cache directory, dimensions, and lookup maps
        self.cache_dir = cache_dir
        self.embedding_dim = embedding_dim
        self.label_to_idx = {}
        self.embeddings_mmap = None
        self.current_count = 0
        self.max_embeddings = 3_000_000
        self._init_mmap()
    
    def _init_mmap(self):
        # Initializes or loads memory-mapped numpy arrays and label indices from disk
        mmap_path = self.cache_dir / "embeddings.npy"
        index_path = self.cache_dir / "labels.txt"
        
        if mmap_path.exists() and index_path.exists():
            try:
                self.embeddings_mmap = np.load(str(mmap_path), mmap_mode='r+')
                with open(index_path, 'r', encoding='utf-8') as f:
                    for idx, line in enumerate(f):
                        label = line.strip()
                        self.label_to_idx[label] = idx
                        self.current_count += 1
                return
            except:
                pass
        
        self.embeddings_mmap = np.memmap(
            str(mmap_path), dtype='float32', mode='w+', shape=(self.max_embeddings, self.embedding_dim)
        )
        self.current_count = 0
        self.label_to_idx = {}
    
    def get(self, label: str) -> Optional[np.ndarray]:
        # Retrieves a cached embedding for a given label if it exists
        if label in self.label_to_idx:
            idx = self.label_to_idx[label]
            return np.array(self.embeddings_mmap[idx])
        return None
    
    def add_batch(self, labels: List[str], embeddings: np.ndarray):
        # Writes a batch of labels and their new embeddings to the memory-mapped cache
        if len(labels) == 0: return
        start_idx = self.current_count
        end_idx = start_idx + len(labels)
        
        if end_idx > self.max_embeddings:
            labels = labels[:self.max_embeddings - start_idx]
            embeddings = embeddings[:self.max_embeddings - start_idx]
            end_idx = self.max_embeddings
            
        if len(labels) == 0: return
            
        self.embeddings_mmap[start_idx:end_idx] = embeddings
        self.embeddings_mmap.flush()
        
        for i, label in enumerate(labels):
            self.label_to_idx[label] = start_idx + i
        self.current_count = end_idx
    
    def save_index(self):
        # Serializes the label index mappings to text file in memory-mapped index order
        index_path = self.cache_dir / "labels.txt"
        sorted_labels = [None] * self.current_count
        for label, idx in self.label_to_idx.items():
            if idx < self.current_count:
                sorted_labels[idx] = label
        with open(index_path, 'w', encoding='utf-8') as f:
            for label in sorted_labels:
                if label is not None:
                    f.write(f"{label}\n")


class GPUMemoryManager:
    def __init__(self, target_usage_pct=0.85):
        # Initializes thresholds and limits for monitoring and managing GPU VRAM usage
        self.target_usage_pct = target_usage_pct
        self.initial_batch_size = 256
        self.min_batch_size = 32
        self.current_batch_size = self.initial_batch_size
    
    def get_current_memory_usage(self) -> float:
        # Calculates the current fraction of allocated and reserved GPU memory
        if not torch.cuda.is_available(): return 0.0
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory
        return (allocated + reserved) / total
    
    def adapt_batch_size(self) -> int:
        # Decreases batch size dynamically if GPU usage exceeds target threshold
        usage = self.get_current_memory_usage()
        if usage > self.target_usage_pct:
            self.current_batch_size = max(self.min_batch_size, int(self.current_batch_size * 0.75))
        return self.current_batch_size


class MOSAIC:
    _CAMEL_RE = re.compile(r'([a-z])([A-Z])')
    _PCT_RE = re.compile(r'%[0-9A-Fa-f]{2}')
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L12-v2"

    def __init__(self, model_name=None, thresholds=None, device="cuda", chunk_size=100_000, faiss_nprobe=64):
        # Initializes semantic matcher with model config, device settings, caches, and thresholds
        self.thresholds = thresholds or {
            OWL.Class: 0.78, SKOS.Concept: 0.82, OWL.ObjectProperty: 0.75,
            OWL.DatatypeProperty: 0.75, OWL.NamedIndividual: 0.72
        }
        self.default_thres = 0.75
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self.model = None
        
        self.init_model()
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.chunk_size = chunk_size
        self.faiss_nprobe = faiss_nprobe
        self.ontology_size = 0
        self.is_mega_scale = False
        
        self.embedding_cache = EmbeddingDiskCache(CACHE_DIR, embedding_dim=self.embedding_dim)
        self.gpu_memory_mgr = GPUMemoryManager(target_usage_pct=0.80)
        self._ngram_cache = {}
        self._lev_cache = {}
    
    def init_model(self):
        # Loads the SentenceTransformer model onto the target device if not already loaded
        if self.model is not None: return
        self.model = SentenceTransformer(self.model_name, device=self.device)
        try: self.model.max_seq_length = 128
        except: pass
    
    def normalise_label(self, text: str) -> str:
        # Standardizes text by stripping Wiki namespaces, converting camelCase, and lowering
        if not text: return ""
        text = str(text)
        
        # Strip MediaWiki / DBKwik URI namespaces/prefixes to drastically boost matching accuracy
        for prefix in ["Category:", "Template:", "File:", "Property:", "Category_talk:", "User:"]:
            if text.startswith(prefix):
                text = text[len(prefix):]
                
        text = self._CAMEL_RE.sub(r'\1 \2', text)
        text = text.replace('_', ' ').replace('-', ' ').lower().strip()
        return " ".join(text.split()[:20])
    
    def load_ontology(self, path: Path) -> Optional[Graph]:
        # Parses and returns an RDF Graph using multiple format fallbacks
        g = Graph()
        formats = ["turtle", "xml"] if path.suffix == ".ttl" else ["xml", "turtle"]
        for fmt in formats:
            try: g.parse(str(path), format=fmt); return g
            except: continue
        try: g.parse(str(path)); return g
        except: return None
    
    def extract_entities_streaming(self, path: Path, graph: Optional[Graph] = None) -> Iterator[Tuple[URIRef, str, URIRef]]:
        # High-speed entity/label extractor utilizing pyoxigraph if available, or falling back to rdflib
        if HAS_PYOXIGRAPH and path.exists():
            print(" [INFO] Rust-accelerated pyoxigraph parser selected.")
            lbl_map = defaultdict(list)
            type_map = defaultdict(list)
            
            ext = path.suffix.lower()
            if ext == ".ttl":
                ox_format = pyoxigraph.RdfFormat.TURTLE
            elif ext in (".rdf", ".owl", ".xml"):
                ox_format = pyoxigraph.RdfFormat.RDF_XML
            else:
                ox_format = None
            
            try:
                for triple in pyoxigraph.parse(path=str(path), format=ox_format, lenient=True):
                    s, p, o = triple.subject, triple.predicate, triple.object
                    if not isinstance(s, pyoxigraph.NamedNode): continue
                    
                    s_uri = URIRef(s.value)
                    p_val = p.value
                    
                    if p_val in (str(RDFS.label), str(SKOS.prefLabel), str(SKOS.altLabel)):
                        if isinstance(o, pyoxigraph.Literal):
                            lbl_map[s_uri].append((URIRef(p_val), o))
                    elif p_val == str(RDF.type) and isinstance(o, pyoxigraph.NamedNode):
                        type_map[s_uri].append(URIRef(o.value))
            except Exception as e:
                print(f" [WARNING] pyoxigraph failed: {e}. Falling back to standard graph loader.")
                if graph is None: graph = self.load_ontology(path)
                yield from self._extract_entities_rdflib(graph)
                return
            
            yield from self._process_extracted_maps(lbl_map, type_map)
        else:
            if graph is None: graph = self.load_ontology(path)
            yield from self._extract_entities_rdflib(graph)

    def _extract_entities_rdflib(self, graph: Graph) -> Iterator[Tuple[URIRef, str, URIRef]]:
        # Fallback RDFlib entity extractor that maps subjects to labels and types
        if graph is None: return
        lbl_map = defaultdict(list)
        type_map = defaultdict(list)
        for s, p, o in graph:
            if p in (RDFS.label, SKOS.prefLabel, SKOS.altLabel):
                lbl_map[s].append((p, o))
            elif p == RDF.type:
                type_map[s].append(o)
        yield from self._process_extracted_maps(lbl_map, type_map)

    def _process_extracted_maps(self, lbl_map, type_map) -> Iterator[Tuple[URIRef, str, URIRef]]:
        # Categorizes mapped URIs into core OWL/SKOS ontology types and streams validated entities with labels
        skos_concepts = set()
        owl_classes = set()
        props = set()
        all_subjects = set()
        
        for s, types in type_map.items():
            is_skos = False
            is_class = False
            is_prop = False
            for t in types:
                if t == SKOS.Concept: is_skos = True
                elif t == OWL.Class: is_class = True
                elif t in (OWL.ObjectProperty, OWL.DatatypeProperty): is_prop = True
            
            if is_skos: skos_concepts.add(s)
            elif is_class: owl_classes.add(s)
            elif is_prop: props.add(s)
            all_subjects.add(s)
        
        def is_valid(uri):
            uri_str = str(uri)
            return "oboInOwl" not in uri_str and not uri_str.startswith(_CORE_NAMESPACES)
        
        target_langs = ['en', 'de', 'fr']
        preds_order = [SKOS.prefLabel, RDFS.label, SKOS.altLabel]
        
        def get_fast_label(uri) -> str:
            triples = lbl_map.get(uri)
            if triples:
                for lang in target_langs:
                    for pred in preds_order:
                        for p, obj in triples:
                            if p == pred and hasattr(obj, 'language') and obj.language == lang:
                                return self.normalise_label(str(obj.value if hasattr(obj, 'value') else obj))
                for pred in [RDFS.label, SKOS.prefLabel]:
                    for p, obj in triples:
                        if p == pred:
                            return self.normalise_label(str(obj.value if hasattr(obj, 'value') else obj))
                            
            frag = str(uri).split('/')[-1].split('#')[-1]
            return self.normalise_label(self._PCT_RE.sub(' ', frag))

        for uri in filter(is_valid, owl_classes):
            lbl = get_fast_label(uri)
            if lbl: yield uri, lbl, OWL.Class
        for uri in filter(is_valid, skos_concepts):
            lbl = get_fast_label(uri)
            if lbl: yield uri, lbl, SKOS.Concept
        for uri in filter(is_valid, props):
            lbl = get_fast_label(uri)
            if lbl: yield uri, lbl, OWL.ObjectProperty
        for uri in filter(is_valid, all_subjects):
            if uri not in owl_classes and uri not in skos_concepts and uri not in props:
                lbl = get_fast_label(uri)
                if lbl: yield uri, lbl, OWL.NamedIndividual
    
    def get_embeddings_adaptive(self, labels: List[str]) -> np.ndarray:
        # Generates embeddings for a list of text labels, using disk-cache hits or dynamic GPU encoding
        self.init_model()
        
        cached_indices = []
        to_embed = []
        to_embed_indices = []
        
        cache_dict = self.embedding_cache.label_to_idx
        for i, label in enumerate(labels):
            idx = cache_dict.get(label)
            if idx is not None:
                cached_indices.append((i, idx))
            else:
                to_embed.append(label)
                to_embed_indices.append(i)
                
        result = np.zeros((len(labels), self.embedding_dim), dtype='float32')
        
        if cached_indices:
            cached_indices.sort(key=lambda x: x[1])
            orig_indices, cache_idxs = zip(*cached_indices)
            result[list(orig_indices)] = self.embedding_cache.embeddings_mmap[list(cache_idxs)]
        
        if to_embed:
            batch_size = 512
            with torch.inference_mode():
                new_embs_array = self.model.encode(
                    to_embed, 
                    convert_to_tensor=False, 
                    show_progress_bar=False, 
                    batch_size=batch_size
                )
            if len(new_embs_array) > 0:
                self.embedding_cache.add_batch(to_embed, new_embs_array)
                result[to_embed_indices] = new_embs_array
                
        return result
    
    def ngram_similarity(self, s1: str, s2: str, n: int = 2) -> float:
        # Computes normalized n-gram similarity score between two strings with key caching
        key = (s1, s2, n)
        if key in self._ngram_cache: return self._ngram_cache[key]
        
        s1, s2 = s1.lower(), s2.lower()
        if s1 == s2: result = 1.0
        else:
            ngrams1 = set(s1[i:i+n] for i in range(max(0, len(s1)-n+1)))
            ngrams2 = set(s2[i:i+n] for i in range(max(0, len(s2)-n+1)))
            if not ngrams1 or not ngrams2: result = 0.0
            else:
                union = len(ngrams1.union(ngrams2))
                result = len(ngrams1.intersection(ngrams2)) / union if union > 0 else 0.0
        
        self._ngram_cache[key] = result
        return result

    def levenshtein_similarity(self, s1: str, s2: str) -> float:
        # Computes character Levenshtein similarity using RapidFuzz or native DP fallback
        key = (s1, s2)
        if key in self._lev_cache: return self._lev_cache[key]
        
        if s1 == s2: 
            result = 1.0
        elif HAS_RAPIDFUZZ:
            result = fuzz.ratio(s1, s2) / 100.0
        else:
            if not s1 or not s2: 
                result = 0.0
            elif abs(len(s1) - len(s2)) / max(len(s1), len(s2), 1) > 0.4:
                # Fast early cut-off: if string length differs by >40%, similarity cannot be high
                result = 0.0
            else:
                if len(s1) < len(s2): s1, s2 = s2, s1
                previous_row = list(range(len(s2) + 1))
                for i, c1 in enumerate(s1):
                    current_row = [i + 1]
                    for j, c2 in enumerate(s2):
                        insertions = previous_row[j + 1] + 1
                        deletions = current_row[j] + 1
                        substitutions = previous_row[j] + (c1 != c2)
                        current_row.append(min(insertions, deletions, substitutions))
                    previous_row = current_row
                    
                max_len = max(len(s1), len(s2))
                result = (max_len - previous_row[-1]) / max_len
        
        self._lev_cache[key] = result
        return result
    
    def _get_adaptive_k(self) -> int:
        # Scales semantic search neighborhood size 'k' based on target ontology density
        if self.ontology_size < 40000:
            return 1
        elif self.ontology_size < 80000:
            return 3
        elif self.ontology_size < 100000:
            return 5
        elif self.ontology_size < 200000:
            return 8
        else:
            return 20
    
    def _get_threshold_scale(self) -> float:
        # Scales similarity matching thresholds dynamically based on total graph size
        if self.ontology_size < 200:
            return 1.1
        elif self.ontology_size < 275:
            return 0.98
        elif self.ontology_size < 500:
            return 1.02
        elif self.ontology_size < 1000:
            return 1.1
        elif self.ontology_size < 25000:
            return 1.2
        elif self.ontology_size < 50000:
            return 1.08
        elif self.ontology_size < 100000:
            return 1.0
        elif self.ontology_size >= 130000:
            return 1.2
        else:
            return 0.92
    
    def _is_mutual_match(self, src_score: float, tgt_reverse_score: float, threshold: float) -> bool:
        # Validates bidirectional semantic score compatibility to reduce false positives
        strictness = 0.94 if self.is_mega_scale else 0.88
        return src_score >= threshold and tgt_reverse_score >= (threshold * strictness)
    
    def semantic_match_chunked(self, src_labels: List[str], src_uris: List[str],
                               tgt_labels: List[str], tgt_uris: List[str],
                               etype: URIRef, relaxation_factor: float = 1.0) -> Iterator[AlignmentCandidate]:
        # Generates pairwise candidates by executing bidirectional FAISS inner-product vector indexing
        threshold = self.thresholds.get(etype, self.default_thres) * relaxation_factor
        src_embs = self.get_embeddings_adaptive(src_labels)
        tgt_embs = self.get_embeddings_adaptive(tgt_labels)
        
        dim = src_embs.shape[1]
        k = min(self._get_adaptive_k(), len(tgt_labels))
        
        if len(tgt_labels) < 15000:
            index_fwd = faiss.IndexFlatIP(dim)
        else:
            nlist = int(4 * np.sqrt(len(tgt_labels)))
            nlist = max(16, min(16384, nlist))
            if len(tgt_labels) < (nlist * 39):
                index_fwd = faiss.IndexFlatIP(dim)
            else:
                index_fwd = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, nlist, faiss.METRIC_INNER_PRODUCT)
                index_fwd.train(tgt_embs)
                index_fwd.add(tgt_embs)
                index_fwd.nprobe = max(8, min(nlist, nlist // 16))
                
        if isinstance(index_fwd, faiss.IndexFlatIP):
            index_fwd.add(tgt_embs)
        
        if len(src_labels) < 15000:
            index_rev = faiss.IndexFlatIP(dim)
        else:
            nlist_rev = int(4 * np.sqrt(len(src_labels)))
            nlist_rev = max(16, min(16384, nlist_rev))
            if len(src_labels) < (nlist_rev * 39):
                index_rev = faiss.IndexFlatIP(dim)
            else:
                index_rev = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, nlist_rev, faiss.METRIC_INNER_PRODUCT)
                index_rev.train(src_embs)
                index_rev.add(src_embs)
                index_rev.nprobe = max(8, min(nlist_rev, nlist_rev // 16))
                
        if isinstance(index_rev, faiss.IndexFlatIP):
            index_rev.add(src_embs)
        
        if k > 0:
            scores_fwd, indices_fwd = index_fwd.search(src_embs, k=k)
        else:
            return
            
        target_selection_freq = Counter(indices_fwd.flatten().tolist())
        
        tgt_indices_list = list(set(indices_fwd.flatten().tolist()))
        rev_k = min(self._get_adaptive_k(), len(src_labels))
        rev_scores_full, rev_indices_full = index_rev.search(tgt_embs[tgt_indices_list], k=rev_k)
        
        rev_map = {}
        for batch_idx, tgt_idx in enumerate(tgt_indices_list):
            for src_rank, src_idx in enumerate(rev_indices_full[batch_idx]):
                rev_map[(int(src_idx), int(tgt_idx))] = float(rev_scores_full[batch_idx][src_rank])
        
        for src_idx in range(len(src_labels)):
            src_uri = src_uris[src_idx]
            src_lbl = src_labels[src_idx]
            
            for kth in range(k):
                score_fwd = float(scores_fwd[src_idx][kth])
                tgt_idx = int(indices_fwd[src_idx][kth])
                
                if self.is_mega_scale:
                    hits = target_selection_freq.get(tgt_idx, 1)
                    if hits > 3:
                        score_fwd *= max(0.70, 1.0 - (0.02 * hits))
                
                if score_fwd < threshold * 0.88: continue
                
                tgt_uri = tgt_uris[tgt_idx]
                tgt_lbl = tgt_labels[tgt_idx]
                
                reverse_score = rev_map.get((src_idx, tgt_idx), threshold * 0.85)
                if not self._is_mutual_match(score_fwd, reverse_score, threshold):
                    continue
                
                ngram_sim = self.ngram_similarity(src_lbl, tgt_lbl)
                lev_sim = self.levenshtein_similarity(src_lbl, tgt_lbl)
                
                yield AlignmentCandidate(
                    source=str(src_uri), target=str(tgt_uri), etype=str(etype),
                    semantic_score=score_fwd, ngram_score=ngram_sim, levenshtein_score=lev_sim
                )
        
        del index_fwd, index_rev
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    def align_optimized(self, src_entities: Dict, tgt_entities: Dict) -> Set[Tuple[str, str, str]]:
        # Orchestrates the alignment pipeline using exact-matching, core search, and conditional relaxed search passes
        final_alignments = set()
        claimed_src, claimed_tgt = set(), set()
        
        avg_size = (len(src_entities) + len(tgt_entities)) / 2
        self.ontology_size = avg_size
        
        self.is_mega_scale = (self.ontology_size >= 130000)
        if self.is_mega_scale:
            print(f" [INFO] Mega-scale Optimization Mode ACTIVE (size: {self.ontology_size:.0f} nodes)")
        else:
            print(f" [INFO] Standard matching logic active (size: {self.ontology_size:.0f} nodes)")

        original_thresholds = self.thresholds
        scale = self._get_threshold_scale()
        self.thresholds = {k: v * scale for k, v in self.thresholds.items()}
        
        # Exact Label Match Stage
        tgt_lookup = {}
        for uri, meta in tgt_entities.items():
            lbl = meta["label"]
            if lbl not in tgt_lookup: tgt_lookup[lbl] = []
            tgt_lookup[lbl].append((uri, meta["type"]))
            
        for s_uri, s_meta in src_entities.items():
            s_lbl = s_meta["label"]
            if s_lbl in tgt_lookup:
                for t_uri, t_type in tgt_lookup[s_lbl]:
                    if s_meta["type"] == t_type:
                        claimed_src.add(s_uri)
                        claimed_tgt.add(t_uri)
                        final_alignments.add((str(s_uri), str(t_uri), str(s_meta["type"])))
                        break
        del tgt_lookup
        gc.collect()
        
        distinct_types = [OWL.Class, SKOS.Concept, OWL.ObjectProperty, OWL.DatatypeProperty, OWL.NamedIndividual]
        
        # Stage 1: Core Search Pass
        final_alignments, claimed_src, claimed_tgt = self._match_stage(
            src_entities, tgt_entities, distinct_types, claimed_src, claimed_tgt, final_alignments, relaxation_factor=1.0
        )
        
        # Stage 2: Relaxed Match Pass
        if not self.is_mega_scale:
            final_alignments, claimed_src, claimed_tgt = self._match_stage(
                src_entities, tgt_entities, distinct_types, claimed_src, claimed_tgt, final_alignments, relaxation_factor=0.92
            )
        else:
            print(" [INFO] Mega scale detected: Skipping second relaxation pass to protect alignment precision.")
        
        self.thresholds = original_thresholds
        return final_alignments

    def _match_stage(self, src_entities, tgt_entities, distinct_types, claimed_src, claimed_tgt, alignments, relaxation_factor=1.0):
        # Executes search filtering, semantic matching, string token tests, and resolves candidates in priority order
        filtered_src = {k: v for k, v in src_entities.items() if k not in claimed_src}
        filtered_tgt = {k: v for k, v in tgt_entities.items() if k not in claimed_tgt}
        if not filtered_src or not filtered_tgt: return alignments, claimed_src, claimed_tgt
        
        stage_candidates = []
        for etype in distinct_types:
            src_subset = [(uri, meta) for uri, meta in filtered_src.items() if meta["type"] == etype]
            tgt_subset = [(uri, meta) for uri, meta in filtered_tgt.items() if meta["type"] == etype]
            if not src_subset or not tgt_subset: continue
            
            src_uris, src_labels = zip(*[(u, m["label"]) for u, m in src_subset])
            tgt_uris, tgt_labels = zip(*[(u, m["label"]) for u, m in tgt_subset])
            
            for candidate in self.semantic_match_chunked(
                list(src_labels), list(src_uris), list(tgt_labels), list(tgt_uris), etype, relaxation_factor
            ):
                s_lbl = src_entities[candidate.source]["label"]
                t_lbl = tgt_entities[candidate.target]["label"]
                
                # Mega-Scale Precision & Recall Filters
                if self.is_mega_scale:
                    s_tokens = set(s_lbl.split())
                    t_tokens = set(t_lbl.split())
                    intersection = s_tokens.intersection(t_tokens)
                    
                    is_token_match = len(intersection) >= min(len(s_tokens), len(t_tokens)) and len(intersection) >= 2
                    
                    if is_token_match:
                        if candidate.combined_score < 0.70:
                            continue
                    else:
                        if candidate.levenshtein_score < 0.58:
                            continue
                        if candidate.combined_score < 0.81:
                            continue
                
                stage_candidates.append(candidate)
                
        stage_candidates.sort(key=lambda x: x.combined_score, reverse=True)
        for c in stage_candidates:
            if c.source not in claimed_src and c.target not in claimed_tgt:
                claimed_src.add(c.source)
                claimed_tgt.add(c.target)
                alignments.add((c.source, c.target, c.etype))
        return alignments, claimed_src, claimed_tgt
    
    def convert_to_rdf_triples(self, alignments: Set[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        # Maps alignment candidate entities to standard OWL equivalent/identity RDF relation URIs
        rdf_triples = set()
        for src, tgt, etype_str in alignments:
            if "Class" in etype_str: pred = "http://www.w3.org/2002/07/owl#equivalentClass"
            elif "Property" in etype_str: pred = "http://www.w3.org/2002/07/owl#equivalentProperty"
            else: pred = "http://www.w3.org/2002/07/owl#sameAs"
            rdf_triples.add((src, pred, tgt))
        return rdf_triples


class OAEITrackRunner:
    def __init__(self, matcher: MOSAIC, results_dir: Path = None):
        # Initializes the runner with the alignment matcher engine, results directory, and run logs
        self.matcher = matcher
        self.results_dir = results_dir or RESULTS_DIR
        self.log = []
    
    def load_reference_alignments(self, path: Path) -> Set[Tuple[str, str, str]]:
        # Parses canonical sorted triple strings from an OAEI reference alignment TTL/N-Triple file
        ref_set = set()
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'): continue
                    match = re.match(r'<([^>]+)>\s+<([^>]+)>\s+<([^>]+)>\s*\.\s*$', line)
                    if match:
                        s, p, o = match.groups()
                        nodes = tuple(sorted([s, o]))
                        ref_set.add((nodes[0], p, nodes[1]))
        except Exception as e: print(f" [ERROR] Could not read reference: {e}")
        return ref_set
    
    def serialize_alignments_to_ttl(self, alignments: Set[Tuple[str, str, str]], path: Path) -> bool:
        # Serializes alignment triple sets back to disk as a turtle (.ttl) RDF format file
        g = Graph()
        for src, pred, tgt in alignments:
            try: g.add((URIRef(src), URIRef(pred), URIRef(tgt)))
            except: continue
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            g.serialize(destination=str(path), format="turtle")
            return True
        except: return False
    
    def calculate_metrics(self, sys_align, ref_align) -> Tuple[float, float, float, int, int]:
        # Evaluates alignments to produce standard precision, recall, and F1 metrics
        if not ref_align: return 0.0, 0.0, 0.0, 0, 0
        sys_canon = set()
        for s, p, o in sys_align:
            nodes = tuple(sorted([str(s), str(o)]))
            sys_canon.add((nodes[0], str(p), nodes[1]))
        
        tp = len(sys_canon.intersection(ref_align))
        p = tp / len(sys_canon) if sys_canon else 0.0
        r = tp / len(ref_align) if ref_align else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        return round(p, 4), round(r, 4), round(f1, 4), tp, len(ref_align)
    
    def find_ontology_file(self, folder: Path, name: str) -> Optional[Path]:
        # Locates an ontology within a directory matching the specified file extensions
        for ext in [".owl", ".rdf", ".ttl", ".xml"]:
            p = folder / f"{name}{ext}"
            if p.exists(): return p
        return None
    
    def run_all_tracks(self, base_dir: str, csv_out: str = "report_mega_scale.csv"):
        # Executes end-to-end alignment tasks across multiple OAEI benchmark tracks
        base_path = Path(base_dir)
        if not base_path.exists(): return
        
        for track in sorted(base_path.iterdir()):
            if not track.is_dir(): continue
            tasks = sorted(track.glob("*.ttl"))
            
            for tf in tasks:
                parts = tf.stem.split("-")
                if len(parts) != 2:
                    if "human-mouse" in tf.stem: parts = ["human", "mouse"]
                    else: continue
                
                ont_folder = track / "ontologies"
                src_p = self.find_ontology_file(ont_folder, parts[0])
                tgt_p = self.find_ontology_file(ont_folder, parts[1])
                if not src_p or not tgt_p: continue
                
                print(f"\n [TASK] Loading reference alignments: {tf.stem}")
                ref_align = self.load_reference_alignments(tf)
                print(f" [TASK] Reference alignments loaded: {len(ref_align)} pairs")
                
                print(" [TASK] Extracting entities and processing labels...")
                t_ext0 = time.time()
                
                # Direct streaming parsing via PyOxigraph path. Stringify keys immediately to eliminate type lookup bugs.
                src_entities = {str(u): {"label": l, "type": t} for u, l, t in self.matcher.extract_entities_streaming(src_p)}
                tgt_entities = {str(u): {"label": l, "type": t} for u, l, t in self.matcher.extract_entities_streaming(tgt_p)}
                dt_ext = round(time.time() - t_ext0, 2)
                print(f" [TASK] Extracted {len(src_entities)} source entities and {len(tgt_entities)} target entities in {dt_ext}s")
                gc.collect()
                
                print(" [TASK] Starting alignment process...")
                t0 = time.time()
                alignments = self.matcher.align_optimized(src_entities, tgt_entities)
                dt = round(time.time() - t0, 2)
                print(f" [TASK] Alignment completed in {dt}s")
                
                out_ttl = self.results_dir / f"mosaic_{track.name}_{tf.stem}.ttl"
                rdf_triples = self.matcher.convert_to_rdf_triples(alignments)
                self.serialize_alignments_to_ttl(rdf_triples, out_ttl)
                print(f" [TASK] Results serialized to: {out_ttl}")
                
                print(" [TASK] Calculating metrics...")
                p, r, f1, tp, total = self.calculate_metrics(rdf_triples, ref_align)
                print(f" [TASK] RESULTS for {tf.stem}:")
                print(f"         Precision: {p:.4f}")
                print(f"         Recall:    {r:.4f}")
                print(f"         F1-Score:  {f1:.4f}")
                print(f"         Correct:   {tp}/{total}")
                print(f"         Time:      {dt}s")
                
                self.log.append({
                    "Track": track.name, "Task": tf.stem, "Precision": p, "Recall": r, "F1-Score": f1,
                    "Time (s)": dt, "Alignments": len(alignments), "Correct": tp, "Reference": total, "Type": "Task"
                })
        
        self.results_to_csv(csv_out)
        self.matcher.embedding_cache.save_index()

    def results_to_csv(self, filename: str):
        # Writes accumulated benchmark logging and evaluation records to a CSV file
        fields = ["Track", "Task", "Precision", "Recall", "F1-Score", "Time (s)", "Alignments", "Correct", "Reference", "Type"]
        with open(self.results_dir / filename, mode="w", newline="", encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=fields).writerows(self.log)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" [INFO] Running on execution device: {device}")
    
    m = MOSAIC(
        model_name="sentence-transformers/all-MiniLM-L12-v2",
        device=device, chunk_size=100_000, faiss_nprobe=128
    )
    runner = OAEITrackRunner(matcher=m, results_dir=RESULTS_DIR)
    runner.run_all_tracks(str(BASE_DIR), csv_out="mosaic_report.csv")