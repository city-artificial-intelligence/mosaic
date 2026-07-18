import csv
import gc
import re
import time
import warnings
import numpy as np
import torch
import faiss
from pathlib import Path
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
        string_score = (self.ngram_score * 0.5) + (self.levenshtein_score * 0.5)
        return (self.semantic_score * 0.55) + (string_score * 0.45)


class EmbeddingDiskCache:
    def __init__(self, cache_dir: Path, embedding_dim: int):
        self.cache_dir = cache_dir
        self.embedding_dim = embedding_dim
        self.label_to_idx = {}
        self.embeddings_mmap = None
        self.current_count = 0
        self.max_embeddings = 3_000_000
        self._init_mmap()
    
    def _init_mmap(self):
        mmap_path = self.cache_dir / "embeddings.npy"
        index_path = self.cache_dir / "labels.txt"
        
        if mmap_path.exists() and index_path.exists():
            try:
                self.embeddings_mmap = np.load(str(mmap_path), mmap_mode='r+')
                with open(index_path, 'r', encoding='utf-8') as f:
                    for idx, line in enumerate(f):
                        self.label_to_idx[line.strip()] = idx
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
        if label in self.label_to_idx:
            return np.array(self.embeddings_mmap[self.label_to_idx[label]])
        return None
    
    def add_batch(self, labels: List[str], embeddings: np.ndarray):
        if not labels: return
        start_idx = self.current_count
        end_idx = min(start_idx + len(labels), self.max_embeddings)
        
        labels = labels[:end_idx - start_idx]
        embeddings = embeddings[:end_idx - start_idx]
        if not labels: return
            
        self.embeddings_mmap[start_idx:end_idx] = embeddings
        self.embeddings_mmap.flush()
        
        for i, label in enumerate(labels):
            self.label_to_idx[label] = start_idx + i
        self.current_count = end_idx
    
    def save_index(self):
        index_path = self.cache_dir / "labels.txt"
        sorted_labels = [None] * self.current_count
        for label, idx in self.label_to_idx.items():
            if idx < self.current_count:
                sorted_labels[idx] = label
        with open(index_path, 'w', encoding='utf-8') as f:
            f.writelines(f"{lbl}\n" for lbl in sorted_labels if lbl is not None)


class MOSAIC:
    _CAMEL_RE = re.compile(r'([a-z])([A-Z])')
    _PCT_RE = re.compile(r'%[0-9A-Fa-f]{2}')
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L12-v2"

    def __init__(self, model_name=None, thresholds=None, device="cuda"):
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
        self.ontology_size = 0
        self.is_mega_scale = False
        
        self.embedding_cache = EmbeddingDiskCache(CACHE_DIR, embedding_dim=self.embedding_dim)
        self._ngram_cache = {}
        self._lev_cache = {}
    
    def init_model(self):
        if self.model is not None: return
        self.model = SentenceTransformer(self.model_name, device=self.device)
        try: self.model.max_seq_length = 128
        except: pass
    
    def normalise_label(self, text: str) -> str:
        if not text: return ""
        text = str(text)
        
        for prefix in ["Category:", "Template:", "File:", "Property:", "Category_talk:", "User:"]:
            if text.startswith(prefix):
                text = text[len(prefix):]
                
        text = self._CAMEL_RE.sub(r'\1 \2', text)
        text = text.replace('_', ' ').replace('-', ' ').lower().strip()
        return " ".join(text.split()[:20])
    
    def load_ontology(self, path: Path) -> Optional[Graph]:
        g = Graph()
        formats = ["turtle", "xml"] if path.suffix == ".ttl" else ["xml", "turtle"]
        for fmt in formats:
            try: g.parse(str(path), format=fmt); return g
            except: continue
        try: g.parse(str(path)); return g
        except: return None
    
    def extract_entities_streaming(self, path: Path, graph: Optional[Graph] = None) -> Iterator[Tuple[URIRef, str, URIRef]]:
        if HAS_PYOXIGRAPH and path.exists():
            print(" [INFO] Rust-accelerated pyoxigraph parser selected.")
            lbl_map = defaultdict(list)
            type_map = defaultdict(list)
            ox_format = pyoxigraph.RdfFormat.TURTLE if path.suffix.lower() == ".ttl" else pyoxigraph.RdfFormat.RDF_XML
            
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
                print(f" [WARNING] pyoxigraph failed: {e}. Falling back.")
                if graph is None: graph = self.load_ontology(path)
                yield from self._extract_entities_rdflib(graph)
                return
            
            yield from self._process_extracted_maps(lbl_map, type_map)
        else:
            if graph is None: graph = self.load_ontology(path)
            yield from self._extract_entities_rdflib(graph)

    def _extract_entities_rdflib(self, graph: Graph) -> Iterator[Tuple[URIRef, str, URIRef]]:
        if graph is None: return
        lbl_map, type_map = defaultdict(list), defaultdict(list)
        for s, p, o in graph:
            if p in (RDFS.label, SKOS.prefLabel, SKOS.altLabel): lbl_map[s].append((p, o))
            elif p == RDF.type: type_map[s].append(o)
        yield from self._process_extracted_maps(lbl_map, type_map)

    def _process_extracted_maps(self, lbl_map, type_map) -> Iterator[Tuple[URIRef, str, URIRef]]:
        skos_concepts, owl_classes, props, all_subjects = set(), set(), set(), set()
        
        for s, types in type_map.items():
            is_skos = any(t == SKOS.Concept for t in types)
            is_class = any(t == OWL.Class for t in types)
            is_prop = any(t in (OWL.ObjectProperty, OWL.DatatypeProperty) for t in types)
            
            if is_skos: skos_concepts.add(s)
            elif is_class: owl_classes.add(s)
            elif is_prop: props.add(s)
            all_subjects.add(s)
        
        is_valid = lambda uri: "oboInOwl" not in str(uri) and not str(uri).startswith(_CORE_NAMESPACES)
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
                            
            return self.normalise_label(self._PCT_RE.sub(' ', str(uri).split('/')[-1].split('#')[-1]))

        for group, etype in [(owl_classes, OWL.Class), (skos_concepts, SKOS.Concept), (props, OWL.ObjectProperty)]:
            for uri in filter(is_valid, group):
                lbl = get_fast_label(uri)
                if lbl: yield uri, lbl, etype
                
        for uri in filter(is_valid, all_subjects):
            if uri not in owl_classes and uri not in skos_concepts and uri not in props:
                lbl = get_fast_label(uri)
                if lbl: yield uri, lbl, OWL.NamedIndividual
    
    def get_embeddings_adaptive(self, labels: List[str]) -> np.ndarray:
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
                new_embs_array = self.model.encode(to_embed, convert_to_tensor=False, show_progress_bar=False, batch_size=512)
            if len(new_embs_array) > 0:
                self.embedding_cache.add_batch(to_embed, new_embs_array)
                result[to_embed_indices] = new_embs_array
                
        return result
    
    def ngram_similarity(self, s1: str, s2: str, n: int = 2) -> float:
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
        key = (s1, s2)
        if key in self._lev_cache: return self._lev_cache[key]
        
        if s1 == s2: 
            result = 1.0
        elif HAS_RAPIDFUZZ:
            result = fuzz.ratio(s1, s2) / 100.0
        else:
            if not s1 or not s2: result = 0.0
            elif abs(len(s1) - len(s2)) / max(len(s1), len(s2), 1) > 0.4: result = 0.0
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
                result = (max(len(s1), len(s2)) - previous_row[-1]) / max(len(s1), len(s2))
        
        self._lev_cache[key] = result
        return result
    
    def _get_adaptive_k(self) -> int:
        if self.ontology_size < 40000: return 1
        if self.ontology_size < 80000: return 3
        if self.ontology_size < 100000: return 5
        if self.ontology_size < 200000: return 8
        return 20
    
    def _get_threshold_scale(self) -> float:
        if self.ontology_size < 200: return 1.1
        if self.ontology_size < 275: return 0.98
        if self.ontology_size < 500: return 1.02
        if self.ontology_size < 1000: return 1.1
        if self.ontology_size < 25000: return 1.2
        if self.ontology_size < 50000: return 1.08
        if self.ontology_size < 100000: return 1.0
        if self.ontology_size >= 130000: return 1.2
        return 0.92
    
    def _is_mutual_match(self, src_score: float, tgt_reverse_score: float, threshold: float) -> bool:
        return src_score >= threshold and tgt_reverse_score >= (threshold * (0.94 if self.is_mega_scale else 0.88))
    
    def semantic_match_chunked(self, src_labels: List[str], src_uris: List[str],
                               tgt_labels: List[str], tgt_uris: List[str],
                               etype: URIRef, relaxation_factor: float = 1.0) -> Iterator[AlignmentCandidate]:
        threshold = self.thresholds.get(etype, self.default_thres) * relaxation_factor
        src_embs = self.get_embeddings_adaptive(src_labels)
        tgt_embs = self.get_embeddings_adaptive(tgt_labels)
        
        dim = src_embs.shape[1]
        k = min(self._get_adaptive_k(), len(tgt_labels))
        if k <= 0: return
        
        def build_faiss_index(embs, num_labels):
            if num_labels < 15000:
                idx = faiss.IndexFlatIP(dim)
            else:
                nlist = max(16, min(16384, int(4 * np.sqrt(num_labels))))
                if num_labels < (nlist * 39):
                    idx = faiss.IndexFlatIP(dim)
                else:
                    idx = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, nlist, faiss.METRIC_INNER_PRODUCT)
                    idx.train(embs)
                    idx.nprobe = max(8, min(nlist, nlist // 16))
            idx.add(embs)
            return idx

        index_fwd = build_faiss_index(tgt_embs, len(tgt_labels))
        index_rev = build_faiss_index(src_embs, len(src_labels))
        
        scores_fwd, indices_fwd = index_fwd.search(src_embs, k=k)
        target_selection_freq = Counter(indices_fwd.flatten().tolist())
        
        tgt_indices_list = list(set(indices_fwd.flatten().tolist()))
        rev_k = min(self._get_adaptive_k(), len(src_labels))
        rev_scores_full, rev_indices_full = index_rev.search(tgt_embs[tgt_indices_list], k=rev_k)
        
        rev_map = {}
        for batch_idx, tgt_idx in enumerate(tgt_indices_list):
            for src_rank, src_idx in enumerate(rev_indices_full[batch_idx]):
                rev_map[(int(src_idx), int(tgt_idx))] = float(rev_scores_full[batch_idx][src_rank])
        
        for src_idx in range(len(src_labels)):
            src_uri, src_lbl = src_uris[src_idx], src_labels[src_idx]
            
            for kth in range(k):
                score_fwd = float(scores_fwd[src_idx][kth])
                tgt_idx = int(indices_fwd[src_idx][kth])
                
                if self.is_mega_scale:
                    hits = target_selection_freq.get(tgt_idx, 1)
                    if hits > 3: score_fwd *= max(0.70, 1.0 - (0.02 * hits))
                
                if score_fwd < threshold * 0.88: continue
                
                reverse_score = rev_map.get((src_idx, tgt_idx), threshold * 0.85)
                if not self._is_mutual_match(score_fwd, reverse_score, threshold): continue
                
                tgt_lbl = tgt_labels[tgt_idx]
                yield AlignmentCandidate(
                    source=str(src_uri), target=str(tgt_uris[tgt_idx]), etype=str(etype),
                    semantic_score=score_fwd, ngram_score=self.ngram_similarity(src_lbl, tgt_lbl),
                    levenshtein_score=self.levenshtein_similarity(src_lbl, tgt_lbl)
                )
        
        del index_fwd, index_rev
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    def align_optimized(self, src_entities: Dict, tgt_entities: Dict) -> Set[Tuple[str, str, str]]:
        final_alignments, claimed_src, claimed_tgt = set(), set(), set()
        self.ontology_size = (len(src_entities) + len(tgt_entities)) / 2
        self.is_mega_scale = (self.ontology_size >= 130000)
        
        print(f" [INFO] {'Mega-scale Optimization Mode ACTIVE' if self.is_mega_scale else 'Standard matching logic active'} (size: {self.ontology_size:.0f} nodes)")

        original_thresholds = self.thresholds
        self.thresholds = {k: v * self._get_threshold_scale() for k, v in self.thresholds.items()}
        
        tgt_lookup = defaultdict(list)
        for uri, meta in tgt_entities.items():
            tgt_lookup[meta["label"]].append((uri, meta["type"]))
            
        for s_uri, s_meta in src_entities.items():
            if s_meta["label"] in tgt_lookup:
                for t_uri, t_type in tgt_lookup[s_meta["label"]]:
                    if s_meta["type"] == t_type:
                        claimed_src.add(s_uri)
                        claimed_tgt.add(t_uri)
                        final_alignments.add((str(s_uri), str(t_uri), str(s_meta["type"])))
                        break
        del tgt_lookup
        gc.collect()
        
        distinct_types = [OWL.Class, SKOS.Concept, OWL.ObjectProperty, OWL.DatatypeProperty, OWL.NamedIndividual]
        final_alignments, claimed_src, claimed_tgt = self._match_stage(
            src_entities, tgt_entities, distinct_types, claimed_src, claimed_tgt, final_alignments, relaxation_factor=1.0
        )
        
        if not self.is_mega_scale:
            final_alignments, claimed_src, claimed_tgt = self._match_stage(
                src_entities, tgt_entities, distinct_types, claimed_src, claimed_tgt, final_alignments, relaxation_factor=0.92
            )
        
        self.thresholds = original_thresholds
        return final_alignments

    def _match_stage(self, src_entities, tgt_entities, distinct_types, claimed_src, claimed_tgt, alignments, relaxation_factor=1.0):
        filtered_src = {k: v for k, v in src_entities.items() if k not in claimed_src}
        filtered_tgt = {k: v for k, v in tgt_entities.items() if k not in claimed_tgt}
        if not filtered_src or not filtered_tgt: return alignments, claimed_src, claimed_tgt
        
        stage_candidates = []
        for etype in distinct_types:
            src_subset = [(u, m["label"]) for u, m in filtered_src.items() if m["type"] == etype]
            tgt_subset = [(u, m["label"]) for u, m in filtered_tgt.items() if m["type"] == etype]
            if not src_subset or not tgt_subset: continue
            
            src_uris, src_labels = zip(*src_subset)
            tgt_uris, tgt_labels = zip(*tgt_subset)
            
            for candidate in self.semantic_match_chunked(list(src_labels), list(src_uris), list(tgt_labels), list(tgt_uris), etype, relaxation_factor):
                if self.is_mega_scale:
                    s_tokens = set(src_entities[candidate.source]["label"].split())
                    t_tokens = set(tgt_entities[candidate.target]["label"].split())
                    intersection = s_tokens.intersection(t_tokens)
                    if len(intersection) >= min(len(s_tokens), len(t_tokens)) and len(intersection) >= 2:
                        if candidate.combined_score < 0.70: continue
                    else:
                        if candidate.levenshtein_score < 0.58 or candidate.combined_score < 0.81: continue
                stage_candidates.append(candidate)
                
        stage_candidates.sort(key=lambda x: x.combined_score, reverse=True)
        for c in stage_candidates:
            if c.source not in claimed_src and c.target not in claimed_tgt:
                claimed_src.add(c.source)
                claimed_tgt.add(c.target)
                alignments.add((c.source, c.target, c.etype))
        return alignments, claimed_src, claimed_tgt
    
    def convert_to_rdf_triples(self, alignments: Set[Tuple[str, str, str]]) -> Set[Tuple[str, str, str]]:
        rdf_triples = set()
        for src, tgt, etype_str in alignments:
            pred = "http://www.w3.org/2002/07/owl#equivalentClass" if "Class" in etype_str else \
                   "http://www.w3.org/2002/07/owl#equivalentProperty" if "Property" in etype_str else \
                   "http://www.w3.org/2002/07/owl#sameAs"
            rdf_triples.add((src, pred, tgt))
        return rdf_triples


class OAEITrackRunner:
    def __init__(self, matcher: MOSAIC, results_dir: Path = None):
        self.matcher = matcher
        self.results_dir = results_dir or RESULTS_DIR
        self.log = []
    
    def load_reference_alignments(self, path: Path) -> Set[Tuple[str, str, str]]:
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
        g = Graph()
        for src, pred, tgt in alignments:
            try: g.add((URIRef(src), URIRef(pred), URIRef(tgt)))
            except: pass
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            g.serialize(destination=str(path), format="turtle")
            return True
        except: return False
    
    def calculate_metrics(self, sys_align, ref_align) -> Tuple[float, float, float, int, int]:
        if not ref_align: return 0.0, 0.0, 0.0, 0, 0
        sys_canon = {(tuple(sorted([str(s), str(o)]))[0], str(p), tuple(sorted([str(s), str(o)]))[1]) for s, p, o in sys_align}
        tp = len(sys_canon.intersection(ref_align))
        p = tp / len(sys_canon) if sys_canon else 0.0
        r = tp / len(ref_align) if ref_align else 0.0
        return round(p, 4), round(r, 4), round((2 * p * r) / (p + r) if (p + r) > 0 else 0.0, 4), tp, len(ref_align)
    
    def find_ontology_file(self, folder: Path, name: str) -> Optional[Path]:
        for ext in [".owl", ".rdf", ".ttl", ".xml"]:
            p = folder / f"{name}{ext}"
            if p.exists(): return p
        return None
    
    def run_all_tracks(self, base_dir: str, csv_out: str = "report_mega_scale.csv"):
        base_path = Path(base_dir)
        if not base_path.exists(): return
        
        for track in sorted(base_path.iterdir()):
            if not track.is_dir(): continue
            for tf in sorted(track.glob("*.ttl")):
                parts = tf.stem.split("-")
                if len(parts) != 2:
                    if "human-mouse" in tf.stem: parts = ["human", "mouse"]
                    else: continue
                
                ont_folder = track / "ontologies"
                src_p, tgt_p = self.find_ontology_file(ont_folder, parts[0]), self.find_ontology_file(ont_folder, parts[1])
                if not src_p or not tgt_p: continue
                
                print(f"\n [TASK] Loading reference alignments: {tf.stem}")
                ref_align = self.load_reference_alignments(tf)
                t_ext0 = time.time()
                
                src_entities = {str(u): {"label": l, "type": t} for u, l, t in self.matcher.extract_entities_streaming(src_p)}
                tgt_entities = {str(u): {"label": l, "type": t} for u, l, t in self.matcher.extract_entities_streaming(tgt_p)}
                print(f" [TASK] Extracted {len(src_entities)} source and {len(tgt_entities)} target entities in {round(time.time() - t_ext0, 2)}s")
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
        self.matcher.embedding_cache.save_index()

    def results_to_csv(self, filename: str):
        fields = ["Track", "Task", "Precision", "Recall", "F1-Score", "Time (s)", "Alignments", "Correct", "Reference", "Type"]
        with open(self.results_dir / filename, mode="w", newline="", encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=fields).writerows(self.log)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" [INFO] Running on execution device: {device}")
    
    m = MOSAIC(model_name="sentence-transformers/all-MiniLM-L12-v2", device=device)
    runner = OAEITrackRunner(matcher=m, results_dir=RESULTS_DIR)
    runner.run_all_tracks(str(BASE_DIR), csv_out="mosaic_report.csv")