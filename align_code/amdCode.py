import csv, gc, re, time, warnings, logging, math, unicodedata
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator, List, Tuple, Set, Dict, Optional
from collections import defaultdict, Counter
from functools import lru_cache
from difflib import SequenceMatcher
import numpy as np
import torch
import faiss

from sentence_transformers import SentenceTransformer
from rdflib import Graph, RDF, RDFS, URIRef
from rdflib.namespace import OWL, SKOS

try:
    import pyoxigraph
    HAS_PYOXIGRAPH = True
except ImportError:
    HAS_PYOXIGRAPH = False

try:
    import pyhornedowl
    HAS_HORNED_OWL = True
except ImportError:
    HAS_HORNED_OWL = False

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

RDFS_LABEL, SKOS_PREF_LABEL, SKOS_ALT_LABEL = str(RDFS.label), str(SKOS.prefLabel), str(SKOS.altLabel)
SKOS_BROADER = "http://www.w3.org/2004/02/skos/core#broader"

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

_FUNCTIONAL_SNIFF_RE = re.compile(r'\b(Ontology|Prefix|Declaration|SubClassOf|EquivalentClasses|AnnotationAssertion)\s*\(')
_MANCHESTER_SNIFF_RE = re.compile(r'^\s*(Ontology|Class|ObjectProperty)\s*:', re.M)
_XML_SNIFF_RE = re.compile(r'<\?xml|<rdf:RDF|<Ontology[\s>]')

@dataclass
class MOSAICConfig:
    """Master parameter tuning configuration for MOSAIC alignment engine."""
    # FAISS & Search parameters
    ivf_training_ratio: int = 39
    ivf_nprobe_divisor: int = 16

    # Precision-Optimized Score Weights
    semantic_weight: float = 0.60
    string_weight: float = 0.40
    ngram_weight: float = 0.30
    levenshtein_weight: float = 0.35
    isub_weight: float = 0.35
    structural_bonus: float = 0.10

    # Hub Penalty tuning
    hub_freq_threshold: int = 4
    hub_penalty_step: float = 0.03
    hub_penalty_max_discount: float = 0.65

    # Filter / Gating Threshold Ratios (Tightened for Precision)
    cutoff_ratio_floor: float = 0.72
    symmetry_diff_tolerance: float = 0.20
    general_string_gate: float = 0.45
    general_overlap_gate: float = 0.25

    # Length Imbalance Guard Threshold
    max_length_ratio_imbalance: float = 2.5

    # Model defaults
    default_model: str = "sentence-transformers/all-MiniLM-L12-v2"
    bgem3_model: str = "BAAI/bge-m3"
    sapbert_model: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

    # Memory/Performance tuning
    use_fp16_cache: bool = True
    use_fp16_compute: bool = True
    mega_scale_pq_threshold: int = 60000
    pq_m_subquantizers: int = 16
    pq_bits: int = 8
    ivf_pq_training_ratio: int = 40
    encode_batch_size_gpu: int = 1024
    encode_batch_size_cpu: int = 256
    max_bucket_chunk: int = 150000

def sniff_owl_format(path: Path, sample_size: int = 8192) -> str:
    try:
        with open(path, 'rb') as f:
            raw = f.read(sample_size)
        text = raw.decode('utf-8', errors='ignore')
    except Exception:
        return "unknown"

    stripped = text.lstrip()
    if _XML_SNIFF_RE.search(text[:2048]):
        return "rdfxml"
    if _FUNCTIONAL_SNIFF_RE.search(text[:2048]):
        return "functional"
    if _MANCHESTER_SNIFF_RE.search(text):
        return "manchester"
    if stripped.startswith(("@prefix", "@base", "PREFIX", "BASE")) or re.search(r'^\s*<[^>]+>\s+a\s+', text, re.M):
        return "turtle"
    return "unknown"

BIO_ML_ALIGNMENT_TASKS = [
    "ncit-doid.rdf",
    "snomed-fma.rdf",
    "snomed-ncit.rdf"
]

class MedicalDomainDetector:
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
        combined_id = f"{track_name} {task_name}".lower()
        if any(kw in combined_id for kw in ["anatomy", "disease", "phenotype", "human-mouse", "pharmaceutical", "med", "bio"]):
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

class EntityMeta:
    __slots__ = ("label", "type", "parents")

    def __init__(self, label: str, type_, parents):
        self.label = label
        self.type = type_
        self.parents = parents

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

@dataclass
class AlignmentCandidate:
    source: str
    target: str
    etype: str
    semantic_score: float
    ngram_score: float = 0.0
    levenshtein_score: float = 0.0
    isub_score: float = 0.0
    structural_bonus: float = 0.0
    is_exact_match: bool = False
    config: MOSAICConfig = field(default_factory=MOSAICConfig)

    @property
    def combined_score(self) -> float:
        if self.is_exact_match:
            return 1.0
        c = self.config
        string_score = (self.ngram_score * c.ngram_weight) + (self.levenshtein_score * c.levenshtein_weight) + (self.isub_score * c.isub_weight)
        base = (self.semantic_score * c.semantic_weight) + (string_score * c.string_weight)
        return max(0.0, min(1.0, base + self.structural_bonus))

class EmbeddingDiskCache:
    def __init__(self, cache_dir: Path, embedding_dim: int, max_embeddings: int = 3_000_000,
                 use_fp16: bool = True):
        self.cache_dir, self.embedding_dim = cache_dir, embedding_dim
        self.label_to_idx, self.embeddings_mmap = {}, None
        self.current_count, self.max_embeddings = 0, max_embeddings
        self.dtype = np.float16 if use_fp16 else np.float32
        self._init_mmap()

    def _init_mmap(self):
        mmap_path, index_path = self.cache_dir / "embeddings.npy", self.cache_dir / "labels.txt"
        if mmap_path.exists() and index_path.exists():
            try:
                self.embeddings_mmap = np.load(str(mmap_path), mmap_mode='r+')
                if self.embeddings_mmap.dtype != self.dtype:
                    raise ValueError("cache dtype mismatch, rebuilding")
                with open(index_path, 'r', encoding='utf-8') as f:
                    self.label_to_idx = {line.strip(): idx for idx, line in enumerate(f)}
                self.current_count = len(self.label_to_idx)
                return
            except Exception:
                pass

        self.embeddings_mmap = np.memmap(str(mmap_path), dtype=self.dtype, mode='w+',
                                          shape=(self.max_embeddings, self.embedding_dim))
        self.current_count, self.label_to_idx = 0, {}

    def add_batch(self, labels: List[str], embeddings: np.ndarray):
        if not labels or self.embeddings_mmap is None:
            return
        start_idx = self.current_count
        end_idx = min(start_idx + len(labels), self.max_embeddings)
        labels, embeddings = labels[:end_idx - start_idx], embeddings[:end_idx - start_idx]
        if not labels:
            return

        if embeddings.dtype != self.dtype:
            embeddings = embeddings.astype(self.dtype)
        self.embeddings_mmap[start_idx:end_idx] = embeddings
        self.embeddings_mmap.flush()
        for i, label in enumerate(labels):
            self.label_to_idx[label] = start_idx + i
        self.current_count = end_idx

    def save_index(self):
        sorted_labels = [None] * self.current_count
        for label, idx in self.label_to_idx.items():
            if idx < self.current_count:
                sorted_labels[idx] = label
        with open(self.cache_dir / "labels.txt", 'w', encoding='utf-8') as f:
            f.writelines(f"{lbl}\n" for lbl in sorted_labels if lbl is not None)

class MOSAIC:
    _CAMEL_RE = re.compile(r'([a-z0-9])([A-Z])')
    _PCT_RE = re.compile(r'%[0-9A-Fa-f]{2}')
    _STOPWORDS = {
        "a", "an", "the", "of", "in", "on", "at", "by", "for", "with",
        "about", "against", "between", "into", "through", "during",
        "before", "after", "above", "below", "to", "from", "up", "down",
        "and", "or", "is", "are", "was", "were", "be", "been", "being",
        "has", "have", "had", "do", "does", "did", "not"
    }

    def __init__(self, model_name=None, config: MOSAICConfig = None, thresholds=None, device="cuda"):
        self.config = config or MOSAICConfig()
        self.thresholds = thresholds or {
            OWL.Class: 0.78, SKOS.Concept: 0.85, OWL.ObjectProperty: 0.82,
            OWL.DatatypeProperty: 0.82, OWL.NamedIndividual: 0.80
        }
        self.default_thres = 0.80
        self.default_model_name = model_name or self.config.default_model
        self.current_model_name, self.is_medical_domain, self.device = None, False, device
        self.model, self.embedding_dim, self.ontology_size, self.is_mega_scale = None, None, 0, False
        self.is_small_scale = False
        self.embedding_cache = None

    def update_ontology_size(self, src_entities: Dict, tgt_entities: Dict) -> float:
        n_src, n_tgt = len(src_entities), len(tgt_entities)
        if n_src > 100 and n_tgt > 100:
            self.ontology_size = (n_src + n_tgt) / 2
        else:
            self.ontology_size = (n_src + n_tgt) / 4
        self.is_mega_scale = (self.ontology_size >= 130000)
        return self.ontology_size

    def apply_domain_model(self, is_medical: bool, total_entities: int = 0):
        self.is_medical_domain = is_medical
        self.is_small_scale = (0 < total_entities < 3000)

        if self.is_small_scale:
            target_model = self.config.bgem3_model
            model_key = "bgem3"
        elif self.is_medical_domain:
            target_model = self.config.sapbert_model
            model_key = "sapbert"
        else:
            target_model = self.default_model_name
            model_key = "minilm"

        if self.current_model_name != target_model:
            print(f" [INFO] Switching Backbone Model to: {target_model}")
            self.model = SentenceTransformer(target_model, device=self.device)
            try:
                if target_model == self.config.bgem3_model:
                    self.model.max_seq_length = 64
                else:
                    self.model.max_seq_length = 512 if self.is_small_scale else 128
            except Exception:
                pass

            if self.config.use_fp16_compute and str(self.device).startswith("cuda"):
                try:
                    self.model.half()
                except Exception:
                    pass

            self.current_model_name = target_model
            self.embedding_dim = (self.model.get_embedding_dimension()
                                   if hasattr(self.model, "get_embedding_dimension")
                                   else self.model.get_sentence_embedding_dimension())
            model_cache_dir = CACHE_DIR / model_key
            model_cache_dir.mkdir(parents=True, exist_ok=True)
            self.embedding_cache = EmbeddingDiskCache(model_cache_dir, embedding_dim=self.embedding_dim,
                                                        use_fp16=self.config.use_fp16_cache)

    def init_model(self):
        if self.model is None:
            self.apply_domain_model(is_medical=False)

    @classmethod
    def normalise_label(cls, text: str) -> str:
        if not text:
            return ""
        text = str(text)

        for prefix in ("Category:", "Template:", "File:", "Property:", "Category_talk:", "User:"):
            if text.startswith(prefix):
                text = text[len(prefix):]

        text = cls._CAMEL_RE.sub(r'\1 \2', text)
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text).lower().strip()
        
        tokens = [t for t in text.split() if t]
        return " ".join(tokens[:20])

    def load_ontology(self, path: Path) -> Optional[Graph]:
        g = Graph()
        detected = sniff_owl_format(path)
        formats = ["turtle", "xml"] if detected == "turtle" else ["xml", "turtle"]

        for fmt in formats:
            try:
                g.parse(str(path), format=fmt)
                return g
            except Exception:
                continue
        try:
            g.parse(str(path))
            return g
        except Exception:
            return None

    def extract_entities_streaming(self, path: Path, graph: Optional[Graph] = None) -> Iterator[Tuple[URIRef, str, URIRef, Set[str]]]:
        if not path.exists():
            return

        detected = sniff_owl_format(path)

        if detected in ("functional", "manchester"):
            if not HAS_HORNED_OWL:
                raise RuntimeError(
                    f"'{path.name}' is OWL {detected} syntax; install py-horned-owl to parse it."
                )
            yield from self._extract_entities_horned_owl(path, detected)
            return

        if HAS_PYOXIGRAPH:
            lbl_map, type_map, parent_map = defaultdict(list), defaultdict(list), defaultdict(set)
            ox_format = pyoxigraph.RdfFormat.TURTLE if detected == "turtle" else pyoxigraph.RdfFormat.RDF_XML

            try:
                for triple in pyoxigraph.parse(path=str(path), format=ox_format, lenient=True):
                    s, p, o = triple.subject, triple.predicate, triple.object
                    if not isinstance(s, pyoxigraph.NamedNode):
                        continue
                    s_uri, p_val = URIRef(s.value), p.value

                    if p_val in (RDFS_LABEL, SKOS_PREF_LABEL, SKOS_ALT_LABEL) and isinstance(o, pyoxigraph.Literal):
                        lbl_map[s_uri].append((URIRef(p_val), o))
                    elif p_val == str(RDF.type) and isinstance(o, pyoxigraph.NamedNode):
                        type_map[s_uri].append(URIRef(o.value))
                    elif p_val in (str(RDFS.subClassOf), SKOS_BROADER) and isinstance(o, pyoxigraph.NamedNode):
                        parent_map[s_uri].add(str(o.value))

                if lbl_map or type_map:
                    yield from self._process_extracted_maps(lbl_map, type_map, parent_map)
                    return
            except Exception:
                pass

        if graph is None:
            graph = self.load_ontology(path)
        yield from self._extract_entities_rdflib(graph)

    def _extract_entities_horned_owl(self, path: Path, detected: str) -> Iterator[Tuple[URIRef, str, URIRef, Set[str]]]:
        serialization = "ofn" if detected == "functional" else "omn"
        try:
            onto = pyhornedowl.open_ontology_from_file(str(path), serialization=serialization)
        except Exception as e:
            raise RuntimeError(f"py_horned_owl failed to parse '{path.name}' ({detected} syntax): {e}")

        def best_label(iri: str) -> str:
            for ann_iri in (SKOS_PREF_LABEL, RDFS_LABEL, SKOS_ALT_LABEL):
                try:
                    val = onto.get_annotation(iri, ann_iri)
                except Exception:
                    val = None
                if val:
                    return self.normalise_label(val)
            return self.normalise_label(self._PCT_RE.sub(' ', iri.split('/')[-1].split('#')[-1]))

        def parents_of(iri: str) -> Set[str]:
            try:
                return set(onto.get_superclasses(iri))
            except Exception:
                return set()

        for fetcher, etype in [
            (onto.get_classes, OWL.Class),
            (onto.get_object_properties, OWL.ObjectProperty),
            (onto.get_data_properties, OWL.DatatypeProperty),
            (onto.get_named_individuals, OWL.NamedIndividual)
        ]:
            try:
                iris = fetcher()
            except Exception:
                continue
            for iri in iris:
                if "oboInOwl" in iri or iri.startswith(_CORE_NAMESPACES):
                    continue
                lbl = best_label(iri)
                if lbl:
                    yield URIRef(iri), lbl, etype, parents_of(iri) if etype == OWL.Class else set()

    def _extract_entities_rdflib(self, graph: Graph) -> Iterator[Tuple[URIRef, str, URIRef, Set[str]]]:
        if graph is None:
            return
        lbl_map, type_map, parent_map = defaultdict(list), defaultdict(list), defaultdict(set)
        for s, p, o in graph:
            if p in (RDFS.label, SKOS.prefLabel, SKOS.altLabel):
                lbl_map[s].append((p, o))
            elif p == RDF.type:
                type_map[s].append(o)
            elif p in (RDFS.subClassOf, URIRef(SKOS_BROADER)) and isinstance(o, URIRef):
                parent_map[s].add(str(o))
        yield from self._process_extracted_maps(lbl_map, type_map, parent_map)

    def _process_extracted_maps(self, lbl_map, type_map, parent_map) -> Iterator[Tuple[URIRef, str, URIRef, Set[str]]]:
        skos_concepts, owl_classes, obj_props, data_props, named_individuals = set(), set(), set(), set(), set()
        all_subjects = set(type_map.keys()) | set(lbl_map.keys())

        for s in all_subjects:
            types = type_map.get(s, [])
            if any(t == SKOS.Concept for t in types):
                skos_concepts.add(s)
            elif any(t == OWL.Class for t in types):
                owl_classes.add(s)
            elif any(t == OWL.ObjectProperty for t in types):
                obj_props.add(s)
            elif any(t == OWL.DatatypeProperty for t in types):
                data_props.add(s)
            else:
                named_individuals.add(s)

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

        for group, etype in [(owl_classes, OWL.Class), (skos_concepts, SKOS.Concept),
                              (obj_props, OWL.ObjectProperty), (data_props, OWL.DatatypeProperty)]:
            for uri in filter(is_valid, group):
                lbl = get_fast_label(uri)
                if lbl:
                    yield uri, lbl, etype, parent_map.get(uri, set())

        for uri in filter(is_valid, named_individuals):
            lbl = get_fast_label(uri)
            if lbl:
                yield uri, lbl, OWL.NamedIndividual, parent_map.get(uri, set())

    def get_embeddings_adaptive(self, labels: List[str]) -> np.ndarray:
        self.init_model()
        cached_indices, to_embed, to_embed_indices = [], [], []
        cache_dict = self.embedding_cache.label_to_idx if self.embedding_cache else {}

        for i, label in enumerate(labels):
            idx = cache_dict.get(label)
            if idx is not None:
                cached_indices.append((i, idx))
            else:
                to_embed.append(label)
                to_embed_indices.append(i)

        result = np.zeros((len(labels), self.embedding_dim), dtype='float32')
        if cached_indices and self.embedding_cache:
            cached_indices.sort(key=lambda x: x[1])
            orig_indices, cache_idxs = zip(*cached_indices)
            cached_vals = self.embedding_cache.embeddings_mmap[list(cache_idxs)]
            result[list(orig_indices)] = cached_vals.astype('float32', copy=False)

        if to_embed:
            is_gpu = str(self.device).startswith("cuda")
            batch_size = self.config.encode_batch_size_gpu if is_gpu else (self.config.encode_batch_size_cpu if not self.is_small_scale else 64)
            with torch.inference_mode():
                new_embs = self.model.encode(to_embed, convert_to_tensor=False, show_progress_bar=False, batch_size=batch_size)
            new_embs = np.asarray(new_embs, dtype='float32')
            if len(new_embs) > 0:
                if self.embedding_cache:
                    self.embedding_cache.add_batch(to_embed, new_embs)
                result[to_embed_indices] = new_embs

        faiss.normalize_L2(result)
        return result

    @staticmethod
    def _sym_key(s1: str, s2: str) -> Tuple[str, str]:
        return (s1, s2) if s1 <= s2 else (s2, s1)

    @classmethod
    @lru_cache(maxsize=150000)
    def isub_similarity(cls, s1: str, s2: str) -> float:
        a, b = cls._sym_key(s1, s2)
        if a == b:
            return 1.0
        if not a or not b:
            return 0.0

        l1, l2 = len(a), len(b)
        # Precision Guard: Heavily penalize asymmetric string length mismatches
        if max(l1, l2) / max(min(l1, l2), 1) > MOSAICConfig.max_length_ratio_imbalance:
            return 0.0

        sm = SequenceMatcher(None, a, b, autojunk=False)
        a_w, b_w, common_len = a, b, 0

        for _ in range(6):
            sm.set_seqs(a_w, b_w)
            match = sm.find_longest_match(0, len(a_w), 0, len(b_w))
            if match.size < 2:
                break
            common_len += match.size
            a_w = a_w[:match.a] + a_w[match.a + match.size:]
            b_w = b_w[:match.b] + b_w[match.b + match.size:]

        if common_len == 0:
            return 0.0

        s_comm = (2.0 * common_len) / (l1 + l2)
        p_unmatched = (len(a_w) * len(b_w)) / (l1 * l2) if (l1 * l2) > 0 else 0.0
        w_sub = s_comm - (p_unmatched * 0.3)

        return max(0.0, min(1.0, w_sub))

    @classmethod
    @lru_cache(maxsize=150000)
    def ngram_similarity(cls, s1: str, s2: str, n: int = 3) -> float:
        a, b = cls._sym_key(s1.lower(), s2.lower())
        if a == b:
            return 1.0
        if len(a) < n or len(b) < n:
            n = 2
            if len(a) < n or len(b) < n:
                return 1.0 if a == b else 0.0

        ng1 = set(a[i:i+n] for i in range(len(a) - n + 1))
        ng2 = set(b[i:i+n] for i in range(len(b) - n + 1))
        
        inter = len(ng1 & ng2)
        if not inter:
            return 0.0
        union = len(ng1) + len(ng2) - inter
        return inter / union if union else 0.0

    @classmethod
    @lru_cache(maxsize=150000)
    def levenshtein_similarity(cls, s1: str, s2: str) -> float:
        a, b = cls._sym_key(s1, s2)
        if a == b:
            return 1.0
        
        len_a, len_b = len(a), len(b)
        if len_a == 0 or len_b == 0:
            return 0.0

        # Precision Guard: Ratio Imbalance Check
        if max(len_a, len_b) / max(min(len_a, len_b), 1) > MOSAICConfig.max_length_ratio_imbalance:
            return 0.0

        if HAS_RAPIDFUZZ:
            raw_score = fuzz.ratio(a, b) / 100.0
        else:
            if abs(len_a - len_b) / max(len_a, len_b) > 0.4:
                return 0.0

            if len_a < len_b:
                a, b = b, a
                len_a, len_b = len_b, len_a

            prev = list(range(len_b + 1))
            for i, c1 in enumerate(a):
                curr = [i + 1]
                for j, c2 in enumerate(b):
                    curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
                prev = curr
            raw_score = (len_a - prev[-1]) / len_a

        # Dampen short-token non-exact matches to prevent false positives
        min_len = min(len_a, len_b)
        if min_len <= 3 and raw_score < 1.0:
            return raw_score * 0.4
        elif min_len <= 5 and raw_score < 0.88:
            return raw_score * 0.70

        return raw_score

    @classmethod
    def token_sort_similarity(cls, s1: str, s2: str) -> float:
        t1 = set(w for w in s1.split() if w not in cls._STOPWORDS)
        t2 = set(w for w in s2.split() if w not in cls._STOPWORDS)

        if not t1 or not t2:
            return 0.0

        inter = t1 & t2
        if not inter:
            return 0.0

        union = t1 | t2
        jaccard = len(inter) / len(union)
        
        if len(inter) == len(t1) == len(t2):
            return 1.0
            
        return jaccard

    @classmethod
    def acronym_bonus(cls, s1: str, s2: str) -> float:
        a, b = s1.strip(), s2.strip()
        if not a or not b:
            return 0.0
        
        short, long_ = (a, b) if len(a) <= len(b) else (b, a)
        short_compact = short.replace(" ", "")

        if not (2 <= len(short_compact) <= 8) or " " not in long_:
            return 0.0

        long_words = [w for w in long_.split() if w.lower() not in cls._STOPWORDS]
        if len(long_words) < 2:
            long_words = [w for w in long_.split() if w]

        initials = "".join(w[0] for w in long_words).lower()
        short_lower = short_compact.lower()

        if short_lower == initials:
            return 1.0
        if len(short_compact) >= 3 and short_lower == initials[:len(short_compact)]:
            return 0.8

        return 0.0

    def evaluate_structural_similarity(self, src_uri: str, tgt_uri: str,
                                       src_entities: Dict, tgt_entities: Dict) -> float:
        sp = src_entities.get(src_uri, {}).get("parents", set())
        tp = tgt_entities.get(tgt_uri, {}).get("parents", set())
        if not sp or not tp:
            return 0.0
        inter = len(sp & tp)
        if inter == 0:
            return 0.0
        union = len(sp | tp)
        jaccard = inter / union if union else 0.0
        return self.config.structural_bonus * jaccard

    def _get_adaptive_k(self, relaxation_factor: float = 1.0, bucket_size: Optional[int] = None,
                         ambiguity: Optional[float] = None) -> int:
        s = self.ontology_size

        if s < 1000:
            global_k = 1
        elif s < 10000:
            global_k = 1
        elif s < 50000:
            global_k = 2
        elif s < 130000:
            global_k = 3
        elif s < 400000:
            global_k = 5
        else:
            global_k = 25

        if bucket_size is not None:
            local_k = int(max(3, min(50, round(2.0 * (bucket_size ** 0.5)))))
            base_k = int(round((0.35 * global_k) + (0.65 * local_k)))
        else:
            base_k = global_k

        if ambiguity is not None:
            base_k = int(round(base_k * (1.0 + 0.6 * max(0.0, min(1.0, ambiguity)))))

        if relaxation_factor < 1.0:
            base_k = int(base_k * 1.3)

        return max(1, min(base_k, 150))

    @staticmethod
    def _estimate_ambiguity(embs: np.ndarray, sample_size: int = 64) -> float:
        n = embs.shape[0]
        if n < 8:
            return 0.0
        sample_n = min(sample_size, n)
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(n, size=sample_n, replace=False)
        probe_k = min(5, n)

        try:
            idx = faiss.IndexFlatIP(embs.shape[1])
            idx.add(embs)
            scores, _ = idx.search(embs[sample_idx], k=probe_k)
        except Exception:
            return 0.0

        if scores.shape[1] < 2:
            return 0.0

        top = scores[:, 0]
        tail = scores[:, -1]
        decay = np.clip(top - tail, 0.0, 2.0)
        mean_decay = float(np.mean(decay))
        ambiguity = 1.0 - min(1.0, mean_decay / 0.5)
        return max(0.0, min(1.0, ambiguity))

    def _get_threshold_scale(self) -> float:
        base = (1.12 if self.is_medical_domain else 1.05) if self.is_small_scale else (1.08 if self.is_medical_domain else 1.00)
        s = self.ontology_size
        if s < 1000:
            scale = 0.95
        elif s < 12500:
            scale = 1.00
        elif s < 50000:
            scale = 1.05
        elif s < 130000:
            scale = 1.08
        else:
            scale = 1.18
        return scale * base

    @staticmethod
    def _build_idf_weights(*token_sets_lists) -> Dict[str, float]:
        df, n_docs = Counter(), 0
        for tok_sets in token_sets_lists:
            for toks in tok_sets:
                if toks:
                    n_docs += 1
                    df.update(toks)
        if n_docs == 0:
            return {}
        return {tok: math.log((n_docs + 1) / (c + 1)) + 1.0 for tok, c in df.items()}

    @staticmethod
    def _weighted_jaccard(s_toks: Set[str], t_toks: Set[str], idf: Dict[str, float]) -> float:
        if not s_toks or not t_toks:
            return 0.0
        inter = s_toks & t_toks
        if not inter:
            return 0.0
        union = s_toks | t_toks
        w_inter = sum(idf.get(t, 1.0) for t in inter)
        w_union = sum(idf.get(t, 1.0) for t in union)
        return w_inter / w_union if w_union else 0.0

    def semantic_match_chunked(self, src_labels: List[str], src_uris: List[str],
                               tgt_labels: List[str], tgt_uris: List[str],
                               etype: URIRef, src_entities: Dict, tgt_entities: Dict,
                               relaxation_factor: float = 1.0,
                               src_chunk_size: int = 50000) -> Iterator[AlignmentCandidate]:
        cfg = self.config
        raw_cutoff = self.thresholds.get(etype, self.default_thres) * relaxation_factor

        n_tgt = len(tgt_labels)
        tgt_embs = self.get_embeddings_adaptive(list(tgt_labels))
        dim = tgt_embs.shape[1]

        bucket_size = max(len(src_labels), n_tgt)
        ambiguity = self._estimate_ambiguity(tgt_embs)
        k = min(self._get_adaptive_k(relaxation_factor, bucket_size=bucket_size, ambiguity=ambiguity), n_tgt)
        if k <= 0:
            del tgt_embs
            return

        def build_faiss_index_streamed(labels, num_labels, chunk):
            use_pq = num_labels >= cfg.mega_scale_pq_threshold
            if num_labels < 15000:
                idx = faiss.IndexFlatIP(dim)
                for s in range(0, num_labels, chunk):
                    e = min(s + chunk, num_labels)
                    idx.add(self.get_embeddings_adaptive(list(labels[s:e])))
                return idx

            nlist = max(64, min(16384, int(4 * np.sqrt(num_labels))))
            if use_pq:
                m = cfg.pq_m_subquantizers
                while dim % m != 0 and m > 1:
                    m -= 1
                quantizer = faiss.IndexFlatIP(dim)
                idx = faiss.IndexIVFPQ(quantizer, dim, nlist, m, cfg.pq_bits, faiss.METRIC_INNER_PRODUCT)
                train_n = min(num_labels, max(nlist * cfg.ivf_pq_training_ratio, nlist * 40))
            elif num_labels >= (nlist * cfg.ivf_training_ratio):
                idx = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, nlist, faiss.METRIC_INNER_PRODUCT)
                train_n = min(num_labels, nlist * cfg.ivf_training_ratio)
            else:
                idx = faiss.IndexFlatIP(dim)
                for s in range(0, num_labels, chunk):
                    e = min(s + chunk, num_labels)
                    idx.add(self.get_embeddings_adaptive(list(labels[s:e])))
                return idx

            train_idx = np.random.default_rng(0).choice(num_labels, size=train_n, replace=False)
            train_idx.sort()
            train_sample = self.get_embeddings_adaptive([labels[i] for i in train_idx.tolist()])
            idx.train(train_sample)
            idx.nprobe = max(8, min(nlist, nlist // cfg.ivf_nprobe_divisor))
            del train_sample

            for s in range(0, num_labels, chunk):
                e = min(s + chunk, num_labels)
                idx.add(self.get_embeddings_adaptive(list(labels[s:e])))
            return idx

        n_src = len(src_labels)
        chunk_size = max(1000, min(src_chunk_size, n_src))
        index_fwd = build_faiss_index_streamed(list(tgt_labels), n_tgt, chunk_size)

        target_selection_freq = np.zeros(n_tgt, dtype=np.int64)
        all_indices_fwd_chunks, all_scores_fwd_chunks = [], []

        for start in range(0, n_src, chunk_size):
            end = min(start + chunk_size, n_src)
            chunk_embs = self.get_embeddings_adaptive(list(src_labels[start:end]))
            scores_c, indices_c = index_fwd.search(chunk_embs, k=k)
            all_scores_fwd_chunks.append(scores_c)
            all_indices_fwd_chunks.append(indices_c)
            counts = np.bincount(indices_c.flatten(), minlength=n_tgt)
            target_selection_freq += counts
            del chunk_embs, scores_c, indices_c, counts

        scores_fwd = np.concatenate(all_scores_fwd_chunks, axis=0)
        indices_fwd = np.concatenate(all_indices_fwd_chunks, axis=0)
        del all_indices_fwd_chunks, all_scores_fwd_chunks

        expected_baseline = (n_src * k) / max(n_tgt, 1)
        effective_hub_threshold = max(cfg.hub_freq_threshold, expected_baseline * 2.5)

        tgt_indices_arr = np.flatnonzero(target_selection_freq > 0)
        rev_k = min(20 if bucket_size >= cfg.mega_scale_pq_threshold else k, n_src)

        rev_map: Dict[Tuple[int, int], float] = {}
        if len(tgt_indices_arr) > 0 and rev_k > 0:
            index_rev = build_faiss_index_streamed(list(src_labels), n_src, chunk_size)
            rev_scores_full, rev_indices_full = index_rev.search(tgt_embs[tgt_indices_arr], k=rev_k)
            for b_idx, tgt_idx in enumerate(tgt_indices_arr.tolist()):
                row_idx, row_score = rev_indices_full[b_idx], rev_scores_full[b_idx]
                for s_rank in range(len(row_idx)):
                    rev_map[(int(row_idx[s_rank]), int(tgt_idx))] = float(row_score[s_rank])
            del rev_scores_full, rev_indices_full, index_rev

        src_token_sets = [set(s.split()) for s in src_labels]
        tgt_token_sets = [set(t.split()) for t in tgt_labels]
        idf = self._build_idf_weights(src_token_sets, tgt_token_sets)

        for src_idx in range(n_src):
            src_uri, src_lbl, s_toks = src_uris[src_idx], src_labels[src_idx], src_token_sets[src_idx]

            for kth in range(k):
                score_fwd, tgt_idx = float(scores_fwd[src_idx][kth]), int(indices_fwd[src_idx][kth])

                if k > 1 and target_selection_freq[tgt_idx] > effective_hub_threshold:
                    excess = target_selection_freq[tgt_idx] - effective_hub_threshold
                    score_fwd *= max(cfg.hub_penalty_max_discount, 1.0 - (cfg.hub_penalty_step * excess))

                rev_score = rev_map.get((src_idx, tgt_idx), 0.0)

                if score_fwd < raw_cutoff * cfg.cutoff_ratio_floor:
                    continue
                # Precision Filter: Reject asymmetric matches where reciprocal target score drops off heavily
                if score_fwd >= 0.75 and rev_score > 0.0 and rev_score < min(score_fwd - cfg.symmetry_diff_tolerance, raw_cutoff * 0.75):
                    continue

                tgt_lbl, t_toks, target_uri = tgt_labels[tgt_idx], tgt_token_sets[tgt_idx], str(tgt_uris[tgt_idx])

                is_exact = (src_lbl == tgt_lbl) and len(src_lbl) > 2
                isub = self.isub_similarity(src_lbl, tgt_lbl)
                lev = self.levenshtein_similarity(src_lbl, tgt_lbl)
                tsort_sim = self.token_sort_similarity(src_lbl, tgt_lbl)
                w_overlap = self._weighted_jaccard(s_toks, t_toks, idf)
                acronym = self.acronym_bonus(src_lbl, tgt_lbl)

                max_string_sim = max(lev, isub, tsort_sim)

                # Precision Gate: Require either high semantic score OR validated lexical evidence
                if not is_exact and score_fwd < 0.88:
                    if max_string_sim < cfg.general_string_gate and w_overlap < cfg.general_overlap_gate and acronym < 0.8:
                        continue

                structural = self.evaluate_structural_similarity(str(src_uri), target_uri, src_entities, tgt_entities)
                structural += acronym * cfg.structural_bonus

                yield AlignmentCandidate(
                    source=str(src_uri), target=target_uri, etype=str(etype),
                    semantic_score=score_fwd, ngram_score=self.ngram_similarity(src_lbl, tgt_lbl),
                    levenshtein_score=lev, isub_score=isub,
                    structural_bonus=structural,
                    is_exact_match=is_exact,
                    config=self.config
                )

        del index_fwd, tgt_embs, scores_fwd, indices_fwd, rev_map, target_selection_freq
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def align_optimized(self, src_entities: Dict, tgt_entities: Dict) -> Set[Tuple[str, str, str]]:
        final_alignments, claimed_src, claimed_tgt = set(), set(), set()
        self.update_ontology_size(src_entities, tgt_entities)

        print(f" [INFO] {'Mega-scale Optimization Mode ACTIVE' if self.is_mega_scale else 'Standard matching logic active'} (size: {self.ontology_size:.0f} nodes)")

        # Exact Match Pre-Pass: Priority Queue to lock 100% exact entity label/type pairs
        tgt_lookup = defaultdict(list)
        for uri, meta in tgt_entities.items():
            tgt_lookup[(meta["label"], meta["type"])].append(uri)

        src_lookup = defaultdict(list)
        for uri, meta in src_entities.items():
            src_lookup[(meta["label"], meta["type"])].append(uri)

        for key, s_uris in src_lookup.items():
            if len(s_uris) != 1 or len(key[0]) <= 2:
                continue
            t_uris = tgt_lookup.get(key)
            if t_uris and len(t_uris) == 1:
                s_uri, t_uri = s_uris[0], t_uris[0]
                claimed_src.add(s_uri)
                claimed_tgt.add(t_uri)
                final_alignments.add((str(s_uri), str(t_uri), str(key[1]), 1.0))
        del tgt_lookup, src_lookup
        gc.collect()

        types = [OWL.Class, SKOS.Concept, OWL.ObjectProperty, OWL.DatatypeProperty, OWL.NamedIndividual]
        final_alignments, claimed_src, claimed_tgt = self._match_stage(src_entities, tgt_entities, types, claimed_src, claimed_tgt, final_alignments, 1.0)

        if self.ontology_size < 50000:
            final_alignments, claimed_src, claimed_tgt = self._match_stage(src_entities, tgt_entities, types, claimed_src, claimed_tgt, final_alignments, 0.92)

        return final_alignments

    def _match_stage(self, src_entities, tgt_entities, distinct_types, claimed_src, claimed_tgt, alignments, relaxation_factor=1.0):
        filtered_src_keys = set(src_entities.keys()) - claimed_src
        filtered_tgt_keys = set(tgt_entities.keys()) - claimed_tgt
        if not filtered_src_keys or not filtered_tgt_keys:
            return alignments, claimed_src, claimed_tgt

        scale_factor = self._get_threshold_scale()
        stage_candidates = []

        for etype in distinct_types:
            src_sub = [(u, src_entities[u]["label"]) for u in filtered_src_keys if src_entities[u]["type"] == etype]
            tgt_sub = [(u, tgt_entities[u]["label"]) for u in filtered_tgt_keys if tgt_entities[u]["type"] == etype]
            if not src_sub or not tgt_sub:
                continue

            src_uris, src_labels = zip(*src_sub)
            tgt_uris, tgt_labels = zip(*tgt_sub)
            del src_sub, tgt_sub
            required_cutoff = self.thresholds.get(etype, self.default_thres) * scale_factor * relaxation_factor

            type_candidates = [
                c for c in self.semantic_match_chunked(
                    list(src_labels), list(src_uris), list(tgt_labels), list(tgt_uris),
                    etype, src_entities, tgt_entities, relaxation_factor)
                if c.combined_score >= required_cutoff
            ]
            stage_candidates.extend(type_candidates)
            del src_uris, src_labels, tgt_uris, tgt_labels, type_candidates
            gc.collect()

        del filtered_src_keys, filtered_tgt_keys

        stage_candidates.sort(key=lambda x: x.combined_score, reverse=True)
        for c in stage_candidates:
            if c.source not in claimed_src and c.target not in claimed_tgt:
                claimed_src.add(c.source)
                claimed_tgt.add(c.target)
                alignments.add((c.source, c.target, c.etype, round(c.combined_score, 4)))
        del stage_candidates
        gc.collect()
        return alignments, claimed_src, claimed_tgt

    def convert_to_rdf_triples(self, alignments: Set[Tuple]) -> Set[Tuple[str, str, str]]:
        return {
            (src, "http://www.w3.org/2002/07/owl#equivalentClass" if "Class" in etype else
                  "http://www.w3.org/2002/07/owl#equivalentProperty" if "Property" in etype else
                  "http://www.w3.org/2002/07/owl#sameAs", tgt)
            for src, tgt, etype, *_score in alignments
        }

class OAEITrackRunner:
    def __init__(self, matcher: MOSAIC, results_dir: Path = None):
        self.matcher, self.results_dir, self.log = matcher, results_dir or RESULTS_DIR, []

    def load_reference_alignments(self, path: Path) -> Set[Tuple[str, str, str]]:
        if path.suffix.lower() in (".tsv", ".csv"):
            ref_set = self._load_reference_alignments_tsv(path)
            if ref_set:
                return ref_set
        ref_set = self._load_reference_alignments_ntriples(path)
        if ref_set:
            return ref_set
        return self._load_reference_alignments_oaei_xml(path)

    def _load_reference_alignments_tsv(self, path: Path) -> Set[Tuple[str, str, str]]:
        ref_set = set()
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        try:
            with open(path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.reader(f, delimiter=delimiter)
                rows = list(reader)
            if not rows:
                return ref_set

            header = [h.strip().lower() for h in rows[0]]
            has_header = any(h in ("srcentity", "tgtentity", "src", "tgt", "source", "target", "score", "relation") for h in header)

            if has_header:
                col_idx = {name: i for i, name in enumerate(header)}
                src_i = col_idx.get("srcentity", col_idx.get("src", col_idx.get("source", 0)))
                tgt_i = col_idx.get("tgtentity", col_idx.get("tgt", col_idx.get("target", 1)))
                rel_i = col_idx.get("relation")
                data_rows = rows[1:]
            else:
                src_i, tgt_i, rel_i = 0, 1, None
                data_rows = rows

            RELATION_TO_PREDICATE = {
                "=": "http://www.w3.org/2002/07/owl#equivalentClass",
                "equivalentclass": "http://www.w3.org/2002/07/owl#equivalentClass",
                "equivalentproperty": "http://www.w3.org/2002/07/owl#equivalentProperty",
                "sameas": "http://www.w3.org/2002/07/owl#sameAs",
            }

            for row in data_rows:
                if not row or len(row) <= max(src_i, tgt_i):
                    continue
                s, o = row[src_i].strip(), row[tgt_i].strip()
                if not s or not o:
                    continue
                predicate = "http://www.w3.org/2002/07/owl#equivalentClass"
                if rel_i is not None and rel_i < len(row):
                    predicate = RELATION_TO_PREDICATE.get(row[rel_i].strip().lower(), predicate)
                nodes = tuple(sorted([s, o]))
                ref_set.add((nodes[0], predicate, nodes[1]))
        except Exception as e:
            print(f" [ERROR] Could not read reference (TSV): {e}")
        return ref_set

    def _load_reference_alignments_ntriples(self, path: Path) -> Set[Tuple[str, str, str]]:
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
        except Exception as e:
            print(f" [ERROR] Could not read reference (N-Triples): {e}")
        return ref_set

    def _load_reference_alignments_oaei_xml(self, path: Path) -> Set[Tuple[str, str, str]]:
        ref_set = set()
        RELATION_TO_PREDICATE = {
            "=": "http://www.w3.org/2002/07/owl#equivalentClass",
            "equivalentClass": "http://www.w3.org/2002/07/owl#equivalentClass",
            "equivalentProperty": "http://www.w3.org/2002/07/owl#equivalentProperty",
            "sameAs": "http://www.w3.org/2002/07/owl#sameAs",
        }
        try:
            g = Graph()
            parsed = False
            for fmt in ("xml", "turtle"):
                try:
                    g.parse(str(path), format=fmt)
                    parsed = True
                    break
                except Exception:
                    continue
            if not parsed:
                return ref_set

            ALIGN_NS = "http://knowledgeweb.semanticweb.org/heterogeneity/alignment"
            cell_type = URIRef(f"{ALIGN_NS}Cell")
            entity1_pred = URIRef(f"{ALIGN_NS}entity1")
            entity2_pred = URIRef(f"{ALIGN_NS}entity2")
            relation_pred = URIRef(f"{ALIGN_NS}relation")

            cells = set(g.subjects(RDF.type, cell_type))
            if not cells:
                cells = set(g.subjects(entity1_pred, None)) & set(g.subjects(entity2_pred, None))

            for cell in cells:
                e1 = g.value(cell, entity1_pred)
                e2 = g.value(cell, entity2_pred)
                if e1 is None or e2 is None:
                    continue
                relation = g.value(cell, relation_pred)
                relation_str = str(relation).strip() if relation is not None else "="
                predicate = RELATION_TO_PREDICATE.get(relation_str, RELATION_TO_PREDICATE["="])

                s, o = str(e1), str(e2)
                nodes = tuple(sorted([s, o]))
                ref_set.add((nodes[0], predicate, nodes[1]))
        except Exception as e:
            print(f" [ERROR] Could not read reference (OAEI XML): {e}")
        return ref_set

    def serialize_alignments_to_ttl(self, alignments: Set[Tuple[str, str, str]], path: Path) -> bool:
        g = Graph()
        for src, pred, tgt in alignments:
            try:
                g.add((URIRef(src), URIRef(pred), URIRef(tgt)))
            except Exception:
                pass
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            g.serialize(destination=str(path), format="turtle")
            return True
        except Exception:
            return False

    @staticmethod
    def _xml_escape(s: str) -> str:
        return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                 .replace('"', "&quot;"))

    def serialize_alignments_to_bioml_rdf(self, alignments: Set[Tuple], path: Path) -> bool:
        ALIGN_NS = "http://knowledgeweb.semanticweb.org/heterogeneity/alignment"
        lines = [
            '<?xml version="1.0" encoding="utf-8"?>',
            '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"',
            '         xmlns:xsd="http://www.w3.org/2001/XMLSchema#"',
            f'         xmlns:align="{ALIGN_NS}#">',
            '<align:Alignment>',
            '  <align:xml>yes</align:xml>',
            '  <align:level>0</align:level>',
            '  <align:type>**</align:type>',
            '  <align:onto1>source</align:onto1>',
            '  <align:onto2>target</align:onto2>',
        ]
        for i, entry in enumerate(alignments):
            src, tgt, etype, score = (entry + (1.0,))[:4] if len(entry) < 4 else entry
            lines.append(f'  <align:map>')
            lines.append(f'    <align:Cell rdf:about="#cell{i}">')
            lines.append(f'      <align:entity1 rdf:resource="{self._xml_escape(str(src))}"/>')
            lines.append(f'      <align:entity2 rdf:resource="{self._xml_escape(str(tgt))}"/>')
            lines.append('      <align:relation>=</align:relation>')
            lines.append(f'      <align:measure rdf:datatype="http://www.w3.org/2001/XMLSchema#float">{float(score):.4f}</align:measure>')
            lines.append('    </align:Cell>')
            lines.append('  </align:map>')
        lines += ['</align:Alignment>', '</rdf:RDF>']
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(lines), encoding="utf-8")
            return True
        except Exception:
            return False

    def serialize_alignments_to_bioml_tsv(self, alignments: Set[Tuple], path: Path) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(["SrcEntity", "TgtEntity", "Relation", "Score"])
                for entry in alignments:
                    src, tgt, etype, score = (entry + (1.0,))[:4] if len(entry) < 4 else entry
                    writer.writerow([str(src), str(tgt), "=", f"{float(score):.4f}"])
            return True
        except Exception:
            return False

    def calculate_metrics(self, sys_align, ref_align) -> Optional[Tuple[float, float, float, int, int]]:
        if not ref_align:
            return None
        ref_pairs = {tuple(sorted([str(s), str(o)])) for s, p, o in ref_align}
        sys_pairs = {tuple(sorted([str(s), str(o)])) for s, p, o in sys_align}

        tp = len(sys_pairs.intersection(ref_pairs))
        p = tp / len(sys_pairs) if sys_pairs else 0.0
        r = tp / len(ref_pairs) if ref_pairs else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        return round(p, 4), round(r, 4), round(f1, 4), tp, len(ref_pairs)

    def find_ontology_file(self, folder: Path, name: str) -> Optional[Path]:
        for ext in [".owl", ".rdf", ".ttl", ".xml"]:
            if (folder / f"{name}{ext}").exists():
                return folder / f"{name}{ext}"
        return None

    def find_tsv_reference_file(self, track: Path, task_stem: str) -> Optional[Path]:
        parts = task_stem.split("-", 1)
        reversed_stem = f"{parts[1]}-{parts[0]}" if len(parts) == 2 else None

        name_variants = [task_stem] + ([reversed_stem] if reversed_stem else [])
        search_dirs = [track, track / "ontologies", track / "refs_equiv", track / "refs",
                       track / "references", track / "reference"]

        candidates = []
        for name in name_variants:
            for ext in (".tsv", ".csv"):
                for d in search_dirs:
                    candidates.append(d / f"{name}{ext}")

        for d in search_dirs:
            for fname in ("train.tsv", "test.tsv", "full.tsv", "reference.tsv"):
                candidates.append(d / fname)

        for c in candidates:
            if c.exists() and c.is_file():
                return c
        return None

    def _run_single_task(self, track_name: str, task_name: str, src_p: Path, tgt_p: Path, ref_p: Optional[Path]):
        if ref_p and ref_p.exists():
            print(f"\n [TASK] Loading reference alignments: {task_name}")
            ref_align = self.load_reference_alignments(ref_p)
        else:
            print(f"\n [TASK] Skipping reference score check (No reference file found for {task_name})")
            ref_align = None

        task_t0 = time.time()
        t_ext0 = time.time()

        src_entities = {}
        tgt_entities = {}
        for u, l, t, par in self.matcher.extract_entities_streaming(src_p):
            src_entities[str(u)] = EntityMeta(l, t, par)
        for u, l, t, par in self.matcher.extract_entities_streaming(tgt_p):
            tgt_entities[str(u)] = EntityMeta(l, t, par)

        approx_size = (len(src_entities) + len(tgt_entities)) / 2
        if approx_size >= 130000:
            for meta in src_entities.values():
                meta.parents = None
            for meta in tgt_entities.values():
                meta.parents = None
            gc.collect()
        total_entities = len(src_entities) + len(tgt_entities)
        print(f" [TASK] Extracted {len(src_entities)} source and {len(tgt_entities)} target entities in {round(time.time() - t_ext0, 2)}s")

        self.matcher.update_ontology_size(src_entities, tgt_entities)

        is_med, reason = MedicalDomainDetector.evaluate_is_medical(track_name, task_name, src_entities, tgt_entities)
        print(f" [DOMAIN DETECTOR] Result: {'MEDICAL' if is_med else 'GENERAL'} | Reason: {reason}")

        self.matcher.apply_domain_model(is_medical=is_med, total_entities=total_entities)
        gc.collect()

        t0 = time.time()
        alignments = self.matcher.align_optimized(src_entities, tgt_entities)
        dt = round(time.time() - t0, 2)

        out_rdf = self.results_dir / f"{task_name}.rdf"
        out_tsv = self.results_dir / f"{task_name}.tsv"
        out_ttl = self.results_dir / f"mosaic_{track_name}_{task_name}.ttl"
        rdf_triples = self.matcher.convert_to_rdf_triples(alignments)
        self.serialize_alignments_to_ttl(rdf_triples, out_ttl)
        self.serialize_alignments_to_bioml_rdf(alignments, out_rdf)
        self.serialize_alignments_to_bioml_tsv(alignments, out_tsv)

        metrics = self.calculate_metrics(rdf_triples, ref_align) if ref_align else None
        total_task_time = round(time.time() - task_t0, 2)
        if metrics:
            p, r, f1, tp, total = metrics
            print(f"         Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f} | Correct: {tp}/{total} | Match Time: {dt}s | Total Time: {total_task_time}s")
            self.log.append({
                "Track": track_name, "Task": task_name, "Precision": p, "Recall": r, "F1-Score": f1,
                "Time (s)": dt, "Total Time (s)": total_task_time, "Alignments": len(alignments),
                "Correct": tp, "Reference": total, "Type": "Task"
            })
        else:
            print(f"         Alignments Generated: {len(alignments)} | Match Time: {dt}s | Total Time: {total_task_time}s (Reference Evaluation Skipped)")
            self.log.append({
                "Track": track_name, "Task": task_name, "Precision": "N/A", "Recall": "N/A", "F1-Score": "N/A",
                "Time (s)": dt, "Total Time (s)": total_task_time, "Alignments": len(alignments),
                "Correct": "N/A", "Reference": "N/A", "Type": "Task"
            })

        del src_entities, tgt_entities, alignments, rdf_triples
        MOSAIC.isub_similarity.cache_clear()
        MOSAIC.ngram_similarity.cache_clear()
        MOSAIC.levenshtein_similarity.cache_clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run_all_tracks(self, base_dir: str, csv_out: str = "mosaic_report.csv"):
        base_path = Path(base_dir)
        if not base_path.exists():
            return

        for track in sorted(base_path.iterdir()):
            if not track.is_dir():
                continue

            ran_legacy_task = False
            candidate_exts = (".rdf", ".owl", ".ttl", ".xml")

            for tf in sorted(track.glob("*.ttl")):
                parts = tf.stem.split("-")
                if len(parts) != 2:
                    if "human-mouse" in tf.stem:
                        parts = ["human", "mouse"]
                    else:
                        continue

                src_p, tgt_p = self.find_ontology_file(track / "ontologies", parts[0]), self.find_ontology_file(track / "ontologies", parts[1])
                if not src_p or not tgt_p:
                    continue

                ran_legacy_task = True
                self._run_single_task(track.name, tf.stem, src_p, tgt_p, tf)

            if not ran_legacy_task:
                sibling_stems = {f.stem for f in track.iterdir() if f.is_file() and f.suffix.lower() in candidate_exts}

                for rf in sorted(track.iterdir()):
                    if not rf.is_file() or rf.suffix.lower() not in candidate_exts:
                        continue
                    parts = rf.stem.split("-", 1)
                    if len(parts) != 2:
                        continue
                    name1, name2 = parts
                    if name1 not in sibling_stems or name2 not in sibling_stems:
                        continue
                    if name1 == rf.stem or name2 == rf.stem:
                        continue

                    src_p, tgt_p = self.find_ontology_file(track, name1), self.find_ontology_file(track, name2)
                    if not src_p or not tgt_p or src_p == rf or tgt_p == rf:
                        continue

                    ran_legacy_task = True
                    self._run_single_task(track.name, rf.stem, src_p, tgt_p, rf)

            if not ran_legacy_task:
                for bio_task in BIO_ML_ALIGNMENT_TASKS:
                    task_stem = Path(bio_task).stem
                    parts = task_stem.split("-")
                    if len(parts) == 2:
                        src_p, tgt_p = self.find_ontology_file(track, parts[0]), self.find_ontology_file(track, parts[1])
                        if src_p and tgt_p:
                            ran_legacy_task = True
                            ref_p = self.find_tsv_reference_file(track, task_stem)
                            self._run_single_task(track.name, task_stem, src_p, tgt_p, ref_p)

            if not ran_legacy_task:
                onto_files = sorted(f for f in track.iterdir() if f.is_file() and f.suffix.lower() in candidate_exts)
                if len(onto_files) == 2:
                    src_p, tgt_p = onto_files
                    task_stem = f"{src_p.stem}-{tgt_p.stem}"
                    ref_p = self.find_tsv_reference_file(track, task_stem)
                    self._run_single_task(track.name, task_stem, src_p, tgt_p, ref_p)

        self.results_to_csv(csv_out)
        if self.matcher.embedding_cache:
            self.matcher.embedding_cache.save_index()

    def results_to_csv(self, filename: str):
        fields = ["Track", "Task", "Precision", "Recall", "F1-Score", "Time (s)", "Total Time (s)",
                   "Alignments", "Correct", "Reference", "Type"]
        with open(self.results_dir / filename, mode="w", newline="", encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.log)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" [INFO] Running on execution device: {device}")

    custom_config = MOSAICConfig()
    m = MOSAIC(config=custom_config, device=device)
    runner = OAEITrackRunner(matcher=m, results_dir=RESULTS_DIR)
    runner.run_all_tracks(str(BASE_DIR), csv_out="mosaic_report.csv")