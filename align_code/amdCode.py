# interpreter path used for reference: /home/linuxbrew/.linuxbrew/bin/python3.12
import csv
import gc
import os
import re
import time
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import logging

import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer

from rdflib import Graph, RDF, RDFS, URIRef
from rdflib.namespace import OWL, SKOS

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("rdflib").setLevel(logging.ERROR)

_CORE_NAMESPACES = (str(RDF), str(RDFS), str(OWL), str(SKOS))


class MOSAICGPUOptimized:
    """
    GPU-accelerated version for massive ontologies (100K+ entities).
    
    Key features for massive scale:
    - GPU embedding with larger batch sizes
    - GPU-accelerated FAISS (IndexGPU)
    - Chunked processing to manage GPU memory
    - Streaming with explicit garbage collection
    - Progress tracking for long-running jobs
    """
    
    _CAMEL_RE = re.compile(r'([a-z])([A-Z])')
    _PCT_RE = re.compile(r'%[0-9A-Fa-f]{2}')
    DEFAULT_MODEL = "sentence-transformers/all-minilm-l12-v2"

    def __init__(self, model_name=None, thresholds=None, max_cache_labels=50_000,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 gpu_batch_size=512, use_gpu_faiss=True):
        
        self.thresholds = thresholds or {
            OWL.Class: 0.80, SKOS.Concept: 0.80,
            OWL.ObjectProperty: 0.88, OWL.DatatypeProperty: 0.88,
            OWL.NamedIndividual: 0.82
        }
        self.default_thres = 0.80
        self.model = None
        self.model_name = model_name or self.DEFAULT_MODEL
        self.max_cache_labels = max_cache_labels
        self.string_emb_cache = {}
        self.device = device
        self.gpu_batch_size = gpu_batch_size
        self.use_gpu_faiss = use_gpu_faiss and device == "cuda"
        
        # GPU memory management
        self.gpu_available_gb = self._get_gpu_memory() if device == "cuda" else 0
        
        cores = min(os.cpu_count() or 1, 8)
        if hasattr(os, 'sched_getaffinity'):
            try:
                cores = len(os.sched_getaffinity(0))
            except:
                pass
        torch.set_num_threads(max(2, cores))
        
        print(f" [MOSAIC-GPU] Using device: {device}")
        if device == "cuda":
            print(f" [MOSAIC-GPU] GPU memory: {self.gpu_available_gb:.1f} GB available")

    def _get_gpu_memory(self) -> float:
        """Get available GPU memory in GB."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory / 1e9
        except:
            return 0
        return 0

    def init_model(self):
        """Lazy initialization on specified device."""
        if self.model is not None:
            return
        
        print(f" [MOSAIC-GPU] Initializing: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        try:
            self.model.max_seq_length = 64
        except:
            pass

    def normalise_label(self, text: str) -> str:
        """Normalize label with minimal allocation."""
        if not text:
            return ""
        text = self._CAMEL_RE.sub(r'\1 \2', str(text))
        text = text.replace('_', ' ').replace('-', ' ').lower().strip()
        words = text.split()
        return " ".join(words[:15])

    def _get_label_streaming(self, uri, graph) -> str:
        """Stream label extraction."""
        target_langs = ['en', 'de', 'fr']
        
        for lang in target_langs:
            for label in graph.objects(uri, SKOS.prefLabel):
                if hasattr(label, 'language') and label.language == lang:
                    return self.normalise_label(label)
            for label in graph.objects(uri, RDFS.label):
                if hasattr(label, 'language') and label.language == lang:
                    return self.normalise_label(label)

        label = graph.value(uri, RDFS.label) or graph.value(uri, SKOS.prefLabel)
        if label:
            return self.normalise_label(label)

        frag = str(uri).split('/')[-1].split('#')[-1]
        frag = self._PCT_RE.sub(' ', frag)
        return self.normalise_label(frag)

    def load_ontology(self, path: Path) -> Graph:
        """Load ontology with format detection."""
        g = Graph()
        formats = ["turtle", "xml"] if path.suffix == ".ttl" else ["xml", "turtle"]
        
        for fmt in formats:
            try:
                g.parse(str(path), format=fmt)
                return g
            except:
                continue
        
        try:
            g.parse(str(path))
            return g
        except Exception as e:
            print(f" [ERROR] Failed to load {path.name}: {e}")
            return None

    def extract_entities_streaming(self, graph: Graph, max_entities=None):
        """Stream entities without full materialization."""
        if graph is None:
            return
        
        skos_concepts = set(graph.subjects(RDF.type, SKOS.Concept))
        owl_classes = set(graph.subjects(RDF.type, OWL.Class)) - skos_concepts
        props = (set(graph.subjects(RDF.type, OWL.ObjectProperty)) |
                 set(graph.subjects(RDF.type, OWL.DatatypeProperty)))
        insts = set(graph.subjects()) - skos_concepts - owl_classes - props

        def is_valid(uri):
            if not isinstance(uri, URIRef):
                return False
            s = str(uri)
            return ("oboInOwl" not in s and not s.startswith(_CORE_NAMESPACES))

        count = 0
        for uri in filter(is_valid, owl_classes):
            if max_entities and count >= max_entities:
                break
            lbl = self._get_label_streaming(uri, graph)
            if lbl:
                yield uri, lbl, OWL.Class
                count += 1

        for uri in filter(is_valid, skos_concepts):
            if max_entities and count >= max_entities:
                break
            lbl = self._get_label_streaming(uri, graph)
            if lbl:
                yield uri, lbl, SKOS.Concept
                count += 1

        for uri in filter(is_valid, props):
            if max_entities and count >= max_entities:
                break
            lbl = self._get_label_streaming(uri, graph)
            if lbl:
                yield uri, lbl, OWL.ObjectProperty
                count += 1

        for uri in filter(is_valid, insts):
            if max_entities and count >= max_entities:
                break
            lbl = self._get_label_streaming(uri, graph)
            if lbl:
                yield uri, lbl, OWL.NamedIndividual
                count += 1

    def materialize_entities(self, graph: Graph, max_entities=None) -> dict:
        """Materialize entities from streaming extraction."""
        entities = {}
        for uri, lbl, etype in self.extract_entities_streaming(graph, max_entities):
            entities[uri] = {"label": lbl, "type": etype}
        return entities

    def get_embeddings_batch_gpu(self, labels: list, batch_size=None) -> torch.Tensor:
        """GPU-accelerated batch embedding with larger batches."""
        if not labels:
            return torch.tensor([], dtype=torch.float32).to(self.device)
        
        if batch_size is None:
            batch_size = self.gpu_batch_size
        
        self.init_model()
        unique_labels = list(dict.fromkeys(labels))
        
        # Embed missing labels
        to_embed = [lbl for lbl in unique_labels if lbl not in self.string_emb_cache]
        
        if to_embed:
            print(f"  [GPU] Embedding {len(to_embed)} unique labels (batch_size={batch_size})...")
            
            with torch.inference_mode():
                # GPU encoding with larger batches
                embeddings = self.model.encode(
                    to_embed,
                    convert_to_tensor=True,
                    show_progress_bar=True,
                    batch_size=batch_size,
                    device=self.device
                )
            
            # Store on GPU if enough VRAM, otherwise CPU
            for lbl, emb in zip(to_embed, embeddings):
                if len(self.string_emb_cache) < self.max_cache_labels:
                    self.string_emb_cache[lbl] = emb.detach()
            
            del embeddings
        
        # Gather results
        result = []
        for lbl in labels:
            if lbl in self.string_emb_cache:
                result.append(self.string_emb_cache[lbl])
            else:
                # Fallback: compute on-the-fly
                with torch.inference_mode():
                    emb = self.model.encode(lbl, convert_to_tensor=True, 
                                           device=self.device)
                result.append(emb.detach())
        
        if result:
            embeddings = torch.stack(result)
            if self.device == "cpu":
                return embeddings
            return embeddings.to(self.device)
        
        return torch.tensor([], dtype=torch.float32).to(self.device)

    def semantic_match_type_gpu(self, src_ents: dict, tgt_ents: dict, etype, k=1):
        """GPU-accelerated matching with FAISS GPU index."""
        src_subset = [(uri, meta) for uri, meta in src_ents.items() 
                      if meta["type"] == etype]
        tgt_subset = [(uri, meta) for uri, meta in tgt_ents.items() 
                      if meta["type"] == etype]
        
        if not src_subset or not tgt_subset:
            return []

        threshold = self.thresholds.get(etype, self.default_thres)
        
        src_uris, src_labels = zip(*[(u, m["label"]) for u, m in src_subset])
        tgt_uris, tgt_labels = zip(*[(u, m["label"]) for u, m in tgt_subset])

        # GPU embedding
        print(f"  [GPU] Embedding {len(src_labels)} source + {len(tgt_labels)} target labels...")
        with torch.inference_mode():
            src_emb = self.get_embeddings_batch_gpu(list(src_labels))
            tgt_emb = self.get_embeddings_batch_gpu(list(tgt_labels))

        if len(src_emb) == 0 or len(tgt_emb) == 0:
            return []

        # Convert to CPU for FAISS (or use GPU FAISS if available)
        tgt_matrix = tgt_emb.cpu().contiguous().numpy().astype('float32')
        dim = tgt_matrix.shape[1]
        
        print(f"  [GPU] Building FAISS index for {len(tgt_matrix)} target entities...")
        
        # GPU-accelerated FAISS index
        if self.use_gpu_faiss:
            try:
                import faiss.gpu as gpu_faiss
                res = faiss.StandardGpuResources()
                index_cpu = faiss.IndexFlatIP(dim)
                index = gpu_faiss.index_cpu_to_gpu(res, 0, index_cpu)
                index.add(tgt_matrix)
                use_gpu = True
            except Exception as e:
                print(f"  [GPU] GPU FAISS unavailable ({e}), using CPU index")
                index = faiss.IndexFlatIP(dim)
                index.add(tgt_matrix)
                use_gpu = False
        else:
            index = faiss.IndexFlatIP(dim)
            index.add(tgt_matrix)
            use_gpu = False

        # Search in batches for large src
        src_matrix = src_emb.cpu().contiguous().numpy().astype('float32')
        batch_search_size = 10_000
        
        candidates = []
        print(f"  [GPU] Searching {len(src_matrix)} source entities against index...")
        
        for start_idx in range(0, len(src_matrix), batch_search_size):
            end_idx = min(start_idx + batch_search_size, len(src_matrix))
            batch = src_matrix[start_idx:end_idx]
            
            scores, indices = index.search(batch, k)
            
            for local_i in range(len(batch)):
                i = start_idx + local_i
                score = float(scores[local_i][0])
                
                if score < threshold:
                    continue
                
                s_uri = src_uris[i]
                best_idx = int(indices[local_i][0])
                t_uri = tgt_uris[best_idx]
                t_lbl = tgt_labels[best_idx]
                s_lbl = src_labels[i]
                
                # Length heuristic
                len_diff = abs(len(s_lbl) - len(t_lbl))
                if len_diff > max(len(s_lbl), len(t_lbl)) * 0.6:
                    continue
                
                candidates.append({
                    "source": s_uri,
                    "target": t_uri,
                    "type": etype,
                    "combined_score": score
                })
            
            if (end_idx % 50_000) == 0 or end_idx == len(src_matrix):
                print(f"    Processed {end_idx}/{len(src_matrix)} source entities")

        if use_gpu:
            del index  # Free GPU memory
            torch.cuda.empty_cache()
        
        del src_emb, tgt_emb, index
        gc.collect()
        
        return candidates

    def align_optimized(self, src_ents: dict, tgt_ents: dict,
                       preferred_skos_pred="http://www.w3.org/2002/07/owl#sameAs"):
        """Two-phase GPU-optimized alignment."""
        final_pool = []
        claimed_src = set()
        claimed_tgt = set()

        print(" [Phase 1] Exact string matching...")
        # Phase 1: Exact matching
        tgt_lookup = {meta["label"]: uri for uri, meta in tgt_ents.items() 
                      if meta["label"]}
        
        for s_uri, s_meta in src_ents.items():
            s_lbl = s_meta["label"]
            if s_lbl in tgt_lookup:
                t_uri = tgt_lookup[s_lbl]
                if s_meta["type"] == tgt_ents[t_uri]["type"]:
                    claimed_src.add(s_uri)
                    claimed_tgt.add(t_uri)
                    final_pool.append({
                        "source": s_uri,
                        "target": t_uri,
                        "type": s_meta["type"],
                        "combined_score": 1.0
                    })
        
        print(f"  Exact matches: {len(final_pool)} ({100*len(final_pool)/len(src_ents):.1f}% of source)")
        
        del tgt_lookup
        gc.collect()

        # Phase 2: Semantic matching on remainder
        filtered_src = {k: v for k, v in src_ents.items() if k not in claimed_src}
        filtered_tgt = {k: v for k, v in tgt_ents.items() if k not in claimed_tgt}

        if filtered_src and filtered_tgt:
            print(f" [Phase 2] Semantic matching on {len(filtered_src)} src × {len(filtered_tgt)} tgt...")
            
            candidates = []
            distinct_types = [OWL.Class, SKOS.Concept, OWL.ObjectProperty,
                            OWL.DatatypeProperty, OWL.NamedIndividual]
            
            for etype in distinct_types:
                print(f"  Processing {etype}...")
                candidates.extend(self.semantic_match_type_gpu(
                    filtered_src, filtered_tgt, etype, k=1
                ))
            
            candidates.sort(key=lambda x: x["combined_score"], reverse=True)
            
            for c in candidates:
                if c["source"] not in claimed_src and c["target"] not in claimed_tgt:
                    claimed_src.add(c["source"])
                    claimed_tgt.add(c["target"])
                    final_pool.append(c)

        # Generate RDF triples
        alignments = set()
        for c in final_pool:
            s_uri, t_uri, etype = c["source"], c["target"], c["type"]
            if etype == OWL.Class:
                alignments.add((str(s_uri), "http://www.w3.org/2002/07/owl#equivalentClass", str(t_uri)))
            elif etype == SKOS.Concept:
                alignments.add((str(s_uri), preferred_skos_pred, str(t_uri)))
            elif etype in [OWL.ObjectProperty, OWL.DatatypeProperty]:
                alignments.add((str(s_uri), "http://www.w3.org/2002/07/owl#equivalentProperty", str(t_uri)))
            else:
                alignments.add((str(s_uri), "http://www.w3.org/2002/07/owl#sameAs", str(t_uri)))

        return alignments


class OAEITrackRunnerGPU:
    """GPU-optimized runner for large-scale evaluations."""
    
    def __init__(self, matcher: MOSAICGPUOptimized):
        self.matcher = matcher
        self.log = []

    def load_reference_alignments(self, path: Path) -> set:
        ref_set = set()
        g = Graph()
        try:
            g.parse(str(path), format="turtle")
            valid_preds = {
                "http://www.w3.org/2002/07/owl#equivalentClass",
                "http://www.w3.org/2000/01/rdf-schema#subClassOf",
                "http://www.w3.org/2002/07/owl#equivalentProperty",
                "http://www.w3.org/2000/01/rdf-schema#subPropertyOf",
                "http://www.w3.org/2002/07/owl#sameAs",
            }
            for s, p, o in g:
                if str(p) in valid_preds:
                    nodes = sorted([str(s), str(o)])
                    ref_set.add((nodes[0], str(p), nodes[1]))
        except Exception as e:
            print(f" [ERROR] Could not read reference: {e}")
        return ref_set

    def serialize_alignments_to_ttl(self, alignments: set, path: Path):
        g = Graph()
        for src, pred, tgt in alignments:
            g.add((URIRef(src), URIRef(pred), URIRef(tgt)))
        try:
            g.serialize(destination=str(path), format="turtle")
            print(f"   Output: {path.parent.name}/{path.name}")
        except Exception as e:
            print(f"   Serialization error: {e}")

    def calculate_metrics(self, sys_align, ref_align):
        if not ref_align:
            return 0.0, 0.0, 0.0
        
        sys_canon = set()
        for s, p, o in sys_align:
            nodes = sorted([str(s), str(o)])
            sys_canon.add((nodes[0], str(p), nodes[1]))

        tp = len(sys_canon.intersection(ref_align))
        p = tp / len(sys_canon) if sys_canon else 0.0
        r = tp / len(ref_align) if ref_align else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        
        return round(p, 4), round(r, 4), round(f1, 4)

    def find_ontology_file(self, folder: Path, name: str) -> Path:
        for ext in [".owl", ".rdf", ".ttl", ".xml"]:
            p = folder / f"{name}{ext}"
            if p.exists():
                return p
        return None

    def run_all_tracks(self, base_dir: str, csv_out: str = "report_gpu.csv"):
        start_global = time.time()
        base_path = Path(base_dir)
        res_dir = Path("../results")
        res_dir.mkdir(parents=True, exist_ok=True)

        if not base_path.exists():
            print(f"Error: Base directory not found: {base_dir}")
            return

        for track in sorted(base_path.iterdir()):
            if not track.is_dir():
                continue

            print(f"\n{'='*60}\n TRACK: {track.name.upper()}\n{'='*60}")

            tasks = sorted(track.glob("*.ttl"))
            metrics = []

            for tf in tasks:
                parts = tf.stem.split("-")
                if len(parts) != 2:
                    if "human-mouse" in tf.stem:
                        parts = ["human", "mouse"]
                    else:
                        continue

                ont_folder = track / "ontologies"
                src_p = self.find_ontology_file(ont_folder, parts[0])
                tgt_p = self.find_ontology_file(ont_folder, parts[1])

                print(f"\nTask: {parts[0]} → {parts[1]}")

                if not src_p or not tgt_p:
                    print(" Skipping (missing files).")
                    continue

                ref_align = self.load_reference_alignments(tf)
                
                # Parallel load
                print(" Loading ontologies...")
                with ThreadPoolExecutor(max_workers=2) as executor:
                    src_g = executor.submit(self.matcher.load_ontology, src_p).result()
                    tgt_g = executor.submit(self.matcher.load_ontology, tgt_p).result()

                if not src_g or not tgt_g:
                    print(" Skipping (load failed).")
                    continue

                t0 = time.time()
                src_ents = self.matcher.materialize_entities(src_g)
                tgt_ents = self.matcher.materialize_entities(tgt_g)
                del src_g, tgt_g
                gc.collect()
                extract_time = time.time() - t0

                print(f" Extracted: {len(src_ents)} src / {len(tgt_ents)} tgt entities ({extract_time:.1f}s)")

                t0 = time.time()
                alignments = self.matcher.align_optimized(src_ents, tgt_ents)
                dt = round(time.time() - t0, 2)

                print(f" Matched: {len(alignments)} pairs in {dt}s")

                out_ttl = res_dir / f"mosaic_{track.name}_{tf.name}"
                self.serialize_alignments_to_ttl(alignments, out_ttl)

                p, r, f1 = self.calculate_metrics(alignments, ref_align)
                print(f" Metrics → P: {p}, R: {r}, F1: {f1}")

                metrics.append((p, r, f1, dt))
                self.log.append({
                    "Track": track.name, "Task": tf.stem,
                    "Precision": p, "Recall": r, "F1-Score": f1,
                    "Time (s)": dt, "Type": "Task"
                })

                del src_ents, tgt_ents, alignments
                gc.collect()

            if metrics:
                avg_p = round(sum(m[0] for m in metrics) / len(metrics), 4)
                avg_r = round(sum(m[1] for m in metrics) / len(metrics), 4)
                avg_f1 = round(sum(m[2] for m in metrics) / len(metrics), 4)
                avg_t = round(sum(m[3] for m in metrics) / len(metrics), 2)

                print(f"\nTrack avg → P: {avg_p}, R: {avg_r}, F1: {avg_f1} | Time: {avg_t}s")
                
                self.log.append({
                    "Track": track.name, "Task": "AVERAGE",
                    "Precision": avg_p, "Recall": avg_r, "F1-Score": avg_f1,
                    "Time (s)": avg_t, "Type": "Average"
                })

            self.matcher.string_emb_cache.clear()
            if self.matcher.device == "cuda":
                torch.cuda.empty_cache()

        total_runtime = round(time.time() - start_global, 2)
        print(f"\n{'='*60}\nTotal time: {total_runtime}s\n{'='*60}")

        self.log.append({
            "Track": "ALL", "Task": "TOTAL",
            "Precision": "", "Recall": "", "F1-Score": "",
            "Time (s)": total_runtime, "Type": "Summary"
        })

        self.results_to_csv(csv_out)

    def results_to_csv(self, filename: str):
        fields = ["Track", "Task", "Precision", "Recall", "F1-Score", "Time (s)", "Type"]
        with open(filename, mode="w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerows(self.log)
        print(f"Results: {filename}")


if __name__ == "__main__":
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    thresholds = {
        OWL.Class: 0.8, SKOS.Concept: 0.92,
        OWL.ObjectProperty: 0.82, OWL.DatatypeProperty: 0.78,
        OWL.NamedIndividual: 0.82
    }

    m = MOSAICGPUOptimized(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        thresholds=thresholds,
        max_cache_labels=50_000,
        device=device,
        gpu_batch_size=512 if device == "cuda" else 128,
        use_gpu_faiss=False,
    )
    
    runner = OAEITrackRunnerGPU(matcher=m)
    runner.run_all_tracks("../tracks", csv_out="mosaic_evaluation_report_gpu.csv")