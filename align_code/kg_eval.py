import csv
import math
from pathlib import Path
from typing import Optional, Set, Tuple
from rdflib import Graph, RDF, URIRef

# Local paths matching amdcode structure
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR / "../tracks"
RESULTS_DIR = SCRIPT_DIR / "results"
KG_TRACK_DIR = BASE_DIR / "knowledge-graph"

TARGET_ALIGNMENTS = [
    "marvelcinematicuniverse-marvel",
    "memoryalpha-memorybeta",
    "memoryalpha-stexpanded",
    "starwars-swg",
    "starwars-swtor"
]

def calc_precision(tp: int, fp: int) -> float:
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def calc_recall(tp: int, fn: int) -> float:
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def calc_f1(precision: float, recall: float) -> float:
    return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

def compute_kg_partial_prf(system_alignment: set[tuple[str, str]], reference_alignment: set[tuple[str, str]]) -> dict:
    """ P/R/F1 under the OAEI KG track partial gold standard semantics. """
    if not reference_alignment:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "ignored": len(system_alignment),
            "system_size": len(system_alignment),
            "evaluated_size": 0,
            "reference_size": 0,
            "source": "kg_partial",
        }

    reference_sources = {m_src for m_src, _m_tgt in reference_alignment}
    reference_targets = {_m_src for _m_src, m_tgt in reference_alignment}

    true_positives = 0
    false_positives = 0
    number_of_ignored_mappings = 0

    for system_src_mapping, system_target_mapping in system_alignment:
        if (system_src_mapping in reference_sources) or (system_target_mapping in reference_targets):
            if (system_src_mapping, system_target_mapping) in reference_alignment:
                true_positives += 1
            else:
                false_positives += 1
        else:
            number_of_ignored_mappings += 1

    false_negatives = len(reference_alignment) - true_positives
    total_number_of_evaluated_mappings = len(system_alignment) - number_of_ignored_mappings

    precision = calc_precision(true_positives, false_positives)
    recall = calc_recall(true_positives, false_negatives)
    f1 = calc_f1(precision, recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "ignored": number_of_ignored_mappings,
        "system_size": len(system_alignment),
        "evaluated_size": total_number_of_evaluated_mappings,
        "reference_size": len(reference_alignment),
        "source": "kg_partial",
    }

def find_reference_file(track: Path, task_stem: str) -> Optional[Path]:
    """ Direct adaptation of the reference discovery logic from amdcode.py """
    candidate_exts = (".ttl", ".rdf", ".owl", ".xml", ".tsv", ".csv")

    parts = task_stem.split("-", 1)
    reversed_stem = f"{parts[1]}-{parts[0]}" if len(parts) == 2 else None
    name_variants = [task_stem] + ([reversed_stem] if reversed_stem else [])

    # Target directories: both track root and task subdirectories
    search_base_dirs = [track]
    for variant in name_variants:
        search_base_dirs.append(track / variant)

    sub_dirs = [
        "",
        "refs_equiv",
        "refs",
        "references",
        "reference",
        "ontologies"
    ]

    # 1. Search for task name variant matches (e.g. marvelcinematicuniverse-marvel.ttl)
    for base_dir in search_base_dirs:
        for sub in sub_dirs:
            target_dir = base_dir / sub if sub else base_dir
            if not target_dir.exists():
                continue
            for name in name_variants:
                for ext in candidate_exts:
                    candidate = target_dir / f"{name}{ext}"
                    if candidate.exists() and candidate.is_file():
                        return candidate

    # 2. Search for generic/standard reference names (e.g. onto1-onto2.ttl)
    generic_ref_names = (
        "onto1-onto2.ttl",
        "onto1-onto2.rdf",
        "ref-align.rdf",
        "reference.rdf",
        "reference.ttl",
        "reference.tsv",
        "train.tsv",
        "test.tsv",
        "full.tsv"
    )

    for base_dir in search_base_dirs:
        for sub in sub_dirs:
            target_dir = base_dir / sub if sub else base_dir
            if not target_dir.exists():
                continue
            for fname in generic_ref_names:
                candidate = target_dir / fname
                if candidate.exists() and candidate.is_file():
                    return candidate

    return None

from rdflib import Graph, RDF, URIRef, OWL

def parse_alignment_file(path: Path) -> Set[Tuple[str, str]]:
    """ Extracts (source_entity, target_entity) pairs from OAEI RDF/XML, Turtle (.ttl), or TSV/CSV alignments """
    pairs = set()
    if not path.exists():
        return pairs

    # 1. TSV / CSV Parsing
    if path.suffix.lower() in (".tsv", ".csv"):
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f, delimiter=delimiter)
                rows = list(reader)
            if not rows:
                return pairs
            
            header = [h.strip().lower() for h in rows[0]]
            has_header = any(h in ("srcentity", "tgtentity", "src", "tgt", "source", "target") for h in header)
            
            if has_header:
                col_idx = {name: i for i, name in enumerate(header)}
                src_i = col_idx.get("srcentity", col_idx.get("src", col_idx.get("source", 0)))
                tgt_i = col_idx.get("tgtentity", col_idx.get("tgt", col_idx.get("target", 1)))
                data_rows = rows[1:]
            else:
                src_i, tgt_i = 0, 1
                data_rows = rows

            for row in data_rows:
                if row and len(row) > max(src_i, tgt_i):
                    s, t = row[src_i].strip(), row[tgt_i].strip()
                    if s and t:
                        pairs.add((s, t))
        except Exception as e:
            print(f" [ERROR] Failed to parse TSV alignment '{path.name}': {e}")
        return pairs

    # 2. RDF / TTL Parsing
    try:
        g = Graph()
        fmt = "turtle" if path.suffix.lower() == ".ttl" else "xml"
        try:
            g.parse(str(path), format=fmt)
        except Exception:
            alt_fmt = "xml" if fmt == "turtle" else "turtle"
            g.parse(str(path), format=alt_fmt)

        # Correct Alignment NS with trailing '#'
        ALIGN_NS = "http://knowledgeweb.semanticweb.org/heterogeneity/alignment#"
        cell_type = URIRef(f"{ALIGN_NS}Cell")
        e1_pred = URIRef(f"{ALIGN_NS}entity1")
        e2_pred = URIRef(f"{ALIGN_NS}entity2")

        # Extraction Method A: Standard OAEI Alignment Format
        cells = set(g.subjects(RDF.type, cell_type))
        if not cells:
            cells = set(g.subjects(e1_pred, None)) & set(g.subjects(e2_pred, None))

        for cell in cells:
            e1 = g.value(cell, e1_pred)
            e2 = g.value(cell, e2_pred)
            if e1 and e2:
                pairs.add((str(e1).strip(), str(e2).strip()))

        # Extraction Method B: Direct owl:sameAs triples (Common in onto1-onto2.ttl)
        if not pairs:
            for s, o in g.subject_objects(OWL.sameAs):
                if isinstance(s, URIRef) and isinstance(o, URIRef):
                    pairs.add((str(s).strip(), str(o).strip()))

        # Extraction Method C: Generic triple fallback
        if not pairs:
            for s, p, o in g:
                if isinstance(s, URIRef) and isinstance(o, URIRef) and str(p) != str(RDF.type):
                    pairs.add((str(s).strip(), str(o).strip()))

    except Exception as e:
        print(f" [ERROR] Failed to parse RDF/TTL alignment '{path.name}': {e}")

    return pairs

def evaluate_knowledge_graph_alignments(output_csv: str = "kg_eval_report.csv"):
    log_reports = []

    print("=" * 70)
    print(" Running Knowledge Graph Track Re-Evaluation")
    print("=" * 70)

    for task_name in TARGET_ALIGNMENTS:
        print(f"\n[Evaluating Task]: {task_name}")

        # Locate system generated file
        system_file = RESULTS_DIR / f"{task_name}.rdf"
        if not system_file.exists():
            system_file = RESULTS_DIR / f"{task_name}.tsv"

        if not system_file.exists():
            print(f" [WARNING] No generated result found for {task_name} in {RESULTS_DIR}")
            continue

        # Locate reference file in tracks/knowledge-graphs
        ref_file = find_reference_file(KG_TRACK_DIR, task_name)
        if not ref_file or not ref_file.exists():
            print(f" [WARNING] Reference file not found for {task_name} under {KG_TRACK_DIR}")
            continue

        print(f"  System Alignment: {system_file.name}")
        print(f"  Reference File:   {ref_file.relative_to(BASE_DIR)}")

        system_pairs = parse_alignment_file(system_file)
        reference_pairs = parse_alignment_file(ref_file)

        metrics = compute_kg_partial_prf(system_pairs, reference_pairs)

        p = round(metrics["precision"], 4)
        r = round(metrics["recall"], 4)
        f1 = round(metrics["f1"], 4)
        tp = metrics["true_positives"]
        ref_size = metrics["reference_size"]
        sys_size = metrics["system_size"]

        print(f"  Results -> Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
        print(f"             TP: {tp} | Reference Size: {ref_size} | System Alignments: {sys_size}")

        log_reports.append({
            "Track": "knowledge-graph",
            "Task": task_name,
            "Precision": p,
            "Recall": r,
            "F1-Score": f1,
            "Time (s)": "N/A",
            "Total Time (s)": "N/A",
            "Alignments": sys_size,
            "Correct": tp,
            "Reference": ref_size,
            "Type": "Task"
        })

    # Save to CSV
    output_path = RESULTS_DIR / output_csv
    fields = [
        "Track", "Task", "Precision", "Recall", "F1-Score",
        "Time (s)", "Total Time (s)", "Alignments", "Correct",
        "Reference", "Type"
    ]

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(log_reports)

    print("\n" + "=" * 70)
    print(f" Evaluation Completed. Report saved to: {output_path}")
    print("=" * 70)

if __name__ == "__main__":
    evaluate_knowledge_graph_alignments()