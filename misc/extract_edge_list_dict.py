"""
Extract edge-list CSVs from NEW_edge_list_individual into two aligned JSON
dictionary files keyed by run_id.

File 1 — edge_list_dict.json:
    {run_id: [[cue_word, association_word], ...]}

File 2 — valence_dict.json:
    {run_id: {"cue_valences": [...], "assoc_valences": [...]}}

Both files are produced in a single pass so that positional indices are
guaranteed aligned: edges[i] ↔ (cue_valences[i], assoc_valences[i]).

Usage:
    uv run python misc/extract_edge_list_dict.py
"""

import glob
import json
import os
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/processed/NEW_edge_list_individual")
OUTPUT_EDGE_PATH = Path("data/processed/edge_list_dict.json")
OUTPUT_VALENCE_PATH = Path("data/processed/valence_dict.json")


def build_dicts(
    models: list[str] | None = None,
) -> tuple[dict[str, list[list[str]]], dict[str, dict[str, list[int]]]]:
    """Build edge_dict and valence_dict in a single pass over all CSV files."""
    if models is None:
        model_dirs = sorted(glob.glob(str(DATA_DIR / "*/")))
    else:
        model_dirs = [str(DATA_DIR / m) for m in models]

    edge_dict: dict[str, list[list[str]]] = {}
    valence_dict: dict[str, dict[str, list[int]]] = {}
    total_runs = 0

    for model_dir in model_dirs:
        model_name = Path(model_dir).name
        run_dirs = sorted(glob.glob(os.path.join(model_dir, "*/")))

        print(f"  {model_name}: {len(run_dirs)} runs ...", end="", flush=True)
        for rd in run_dirs:
            run_id = Path(rd).name
            csv_path = os.path.join(rd, "edgelist.csv")
            df = pd.read_csv(csv_path)

            edge_dict[run_id] = df[["cue_word", "association_word"]].values.tolist()
            valence_dict[run_id] = {
                "cue_valences": df["cue_valence"].tolist(),
                "assoc_valences": df["associated_valence"].tolist(),
            }
        print(f" done ({len(edge_dict) - total_runs} rows)")
        total_runs = len(edge_dict)

    return edge_dict, valence_dict


def _write_json(data, path: Path, label: str):
    print(f"Writing {label} to {path} ...")
    with open(path, "w") as f:
        json.dump(data, f)
    size = path.stat().st_size
    print(f"  -> {size / 1024 / 1024:.1f} MB")


def main():
    print("Building edge-list and valence dictionaries from:")
    print(f"  Source: {DATA_DIR.resolve()}")
    print()

    edge_dict, valence_dict = build_dicts()

    n_runs = len(edge_dict)
    n_edges = sum(len(v) for v in edge_dict.values())
    print(f"\nTotal run_ids: {n_runs}")
    print(f"Total edges:   {n_edges}")
    print(f"Avg edges/run: {n_edges / n_runs:.1f}")
    print()

    _write_json(edge_dict, OUTPUT_EDGE_PATH, "edge_list_dict.json")
    _write_json(valence_dict, OUTPUT_VALENCE_PATH, "valence_dict.json")
    print("Done. Both files are aligned by run_id and positional index.")


if __name__ == "__main__":
    main()
