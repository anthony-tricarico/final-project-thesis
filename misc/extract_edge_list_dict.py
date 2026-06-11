"""
Extract edge-list CSVs from NEW_edge_list_individual into a single JSON
dictionary keyed by run_id.

Each value is a list of 2-tuples [(cue_word, association_word), ...]
representing all edges for that run.

Usage:
    uv run python misc/extract_edge_list_dict.py
"""

import glob
import json
import os
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/processed/NEW_edge_list_individual")
OUTPUT_PATH = Path("data/processed/edge_list_dict.json")


def build_edge_dict(models: list[str] | None = None) -> dict[str, list[list[str]]]:
    """Build {run_id: [[cue, assoc], ...]} from all edgelist CSVs."""
    if models is None:
        model_dirs = sorted(glob.glob(str(DATA_DIR / "*/")))
    else:
        model_dirs = [str(DATA_DIR / m) for m in models]

    edge_dict: dict[str, list[list[str]]] = {}
    total_runs = 0

    for model_dir in model_dirs:
        model_name = Path(model_dir).name
        run_dirs = sorted(glob.glob(os.path.join(model_dir, "*/")))

        print(f"  {model_name}: {len(run_dirs)} runs ...", end="", flush=True)
        for rd in run_dirs:
            run_id = Path(rd).name
            csv_path = os.path.join(rd, "edgelist.csv")
            df = pd.read_csv(csv_path)
            edges = df[["cue_word", "association_word"]].values.tolist()
            edge_dict[run_id] = edges
        print(f" done ({len(edge_dict) - total_runs} rows)")
        total_runs = len(edge_dict)

    return edge_dict


def main():
    print("Building edge-list dictionary from:")
    print(f"  Source: {DATA_DIR.resolve()}")
    print()

    edge_dict = build_edge_dict()

    print(f"\nTotal run_ids: {len(edge_dict)}")
    total_edges = sum(len(v) for v in edge_dict.values())
    print(f"Total edges:   {total_edges}")
    print(f"Avg edges/run: {total_edges / len(edge_dict):.1f}")
    print(f"Writing to {OUTPUT_PATH} ...")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(edge_dict, f, indent=2)

    file_size = OUTPUT_PATH.stat().st_size
    print(f"Done. File size: {file_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
