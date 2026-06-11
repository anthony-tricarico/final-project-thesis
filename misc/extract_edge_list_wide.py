"""
Extract edge-list CSVs from NEW_edge_list_individual into a single wide
dataset with one row per run_id.

Columns per cue (e.g. cue="work"):
    {cue}_association_1    {cue}_association_2    {cue}_association_3
    {cue}_valence
    {cue}_associated_valence_1    {cue}_associated_valence_2    {cue}_associated_valence_3

Usage:
    uv run python misc/extract_edge_list_wide.py
    uv run python misc/extract_edge_list_wide.py --model MANX_LLM_anitamistral
"""

import argparse
import glob
import os
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/processed/NEW_edge_list_individual")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "edge_list_wide.csv"


def _pivot_run(edgelist_path: str, model_name: str, run_id: str) -> dict:
    """Read a single edgelist.csv and return a wide row as a dict."""
    df = pd.read_csv(edgelist_path)

    row = {"run_id": run_id, "model": model_name}

    for cue, grp in df.groupby("cue_word", sort=False):
        associations = grp["association_word"].tolist()
        assoc_valences = grp["associated_valence"].tolist()
        cue_val = grp["cue_valence"].iloc[0]

        for i in range(3):
            col = f"{cue}_association_{i + 1}"
            row[col] = associations[i] if i < len(associations) else None

        row[f"{cue}_valence"] = cue_val

        for i in range(3):
            col = f"{cue}_associated_valence_{i + 1}"
            row[col] = assoc_valences[i] if i < len(assoc_valences) else None

    return row


def build_wide_dataset(models: list[str] | None = None) -> pd.DataFrame:
    """Iterate over model folders and run_ids, pivot each edgelist to wide."""
    if models is None:
        model_dirs = sorted(glob.glob(str(DATA_DIR / "*/")))
    else:
        model_dirs = [str(DATA_DIR / m) for m in models]

    all_rows = []
    total_runs = 0

    for model_dir in model_dirs:
        model_name = Path(model_dir).name
        run_dirs = sorted(glob.glob(os.path.join(model_dir, "*/")))

        print(f"  {model_name}: {len(run_dirs)} runs ...", end="", flush=True)
        for rd in run_dirs:
            run_id = Path(rd).name
            csv_path = os.path.join(rd, "edgelist.csv")
            row = _pivot_run(csv_path, model_name, run_id)
            all_rows.append(row)
        print(f" done ({len(all_rows) - total_runs} rows)")
        total_runs = len(all_rows)

    print(f"\nBuilding DataFrame ({len(all_rows)} rows)...")
    result = pd.DataFrame(all_rows)
    # sort columns: run_id, model, then alphabetically
    cols = [c for c in result.columns if c not in ("run_id", "model")]
    cols.sort()
    result = result[["run_id", "model"] + cols]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Restrict to one or more model folder names (repeatable).",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_FILE,
        help=f"Output CSV path (default: {OUTPUT_FILE})",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building wide edge-list dataset from:")
    print(f"  Source: {DATA_DIR.resolve()}")
    if args.models:
        print(f"  Models: {args.models}")
    print()

    df = build_wide_dataset(models=args.models)

    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Writing to {args.output} ...")
    df.to_csv(args.output, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
