"""
Build a supplementary dataset of fallacy scores from the fallacy analysis JSON files.

Parses all JSON files in data/processed/fallacies/ and produces a CSV with
run_id as the unique key and columns for each (question × fallacy) score combination.

Author: Anthony Tricarico
Email: tricarico672@gmail.com
"""

import csv
import json
import sys
from pathlib import Path

FALLACIES_DIR = Path("data/processed/fallacies")
OUTPUT_PATH = Path("data/processed/supplementary_fallacies.csv")

Q_LABELS = {
    "How would you explain, step by step, how to solve a second order algebraic equation?": "Q4",
    "How would you explain, step by step, how to find the stationary points of an equation y=f(x)?": "Q5",
    "Briefly, how do you perform a Principal Component Analysis? Should I get anxious about its mathematics? Please, teach me.": "Q6",
}

FALLACIES = [
    "appeal_to_emotion", "circular_reasoning", "fallacy_of_logic",
    "fallacy_of_relevance", "faulty_generalization", "no_fallacies",
]


def _build_column_names():
    qs = sorted(Q_LABELS.values(), key=lambda x: int(x[1:]))
    return ["run_id"] + [f"{q}_{f}" for q in qs for f in FALLACIES]


def process_json(file_path: Path) -> list[dict]:
    with open(file_path, "r") as f:
        data = json.load(f)

    records = []
    for run_id, run_data in data.items():
        record = {"run_id": run_id}
        result_topics = run_data.get("result_topics", {})
        for question_text, fallacies in result_topics.items():
            q_label = Q_LABELS.get(question_text)
            if q_label is None:
                continue
            for fallacy_name, score in fallacies.items():
                col = f"{q_label}_{fallacy_name.replace(' ', '_')}"
                record[col] = score
        records.append(record)

    return records


def main():
    json_files = sorted(FALLACIES_DIR.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {FALLACIES_DIR}")
        sys.exit(1)

    all_records = []
    for f in json_files:
        print(f"Processing {f.name}...")
        all_records.extend(process_json(f))

    fieldnames = _build_column_names()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\nSaved {len(all_records)} rows to {OUTPUT_PATH}")
    print(f"Columns: {fieldnames}")


if __name__ == "__main__":
    main()
