"""
This script creates the supporting dataset for the validation of Task 2.

Run with uv from the root of the directory using:

uv run code/data_processing/generate_call2_dataset.py

Author: Anthony Tricarico
Email: tricarico672@gmail.com
"""

import pandas as pd
import json
from pathlib import Path
import sys

from mathanx.constants import DATA_PATH

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))


def extract_scales_to_dataframe(base_directory: str) -> pd.DataFrame:
    """
    Extracts run_id, model, scale, item number, and rating from call 2  JSON files , and returns them as a pandas DataFrame.
    """
    base_path = Path(base_directory)
    extracted_data = []

    # Iterate through all subdirectories (Model folders)
    for model_dir in base_path.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        # Find all JSON files in the model directory that include "call2" in the name
        for json_file in model_dir.glob("*call2*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {json_file.name}. Skipping.")
                continue

            # Extract the top-level run_id and mode
            run_id = data.get("run_id")
            mode = data.get("mode")

            # Navigate through the schema to reach the 'scales' dictionary
            response_parsed = data.get("response_parsed", {})
            parsed = response_parsed.get("parsed", {})
            scales = parsed.get("scales", {})

            # Iterate through each scale (e.g., 'maes', 'amas', 'mseaq')
            for scale_name, scale_data in scales.items():

                # Access the "items" dictionary inside the specific scale
                items = scale_data.get("items", {})

                # Iterate through each item number and grab the rating
                for item_num, item_details in items.items():
                    rating = item_details.get("rating")

                    # Append the flattened data to our list
                    extracted_data.append({
                        "run_id": run_id,
                        "mode": mode,
                        "Model": model_name,
                        "scale": scale_name,
                        "item number": item_num,
                        "rating": rating
                    })

    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(extracted_data)

    return df

# --- Execution Example ---


data_path = Path(DATA_PATH).resolve().absolute()

df_results = extract_scales_to_dataframe(data_path)
# save as csv
path_to_csv = Path(
    "data/processed/validations/task-2/call2_dataset.csv").resolve().absolute()
df_results.to_csv(path_to_csv, index=False)
print(df_results.head(15))
