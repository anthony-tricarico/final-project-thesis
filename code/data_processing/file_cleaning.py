# /// script
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "pydantic",
#     "torch",
#     "transformers",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import sys
    sys.path.append("scripts")

    from datetime import datetime
    import json
    from pathlib import Path
    from typing import Any, Dict, Set
    import os

    from scripts.schemas import Call1Schema, Call2Schema, Call3Schema, Call4Schema
    from scripts.utils.constants import FOLDER_NAME_MAPPING
    from scripts.validator import DataValidator

    color_human = '#648fff' 
    color_llm = '#fe6100'
    return (
        DataValidator,
        Dict,
        FOLDER_NAME_MAPPING,
        Path,
        color_human,
        color_llm,
        json,
    )


@app.cell
def _(Dict, Path, json):
    LOG_PATH = "test-logs/"

    def write_json_output_file(filename: str, data: Dict, output_dir: Path = LOG_PATH):
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir).resolve().absolute()

        complete_path = output_dir.joinpath(filename)

        with open(complete_path, "w+") as f:
            json.dump(data, f, indent=2)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cleaning Logic

    This section details how raw data was cleaned and how the DataValidator object and its methods have been used in the data processing pipeline.
    """)
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md("""## Warning
        This section might take some time to run due to the validation being run on all the files in the dataset."""),
        kind="warn"
    )
    return


@app.cell
def _(DataValidator):
    DATA_PATH = "../../01-original_data/"
    # Initialize the DataValidator object
    dv = DataValidator(DATA_PATH)
    # This is the main public method of the validator object. This method is in charge of carrying out the entire validation / cleaning procedure.
    dv.validate_data()
    return (dv,)


@app.cell
def _():
    # Get the total number of correct files
    # dv.get_correct_files()
    return


@app.cell
def _():
    # Get the total number of correct files split by model
    # dv.get_number_correct_files_by_model()
    return


@app.cell
def _():
    # Get details of correct files split by calls for each model.
    # dv.get_detail_correct_files_by_model()
    return


@app.cell
def _(dv):
    # Extract the proportion of Human and LLMS.
    dv.get_mode_proportions()
    _prop = dv.proportions
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Plots
    """)
    return


@app.cell
def _():
    # proportions for final dataset
    prop = {'files': {'MANX_LLM_MistralSmall4': {'llm': 483, 'human': 1517},
      'MANX_LLM_DeepSeekLarge': {'llm': 500, 'human': 1500},
      'MANX_LLM_Grok41FastReasoning': {'llm': 500, 'human': 1500},
      'MANX_LLM_qwen4bthink': {'llm': 500, 'human': 1500},
      'MANX_LLM_ministral3b': {'llm': 500, 'human': 1500},
      'MANX_LLM_mistralsmall': {'llm': 500, 'human': 1500},
      'MANX_LLM_anitamistral': {'llm': 500, 'human': 1500},
      'MANX_LLM_qwen4bunce': {'llm': 500, 'human': 1500},
      'MANX_LLM_granite4h': {'llm': 500, 'human': 1500},
      'MANX_LLM_qwen35_9b': {'llm': 500, 'human': 1500},
      'MANX_LLM_qwen34binstruct': {'llm': 500, 'human': 1500},
      'MANX_LLM_ministral14b': {'llm': 500, 'human': 1500},
      'MANX_LLM_phi4reasoning': {'llm': 500, 'human': 1500},
      'MANX_LLM_magistralsmall': {'llm': 500, 'human': 1500}}}
    return (prop,)


@app.cell
def _(FOLDER_NAME_MAPPING, color_human, color_llm, prop):

    import pandas as pd
    import matplotlib.pyplot as plt

    # Parse the dictionary, clean the names, and calculate percentages
    records = []
    for k, v in prop['files'].items():
        # Remove the prefix case-insensitively
        # name = k.replace('MANX_LLM_', '').replace('manx_LLM_', '')
        name = FOLDER_NAME_MAPPING[k]
        total = v['llm'] + v['human']

        pct_llm = (v['llm'] / total) * 100
        pct_human = (v['human'] / total) * 100

        records.append({'Model': name, 'LLM (%)': pct_llm, 'Human (%)': pct_human})

    df = pd.DataFrame(records)

    # Sort the dataframe by LLM percentage for a cleaner waterfall look
    df = df.sort_values(by='LLM (%)', ascending=True)

    # Create the stacked horizontal bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    df.set_index('Model').plot(
        kind='barh', 
        stacked=True, 
        color=[color_llm, color_human], # Orange for LLM, Blue for Human
        ax=ax,
        width=0.7,
        edgecolor='black',
        linewidth=0.5
    )

    # Add the exact percentages as text inside the bars
    for p in ax.patches:
        width = p.get_width()
        if width > 4: # Only label if there's actually a bar
            x = p.get_x() + width / 2
            y = p.get_y() + p.get_height() / 2
            ax.text(x, y, f'{width:.2f}%', ha='center', va='center', color='white', fontweight='bold', fontsize=9)

    # Formatting
    # plt.title('Proportion of LLM vs. Human by Model', fontweight='bold', pad=15, fontsize=14)
    plt.xlabel('Percentage (%)', fontweight='bold')
    plt.ylabel('') # Remove the generic 'Model' label
    plt.xlim(0, 100) # Force the x-axis to be exactly 0 to 100%

    # Legend adjustments
    plt.legend(['LLM', 'Human'], loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=True)

    plt.tight_layout()

    plt.show()
    return


if __name__ == "__main__":
    app.run()
