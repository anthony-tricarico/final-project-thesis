# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "seaborn",
#     "pydantic",
#     "torch",
#     "transformers",
# ]
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import os
    import sys
    from pathlib import Path
    from typing import Any, Dict, List, Optional

    import math
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Patch

    import numpy as np
    import pandas as pd
    import seaborn as sns

    from mathanx.schemas import Call4Schema
    from mathanx.constants import (
        MSESR_CORRECT_ANSWERS,
        MODEL_NAME_MAPPING,
    )
    from mathanx.ml.config import FIG_PATH

    COLOR_PALETTE_1 = "#648fff"
    COLOR_PALETTE_2 = "#fe6100"
    return (
        Any,
        Call4Schema,
        Dict,
        FIG_PATH,
        List,
        MODEL_NAME_MAPPING,
        MSESR_CORRECT_ANSWERS,
        Optional,
        Path,
        mo,
        np,
        os,
        pd,
        plt,
    )


@app.cell
def _(mo):
    mo.md("""
    # Task 4: Accuracy

    This notebook shows the computations performed to compute and plot both accuracy and overconfidence.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset construction

    This section illustrates how the supporting dataset for this section (saved in `03-processed_data/validations/task-4_accuracy/call4.pkl`) was created from the data.
    """)
    return


@app.cell
def _(Call4Schema, Dict, pd):
    def build_df_call4(data: Dict[str, Call4Schema]) -> pd.DataFrame:
        """
        Build a pandas DataFrame given a dictionary of call4 records.

        Args:
            data: a dictionary made of `filename: extracted call 4 schema data`.
            This can easily be obtained by using the JSONExtractor object by calling
            the _load_data() method.

            ```
            # Load the data
            ext = JSONExtractor(PROCESSED_DATA_DIR_PATH)
            ext._load_data()

            # Save the extracted data inside a variable
            processed_dict = ext.data_dict
            # Filter the extracted data to only get results related to call4
            data = {k: v for k, v in processed_dict.items() if "call4" in k}

            # Call the function
            df = build_df_call(data)
            ```

        Outputs:
            A pandas DataFrame containing the columns ["persona", "gender", "run_id", "model", "mode", "chosen_option", "reasoning", 
            "confidence_score", "question_number"]
        """
        persona_list = []
        run_id_list = []
        gender_list = []
        model_list = []
        mode_list = []
        chosen_option_list = []
        reasoning_list = []
        confidence_score_list = []
        question_number_list = []

        for k in data:
            data_cur = data[k]

            run_id = [data_cur["run_id"]]
            mode = [data_cur["mode"]]
            model = [data_cur["model"]]

            # print(f"reading {run_id} from model {model}")
            if data_cur["mode"] == "human": 
                persona = [data_cur["persona"]]
                try:
                    gender = [data_cur["persona"]["gender"]]
                except KeyError:
                    gender = [data_cur["persona"]["gender_identity"]]

            else:
                persona = ["None"]
                gender = ["None"]

            nested = data_cur["response_parsed"]["parsed"]["msesr_problem_solving"]

            cnt = 0
            for ans in nested:
                cnt += 1
                question_number_list.append(ans)
                chosen_option = nested[ans]["chosen_option"]
                reasoning = nested[ans]["reasoning"]
                confidence_score = nested[ans]["confidence_score"]

                chosen_option_list.append(chosen_option)
                reasoning_list.append(reasoning)
                confidence_score_list.append(confidence_score)

            persona_list.extend(persona * cnt)
            run_id_list.extend(run_id * cnt)
            gender_list.extend(gender * cnt)
            mode_list.extend(mode * cnt)
            model_list.extend(model * cnt)

        # The column names and the values associated to them must be aligned
        col_names = ["persona", "gender", "run_id", "model", "mode", "chosen_option", "reasoning", "confidence_score", "question_number"]
        values = [persona_list, gender_list, run_id_list, model_list, mode_list, chosen_option_list, reasoning_list, confidence_score_list, question_number_list]

        dct = {k:v for k, v in zip(col_names, values)}

        return pd.DataFrame(dct)


    return (build_df_call4,)


@app.cell
def _(Any, Dict, List, Path, os):
    from abc import ABC, abstractmethod
    import json

    class Extractor(ABC):
        """
        Define the generic Extractor interface using an abstract class.
        """
        def __init__(self, data_path) -> None:
            self.data_dict = {}
            self.data_path = data_path

        def _convert_path(self, path: str | Path | None = None) -> Path:
            """
            Checks if path is an instance of Path.
            If not, convert it to Path and return its absolute path to avoid ambiguity.
            """
            if not path:
                return Path(self.data_path).resolve().absolute()

            if not isinstance(path, Path):
                return Path(path).resolve().absolute()
            else:
                return path.resolve().absolute()

        def _yield_data_paths(self, path: str | Path | None = None):
            """
            This function yields data paths to files in a single
            folder with a depth of 2.
            """
            if not path:
                path = self._convert_path(self.data_path)
            else:
                path = self._convert_path(path)

            # List the top-level directory (e.g., /data/processed/)
            for item in os.listdir(path):
                dir_path = path.joinpath(item)

                if dir_path.is_dir():
                    for file in os.listdir(dir_path):
                        # Optional: skip hidden files like .DS_Store here too
                        if not file.startswith('.'):
                            file_path = dir_path.joinpath(file)
                            yield file_path

        # Method to load dataset
        @abstractmethod
        def _load_data(self, path_to_data: str | Path) -> Any:
            ...

        # Method to extract information of interest
        @abstractmethod
        def _extract_info(self):
            ...

        # Method to plot information of interest
        @abstractmethod
        def plot(self):
            ...

    class JSONExtractor(Extractor):

        def __init__(self, data_path) -> None:
            super().__init__(data_path)

        def _load_data(
                self,
                path_to_data: str | Path | None = None
                ) -> None:
            """
            Load data assuming current directory structure having
            only a depth of 2.

            Args:
                path_to_data: A path to the directory containing the data

            Returns:
                None
            """

            if not path_to_data:
                path = self._convert_path(path=self.data_path)
            else: 
                path = self._convert_path(path_to_data)

            # Check that path is a valid directory
            if not path.is_dir():
                raise OSError(f"The path {path} does not point to a valid directory!")

            for file in self._yield_data_paths(path):
                try:
                    with open(file, "r") as f:
                        data = json.load(f)
                    # Writes a dictionary with file name as key and read data as value
                    self.data_dict[file.name] = data
                except Exception as e:
                    print(f"An error occurred while reading {file}:", e)

        def filter_data_by_name(self, filter_name: str) -> List[Dict]:
            """
            Filter data by matching parts of their names.

            """
            # Checks if data was not loaded before
            if not self.data_dict:
                self._load_data()

            return list(filter(lambda x: filter_name in x, self.data_dict))

        def _extract_info(self):
            ...

        def plot(self):
            ...

    return (JSONExtractor,)


@app.cell
def _(JSONExtractor, Path, build_df_call4):
    # Load the data
    path_to_data = Path("data/raw").resolve().absolute()
    ext = JSONExtractor(path_to_data) 
    ext._load_data()

    # Save the extracted data inside a variable
    processed_dict = ext.data_dict
    # Filter the extracted data to only get results related to call4
    filtered_data = {k: v for k, v in processed_dict.items() if "call4" in k}

    df_confidence_extracted = build_df_call4(filtered_data)

    return


@app.cell
def _():
    # df_confidence_extracted.to_pickle(Path('data/processed/validations/task-4_accuracy/call4.pkl').resolve().absolute())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Accuracy and confidence computations

    This section outlines the main steps carried out to compute the accuracy and confidence scores.
    """)
    return


@app.cell
def _(MODEL_NAME_MAPPING, MSESR_CORRECT_ANSWERS, Path, pd):
    df_confidence = pd.read_pickle(Path('data/processed/validations/task-4_accuracy/call4.pkl').resolve().absolute())
    df_confidence['question_id'] = df_confidence["question_number"].apply(lambda x: "Q"+x)
    df_confidence["model"] = df_confidence["model"].replace(MODEL_NAME_MAPPING)
    df_confidence['correct'] = df_confidence["question_id"].map(MSESR_CORRECT_ANSWERS)
    df_confidence["flg_correct"] = df_confidence["chosen_option"] == df_confidence["correct"]
    return (df_confidence,)


@app.cell
def _(df_confidence):
    df_confidence.head(10)
    return


@app.cell
def _(List, pd):
    def group_compute_stats(df: pd.DataFrame, grouping_vars: List[str]) -> pd.DataFrame:

        grouped_df_model= df.groupby(grouping_vars).agg(
            total_correct=("flg_correct", "sum"),
            n_observations=("model", "size"),
            confidence = ("confidence_score", "mean")
        ).reset_index()

        grouped_df_model["accuracy"] = (
            grouped_df_model["total_correct"] / grouped_df_model["n_observations"]
        )

        grouped_df_model["confidence_scaled"] = (grouped_df_model["confidence"] - grouped_df_model["confidence"].min()) /\
            (grouped_df_model["confidence"].max() - grouped_df_model["confidence"].min())

        # Compute the difference between confidence scores and accuracy
        grouped_df_model["delta_confidence"] = grouped_df_model["accuracy"] - grouped_df_model["confidence_scaled"]

        return grouped_df_model


    return (group_compute_stats,)


@app.cell
def _(df_confidence, group_compute_stats):
    grouped_df_model = group_compute_stats(df_confidence, ["model", "question_id"])
    grouped_df_model.head()
    return (grouped_df_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Extract data for ML

    This section has the goal of producing a dataset that can be used for ML tasks.
    """)
    return


@app.cell
def _(df_confidence, group_compute_stats):
    df_ml = group_compute_stats(df_confidence, ["run_id"])
    df_ml.head()
    return (df_ml,)


@app.cell
def _(df_ml):
    # All personas replied to exactly 18 questions
    df_ml["n_observations"].nunique()
    return


@app.cell
def _(Path, df_ml):
    if False:
        path_to_save = Path("data/processed/validations/task-4_accuracy/task4_ml.csv")
        df_ml.to_csv(path_to_save, index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Plots

    This section details how the plots were produced.
    """)
    return


@app.cell
def _(Optional, Path, np, pd, plt):


    # ==========================================
    # Plotting Configuration & Function
    # ==========================================
    _COLOR_PALETTE_2 = "#785ef0"
    _COLOR_PALETTE_1 = "#ffb000"

    def plot_accuracy_vs_confidence_custom_ci(
        data: pd.DataFrame, 
        save_path: Optional[str] = None,
        legend_loc: str = "upper left", 
        bbox_to_anchor: tuple = (1.02, 1)
    ):
        """
        Plots a grouped bar chart comparing accuracy and scaled confidence,
        where bars are side-by-side but have different widths.

        Parameters:
        - data: The dataset containing model, accuracy, and confidence metrics.
        - save_path: Directory path to save the SVG.
        - legend_loc: The anchor point of the legend box (default: "upper left").
        - bbox_to_anchor: The coordinates to place the anchor (default: (1.02, 1) places it outside on the right).
        """
        if data.empty:
            raise ValueError("The dataset passed is empty!")

        # Calculate the means and standard error for the metrics
        agg_df = data.groupby("model").agg(
            Accuracy=("accuracy", "mean"),
            Confidence=("confidence_scaled", "mean"),
            Conf_SEM=("confidence_scaled", "sem")
        ).reset_index()

        agg_df["Conf_CI"] = agg_df["Conf_SEM"] * 1.96
        agg_df = agg_df.sort_values(by="Accuracy", ascending=False).set_index("model")

        # Prepare manual X coordinates and widths
        models_labels = agg_df.index
        x = np.arange(len(models_labels))

        # Set your desired widths here
        width_acc = 0.50   # Wider Accuracy bar
        width_conf = 0.30  # Narrower Confidence bar

        # Calculate perfectly centered side-by-side positions
        x_acc = x - (width_conf / 2)
        x_conf = x + (width_acc / 2)

        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot Accuracy
        ax.bar(
            x_acc, agg_df["Accuracy"], 
            width=width_acc,           
            color=_COLOR_PALETTE_1, 
            edgecolor="black",
            linewidth=0.5,
            label="Accuracy"
        )

        # Plot Confidence with Error Bars
        ax.bar(
            x_conf, agg_df["Confidence"], 
            yerr=agg_df["Conf_CI"],
            width=width_conf,          
            color=_COLOR_PALETTE_2, 
            capsize=4, 
            edgecolor="black",
            linewidth=0.5,
            label="Confidence"
        )

        # Formatting
        ax.set_ylabel("Score", fontsize=17)

        # Set the ticks exactly at our base 'x' array to center the labels
        ax.set_xticks(x)
        ax.set_xticklabels(models_labels, rotation=45, ha="right", fontsize=13)
        ax.tick_params(axis='y', labelsize=13)

        ax.set_ylim(0, 1.05) 

        # ---------------------------------------------------------
        # UPDATED: Use the passed parameters for legend positioning
        # ---------------------------------------------------------
        ax.legend(
            title="", 
            loc=legend_loc, 
            bbox_to_anchor=bbox_to_anchor, 
            frameon=True, 
            fontsize=13
        )

        plt.tight_layout()

        if save_path:
            path = Path(save_path).resolve().absolute()
            if not path.is_dir(): raise OSError("Given path is not a directory")
            # bbox_inches="tight" is crucial here, as it ensures the external legend isn't cut off when saving
            plt.savefig(path.joinpath("accuracy_vs_confidence.pdf"), format="pdf", bbox_inches="tight")

        return fig

    return (plot_accuracy_vs_confidence_custom_ci,)


@app.cell
def _(FIG_PATH, grouped_df_model, plot_accuracy_vs_confidence_custom_ci):
    plot_accuracy_vs_confidence_custom_ci(grouped_df_model, save_path=FIG_PATH)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Splitting between human and LLM mode
    """)
    return


@app.cell
def _(df_confidence, group_compute_stats):
    grouped_df_model_mode = group_compute_stats(df_confidence, ["model", "question_id", "mode"])
    grouped_df_model_mode["mode"].unique()
    return (grouped_df_model_mode,)


@app.cell
def _(grouped_df_model_mode):
    human_filter = grouped_df_model_mode["mode"] == "human"
    llm_filter = grouped_df_model_mode["mode"] == "llm"
    grouped_df_model_mode_human = grouped_df_model_mode[human_filter]
    grouped_df_model_mode_llm = grouped_df_model_mode[llm_filter]
    return grouped_df_model_mode_human, grouped_df_model_mode_llm


@app.cell
def _(
    FIG_PATH,
    grouped_df_model_mode_human,
    plot_accuracy_vs_confidence_custom_ci,
):
    _fig = plot_accuracy_vs_confidence_custom_ci(grouped_df_model_mode_human)
    _fig.savefig(FIG_PATH / "confidence_vs_accuracy_human.pdf")
    _fig
    return


@app.cell
def _(
    FIG_PATH,
    grouped_df_model_mode_llm,
    plot_accuracy_vs_confidence_custom_ci,
):
    _fig = plot_accuracy_vs_confidence_custom_ci(grouped_df_model_mode_llm)
    _fig.savefig(FIG_PATH / "confidence_vs_accuracy_llm.pdf")
    _fig
    return


if __name__ == "__main__":
    app.run()
