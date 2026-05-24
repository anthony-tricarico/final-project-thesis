# /// script
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "seaborn",
#     "pydantic",
#     "torch",
#     "transformers",
#     "scipy"
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Task 2: Distribution of psychometric scores

    This notebook provides the code to reproduce the analysis that is proposed as validation for Task 2.
    """)
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md("Note that the dataset used throughout this notebook (`call2_dataset.csv`) is created following the script `generate_call2_dataset.py`. You can refer to the specific script for additional details on how this supporting dataset was created from the original data."),
        kind="warn"
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from pathlib import Path
    import sys

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy.stats import gaussian_kde

    from mathanx.constants import FOLDER_NAME_MAPPING

    color_human = '#648fff'
    color_llm = '#fe6100'
    return (
        FOLDER_NAME_MAPPING,
        Patch,
        Path,
        color_human,
        color_llm,
        gaussian_kde,
        np,
        pd,
        plt,
    )


@app.cell
def _(FOLDER_NAME_MAPPING, Path, pd):
    path_call2_dataset = Path("data/processed/validations/task-2/call2_dataset.csv").resolve().absolute()
    df = pd.read_csv(path_call2_dataset)
    df["Model"] = df["Model"].replace(FOLDER_NAME_MAPPING)
    df.head()
    return df, path_call2_dataset


@app.cell
def _(df):
    ITEMS_TO_REVERSE = ["13", "14", "17", "22", "25", "27"]
    REVERSE_ITEMS_MAP = {
        "1": "5",
        "2": "4",
        "3": "3",
        "4": "2",
        "5": "1"
    }

    def map_if_mseaq(df):
        # Create a copy to avoid SettingWithCopyWarning if passing a slice
        df = df.copy()

        # Create a boolean mask for the specific conditions
        condition = (df['scale'] == 'mseaq') & (
            df['item number'].astype(str).isin(ITEMS_TO_REVERSE))

        # Apply the mapping only to the rows that meet the condition
        df.loc[condition, 'rating'] = df.loc[condition, 'rating'].astype(
            str).map(REVERSE_ITEMS_MAP).astype(int)

        return df

    new_df = map_if_mseaq(df)
    return (new_df,)


@app.cell
def _(new_df):
    df_viz = new_df.groupby(["Model", "scale", "mode", "run_id"]).agg(
        sum_of_scores=("rating", "sum")
    ).reset_index()
    return (df_viz,)


@app.cell
def _():
    # all the items that are part of the anxiety subscale
    anxiety_items = [str(x) for x in range(8, 29)]

    # all the items that are part of the self-efficacy subscale
    self_efficacy_items = [str(x) for x in range(1, 8)]
    return anxiety_items, self_efficacy_items


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Split MSEAQ into its subscales

    Items from the MSEAQ scale are made of multiple subscales. It is important to split these different subscales to avoid a collpase of all dimensions into a single, meaningless, dimension.
    """)
    return


@app.cell
def _(anxiety_items, new_df, self_efficacy_items):
    anxiety_filter = (new_df["scale"] == "mseaq") & (
        new_df["item number"].astype(str).isin(anxiety_items))
    self_efficacy_filter = (new_df["scale"] == "mseaq") & (
        new_df["item number"].astype(str).isin(self_efficacy_items))

    mseaq_anxiety_dataset = new_df[anxiety_filter]
    mseaq_efficacy_dataset = new_df[self_efficacy_filter]
    return mseaq_anxiety_dataset, mseaq_efficacy_dataset


@app.cell
def _(mseaq_anxiety_dataset, mseaq_efficacy_dataset):
    df_viz_anxiety = mseaq_anxiety_dataset.groupby(["Model", "scale", "mode", "run_id"]).agg(
        sum_of_scores=("rating", "sum")
    ).reset_index()

    df_viz_self_efficacy = mseaq_efficacy_dataset.groupby(["Model", "scale", "mode", "run_id"]).agg(
        sum_of_scores=("rating", "sum")
    ).reset_index()
    return df_viz_anxiety, df_viz_self_efficacy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Preparing dataset with anxiety and self-efficacy scores for export

    This step is used to derive anxiety and self-efficacy scores by taking advantage of the fact that most of the pre-processing needed to achieve this goal is already carried out in this notebook.
    """)
    return


@app.cell
def _(df_viz, df_viz_anxiety, df_viz_self_efficacy, pd):
    # Change the name of the scales to recognize the appropriate subscales
    df_viz_anxiety["scale"] = "mseaq_anx"
    df_viz_self_efficacy["scale"] = "mseaq_se"

    # Remove duplicate mseaq data from df_viz
    final_df = df_viz[df_viz["scale"] != "mseaq"]

    # Concatenate the two datasets to the final dataset to add MSEAQ observations
    export_df = pd.concat([final_df, df_viz_anxiety, df_viz_self_efficacy], axis = 0, ignore_index=True)
    export_df.head()
    return (export_df,)


@app.cell
def _(df, export_df):
    # Pivot the DataFrame
    # We set run_id, Model, and mode as the index so they remain as standard columns
    df_wide = export_df.pivot(
        index=['run_id', 'Model', 'mode'], 
        columns='scale', 
        values='sum_of_scores'
    ).reset_index()

    # Clean up the column index name (pandas adds the name of the columns argument, 'scale')
    df_wide.columns.name = None

    # Rename the newly created columns to append '_score'
    # dynamically map the unique values from the original 'scale' column
    scale_values = df['scale'].unique()
    rename_mapping = {col: f"{col}_score" for col in scale_values}

    df_wide = df_wide.rename(columns=rename_mapping)
    df_wide.head()
    return (df_wide,)


@app.cell
def _(df_wide, path_call2_dataset):
    # Save the dataset as csv
    if False:
        save_path = path_call2_dataset.parent / "task2_ml_dataset.csv"
        df_wide.to_csv(save_path, index = False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plots

    The plots below show the distribution of the scores for each model for each psychometric (sub)scale.
    """)
    return


@app.cell
def _(Patch, Path, color_human, color_llm, gaussian_kde, np, plt):
    def create_custom_ridgeline(
            df,
            target_scale,
            scale_col="scale",
            model_col="Model",
            mode_col="mode",
            score_col="rating",
            save_path=None,
            custom_name=None):
        """
        Creates a clean, overlapping ridgeline plot on a single axis using KDEs,
        filtered to show only data for a specific scale.
        """
        # Filter the dataframe for the target scale
        df_filtered = df[df[scale_col] == target_scale].copy()

        if df_filtered.empty:
            print(f"Warning: No data found for scale '{
                  target_scale}'. Check your spelling or dataframe.")
            return

        # Setup and configurations
        models = df_filtered[model_col].unique()
        n_models = len(models)

        # SPACING and HEIGHT_SCALE can be used to control how much the distributions overlap
        SPACING = 0.6
        HEIGHT_SCALE = 1.0

        fig, ax = plt.subplots(figsize=(12, 6))

        # Define a smooth x-axis range based on your filtered data's min/max
        x_min, x_max = df_filtered[score_col].min(
        ), df_filtered[score_col].max()
        x_smooth = np.linspace(x_min - 0.5, x_max + 0.5, 500)

        # Iterate and plot each model on the same axis
        for i, model in enumerate(models):
            base = i * SPACING

            # z-order: lower rows are drawn last so they overlap the rows behind them
            z_fill = n_models - i
            z_line = z_fill + 0.1

            # --- HUMAN MODE (Blue) ---
            human_data = df_filtered[(df_filtered[model_col] == model) & (
                df_filtered[mode_col].str.lower() == "human")][score_col].dropna().values

            # Ensure we have enough variance to compute a KDE
            if len(human_data) > 1 and np.var(human_data) > 0:
                kde_human = gaussian_kde(human_data, bw_method=0.3)
                y_human = kde_human(x_smooth)
                y_human = (y_human / y_human.max()) * HEIGHT_SCALE

                ax.fill_between(x_smooth, base, base + y_human,
                                color=color_human, alpha=0.85, zorder=z_fill, linewidth=0)
                ax.plot(x_smooth, base + y_human, color="black",
                        linewidth=1, zorder=z_line)

            # --- LLM MODE (Orange) ---
            llm_data = df_filtered[(df_filtered[model_col] == model) & (
                df_filtered[mode_col].str.lower() == "llm")][score_col].dropna().values

            if len(llm_data) > 1 and np.var(llm_data) > 0:
                kde_llm = gaussian_kde(llm_data, bw_method=0.3)
                y_llm = kde_llm(x_smooth)
                y_llm = (y_llm / y_llm.max()) * HEIGHT_SCALE

                ax.fill_between(x_smooth, base, base + y_llm, color=color_llm,
                                alpha=0.85, zorder=z_fill - 0.05, linewidth=0)
                ax.plot(x_smooth, base + y_llm, color="black",
                        linewidth=1, zorder=z_line)

        # Axis formatting
        ax.set_yticks(np.arange(n_models) * SPACING)
        ax.set_yticklabels(models, fontsize=14)

        ax.set_xlabel("Score", fontsize=14)
        # ax.set_ylabel("Models", fontsize=14)
        ax.tick_params(axis='x', labelsize=14)

        # Uncomment this to show title of the scale.
        # ax.set_title(f"Ridgeline Distribution: {str(target_scale).upper()}", fontsize=14, fontweight="bold", pad=15)

        # Remove the bulky borders for a cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)

        # Custom Legend
        legend_handles = [
            Patch(facecolor=color_human, edgecolor="black", label="Human"),
            Patch(facecolor=color_llm, edgecolor="black", label="LLM")
        ]

        ax.legend(handles=legend_handles, loc="upper right",
                  frameon=False, fontsize=14)

        plt.tight_layout()

        if save_path:
            if custom_name:
                path = Path(save_path).resolve().absolute()
                plt.savefig(path.joinpath(str(custom_name) +
                            "_joyplot.svg"), format="svg")

        plt.show()

    return (create_custom_ridgeline,)


@app.cell
def _(create_custom_ridgeline, df_viz):
    create_custom_ridgeline(df_viz, target_scale="amas", scale_col="scale",
                            model_col="Model", mode_col="mode", score_col="sum_of_scores")
    return


@app.cell
def _(create_custom_ridgeline, df_viz):
    create_custom_ridgeline(df_viz, target_scale="maes", scale_col="scale",
                            model_col="Model", mode_col="mode", score_col="sum_of_scores")
    return


@app.cell
def _(create_custom_ridgeline, df_viz_anxiety):
    create_custom_ridgeline(df_viz_anxiety, target_scale="mseaq", scale_col="scale",
                            model_col="Model", mode_col="mode", score_col="sum_of_scores")
    return


@app.cell
def _(create_custom_ridgeline, df_viz_self_efficacy):
    create_custom_ridgeline(df_viz_self_efficacy, target_scale="mseaq", scale_col="scale",
                            model_col="Model", mode_col="mode", score_col="sum_of_scores")
    return


if __name__ == "__main__":
    app.run()
