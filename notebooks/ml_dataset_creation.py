import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd

    return Path, mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Check consistency of datasets

    The main objective of this session is to check that all task-specific datasets have a structure that is compatible with the aim of the analysis that should be carried out.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Demographics dataset
    """)
    return


@app.cell
def _(Path, pd):
    demographics_path = Path("data/processed/demographics/persona_dataset.csv").resolve().absolute()
    demographics_df = pd.read_csv(demographics_path)
    target_cols_demographics = ['run_id', 'age', 'gender', 'sexual_orientation',
           'city_of_living', 'employment_status', 'education_level',
           'marital_status', 'children', 'migration_status', 'religious_beliefs',
           'parent_1_education', 'parent_2_education', 'hobbies', 'fav_subjects',
           'hat_subjects', 'ocean_openness_score', 'ocean_openness_level',
           'ocean_conscientiousness_score', 'ocean_conscientiousness_level',
           'ocean_extraversion_score', 'ocean_extraversion_level',
           'ocean_agreeableness_score', 'ocean_agreeableness_level',
           'ocean_neuroticism_score', 'ocean_neuroticism_level']
    demographics_reduced_df = demographics_df.loc[:, target_cols_demographics]
    demographics_reduced_df.head()
    return (demographics_reduced_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task 2 Dataset
    """)
    return


@app.cell
def _(Path, pd):
    task2_path = Path("data/processed/validations/task-2/task2_ml_dataset.csv").resolve().absolute()
    task2_df = pd.read_csv(task2_path)
    human_task2_df = task2_df[task2_df["mode"].str.lower() == "human"]
    human_task2_df.head()
    return human_task2_df, task2_df


@app.cell
def _(task2_df):
    # This dataset contains the correct number of models
    task2_df["Model"].nunique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task 4 Dataset
    """)
    return


@app.cell
def _(Path, pd):
    path_task4_df = Path("data/processed/validations/task-4_accuracy/task4_ml.csv").resolve().absolute()
    task4_df = pd.read_csv(path_task4_df)
    task4_df.head()
    return (task4_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ML dataset merge

    In this section we merge the three dataframes to produce a single tabular representation of the data to train ML models later on.
    """)
    return


@app.cell
def _(demographics_reduced_df, human_task2_df, task4_df):
    final_ml_dataset = demographics_reduced_df\
        .merge(human_task2_df, how="inner", on="run_id")\
        .merge(task4_df, how="inner", on="run_id")
    return (final_ml_dataset,)


@app.cell
def _(final_ml_dataset):
    # Check how many human personas we have in the final dataset
    len(final_ml_dataset)
    # As expected this is approximately 75% of the total personas generated
    return


@app.cell
def _(final_ml_dataset):
    # Add math_lover_flg and math_hater_flg
    final_ml_dataset["math_lover_flg"] = final_ml_dataset["fav_subjects"].str.contains("math")
    final_ml_dataset["math_hater_flg"] = final_ml_dataset["hat_subjects"].str.contains("math")
    # Remember that being a dummy variable we can only infer potential differences with respect to a baseline (math neutral)
    return


@app.cell
def _(final_ml_dataset):
    final_ml_dataset.head()
    return


@app.cell
def _(Path, final_ml_dataset):
    # save the dataset
    if False:
        path_save = Path("data/processed/ml/ml_dataset.csv").resolve().absolute()
        final_ml_dataset.to_csv(path_save, index=False)
    return


if __name__ == "__main__":
    app.run()
