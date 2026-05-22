import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import re

    import marimo as mo
    import pandas as pd

    return Path, mo, pd, re


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
    final_ml_dataset.head()
    return (final_ml_dataset,)


@app.cell
def _(final_ml_dataset):
    # Check how many human personas we have in the final dataset
    len(final_ml_dataset)
    # As expected this is approximately 75% of the total personas generated
    return


@app.cell
def _(final_ml_dataset, pd, re):
    from typing import Set

    # Set up the ordering for the education level variable
    education_order = {
        "no formal education": 0,
        "primary school": 1,
        "lower secondary": 2,
        "upper secondary": 3,
        "vocational diploma": 4,
        "bachelor's degree": 5,
        "master's degree": 6,
        "phd": 7,
    }

    # Set up the ordering for the education level variable
    ocean_level_order = {"low": 0, "moderate": 1, "high": 2}

    # Declare which columns are categorical
    nominal_cols = [
        "gender",
        "sexual_orientation",
        "city_of_living",
        "employment_status",
        "marital_status",
        "migration_status",
        "religious_beliefs",
        "Model",
    ]

    # Declare which columns have multiple comma-separated entries as values
    multi_label_cols = {
        "hobbies": "hobby",
        "fav_subjects": "fav_subject",
        "hat_subjects": "hat_subject",
    }

    # Declare which columns are ordinal
    ordinal_cols = {
        "education_level": education_order,
        "parent_1_education": education_order,
        "parent_2_education": education_order,
        "ocean_openness_level": ocean_level_order,
        "ocean_conscientiousness_level": ocean_level_order,
        "ocean_extraversion_level": ocean_level_order,
        "ocean_agreeableness_level": ocean_level_order,
        "ocean_neuroticism_level": ocean_level_order,
    }

    def clean_text_series(series):
        """
        Format strings by removing leading and trailing whitespaces
        """
        return series.astype("string").str.strip()

    def normalize_tokens(value: str) -> Set[str]:
        """Split entries for multi-label columns and perform deduplication"""
        if pd.isna(value):
            return set()
        tokens = [token.strip().lower() for token in str(value).split(",")]
        return {token for token in tokens if token}

    def slugify(value):
        """
        Use regex to remove all non-alphanumeric values and replace those with
        a `_` character. The `_` character is stripped from the final str.
        """
        slug = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower())
        return slug.strip("_")

    def encode_ordinal(series, mapping):
        """Clean the ordinal variables and assign their numerical score."""
        normalized = clean_text_series(series).str.lower()
        encoded = normalized.map(mapping)
        return encoded.fillna(-1).astype("int64")

    def expand_multilabel(frame, column, prefix):
        """"""
        token_series = frame[column].map(normalize_tokens)
        vocabulary = sorted({token for tokens in token_series for token in tokens})
        expanded = pd.DataFrame(index=frame.index)
        expanded[f"{prefix}_count"] = token_series.map(len).astype("int64")
        for token in vocabulary:
            feature_name = f"{prefix}_{slugify(token)}"
            expanded[feature_name] = token_series.map(lambda tokens, tok=token: int(tok in tokens)).astype("int64")
        return expanded, token_series

    # Create deep copy of the dataset to avoid overwriting the original df
    engineered = final_ml_dataset.copy()

    # Expand lists and clean the string values in columns that are nominal, ordinal, or multi-label as highlighted before
    for column in [*nominal_cols, *ordinal_cols.keys(), *multi_label_cols.keys()]:
        engineered[column] = clean_text_series(engineered[column])

    # Standardize the migration_status col 
    engineered["migration_status"] = engineered["migration_status"].str.lower()

    # Map levels of ordinal columns 
    for column, mapping in ordinal_cols.items():
        engineered[f"{column}_ord"] = encode_ordinal(engineered[column], mapping)

    # Create has_children_flg which is True if person has at least one child
    engineered["has_children_flg"] = (engineered["children"].astype("int64") > 0).astype("int64")
    # Get the mean of the parent education based on their ordinal values
    engineered["parent_education_mean_ord"] = engineered[["parent_1_education_ord", "parent_2_education_ord"]].mean(axis=1)
    # Compute the education gap between parents
    engineered["parent_education_gap_ord"] = (engineered["parent_1_education_ord"] - engineered["parent_2_education_ord"]).abs()

    # Declare which columns make up the OCEAN attributes
    ocean_score_cols = [
        "ocean_openness_score",
        "ocean_conscientiousness_score",
        "ocean_extraversion_score",
        "ocean_agreeableness_score",
        "ocean_neuroticism_score",
    ]

    # Compute the mean of the five OCEAN scores (might drop this in future version, as it is not grounded in theory)
    # It does not make much sense to aggregate the scores from five, potentially independent psychological traits into a single dimension.
    # engineered["ocean_score_mean"] = engineered[ocean_score_cols].mean(axis=1)
    # engineered["ocean_score_std"] = engineered[ocean_score_cols].std(axis=1)
    # engineered["ocean_score_min"] = engineered[ocean_score_cols].min(axis=1)
    # engineered["ocean_score_max"] = engineered[ocean_score_cols].max(axis=1)
    # engineered["ocean_score_range"] = engineered["ocean_score_max"] - engineered["ocean_score_min"]

    # This accounts for most of the columns in the final dataset. The approach is to check the frequency of each hobby.
    # Since this is not part of the main RQ this can be safely dropped for now.
    # for column, prefix in multi_label_cols.items():
    #     expanded, token_series = expand_multilabel(engineered, column, prefix)
    #     engineered = pd.concat([engineered, expanded], axis=1)

    # By construction this is not possible, we cannot have a subject that is both favorite and hated.
    # fav_subjects_tokens = engineered["fav_subjects"].map(normalize_tokens)
    # hat_subjects_tokens = engineered["hat_subjects"].map(normalize_tokens)
    # engineered["subjects_overlap_count"] = [len(fav & hat) for fav, hat in zip(fav_subjects_tokens, hat_subjects_tokens)]
    # engineered["subjects_union_count"] = [len(fav | hat) for fav, hat in zip(fav_subjects_tokens, hat_subjects_tokens)]
    # engineered["subjects_net_preference"] = engineered["fav_subject_count"] - engineered["hat_subject_count"]

    # Create dummies to identify both math lovers and haters.
    engineered["math_lover_flg"] = engineered["fav_subjects"].str.contains(r"math", case=False, na=False).astype("int64")
    engineered["math_hater_flg"] = engineered["hat_subjects"].str.contains(r"math", case=False, na=False).astype("int64")

    # Create dummy variables out of the categorical variables. Notice that no assumption about the baseline level is made
    # at this point. It is left to the modeling phase to pick a baseline for each categorical variable and use it consistenly throughout.
    engineered = pd.get_dummies(engineered, columns=nominal_cols, drop_first=False, dtype="int64")

    # Finally, drop the original columns.
    engineered = engineered.drop(
        columns=[
            "mode",
            "education_level",
            "parent_1_education",
            "parent_2_education",
            "ocean_openness_level",
            "ocean_conscientiousness_level",
            "ocean_extraversion_level",
            "ocean_agreeableness_level",
            "ocean_neuroticism_level",
            "hobbies",
            "fav_subjects",
            "hat_subjects",
        ]
    )

    feature_engineered_ml_dataset = engineered
    return (feature_engineered_ml_dataset,)


@app.cell
def _(feature_engineered_ml_dataset):
    feature_engineered_ml_dataset.head()
    return


@app.cell
def _(Path, feature_engineered_ml_dataset):
    # save the dataset
    path_save = Path("data/processed/ml/ml_dataset.csv").resolve().absolute()
    path_save.parent.mkdir(parents=True, exist_ok=True)
    feature_engineered_ml_dataset.to_csv(path_save, index=False)
    return


if __name__ == "__main__":
    app.run()
