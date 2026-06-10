import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    import numpy as np


    return Path, mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Call 4 to wide
    """)
    return


@app.cell
def _(Path, pd):
    # read call4 raw dataset
    path_call4_dataset = Path("data/processed/validations/task-4_accuracy/call4.pkl").resolve()
    df_call4 = pd.read_pickle(path_call4_dataset)
    df_call4
    return (df_call4,)


@app.cell
def _(df_call4):
    df_call4["question_number"].value_counts()
    return


@app.cell
def _(df_call4):
    # remove rows for the questions not in the range 1-18

    filter_questions = df_call4["question_number"].isin([str(x) for x in range(1,19)])
    df_call4_filtered = df_call4[filter_questions]
    # check only correct questions are now in the dataset
    df_call4_filtered["question_number"].value_counts()
    return (df_call4_filtered,)


@app.cell
def _():
    # export the corrected dataset
    # df_call4_filtered.to_pickle(path_call4_dataset.parent / "call4_processed.pkl")
    return


@app.cell
def _(df_call4_filtered):
    # Note: Replace 'question_number' with the actual name of your question column
    question_col_name = 'question_number'

    # 2. Pivot the dataframe 
    # We use run_id as the index, the question number as the columns to pivot on, 
    # and the 3 specified columns as the values to populate the new columns.
    df_wide = df_call4_filtered.pivot(
        index='run_id', 
        columns=question_col_name, 
        values=['chosen_option', 'reasoning', 'confidence_score']
    )

    # 3. Flatten the MultiIndex columns and add the question number as a suffix
    # After the pivot, the columns will be hierarchical (e.g., ('chosen_option', '1')).
    # This list comprehension collapses them into a single string like 'chosen_option_1'.
    df_wide.columns = [f"{col[0]}_{col[1]}" for col in df_wide.columns]

    # 4. Reset the index to turn 'run_id' back into a standard column
    df_wide = df_wide.reset_index()

    # 5. (Optional) Inspect the newly transformed DataFrame and save it

    # df_wide_call4 = df_wide.merge(df_call4_filtered.drop_duplicates(subset = ["run_id"])[["run_id", "persona"]], on = "run_id", how="inner")

    df_wide_call4 = df_wide
    print(len(df_wide))
    df_wide_call4.head()
    return (df_wide_call4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Call 2 merge
    """)
    return


@app.cell
def _(Path, pd):
    path_call2_dataset = Path("data/processed/validations/task-2/call2_dataset.csv").resolve()
    df_call2 = pd.read_csv(path_call2_dataset)
    df_call2.head()
    return (df_call2,)


@app.cell
def _(df_call2):
    df_call2.dtypes
    return


@app.cell
def _(df_call2):
    df_call2["item number"].astype(str).value_counts(normalize=False)# .to_csv(path_call2_dataset.parent / "unique_values_items.csv")
    return


@app.cell
def _(df_call2, pd):
    # Assuming your dataframe is named df_call2
    # Step 1: Extract the first consecutive sequence of digits from the string
    extracted_numbers = df_call2['item number'].astype(str).str.extract(r'(\d+)', expand=False)

    # Step 2: Convert to numeric. 
    # We use the nullable integer type 'Int64' so that if there are missing values (NaN), 
    # the column remains an integer type rather than converting entirely to floats.
    df_call2['item number clean'] = pd.to_numeric(extracted_numbers, errors='coerce').astype('Int64')

    # Step 3: Drop the infrequent/garbage string values that did not contain any numbers
    df_call2_cleaned = df_call2.dropna(subset=['item number clean']).copy()

    # Step 4 (Optional): Overwrite the old column and drop the temporary one
    df_call2_cleaned['item number'] = df_call2_cleaned['item number clean']
    range_filter = df_call2_cleaned['item number'].isin(range(1,29))
    df_call2_cleaned = df_call2_cleaned[range_filter]
    df_call2_cleaned = df_call2_cleaned.drop(columns=['item number clean'])
    print(len(df_call2_cleaned))
    df_call2_cleaned.head()
    return (df_call2_cleaned,)


@app.cell
def _():
    return


@app.cell
def _(df_call2_cleaned):
    df_call2_cleaned["item number"].astype(str).value_counts(normalize=False)# .to_csv(path_call2_dataset.parent / "unique_values_items.csv")
    return


@app.cell
def _(df_call2_cleaned):
    # 1. Pivot the dataframe
    # Update 'values' to include both the 'rating' and 'why' columns
    df_wide_call2 = df_call2_cleaned.pivot(
        index=['run_id', 'Model'], 
        columns=['scale', 'item number'], 
        values=['rating', 'why']
    )

    # 2. Flatten the MultiIndex columns
    # The columns are now 3-tuples: (value_name, scale, item_number)
    # We join them to create column names like 'rating_maes_1' and 'why_maes_1'
    df_wide_call2.columns = [f"{val}_{scale}_{item}" for val, scale, item in df_wide_call2.columns]

    # 3. Reset the index
    # This pulls 'run_id' and 'Model' out of the index and back into standard columns.
    df_wide_call2 = df_wide_call2.reset_index()

    # 4. Verify the transformation
    df_wide_call2.head()
    return (df_wide_call2,)


@app.cell
def _(df_wide_call2):
    df_wide_call2.isna().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Call 1 check
    """)
    return


@app.cell
def _(Path, pd):
    path_call1_dataset = Path("data/processed/validations/task-1/tfmn_dataset.csv").resolve()
    call1_df = pd.read_csv(path_call1_dataset)
    call1_df = call1_df.rename({"question_number":"question_text"}, axis = 1)
    call1_df.head()
    return call1_df, path_call1_dataset


@app.cell
def _(call1_df):
    from mathanx.constants import MAPPING_CALL1_QUESTIONS

    call1_df["question_number"] = call1_df["question_text"].map(MAPPING_CALL1_QUESTIONS)
    call1_df.head()
    return


@app.cell
def _(call1_df):
    call1_df["question_number"].value_counts()
    return


@app.cell
def _():
    # call1_df.to_csv(path_call1_dataset.parent / "call1_intermediate.csv")
    return


@app.cell
def _(call1_df):
    # 1. Drop the 'model_name' column so it is not considered in the pivot
    z_score_cols = [x for x in call1_df if "z_scores" in x]
    df_subset = call1_df.drop(columns=['model_name'] + z_score_cols)

    # 2. Pivot the dataframe
    # 'run_id' ensures one row per run.
    # 'question_number' provides the headers for the new columns.
    df_wide_call1 = df_subset.pivot(
        index=['run_id', 'mode'],
        columns='question_number',
        values=df_subset.columns[2:]
    )

    # 3. Flatten the MultiIndex columns to combine the value name and question number
    df_wide_call1.columns = [f"{col[0]}_{col[1]}" for col in df_wide_call1.columns]

    # 4. Reset the index to turn 'run_id' back into a standard column
    question_number_cols = [x for x in df_wide_call1.columns if "question_number" in x]
    df_wide_call1 = df_wide_call1.reset_index().drop(columns = question_number_cols)

    # 5. (Optional) Verify the output
    print(len(df_wide_call1))
    df_wide_call1.head()
    return (df_wide_call1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # add demographics
    """)
    return


@app.cell
def _(Path, pd):
    path_persona_dataset = Path("data/processed/demographics/persona_dataset.csv").resolve()
    df_persona = pd.read_csv(path_persona_dataset)
    return (df_persona,)


@app.cell
def _(df_persona):
    print(len(df_persona))
    df_persona_renamed = df_persona.rename(
        {
            "city_of_living":"city",
            "education_level":"education",        
        },
        axis=1
    ).drop(columns=["mode", "model_name"])
    df_persona_renamed.head()
    return (df_persona_renamed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # final dataset merge
    """)
    return


@app.cell
def _(df_wide_call2):
    df_wide_call2["Model"].unique()
    return


@app.cell
def _(df_persona_renamed, df_wide_call1, df_wide_call2, df_wide_call4):
    from mathanx.constants import FOLDER_NAME_MAPPING
    final_merged = df_wide_call1 \
        .merge(df_wide_call2, how="inner", on="run_id") \
        .merge(df_wide_call4, how="inner", on="run_id") \
        .merge(df_persona_renamed, how="left", on="run_id")

    final_merged["Model"] = final_merged["Model"].map(FOLDER_NAME_MAPPING)
    return (final_merged,)


@app.cell
def _(final_merged):
    final_merged.head()
    return


@app.cell
def _(final_merged):
    len(final_merged)
    return


@app.cell
def _(final_merged):
    final_merged.isna().sum()
    return


@app.cell
def _(final_merged):
    final_merged["Model"].unique()
    return


@app.cell
def _(final_merged, path_call1_dataset):
    final_merged.to_parquet(path_call1_dataset.parent.parent / "df_pooling_system.parquet")
    return


@app.cell
def _(path_call1_dataset, pd):
    df_test = pd.read_parquet(path_call1_dataset.parent.parent / "df_pooling_system.parquet")
    df_test.head()
    return


if __name__ == "__main__":
    app.run()
