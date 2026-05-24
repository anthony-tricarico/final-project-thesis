import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from collections import Counter
    from pathlib import Path
    from typing import Literal

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import shap
    from scipy.cluster.hierarchy import linkage, leaves_list
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from mathanx.ml.config import TARGET, DATASET_PATH, FIG_PATH, LEAKAGE_COLS, SHAP_SAMPLE_SIZE, TREE_MODEL_NAMES

    # Set theme for plots
    sns.set_theme(style="whitegrid", context="notebook")
    return (
        DATASET_PATH,
        FIG_PATH,
        KFold,
        LEAKAGE_COLS,
        Literal,
        Path,
        Pipeline,
        SHAP_SAMPLE_SIZE,
        StandardScaler,
        TARGET,
        TREE_MODEL_NAMES,
        leaves_list,
        linkage,
        mo,
        np,
        pd,
        plt,
        sns,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ML Models Exploration

    This notebook ranks candidate predictors for the `accuracy` target, compares tuned models with nested CV, and adds SHAP-based explainability.
    """)
    return


@app.cell
def _(DATASET_PATH, mo, pd):
    # Issues error in case the dataset is not found.
    if not DATASET_PATH.exists():
        mo.callout(
            mo.md(
                f"The engineered dataset was not found at `{DATASET_PATH}`. Run `ml_dataset_creation.py` first so it can materialize the modeling table."
            ),
            kind="warn",
        )
        raise FileNotFoundError(DATASET_PATH)

    # Import the dataset
    ml_df = pd.read_csv(DATASET_PATH)
    ml_df.head()
    return (ml_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Data sanity checks

    Start by verifying the dataset shape, target distribution, duplicate identifiers, and missing values.
    """)
    return


@app.cell
def _(ml_df):
    missing_summary = (
        ml_df.isna() # Create boolean series, True where missing data is detected
        .sum() # Sum all instances where True = 1 and False = 0, determine total number of missing data
        .rename("missing_count") # Rename the series
        .to_frame() # Coerce to DataFrame
        .assign(missing_rate=lambda df: df["missing_count"] / len(ml_df)) # Create new column containing the ratio of missing entries
        .query("missing_count > 0") # Filter to keep all rows that have at least 1 missing entry
        .sort_values(["missing_count", "missing_rate"], ascending=False) # Sort values in descending order to identify most critical cols
    )
    missing_summary
    return


@app.cell
def _(TARGET, ml_df):
    # Descriptive statistics of the dataset
    target_summary = ml_df[TARGET].describe().to_frame().T
    target_summary["skew"] = ml_df[TARGET].skew()
    target_summary
    return


@app.cell
def _(TARGET, ml_df, pd):
    # Check run ids are unique
    duplicate_run_ids = int(ml_df["run_id"].duplicated().sum())

    sanity_checks = pd.DataFrame(
        {
            "value": [
                len(ml_df),
                ml_df.shape[1],
                duplicate_run_ids,
                ml_df[TARGET].nunique(),
            ]
        },
        index=["rows", "columns", "duplicate_run_id_count", "target_unique_values"],
    )
    sanity_checks
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Feature inventory

    Build a compact schema map before ranking predictors.
    """)
    return


@app.cell
def _(LEAKAGE_COLS, ml_df, pd):
    # Set of names of the features that should be treated as nominal.
    nominal_feature_names = {
        "gender",
        "sexual_orientation",
        "city_of_living",
        "employment_status",
        "marital_status",
        "migration_status",
        "religious_beliefs",
        "Model",
    }

    def infer_feature_type(name: str, series: pd.Series) -> str:
        """Categorize the type of the different features to build a feature inventory"""
        values = series.dropna().unique()
        if name in {"run_id"}:
            return "identifier"
        if name in LEAKAGE_COLS:
            return "target_or_leakage"
        if name in nominal_feature_names:
            return "nominal_raw"
        if pd.api.types.is_bool_dtype(series) or (series.dropna().isin([0, 1]).all() and len(values) <= 2):
            return "binary"
        if name.endswith("_ord"):
            return "ordinal"
        if name.endswith("_count") or name.endswith("_mean") or name.endswith("_std") or name.endswith("_min") or name.endswith("_max") or name.endswith("_range"):
            return "continuous"
        if pd.api.types.is_numeric_dtype(series):
            return "continuous"
        return "other"

    inventory_rows = []
    for _column in ml_df.columns:
        inventory_rows.append(
            {
                "feature": _column,
                "type": infer_feature_type(_column, ml_df[_column]),
                "dtype": str(ml_df[_column].dtype),
                "unique_count": int(ml_df[_column].nunique(dropna=True)),
                "missing_rate": float(ml_df[_column].isna().mean()),
                "example_values": ", ".join(map(str, ml_df[_column].dropna().astype(str).head(3).tolist())),
            }
        )

    feature_inventory_df = pd.DataFrame(inventory_rows).sort_values(["type", "unique_count", "feature"])
    feature_type_counts = feature_inventory_df["type"].value_counts().rename_axis("type").reset_index(name="count")
    return feature_inventory_df, feature_type_counts


@app.cell
def _(feature_inventory_df):
    feature_inventory_df
    return


@app.cell
def _():
    # print the feature inventory as latex table
    # print(feature_inventory_df[~(feature_inventory_df["type"].isin(["target_or_leakage", "identifier"]))].drop(columns=["example_values"]).to_latex(index=False))
    return


@app.cell
def _(feature_type_counts):
    import warnings
    # Ignore int as column name warning
    warnings.filterwarnings('ignore')

    feature_type_counts.T
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Leakage-safe feature split

    Keep `accuracy` as the target and exclude direct outcome columns from the predictor matrix.
    """)
    return


@app.cell
def _(LEAKAGE_COLS, TARGET, ml_df, pd):
    from mathanx.ml.helpers import classify_columns

    # Filter the features to include only those that do not introduce leakage or are the target themselves
    feature_cols = [c for c in ml_df.columns if c not in LEAKAGE_COLS and c not in {TARGET, "education_vs_parent_mean_gap"}]

    # Check which features will be used by the ML models
    print("Included features:\n", feature_cols)

    # Create copy of the data
    X = ml_df.loc[:, feature_cols].copy()
    y = ml_df[TARGET].copy()

    # Classify columns into nominal, binary, numeric, etc.
    col_types = classify_columns(X)
    nominal_features = col_types["nominal_features"]
    binary_features = col_types["binary_features"]
    numeric_features = col_types["numeric_features"]
    ordinal_features = col_types["ordinal_features"]
    continuous_features = col_types["continuous_features"]
    tree_nominal_features = col_types["tree_nominal_features"]

    # Check how many features belong to each of the three main categories
    feature_split_df = pd.DataFrame(
        {
            "group": ["continuous_or_ordinal", "binary_dummies", "nominal_raw", "all_predictors"],
            "count": [len(numeric_features), len(binary_features), len(nominal_features), len(feature_cols)],
        }
    )
    return (
        X,
        continuous_features,
        feature_split_df,
        nominal_features,
        numeric_features,
        tree_nominal_features,
        y,
    )


@app.cell
def _(feature_split_df):
    feature_split_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Target screening for numeric and ordinal features

    Rank predictors by Pearson and Spearman association with `accuracy`.
    """)
    return


@app.cell
def _():
    # def score_numeric_feature(series: pd.Series, target: pd.Series) -> Dict[str, float]:
    #     """
    #     This function computes different association measures for each of the numeric variables employed in
    #     the analysis.
    #     """
    #     valid = series.notna() & target.notna()
    #     x = series.loc[valid]
    #     t = target.loc[valid]
    #     if x.nunique() < 2:
    #         return {"pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan}
    #     pearson_r_value, pearson_p_value = pearsonr(x, t)
    #     spearman_r_value, spearman_p_value = spearmanr(x, t)
    #     return {
    #         "pearson_r": pearson_r_value,
    #         "pearson_p": pearson_p_value,
    #         "spearman_r": spearman_r_value,
    #         "spearman_p": spearman_p_value,
    #     }

    # numeric_rows = []
    # for column in [c for c in X.columns if c not in X.columns[X.dtypes.eq("object")]]:
    #     stats = score_numeric_feature(X[column], y)
    #     numeric_rows.append({"feature": column, **stats, "abs_spearman_r": abs(stats["spearman_r"]) if pd.notna(stats["spearman_r"]) else np.nan})

    # numeric_screen_df = pd.DataFrame(numeric_rows).sort_values("abs_spearman_r", ascending=False)
    # top_numeric_screen_df = numeric_screen_df.head(20)
    # top_numeric_screen_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first features that are highly correlated with the accuracy are mainly the model. Importantly, the ranking and the sign of the correlation is coherent with what already shown in the paper. For instance, it was noticed how `Grok 4.1 Fast (Reasoning)` was the best model in terms of accuracy while `Ministral 3B` was the worst one.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Target screening for binary and one-hot features

    For binary predictors, compare the target mean when the feature is `1` vs `0`.
    """)
    return


@app.cell
def _(X, np, pd, y):
    binary_rows = []
    overall_mean = float(y.mean())

    # Isolate binary columns (those containing only 0 and 1 values)
    for _column in [c for c in X.columns if set(X[c].dropna().unique()).issubset({0, 1})]:
        # Create boolean masks to filter between observations with either 1 or 0
        mask_one = X[_column] == 1
        mask_zero = X[_column] == 0
        # Compute the mean of the two groups for the target variable
        mean_one = y.loc[mask_one].mean() if mask_one.any() else np.nan
        mean_zero = y.loc[mask_zero].mean() if mask_zero.any() else np.nan
        delta = mean_one - mean_zero if pd.notna(mean_one) and pd.notna(mean_zero) else np.nan
        binary_rows.append(
            {
                "feature": _column,
                "mean_target_if_1": mean_one,
                "mean_target_if_0": mean_zero,
                "delta": delta,
                "abs_delta": abs(delta) if pd.notna(delta) else np.nan,
                # Support measures how many observations were used to compute the mean
                "support_1": int(mask_one.sum()),
                "support_0": int(mask_zero.sum()),
            }
        )

    binary_screen_df = pd.DataFrame(binary_rows).sort_values("abs_delta", ascending=False)
    top_binary_screen_df = binary_screen_df
    overall_target_mean = overall_mean
    top_binary_screen_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    No significant differences surface across the different groups as represented by the different dummy variables created.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Clustered correlation map

    Use a clustered heatmap for the numeric/ordinal subset only so the plot stays readable.
    """)
    return


@app.cell
def _(X, continuous_features, leaves_list, linkage, plt, sns):
    corr_features = [c for c in continuous_features if X[c].nunique(dropna=True) > 1]
    corr_df = X.loc[:, corr_features].corr(numeric_only=True)
    corr_plot_df = corr_df.fillna(0)

    if corr_df.shape[0] > 1:
        linkage_matrix = linkage(corr_plot_df, method="average")
        ordered_idx = leaves_list(linkage_matrix)
        ordered_features = corr_df.index[ordered_idx].tolist()
        corr_df_ordered = corr_plot_df.loc[ordered_features, ordered_features]
        corr_grid = sns.clustermap(
            corr_df_ordered,
            cmap="vlag",
            center=0,
            linewidths=0.0,
            figsize=(12, 10),
            xticklabels=True,
            yticklabels=True,
        )
        corr_grid.fig.suptitle("Clustered correlation heatmap", y=1.02)
    else:
        corr_grid = None

    # figs_path = Path("figures/").resolve().absolute()
    # plt.savefig(figs_path / "clustered_corr_heatmap.png")
    plt.show()
    return (corr_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The correlogram above shows some clear structures in the data. First, the different anxiety scores, as measured by the AMAS and MSEAQ anxiety (sub)scales exhibit a correlation of $\approx 0.5$ and tend to cluster together.

    Moreover, the MSES (maes) and the MSEAQ Self-Efficacy (sub)scales cluster together and also exhibit a good degree of association.

    Finally, we also notice that both the MSES (maes) and the MSEAQ Self-Efficacy (sub)scales are negatively correlated to the MSEAQ anxiety subscale.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Collinearity check

    Flag highly correlated feature pairs that may be redundant in linear models.
    """)
    return


@app.cell
def _(corr_df, pd):
    if corr_df.shape[0] > 1:
        corr_abs = corr_df.abs()
        high_pairs = []
        cols = corr_abs.columns.tolist()
        for i, left in enumerate(cols):
            for right in cols[i + 1 :]:
                value = corr_df.loc[left, right]
                if pd.notna(value) and abs(value) >= 0.8:
                    high_pairs.append({"feature_1": left, "feature_2": right, "correlation": value})
        if high_pairs:
            high_corr_pairs_df = pd.DataFrame(high_pairs).sort_values("correlation", key=lambda s: s.abs(), ascending=False)
        else:
            high_corr_pairs_df = pd.DataFrame(columns=["feature_1", "feature_2", "correlation"])
    else:
        high_corr_pairs_df = pd.DataFrame(columns=["feature_1", "feature_2", "correlation"])

    high_corr_pairs_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Nested cross-validation

    Tune the candidate models with an inner CV loop and estimate the generalization error with an outer CV loop.
    """)
    return


@app.cell
def _(nominal_features, numeric_features, tree_nominal_features):
    print("Numeric features common to both models:", numeric_features)
    print("Tree nominal features:", tree_nominal_features)
    print("Nominal features:", nominal_features)
    return


@app.cell
def _(
    X,
    nominal_features,
    numeric_features,
    pd,
    tree_nominal_features,
    y,
):
    from mathanx.ml.config import RANDOM_STATE
    from mathanx.ml.helpers import (
        build_linear_pipeline as _build_linear_pipeline,
        build_tree_pipeline as _build_tree_pipeline,
        make_model_specs as _make_model_specs,
        run_experiment as _run_experiment,
    )

    _build_linear = lambda m: _build_linear_pipeline(m, numeric_features, nominal_features)
    _build_tree = lambda m: _build_tree_pipeline(m, numeric_features, tree_nominal_features)
    model_specs = _make_model_specs(_build_linear, _build_tree, random_state=RANDOM_STATE)

    _result = _run_experiment(X, y, model_specs, random_state=RANDOM_STATE)

    model_summary_df = _result.model_summary
    best_params_by_model = _result.best_params_by_model
    best_model_name = _result.best_model_name
    best_model_pipeline = _result.final_estimator
    tuned_cv_results_df = _result.tuned_cv_results
    permutation_importance_df = _result.permutation_importance
    top_permutation_importance_df = permutation_importance_df.head(20)
    best_params_summary_df = pd.DataFrame(_result.best_params_summary_rows).sort_values("model")

    return (
        RANDOM_STATE,
        best_model_name,
        best_params_by_model,
        best_params_summary_df,
        best_model_pipeline,
        model_specs,
        model_summary_df,
        permutation_importance_df,
        top_permutation_importance_df,
        tuned_cv_results_df,
    )


@app.cell(hide_code=True)
def _():
    return


@app.cell
def _(Path, best_params_summary_df, pd):
    path_best_params = Path("data/processed/ml/best_params_by_model.csv").resolve().absolute()
    # best_params_summary_df.to_csv(path_best_params)

    try:
        print(best_params_summary_df)
    except Exception:
        _best_params_summary_df = pd.read_csv(path_best_params)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Post-tuning CV and importance

    Re-evaluate the tuned models with a second CV pass, then inspect predictor importance for the selected best model.
    """)
    return


@app.cell(hide_code=True)
def _():
    return


@app.cell
def _(top_permutation_importance_df):
    top_permutation_importance_df
    return


@app.cell
def _(model_summary_df):
    model_summary_df.head()
    return


@app.cell
def _(mo):
    METRIC = mo.ui.radio(options=["mean_rmse", "mean_mae", "mean_r2"], 
                               label="Select the metric of interest:",
                        value="mean_r2")
    METRIC
    return (METRIC,)


@app.cell
def _(METRIC, model_summary_df, plt, sns):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=model_summary_df, x=METRIC.value, y="model", ax=ax, color="#648fff")
    ax.set_title(f"Nested CV {METRIC.value} by model")
    ax.set_xlabel(f"{METRIC.value.capitalize().replace("_", " ")}")
    ax.set_ylabel("Model")
    plt.tight_layout()
    fig
    return


@app.cell
def _(model_summary_df, tuned_cv_results_df):
    model_comparison_df = model_summary_df.merge(
        tuned_cv_results_df[["model", "mean_rmse", "mean_mae", "mean_r2"]],
        on="model",
        suffixes=("_nested", "_tuned"),
    )
    model_comparison_df
    return


@app.cell
def _(plt, sns, top_permutation_importance_df):
    _fig, _ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=top_permutation_importance_df, x="importance_mean", y="feature", ax=_ax, color="#fe6100")
    _ax.set_title("Top permutation importances for best tuned model")
    _ax.set_xlabel("Importance")
    _ax.set_ylabel("Feature")
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. SHAP explainability

    Use SHAP beeswarm and global importance plots on the best tree-based model, or on the best tree fallback when the overall winner is linear.
    """)
    return


@app.cell
def _(SHAP_SAMPLE_SIZE, TREE_MODEL_NAMES):
    SHAP_RANDOM_STATE = 42
    return SHAP_RANDOM_STATE, SHAP_SAMPLE_SIZE, TREE_MODEL_NAMES


@app.cell
def _(
    TREE_MODEL_NAMES,
    best_model_name,
    best_params_by_model,
    model_specs,
    tuned_cv_results_df,
    y,
):
    from mathanx.ml.helpers import run_shap_analysis as _run_shap_analysis

    shap_values, shap_model_name = _run_shap_analysis(
        X, y, model_specs, best_model_name, best_params_by_model,
        tuned_cv_results_df, TREE_MODEL_NAMES,
        shap_sample_size=SHAP_SAMPLE_SIZE, shap_random_state=SHAP_RANDOM_STATE,
    )
    return shap_values, shap_model_name


@app.cell
def _(shap_model_name, shap_values):
    from mathanx.ml.helpers import plot_shap_beeswarm as _plot_shap_beeswarm

    _plot_shap_beeswarm(shap_values, f"SHAP beeswarm for {shap_model_name}")
    return


@app.cell
def _(shap_model_name, shap_values):
    from mathanx.ml.helpers import plot_shap_bar as _plot_shap_bar

    _plot_shap_bar(shap_values, f"Global SHAP importance for {shap_model_name}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Without Model feature

    Since the model feature seems to be explaining a lot of the variance, leaving almost no explanatory power to all other features, we can try to refit the models using the same methodology but excluding it from the pool of features.
    """)
    return


@app.cell
def _(RANDOM_STATE, nominal_features, numeric_features, tree_nominal_features):
    from mathanx.ml.helpers import (
        build_linear_pipeline as _build_linear_pipeline,
        build_tree_pipeline as _build_tree_pipeline,
        make_model_specs as _make_model_specs,
    )

    # Here is where Model is excluded from the set of features used to train the model
    nominal_features_no_model = [c for c in nominal_features if c != "Model"]
    tree_nominal_features_no_model = [c for c in tree_nominal_features if c != "Model"]
    numeric_features_no_model = [c for c in numeric_features if c != "Model"]

    _build_linear_no_model = lambda m: _build_linear_pipeline(m, numeric_features_no_model, nominal_features_no_model)
    _build_tree_no_model = lambda m: _build_tree_pipeline(m, numeric_features_no_model, tree_nominal_features_no_model)
    model_specs_no_model = _make_model_specs(_build_linear_no_model, _build_tree_no_model, random_state=RANDOM_STATE)
    return (model_specs_no_model,)


@app.cell
def _(
    X,
    model_specs_no_model,
    y,
):
    from mathanx.ml.config import RANDOM_STATE as RANDOM_STATE_NO_MODEL
    from mathanx.ml.helpers import run_experiment as _run_experiment_no_model

    _result_no_model = _run_experiment_no_model(X, y, model_specs_no_model, random_state=RANDOM_STATE_NO_MODEL)

    model_summary_df_no_model = _result_no_model.model_summary
    best_params_by_model_no_model = _result_no_model.best_params_by_model
    best_model_name_no_model = _result_no_model.best_model_name
    permutation_importance_df_no_model = _result_no_model.permutation_importance

    return (
        best_model_name_no_model,
        best_params_by_model_no_model,
        model_summary_df_no_model,
        permutation_importance_df_no_model,
    )


@app.cell(hide_code=True)
def _():
    return


@app.cell
def _(permutation_importance_df_no_model):
    print(permutation_importance_df_no_model)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. SHAP explainability without `Model`

    Use SHAP beeswarm and global importance plots for the best tree-based no-`Model` refit, or the best tree fallback when the winner is linear.
    """)
    return


@app.cell
def _(SHAP_SAMPLE_SIZE, TREE_MODEL_NAMES):
    SHAP_SAMPLE_SIZE_NO_MODEL = SHAP_SAMPLE_SIZE
    SHAP_RANDOM_STATE_NO_MODEL = 42
    TREE_MODEL_NAMES_NO_MODEL = TREE_MODEL_NAMES
    return (
        SHAP_RANDOM_STATE_NO_MODEL,
        SHAP_SAMPLE_SIZE_NO_MODEL,
        TREE_MODEL_NAMES_NO_MODEL,
    )


@app.cell
def _(
    SHAP_RANDOM_STATE_NO_MODEL,
    SHAP_SAMPLE_SIZE_NO_MODEL,
    TREE_MODEL_NAMES_NO_MODEL,
    X,
    best_model_name_no_model,
    best_params_by_model_no_model,
    model_specs_no_model,
    model_summary_df_no_model,
    y,
):
    from mathanx.ml.helpers import run_shap_analysis as _run_shap_analysis

    shap_values_no_model, shap_model_name_no_model = _run_shap_analysis(
        X, y, model_specs_no_model, best_model_name_no_model,
        best_params_by_model_no_model, model_summary_df_no_model,
        TREE_MODEL_NAMES_NO_MODEL,
        shap_sample_size=SHAP_SAMPLE_SIZE_NO_MODEL,
        shap_random_state=SHAP_RANDOM_STATE_NO_MODEL,
    )
    return shap_values_no_model, shap_model_name_no_model


@app.cell
def _(FIG_PATH, shap_model_name_no_model, shap_values_no_model):
    from mathanx.ml.helpers import plot_shap_beeswarm as _plot_shap_beeswarm

    _fig = _plot_shap_beeswarm(shap_values_no_model, f"SHAP beeswarm for {shap_model_name_no_model} (without Model)")
    _fig.savefig(FIG_PATH / "beeswarm_no_model.pdf", format="pdf")
    return


@app.cell
def _(shap_model_name_no_model, shap_values_no_model):
    from mathanx.ml.helpers import plot_shap_bar as _plot_shap_bar

    _plot_shap_bar(shap_values_no_model, f"Global SHAP importance for {shap_model_name_no_model} (without Model)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. Nested CV with five predictors

    Re-run the nested cross-validation using only `mseaq_anx`, `amas_score`, `maes_score`, `mseaq_se`, and `confidence_scaled`.
    """)
    return


@app.cell
def _(
    Pipeline,
    StandardScaler,
    TARGET,
    ml_df,
    pd,
):
    from mathanx.ml.config import RANDOM_STATE as RANDOM_STATE_FIVE
    from mathanx.ml.helpers import make_model_specs as _make_model_specs_five, run_experiment as _run_experiment_five

    five_feature_columns = ["mseaq_anx", "amas_score", "maes_score", "mseaq_se", "confidence_scaled"]
    X_five = ml_df.loc[:, five_feature_columns].copy()
    y_five = ml_df[TARGET].copy()

    build_linear_five = lambda m: Pipeline([("scale", StandardScaler()), ("model", m)])
    build_tree_five = lambda m: Pipeline([("model", m)])
    model_specs_five = _make_model_specs_five(build_linear_five, build_tree_five, random_state=RANDOM_STATE_FIVE)

    _result_five = _run_experiment_five(X_five, y_five, model_specs_five, random_state=RANDOM_STATE_FIVE)

    model_summary_df_five = _result_five.model_summary
    best_params_by_model_five = _result_five.best_params_by_model
    best_model_name_five = _result_five.best_model_name
    best_model_pipeline_five = _result_five.final_estimator
    best_params_summary_df_five = pd.DataFrame(_result_five.best_params_summary_rows).sort_values("model")
    permutation_importance_df_five = _result_five.permutation_importance
    return (
        X_five,
        best_model_name_five,
        best_model_pipeline_five,
        best_params_summary_df_five,
        model_summary_df_five,
        permutation_importance_df_five,
        y_five,
    )


@app.cell
def _(model_summary_df_five):
    model_summary_df_five
    return


@app.cell
def _(best_params_summary_df_five):
    best_params_summary_df_five
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 13. SHAP explainability for five predictors

    Explain the best five-variable model with SHAP, using the transformed matrix for linear models and the raw matrix for tree models.
    """)
    return


@app.cell
def _(SHAP_SAMPLE_SIZE, TREE_MODEL_NAMES):
    SHAP_SAMPLE_SIZE_FIVE = SHAP_SAMPLE_SIZE
    SHAP_RANDOM_STATE_FIVE = 42
    TREE_MODEL_NAMES_FIVE = TREE_MODEL_NAMES
    return SHAP_RANDOM_STATE_FIVE, SHAP_SAMPLE_SIZE_FIVE, TREE_MODEL_NAMES_FIVE


@app.cell
def _(
    SHAP_RANDOM_STATE_FIVE,
    SHAP_SAMPLE_SIZE_FIVE,
    TREE_MODEL_NAMES_FIVE,
    X_five,
    best_model_name_five,
    best_model_pipeline_five,
    model_summary_df_five,
    y_five,
):
    from mathanx.ml.helpers import run_shap_analysis as _run_shap_analysis

    shap_values_five, shap_model_name_five = _run_shap_analysis(
        X_five, y_five, {}, best_model_name_five, {},
        model_summary_df_five, TREE_MODEL_NAMES_FIVE,
        pipeline=best_model_pipeline_five,
        shap_sample_size=SHAP_SAMPLE_SIZE_FIVE,
        shap_random_state=SHAP_RANDOM_STATE_FIVE,
        check_linear=True,
    )
    return shap_values_five, shap_model_name_five


@app.cell
def _(shap_model_name_five, shap_values_five):
    from mathanx.ml.helpers import plot_shap_beeswarm as _plot_shap_beeswarm

    _plot_shap_beeswarm(shap_values_five, f"SHAP beeswarm for {shap_model_name_five} (selected predictors)")
    return


@app.cell
def _(shap_model_name_five, shap_values_five):
    from mathanx.ml.helpers import plot_shap_bar as _plot_shap_bar

    _plot_shap_bar(shap_values_five, f"Global SHAP importance for {shap_model_name_five} (selected predictors)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 14. Experiment comparison

    Compare best models, parameters, and top permutation importances across all three feature sets.
    """)
    return


@app.cell
def _(
    best_model_name,
    best_model_name_five,
    best_model_name_no_model,
    model_summary_df,
    model_summary_df_five,
    model_summary_df_no_model,
    permutation_importance_df_five,
    permutation_importance_df_no_model,
    pd,
    top_permutation_importance_df,
):
    sections = []

    for label, best, summ, top_perm in [
        ("All features", best_model_name, model_summary_df, top_permutation_importance_df),
        ("Without Model", best_model_name_no_model, model_summary_df_no_model, permutation_importance_df_no_model),
        ("Five predictors", best_model_name_five, model_summary_df_five, permutation_importance_df_five),
    ]:
        row = summ[summ["model"] == best]
        top3 = top_perm.head(3)["feature"].tolist() if top_perm is not None and len(top_perm) > 0 else []
        sections.append(
            {
                "Feature set": label,
                "Best model": best,
                "Nested CV R²": f'{row.iloc[0]["mean_r2"]:.4f}' if len(row) > 0 else "",
                "Nested CV RMSE": f'{row.iloc[0]["mean_rmse"]:.4f}' if len(row) > 0 else "",
                "Top 3 features": ", ".join(top3),
            }
        )

    experiment_comparison_df = pd.DataFrame(sections)
    experiment_comparison_df
    return


if __name__ == "__main__":
    app.run()
