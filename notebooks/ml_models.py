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
        LEAKAGE_COLS,
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
        shap,
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
def _(X, nominal_features, numeric_features, pd, tree_nominal_features, y):
    from pathlib import Path as _Path

    from mathanx.ml.config import RANDOM_STATE
    from mathanx.ml.helpers import (
        build_linear_pipeline as _build_linear_pipeline,
        build_tree_pipeline as _build_tree_pipeline,
        load_experiment as _load_experiment,
        make_model_specs as _make_model_specs,
        run_experiment as _run_experiment,
    )

    _build_linear = lambda m: _build_linear_pipeline(m, numeric_features, nominal_features)
    _build_tree = lambda m: _build_tree_pipeline(m, numeric_features, tree_nominal_features)
    model_specs = _make_model_specs(_build_linear, _build_tree, random_state=RANDOM_STATE)

    _cache_dir = _Path("models/all_features")
    if _cache_dir.exists():
        _result = _load_experiment(_cache_dir)
    else:
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
        model_specs,
        model_summary_df,
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
def _():
    SHAP_RANDOM_STATE = 42
    return (SHAP_RANDOM_STATE,)


@app.cell
def _(
    SHAP_RANDOM_STATE,
    SHAP_SAMPLE_SIZE,
    TREE_MODEL_NAMES,
    X,
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
    return shap_model_name, shap_values


@app.cell
def _(shap_values):
    shap_values.feature_names
    return


@app.cell
def _(FIG_PATH, shap, shap_model_name, shap_values):
    from mathanx.ml.helpers import plot_shap_beeswarm as _plot_shap_beeswarm

    indices = [i for i, name in enumerate(shap_values.feature_names) if name != "Model"]
    shap_values_filtered = shap.Explanation(
        values=shap_values.values[:, indices],
        base_values=shap_values.base_values,
        data=shap_values.data[:, indices] if shap_values.data is not None else None,
        feature_names=[shap_values.feature_names[i] for i in indices],
    )

    _fig = _plot_shap_beeswarm(shap_values_filtered, f"SHAP beeswarm for {shap_model_name} (All predictors, model not plotted)")

    _fig.savefig(FIG_PATH / "beeswarm_all_predictors_model_excluded.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "beeswarm_all_predictors_model_excluded.png", format="png")
    _fig
    return (shap_values_filtered,)


@app.cell
def _(shap_model_name, shap_values_filtered):
    from mathanx.ml.helpers import plot_shap_bar as _plot_shap_bar

    _plot_shap_bar(shap_values_filtered, f"Global SHAP importance for {shap_model_name} (without Model)")
    return


@app.cell
def _(FIG_PATH, shap_model_name, shap_values):
    from mathanx.ml.helpers import plot_shap_beeswarm as _plot_shap_beeswarm

    _fig = _plot_shap_beeswarm(shap_values, f"SHAP beeswarm for {shap_model_name}")

    _fig.savefig(FIG_PATH / "beeswarm_all_predictors.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "beeswarm_all_predictors.png", format="png")
    _fig
    return


@app.cell
def _(shap_model_name, shap_values):
    from mathanx.ml.helpers import plot_shap_bar as _plot_shap_bar

    _plot_shap_bar(shap_values, f"Global SHAP importance for {shap_model_name}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10b. SHAP explainability for all features + confidence_scaled

    Use SHAP beeswarm and global importance plots on the best tree-based model, or on the best tree fallback when the overall winner is linear.
    """)
    return


@app.cell
def _():
    SHAP_RANDOM_STATE_ALL_FEATURES_CS = 42
    return (SHAP_RANDOM_STATE_ALL_FEATURES_CS,)


@app.cell
def _(LEAKAGE_COLS, RANDOM_STATE, TARGET, ml_df, pd):
    from pathlib import Path as _Path

    from mathanx.ml.helpers import (
        build_linear_pipeline as _build_linear_pipeline,
        build_tree_pipeline as _build_tree_pipeline,
        classify_columns as _classify_columns,
        load_experiment as _load_experiment_all_features_cs,
        make_model_specs as _make_model_specs_all_features_cs,
        run_experiment as _run_experiment_all_features_cs,
    )

    _feature_cols = [
        c for c in ml_df.columns
        if c not in (LEAKAGE_COLS - {"confidence_scaled"})
        and c not in {TARGET, "education_vs_parent_mean_gap"}
    ]
    X_all_features_confidence_scaled = ml_df.loc[:, _feature_cols].copy()
    y_all_features_confidence_scaled = ml_df[TARGET].copy()

    _col_types = _classify_columns(X_all_features_confidence_scaled)
    _numeric_features = _col_types["numeric_features"]
    _nominal_features = _col_types["nominal_features"]
    _tree_nominal_features = _col_types["tree_nominal_features"]

    _build_linear = lambda m: _build_linear_pipeline(m, _numeric_features, _nominal_features)
    _build_tree = lambda m: _build_tree_pipeline(m, _numeric_features, _tree_nominal_features)
    model_specs_all_features_confidence_scaled = _make_model_specs_all_features_cs(
        _build_linear,
        _build_tree,
        random_state=RANDOM_STATE,
    )

    _cache_dir = _Path("models/all_features_with_confidence_scaled")
    if _cache_dir.exists():
        _result = _load_experiment_all_features_cs(_cache_dir)
    else:
        _result = _run_experiment_all_features_cs(
            X_all_features_confidence_scaled,
            y_all_features_confidence_scaled,
            model_specs_all_features_confidence_scaled,
            random_state=RANDOM_STATE,
        )

    model_summary_df_all_features_confidence_scaled = _result.model_summary
    best_params_by_model_all_features_confidence_scaled = _result.best_params_by_model
    best_model_name_all_features_confidence_scaled = _result.best_model_name
    best_model_pipeline_all_features_confidence_scaled = _result.final_estimator
    tuned_cv_results_df_all_features_confidence_scaled = _result.tuned_cv_results
    permutation_importance_df_all_features_confidence_scaled = _result.permutation_importance
    top_permutation_importance_df_all_features_confidence_scaled = permutation_importance_df_all_features_confidence_scaled.head(20)
    best_params_summary_df_all_features_confidence_scaled = pd.DataFrame(_result.best_params_summary_rows).sort_values("model")
    return (
        X_all_features_confidence_scaled,
        best_model_name_all_features_confidence_scaled,
        best_params_by_model_all_features_confidence_scaled,
        best_params_summary_df_all_features_confidence_scaled,
        model_specs_all_features_confidence_scaled,
        model_summary_df_all_features_confidence_scaled,
        tuned_cv_results_df_all_features_confidence_scaled,
        y_all_features_confidence_scaled,
    )


@app.cell
def _(model_summary_df_all_features_confidence_scaled):
    model_summary_df_all_features_confidence_scaled
    return


@app.cell
def _(best_params_summary_df_all_features_confidence_scaled):
    best_params_summary_df_all_features_confidence_scaled
    return


@app.cell
def _(
    SHAP_RANDOM_STATE_ALL_FEATURES_CS,
    SHAP_SAMPLE_SIZE,
    TREE_MODEL_NAMES,
    X_all_features_confidence_scaled,
    best_model_name_all_features_confidence_scaled,
    best_params_by_model_all_features_confidence_scaled,
    model_specs_all_features_confidence_scaled,
    tuned_cv_results_df_all_features_confidence_scaled,
    y_all_features_confidence_scaled,
):
    from mathanx.ml.helpers import run_shap_analysis as _run_shap_analysis

    shap_values_all_features_confidence_scaled, shap_model_name_all_features_confidence_scaled = _run_shap_analysis(
        X_all_features_confidence_scaled,
        y_all_features_confidence_scaled,
        model_specs_all_features_confidence_scaled,
        best_model_name_all_features_confidence_scaled,
        best_params_by_model_all_features_confidence_scaled,
        tuned_cv_results_df_all_features_confidence_scaled,
        TREE_MODEL_NAMES,
        shap_sample_size=SHAP_SAMPLE_SIZE,
        shap_random_state=SHAP_RANDOM_STATE_ALL_FEATURES_CS,
    )
    return (
        shap_model_name_all_features_confidence_scaled,
        shap_values_all_features_confidence_scaled,
    )


@app.cell
def _(shap_values_all_features_confidence_scaled):
    shap_values_all_features_confidence_scaled.feature_names
    return


@app.cell
def _(
    FIG_PATH,
    shap,
    shap_model_name_all_features_confidence_scaled,
    shap_values_all_features_confidence_scaled,
):
    from mathanx.ml.helpers import plot_shap_beeswarm as _plot_shap_beeswarm

    _indices = [
        i
        for i, name in enumerate(shap_values_all_features_confidence_scaled.feature_names)
        if name != "Model"
    ]
    shap_values_filtered_all_features_confidence_scaled = shap.Explanation(
        values=shap_values_all_features_confidence_scaled.values[:, _indices],
        base_values=shap_values_all_features_confidence_scaled.base_values,
        data=(
            shap_values_all_features_confidence_scaled.data[:, _indices]
            if shap_values_all_features_confidence_scaled.data is not None
            else None
        ),
        feature_names=[shap_values_all_features_confidence_scaled.feature_names[i] for i in _indices],
    )

    _fig = _plot_shap_beeswarm(
        shap_values_filtered_all_features_confidence_scaled,
        f"SHAP beeswarm for {shap_model_name_all_features_confidence_scaled} (All predictors + confidence_scaled, model not plotted)",
    )

    _fig.savefig(FIG_PATH / "beeswarm_all_predictors_confidence_scaled_model_excluded.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "beeswarm_all_predictors_confidence_scaled_model_excluded.png", format="png")
    _fig
    return (shap_values_filtered_all_features_confidence_scaled,)


@app.cell
def _(
    shap_model_name_all_features_confidence_scaled,
    shap_values_filtered_all_features_confidence_scaled,
):
    from mathanx.ml.helpers import plot_shap_bar as _plot_shap_bar

    _plot_shap_bar(
        shap_values_filtered_all_features_confidence_scaled,
        f"Global SHAP importance for {shap_model_name_all_features_confidence_scaled} (without Model)",
    )
    return


@app.cell
def _(
    FIG_PATH,
    shap_model_name_all_features_confidence_scaled,
    shap_values_all_features_confidence_scaled,
):
    from mathanx.ml.helpers import plot_shap_beeswarm as _plot_shap_beeswarm

    _fig = _plot_shap_beeswarm(
        shap_values_all_features_confidence_scaled,
        f"SHAP beeswarm for {shap_model_name_all_features_confidence_scaled}",
    )

    _fig.savefig(FIG_PATH / "beeswarm_all_predictors_confidence_scaled.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "beeswarm_all_predictors_confidence_scaled.png", format="png")
    _fig
    return


@app.cell
def _(
    shap_model_name_all_features_confidence_scaled,
    shap_values_all_features_confidence_scaled,
):
    from mathanx.ml.helpers import plot_shap_bar as _plot_shap_bar

    _plot_shap_bar(
        shap_values_all_features_confidence_scaled,
        f"Global SHAP importance for {shap_model_name_all_features_confidence_scaled}",
    )
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
def _(X, model_specs_no_model, y):
    from pathlib import Path as _Path

    from mathanx.ml.config import RANDOM_STATE as RANDOM_STATE_NO_MODEL
    from mathanx.ml.helpers import (
        load_experiment as _load_experiment_no_model,
        run_experiment as _run_experiment_no_model,
    )

    _cache_dir = _Path("models/no_model")
    if _cache_dir.exists():
        _result_no_model = _load_experiment_no_model(_cache_dir)
    else:
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
    return shap_model_name_no_model, shap_values_no_model


@app.cell
def _(FIG_PATH, shap_model_name_no_model, shap_values_no_model):
    from mathanx.ml.helpers import plot_shap_beeswarm as _plot_shap_beeswarm

    _fig = _plot_shap_beeswarm(shap_values_no_model, f"SHAP beeswarm for {shap_model_name_no_model} (without Model)")
    _fig.savefig(FIG_PATH / "beeswarm_no_model.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "beeswarm_no_model.png", format="png")
    _fig
    return


@app.cell
def _(shap_model_name_no_model, shap_values_no_model):
    from mathanx.ml.helpers import plot_shap_bar as _plot_shap_bar

    _plot_shap_bar(shap_values_no_model, f"Global SHAP importance for {shap_model_name_no_model} (without Model)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11b. Mistral family — no Model

    Subset the dataset to only Mistral-family models and run the no-Model experiment to measure explanatory power for this architecture family.
    """)
    return


@app.cell
def _(RANDOM_STATE, X, model_specs_no_model, y):
    from pathlib import Path as _Path

    from mathanx.ml.config import MISTRAL_FAMILY
    from mathanx.ml.helpers import (
        load_experiment as _load_experiment_mistral,
        run_experiment as _run_experiment_mistral,
    )

    mistral_mask = X["Model"].isin(MISTRAL_FAMILY)
    X_mistral = X[mistral_mask].reset_index(drop=True)
    y_mistral = y[mistral_mask].reset_index(drop=True)

    _cache_dir = _Path("models/no_model_mistral_family")
    if _cache_dir.exists():
        _result_mistral = _load_experiment_mistral(_cache_dir)
    else:
        _result_mistral = _run_experiment_mistral(X_mistral, y_mistral, model_specs_no_model, random_state=RANDOM_STATE)

    model_summary_df_mistral = _result_mistral.model_summary
    best_params_by_model_mistral = _result_mistral.best_params_by_model
    best_model_name_mistral = _result_mistral.best_model_name
    permutation_importance_df_mistral = _result_mistral.permutation_importance
    return (
        X_mistral,
        best_model_name_mistral,
        best_params_by_model_mistral,
        model_summary_df_mistral,
        permutation_importance_df_mistral,
        y_mistral,
    )


@app.cell
def _(model_summary_df_mistral):
    model_summary_df_mistral
    return


@app.cell
def _(permutation_importance_df_mistral):
    print(permutation_importance_df_mistral)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11c. SHAP explainability for Mistral family

    Explain the best Mistral-family model with SHAP.
    """)
    return


@app.cell
def _(SHAP_SAMPLE_SIZE, TREE_MODEL_NAMES):
    SHAP_SAMPLE_SIZE_MISTRAL = SHAP_SAMPLE_SIZE
    SHAP_RANDOM_STATE_MISTRAL = 42
    TREE_MODEL_NAMES_MISTRAL = TREE_MODEL_NAMES
    return (
        SHAP_RANDOM_STATE_MISTRAL,
        SHAP_SAMPLE_SIZE_MISTRAL,
        TREE_MODEL_NAMES_MISTRAL,
    )


@app.cell
def _(
    SHAP_RANDOM_STATE_MISTRAL,
    SHAP_SAMPLE_SIZE_MISTRAL,
    TREE_MODEL_NAMES_MISTRAL,
    X_mistral,
    best_model_name_mistral,
    best_params_by_model_mistral,
    model_specs_no_model,
    model_summary_df_mistral,
    y_mistral,
):
    from mathanx.ml.helpers import run_shap_analysis as _run_shap_analysis

    shap_values_mistral, shap_model_name_mistral = _run_shap_analysis(
        X_mistral, y_mistral, model_specs_no_model, best_model_name_mistral,
        best_params_by_model_mistral, model_summary_df_mistral,
        TREE_MODEL_NAMES_MISTRAL,
        shap_sample_size=SHAP_SAMPLE_SIZE_MISTRAL,
        shap_random_state=SHAP_RANDOM_STATE_MISTRAL,
    )
    return shap_model_name_mistral, shap_values_mistral


@app.cell
def _(FIG_PATH, shap_model_name_mistral, shap_values_mistral):
    from mathanx.ml.helpers import plot_shap_beeswarm as _plot_shap_beeswarm

    _fig = _plot_shap_beeswarm(shap_values_mistral, f"SHAP beeswarm for {shap_model_name_mistral} (Mistral family, without Model)")
    _fig.savefig(FIG_PATH / "beeswarm_no_model_mistral_family.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "beeswarm_no_model_mistral_family.png", format="png")
    _fig
    return


@app.cell
def _(shap_model_name_mistral, shap_values_mistral):
    from mathanx.ml.helpers import plot_shap_bar as _plot_shap_bar

    _plot_shap_bar(shap_values_mistral, f"Global SHAP importance for {shap_model_name_mistral} (Mistral family, without Model)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11d. Qwen3 family — no Model

    Subset the dataset to only Qwen3-family models and run the no-Model experiment to measure explanatory power for this architecture family.
    """)
    return


@app.cell
def _(RANDOM_STATE, X, model_specs_no_model, y):
    from pathlib import Path as _Path

    from mathanx.ml.config import QWEN3_FAMILY
    from mathanx.ml.helpers import (
        load_experiment as _load_experiment_qwen3,
        run_experiment as _run_experiment_qwen3,
    )

    qwen3_mask = X["Model"].isin(QWEN3_FAMILY)
    X_qwen3 = X[qwen3_mask].reset_index(drop=True)
    y_qwen3 = y[qwen3_mask].reset_index(drop=True)

    _cache_dir = _Path("models/no_model_qwen3_family")
    if _cache_dir.exists():
        _result_qwen3 = _load_experiment_qwen3(_cache_dir)
    else:
        _result_qwen3 = _run_experiment_qwen3(X_qwen3, y_qwen3, model_specs_no_model, random_state=RANDOM_STATE)

    model_summary_df_qwen3 = _result_qwen3.model_summary
    best_params_by_model_qwen3 = _result_qwen3.best_params_by_model
    best_model_name_qwen3 = _result_qwen3.best_model_name
    permutation_importance_df_qwen3 = _result_qwen3.permutation_importance
    return (
        X_qwen3,
        best_model_name_qwen3,
        best_params_by_model_qwen3,
        model_summary_df_qwen3,
        permutation_importance_df_qwen3,
        y_qwen3,
    )


@app.cell
def _(model_summary_df_qwen3):
    model_summary_df_qwen3
    return


@app.cell
def _(permutation_importance_df_qwen3):
    print(permutation_importance_df_qwen3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11e. SHAP explainability for Qwen3 family

    Explain the best Qwen3-family model with SHAP.
    """)
    return


@app.cell
def _(SHAP_SAMPLE_SIZE, TREE_MODEL_NAMES):
    SHAP_SAMPLE_SIZE_QWEN3 = SHAP_SAMPLE_SIZE
    SHAP_RANDOM_STATE_QWEN3 = 42
    TREE_MODEL_NAMES_QWEN3 = TREE_MODEL_NAMES
    return (
        SHAP_RANDOM_STATE_QWEN3,
        SHAP_SAMPLE_SIZE_QWEN3,
        TREE_MODEL_NAMES_QWEN3,
    )


@app.cell
def _(
    SHAP_RANDOM_STATE_QWEN3,
    SHAP_SAMPLE_SIZE_QWEN3,
    TREE_MODEL_NAMES_QWEN3,
    X_qwen3,
    best_model_name_qwen3,
    best_params_by_model_qwen3,
    model_specs_no_model,
    model_summary_df_qwen3,
    y_qwen3,
):
    from mathanx.ml.helpers import run_shap_analysis as _run_shap_analysis

    shap_values_qwen3, shap_model_name_qwen3 = _run_shap_analysis(
        X_qwen3, y_qwen3, model_specs_no_model, best_model_name_qwen3,
        best_params_by_model_qwen3, model_summary_df_qwen3,
        TREE_MODEL_NAMES_QWEN3,
        shap_sample_size=SHAP_SAMPLE_SIZE_QWEN3,
        shap_random_state=SHAP_RANDOM_STATE_QWEN3,
    )
    return shap_model_name_qwen3, shap_values_qwen3


@app.cell
def _(FIG_PATH, shap_model_name_qwen3, shap_values_qwen3):
    from mathanx.ml.helpers import plot_shap_beeswarm as _plot_shap_beeswarm

    _fig = _plot_shap_beeswarm(shap_values_qwen3, f"SHAP beeswarm for {shap_model_name_qwen3} (Qwen3 family, without Model)")
    _fig.savefig(FIG_PATH / "beeswarm_no_model_qwen3_family.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "beeswarm_no_model_qwen3_family.png", format="png")
    _fig
    return


@app.cell
def _(shap_model_name_qwen3, shap_values_qwen3):
    from mathanx.ml.helpers import plot_shap_bar as _plot_shap_bar

    _plot_shap_bar(shap_values_qwen3, f"Global SHAP importance for {shap_model_name_qwen3} (Qwen3 family, without Model)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. Nested CV with five predictors

    Re-run the nested cross-validation using only `mseaq_anx`, `amas_score`, `maes_score`, `mseaq_se`, and `confidence_scaled`.
    """)
    return


@app.cell
def _(Pipeline, StandardScaler, TARGET, ml_df, pd):
    from pathlib import Path as _Path

    from mathanx.ml.config import FIVE_FEATURE_COLUMNS, RANDOM_STATE as RANDOM_STATE_FIVE
    from mathanx.ml.helpers import (
        load_experiment as _load_experiment_five,
        make_model_specs as _make_model_specs_five,
        run_experiment as _run_experiment_five,
    )

    X_five = ml_df.loc[:, FIVE_FEATURE_COLUMNS].copy()
    y_five = ml_df[TARGET].copy()

    build_linear_five = lambda m: Pipeline([("scale", StandardScaler()), ("model", m)])
    build_tree_five = lambda m: Pipeline([("model", m)])
    model_specs_five = _make_model_specs_five(build_linear_five, build_tree_five, random_state=RANDOM_STATE_FIVE)

    _cache_dir = _Path("models/five_predictors")
    if _cache_dir.exists():
        _result_five = _load_experiment_five(_cache_dir)
    else:
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
    return shap_model_name_five, shap_values_five


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
    ## 14. Nested CV with PCA psychometric components
    """)
    return


@app.cell
def _(LEAKAGE_COLS, RANDOM_STATE, TARGET, ml_df):
    import joblib as _joblib

    from mathanx.ml.config import PCA_TRANSFORM_PATH, PSYCH_SCORE_COLUMNS, PCA_COMPONENT_COLUMNS
    from mathanx.ml.helpers import (
        build_linear_pipeline as _build_linear_pipeline,
        build_tree_pipeline as _build_tree_pipeline,
        classify_columns as _classify_columns,
        make_model_specs as _make_model_specs_pca,
    )

    # Build X with PCA components instead of raw psychometric scores
    _feature_cols = [c for c in ml_df.columns if c not in LEAKAGE_COLS and c not in {TARGET}]
    X_pca = ml_df.loc[:, _feature_cols].copy()

    _pca_data = _joblib.load(PCA_TRANSFORM_PATH)
    _pc_scores = _pca_data["pca"].transform(_pca_data["scaler"].transform(ml_df[PSYCH_SCORE_COLUMNS]))

    X_pca = X_pca.drop(columns=PSYCH_SCORE_COLUMNS)
    X_pca[PCA_COMPONENT_COLUMNS[0]] = _pc_scores[:, 0]
    X_pca[PCA_COMPONENT_COLUMNS[1]] = _pc_scores[:, 1]
    y_pca = ml_df[TARGET].copy()

    # Recompute feature types — PC1 and PC2 are now regular numeric features
    _col_types = _classify_columns(X_pca)
    _nominal_features_pca = [c for c in _col_types["nominal_features"] if c != "Model"]
    _tree_nominal_features_pca = [c for c in _col_types["tree_nominal_features"] if c != "Model"]
    _numeric_features_pca = [c for c in _col_types["numeric_features"] if c != "Model"]

    _build_linear_pca = lambda m: _build_linear_pipeline(m, _numeric_features_pca, _nominal_features_pca)
    _build_tree_pca = lambda m: _build_tree_pipeline(m, _numeric_features_pca, _tree_nominal_features_pca)
    model_specs_pca = _make_model_specs_pca(_build_linear_pca, _build_tree_pca, random_state=RANDOM_STATE)
    return X_pca, model_specs_pca, y_pca


@app.cell
def _(X_pca, model_specs_pca, y_pca):
    from pathlib import Path as _Path

    from mathanx.ml.config import RANDOM_STATE as RANDOM_STATE_PCA
    from mathanx.ml.helpers import (
        load_experiment as _load_experiment_pca,
        run_experiment as _run_experiment_pca,
    )

    _cache_dir = _Path("models/pca_predictors")
    if _cache_dir.exists():
        _result_pca = _load_experiment_pca(_cache_dir)
    else:
        _result_pca = _run_experiment_pca(X_pca, y_pca, model_specs_pca, random_state=RANDOM_STATE_PCA)

    model_summary_df_pca = _result_pca.model_summary
    best_model_name_pca = _result_pca.best_model_name
    best_model_pipeline_pca = _result_pca.final_estimator
    permutation_importance_df_pca = _result_pca.permutation_importance
    return (
        best_model_name_pca,
        best_model_pipeline_pca,
        model_summary_df_pca,
        permutation_importance_df_pca,
    )


@app.cell
def _(model_summary_df_pca):
    model_summary_df_pca
    return


@app.cell
def _(
    best_model_name_pca,
    model_summary_df_pca,
    permutation_importance_df_pca,
):
    if permutation_importance_df_pca is not None and len(permutation_importance_df_pca) > 0:
        _best_row = model_summary_df_pca[model_summary_df_pca["model"] == best_model_name_pca]
        _top3 = permutation_importance_df_pca.head(3)["feature"].tolist()
        print(f"Best model: {best_model_name_pca}")
        print(f"Top 3 features: {', '.join(_top3)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 14b. SHAP explainability for PCA components

    Explain the best PCA-based model with SHAP.
    """)
    return


@app.cell
def _(SHAP_SAMPLE_SIZE, TREE_MODEL_NAMES):
    SHAP_SAMPLE_SIZE_PCA = SHAP_SAMPLE_SIZE
    SHAP_RANDOM_STATE_PCA = 42
    TREE_MODEL_NAMES_PCA = TREE_MODEL_NAMES
    return SHAP_RANDOM_STATE_PCA, SHAP_SAMPLE_SIZE_PCA, TREE_MODEL_NAMES_PCA


@app.cell
def _(
    SHAP_RANDOM_STATE_PCA,
    SHAP_SAMPLE_SIZE_PCA,
    TREE_MODEL_NAMES_PCA,
    X_pca,
    best_model_name_pca,
    best_model_pipeline_pca,
    model_summary_df_pca,
    y_pca,
):
    from mathanx.ml.helpers import run_shap_analysis as _run_shap_analysis

    shap_values_pca, shap_model_name_pca = _run_shap_analysis(
        X_pca, y_pca, {}, best_model_name_pca, {},
        model_summary_df_pca, TREE_MODEL_NAMES_PCA,
        pipeline=best_model_pipeline_pca,
        shap_sample_size=SHAP_SAMPLE_SIZE_PCA,
        shap_random_state=SHAP_RANDOM_STATE_PCA,
        check_linear=True,
    )
    return shap_model_name_pca, shap_values_pca


@app.cell
def _(shap_model_name_pca, shap_values_pca):
    from mathanx.ml.helpers import plot_shap_beeswarm as _plot_shap_beeswarm

    _plot_shap_beeswarm(shap_values_pca, f"SHAP beeswarm for {shap_model_name_pca} (PCA predictors)")
    return


@app.cell
def _(shap_model_name_pca, shap_values_pca):
    from mathanx.ml.helpers import plot_shap_bar as _plot_shap_bar

    _plot_shap_bar(shap_values_pca, f"Global SHAP importance for {shap_model_name_pca} (PCA predictors)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 14c. Nested CV with PCA components + Model

    Re-run the experiment using PCA-derived psychometric components while retaining the `Model` variable.
    """)
    return


@app.cell
def _(RANDOM_STATE, X_pca):
    from mathanx.ml.helpers import (
        build_linear_pipeline as _build_linear_pipeline,
        build_tree_pipeline as _build_tree_pipeline,
        classify_columns as _classify_columns,
        make_model_specs as _make_model_specs_pca_wm,
    )

    _col_types_wm = _classify_columns(X_pca)
    _nominal_features_pca_wm = _col_types_wm["nominal_features"]
    _tree_nominal_features_pca_wm = _col_types_wm["tree_nominal_features"]
    _numeric_features_pca_wm = _col_types_wm["numeric_features"]

    _build_linear_pca_wm = lambda m: _build_linear_pipeline(m, _numeric_features_pca_wm, _nominal_features_pca_wm)
    _build_tree_pca_wm = lambda m: _build_tree_pipeline(m, _numeric_features_pca_wm, _tree_nominal_features_pca_wm)
    model_specs_pca_wm = _make_model_specs_pca_wm(_build_linear_pca_wm, _build_tree_pca_wm, random_state=RANDOM_STATE)
    return (model_specs_pca_wm,)


@app.cell
def _(X_pca, model_specs_pca_wm, y_pca):
    from pathlib import Path as _Path

    from mathanx.ml.config import RANDOM_STATE as RANDOM_STATE_PCA_WM
    from mathanx.ml.helpers import (
        load_experiment as _load_experiment_pca_wm,
        run_experiment as _run_experiment_pca_wm,
    )

    _cache_dir = _Path("models/pca_with_model")
    if _cache_dir.exists():
        _result_pca_wm = _load_experiment_pca_wm(_cache_dir)
    else:
        _result_pca_wm = _run_experiment_pca_wm(X_pca, y_pca, model_specs_pca_wm, random_state=RANDOM_STATE_PCA_WM)

    model_summary_df_pca_wm = _result_pca_wm.model_summary
    best_model_name_pca_wm = _result_pca_wm.best_model_name
    best_model_pipeline_pca_wm = _result_pca_wm.final_estimator
    permutation_importance_df_pca_wm = _result_pca_wm.permutation_importance
    return (
        best_model_name_pca_wm,
        best_model_pipeline_pca_wm,
        model_summary_df_pca_wm,
        permutation_importance_df_pca_wm,
    )


@app.cell
def _(model_summary_df_pca_wm):
    model_summary_df_pca_wm
    return


@app.cell
def _(
    best_model_name_pca_wm,
    model_summary_df_pca_wm,
    permutation_importance_df_pca_wm,
):
    if permutation_importance_df_pca_wm is not None and len(permutation_importance_df_pca_wm) > 0:
        _best_row = model_summary_df_pca_wm[model_summary_df_pca_wm["model"] == best_model_name_pca_wm]
        _top3 = permutation_importance_df_pca_wm.head(3)["feature"].tolist()
        print(f"Best model: {best_model_name_pca_wm}")
        print(f"Top 3 features: {', '.join(_top3)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 14d. SHAP explainability for PCA + Model

    Explain the best PCA+Model model with SHAP.
    """)
    return


@app.cell
def _(SHAP_SAMPLE_SIZE, TREE_MODEL_NAMES):
    SHAP_SAMPLE_SIZE_PCA_WM = SHAP_SAMPLE_SIZE
    SHAP_RANDOM_STATE_PCA_WM = 42
    TREE_MODEL_NAMES_PCA_WM = TREE_MODEL_NAMES
    return (
        SHAP_RANDOM_STATE_PCA_WM,
        SHAP_SAMPLE_SIZE_PCA_WM,
        TREE_MODEL_NAMES_PCA_WM,
    )


@app.cell
def _(
    SHAP_RANDOM_STATE_PCA_WM,
    SHAP_SAMPLE_SIZE_PCA_WM,
    TREE_MODEL_NAMES_PCA_WM,
    X_pca,
    best_model_name_pca_wm,
    best_model_pipeline_pca_wm,
    model_summary_df_pca_wm,
    y_pca,
):
    from mathanx.ml.helpers import run_shap_analysis as _run_shap_analysis

    shap_values_pca_wm, shap_model_name_pca_wm = _run_shap_analysis(
        X_pca, y_pca, {}, best_model_name_pca_wm, {},
        model_summary_df_pca_wm, TREE_MODEL_NAMES_PCA_WM,
        pipeline=best_model_pipeline_pca_wm,
        shap_sample_size=SHAP_SAMPLE_SIZE_PCA_WM,
        shap_random_state=SHAP_RANDOM_STATE_PCA_WM,
        check_linear=True,
    )
    return shap_model_name_pca_wm, shap_values_pca_wm


@app.cell
def _(shap_model_name_pca_wm, shap_values_pca_wm):
    from mathanx.ml.helpers import plot_shap_beeswarm as _plot_shap_beeswarm

    _plot_shap_beeswarm(shap_values_pca_wm, f"SHAP beeswarm for {shap_model_name_pca_wm} (PCA + Model)")
    return


@app.cell
def _(shap_model_name_pca_wm, shap_values_pca_wm):
    from mathanx.ml.helpers import plot_shap_bar as _plot_shap_bar

    _plot_shap_bar(shap_values_pca_wm, f"Global SHAP importance for {shap_model_name_pca_wm} (PCA + Model)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 15. Experiment comparison

    Compare best models, parameters, and top permutation importances across all four feature sets.
    """)
    return


@app.cell
def _(
    best_model_name,
    best_model_name_five,
    best_model_name_no_model,
    best_model_name_pca,
    best_model_name_pca_wm,
    model_summary_df,
    model_summary_df_five,
    model_summary_df_no_model,
    model_summary_df_pca,
    model_summary_df_pca_wm,
    pd,
    permutation_importance_df_five,
    permutation_importance_df_no_model,
    permutation_importance_df_pca,
    permutation_importance_df_pca_wm,
    top_permutation_importance_df,
):
    sections = []

    for label, best, summ, top_perm in [
        ("All features", best_model_name, model_summary_df, top_permutation_importance_df),
        ("Without Model", best_model_name_no_model, model_summary_df_no_model, permutation_importance_df_no_model),
        ("Five predictors", best_model_name_five, model_summary_df_five, permutation_importance_df_five),
        ("PCA components", best_model_name_pca, model_summary_df_pca, permutation_importance_df_pca),
        ("PCA + Model", best_model_name_pca_wm, model_summary_df_pca_wm, permutation_importance_df_pca_wm),
    ]:
        _row = summ[summ["model"] == best]
        top3 = top_perm.head(3)["feature"].tolist() if top_perm is not None and len(top_perm) > 0 else []
        sections.append(
            {
                "Feature set": label,
                "Best model": best,
                "Nested CV R²": f'{_row.iloc[0]["mean_r2"]:.4f}' if len(_row) > 0 else "",
                "Nested CV RMSE": f'{_row.iloc[0]["mean_rmse"]:.4f}' if len(_row) > 0 else "",
                "Top 3 features": ", ".join(top3),
            }
        )

    experiment_comparison_df = pd.DataFrame(sections)
    experiment_comparison_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 16. Model confound analysis

    The `Model` feature (which LLM architecture is used) dominates the full model with ~300x the importance of any anxiety feature.
    This section investigates whether the apparent positive anxiety-accuracy relationship is driven by between-model differences
    (better LLMs generate more "anxious" persona responses) rather than a genuine within-person relationship.
    """)
    return


@app.cell
def _(ml_df, pd, plt):
    from statsmodels.nonparametric.smoothers_lowess import lowess

    anxiety_cols = ["amas_score", "maes_score", "mseaq_anx", "mseaq_se"]

    _within_rows = []
    for _model in ml_df["Model"].unique():
        _sub = ml_df[ml_df["Model"] == _model]
        _row_pm = {"Model": _model, "n": len(_sub)}
        for _col in anxiety_cols:
            _row_pm[f"{_col}_r"] = _sub[_col].corr(_sub["accuracy"])
        _within_rows.append(_row_pm)
    within_corr_df = pd.DataFrame(_within_rows).sort_values("Model")

    _corr_cols = [c for c in within_corr_df.columns if c.endswith("_r")]
    _fig, _axes = plt.subplots(4, 4, figsize=(16, 14))
    _models_sorted = sorted(ml_df["Model"].unique())
    for _idx, _model in enumerate(_models_sorted):
        _ax = _axes[_idx // 4, _idx % 4]
        _sub = ml_df[ml_df["Model"] == _model]
        _ax.scatter(_sub["mseaq_anx"], _sub["accuracy"], alpha=0.15, s=3, c="#648fff")
        _smooth = lowess(_sub["accuracy"], _sub["mseaq_anx"], frac=0.5, it=0, return_sorted=True)
        _ax.plot(_smooth[:, 0], _smooth[:, 1], "r-", linewidth=1.5)
        _ax.set_title(f"{_model}", fontsize=8)
        _ax.set_xlabel("")
        _ax.set_ylabel("")
        _ax.tick_params(labelsize=6)
    _fig.suptitle("Accuracy vs MSEAQ Anxiety — one panel per model", fontsize=14, y=1.02)
    _fig.text(0.5, 0.01, "MSEAQ Anxiety", ha="center", fontsize=12)
    _fig.text(0.01, 0.5, "Accuracy", va="center", rotation=90, fontsize=12)
    plt.tight_layout()
    plt.show()
    return anxiety_cols, within_corr_df


@app.cell
def _(corr_cols, within_corr_df):
    styled = within_corr_df.style.background_gradient(cmap="coolwarm", axis=None, subset=corr_cols).format({c: "{:.3f}" for c in corr_cols})
    styled
    return


@app.cell
def _(anxiety_cols, ml_df, pd):
    from statsmodels.formula.api import mixedlm
    import statsmodels.api as sm

    formula = "accuracy ~ " + " + ".join(anxiety_cols)
    mixed_model = mixedlm(formula, ml_df, groups=ml_df["Model"])
    mixed_result = mixed_model.fit()

    ols_model = sm.OLS.from_formula(formula, data=ml_df)
    ols_result = ols_model.fit()

    comparison_rows = []
    for _col in anxiety_cols:
        ols_coef = ols_result.params[_col]
        ols_p = ols_result.pvalues[_col]
        mixed_coef = mixed_result.fe_params[_col]
        mixed_p = mixed_result.pvalues[_col]
        comparison_rows.append({
            "feature": _col,
            "OLS_coef": f"{ols_coef:.5f}",
            "OLS_p": f"{ols_p:.2e}",
            "Mixed_coef": f"{mixed_coef:.5f}",
            "Mixed_p": f"{mixed_p:.2e}",
        })

    comparison_rows.append({
        "feature": "R² / Log-Likelihood",
        "OLS_coef": f"{ols_result.rsquared:.4f}",
        "OLS_p": "",
        "Mixed_coef": f"{mixed_result.llf:.1f}",
        "Mixed_p": "",
    })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df
    return


@app.cell
def _(ml_df, plt):
    anxiety_features = ["amas_score", "maes_score", "mseaq_anx", "mseaq_se"]
    models = sorted(ml_df["Model"].unique())
    model_means = ml_df.groupby("Model")[["accuracy"] + anxiety_features].mean()

    _fig, _axes = plt.subplots(2, 2, figsize=(14, 10))
    for _idx, feat in enumerate(anxiety_features):
        # FIXED: Changed 'axes' to '_axes'
        _ax = _axes[_idx // 2, _idx % 2] 

        for _model in models:
            pt = model_means.loc[_model]
            _ax.scatter(pt[feat], pt["accuracy"], s=80, alpha=0.8)
            _ax.annotate(_model, (pt[feat], pt["accuracy"]), fontsize=7, alpha=0.8)

        _ax.set_xlabel(f"Mean {feat}")
        _ax.set_ylabel("Mean accuracy")
        _ax.set_title(f"Between-model: {feat}", fontsize=12)

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 17. Filtered experiments (excluding top performers)

    The same five experiments re-run on a dataset that excludes `TOP_PERFORMERS`
    (Grok 4.1 Fast (Reasoning) and DeepSeek Chat) to test whether model performance
    insights hold when ceiling-effect models are removed.
    """)
    return


@app.cell
def _(LEAKAGE_COLS, TARGET, ml_df):
    import joblib as _joblib

    from mathanx.ml.config import (
        PCA_COMPONENT_COLUMNS as _PCA_COMPONENT_COLUMNS,
        PCA_TRANSFORM_PATH as _PCA_TRANSFORM_PATH,
        PSYCH_SCORE_COLUMNS as _PSYCH_SCORE_COLUMNS,
        TOP_PERFORMERS,
    )
    from mathanx.ml.helpers import classify_columns as _classify_columns

    ml_df_no_top = ml_df[~ml_df["Model"].isin(TOP_PERFORMERS)].reset_index(drop=True)

    _feature_cols_no_top = [
        c for c in ml_df_no_top.columns
        if c not in LEAKAGE_COLS and c not in {TARGET, "education_vs_parent_mean_gap"}
    ]
    X_no_top = ml_df_no_top.loc[:, _feature_cols_no_top].copy()
    y_no_top = ml_df_no_top[TARGET].copy()

    _col_types_no_top = _classify_columns(X_no_top)
    numeric_features_no_top = _col_types_no_top["numeric_features"]
    nominal_features_no_top = _col_types_no_top["nominal_features"]
    tree_nominal_features_no_top = _col_types_no_top["tree_nominal_features"]

    _feature_cols_pca_no_top = [
        c for c in ml_df_no_top.columns
        if c not in LEAKAGE_COLS and c not in {TARGET}
    ]
    X_pca_no_top = ml_df_no_top.loc[:, _feature_cols_pca_no_top].copy()
    _pca_data = _joblib.load(_PCA_TRANSFORM_PATH)
    _pc_scores = _pca_data["pca"].transform(
        _pca_data["scaler"].transform(ml_df_no_top[_PSYCH_SCORE_COLUMNS])
    )
    X_pca_no_top = X_pca_no_top.drop(columns=_PSYCH_SCORE_COLUMNS)
    X_pca_no_top[_PCA_COMPONENT_COLUMNS[0]] = _pc_scores[:, 0]
    X_pca_no_top[_PCA_COMPONENT_COLUMNS[1]] = _pc_scores[:, 1]
    y_pca_no_top = ml_df_no_top[TARGET].copy()
    return (
        X_no_top,
        X_pca_no_top,
        ml_df_no_top,
        nominal_features_no_top,
        numeric_features_no_top,
        tree_nominal_features_no_top,
        y_no_top,
        y_pca_no_top,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 17a. All features (no top performers)

    Full feature set including the `Model` variable, trained on the filtered dataset.
    """)
    return


@app.cell
def _(
    X_no_top,
    nominal_features_no_top,
    numeric_features_no_top,
    tree_nominal_features_no_top,
    y_no_top,
):
    from pathlib import Path as _Path

    from mathanx.ml.config import RANDOM_STATE as _RANDOM_STATE
    from mathanx.ml.helpers import (
        build_linear_pipeline as _build_linear_pipeline,
        build_tree_pipeline as _build_tree_pipeline,
        load_experiment as _load_experiment,
        make_model_specs as _make_model_specs,
        run_experiment as _run_experiment,
    )

    _build_linear = lambda m: _build_linear_pipeline(m, numeric_features_no_top, nominal_features_no_top)
    _build_tree = lambda m: _build_tree_pipeline(m, numeric_features_no_top, tree_nominal_features_no_top)
    model_specs_no_top = _make_model_specs(_build_linear, _build_tree, random_state=_RANDOM_STATE)

    _cache_dir = _Path("models/all_features_no_top")
    if _cache_dir.exists():
        _result = _load_experiment(_cache_dir)
    else:
        _result = _run_experiment(X_no_top, y_no_top, model_specs_no_top, random_state=_RANDOM_STATE)

    model_summary_df_no_top = _result.model_summary
    best_params_by_model_no_top = _result.best_params_by_model
    best_model_name_no_top = _result.best_model_name
    tuned_cv_results_df_no_top = _result.tuned_cv_results
    permutation_importance_df_no_top = _result.permutation_importance
    top_permutation_importance_df_no_top = (
        permutation_importance_df_no_top.head(20)
        if permutation_importance_df_no_top is not None
        else None
    )
    return (
        best_model_name_no_top,
        best_params_by_model_no_top,
        model_specs_no_top,
        model_summary_df_no_top,
        permutation_importance_df_no_top,
        top_permutation_importance_df_no_top,
        tuned_cv_results_df_no_top,
    )


@app.cell
def _(model_summary_df_no_top):
    model_summary_df_no_top
    return


@app.cell
def _(top_permutation_importance_df_no_top):
    top_permutation_importance_df_no_top
    return


@app.cell
def _(mo):
    METRIC_NO_TOP = mo.ui.radio(
        options=["mean_rmse", "mean_mae", "mean_r2"],
        label="Select the metric of interest (no top performers):",
        value="mean_r2",
    )
    METRIC_NO_TOP
    return (METRIC_NO_TOP,)


@app.cell
def _(METRIC_NO_TOP, model_summary_df_no_top, plt, sns):
    _fig, _ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=model_summary_df_no_top, x=METRIC_NO_TOP.value, y="model", ax=_ax, color="#648fff")
    _ax.set_title(f"Nested CV {METRIC_NO_TOP.value} by model (no top performers)")
    _ax.set_xlabel(f"{METRIC_NO_TOP.value.capitalize().replace('_', ' ')}")
    _ax.set_ylabel("Model")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(model_summary_df_no_top, tuned_cv_results_df_no_top):
    _model_comparison_df = model_summary_df_no_top.merge(
        tuned_cv_results_df_no_top[["model", "mean_rmse", "mean_mae", "mean_r2"]],
        on="model",
        suffixes=("_nested", "_tuned"),
    )
    _model_comparison_df
    return


@app.cell
def _(plt, sns, top_permutation_importance_df_no_top):
    _fig, _ax = plt.subplots(figsize=(10, 8))
    sns.barplot(
        data=top_permutation_importance_df_no_top,
        x="importance_mean", y="feature", ax=_ax, color="#fe6100",
    )
    _ax.set_title("Top permutation importances for best tuned model (no top performers)")
    _ax.set_xlabel("Importance")
    _ax.set_ylabel("Feature")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(
    SHAP_SAMPLE_SIZE,
    TREE_MODEL_NAMES,
    X_no_top,
    best_model_name_no_top,
    best_params_by_model_no_top,
    model_specs_no_top,
    model_summary_df_no_top,
    y_no_top,
):
    from mathanx.ml.helpers import run_shap_analysis as _run_shap_analysis

    shap_values_no_top, shap_model_name_no_top = _run_shap_analysis(
        X_no_top, y_no_top, model_specs_no_top, best_model_name_no_top,
        best_params_by_model_no_top, model_summary_df_no_top,
        TREE_MODEL_NAMES,
        shap_sample_size=SHAP_SAMPLE_SIZE, shap_random_state=42,
    )
    return shap_model_name_no_top, shap_values_no_top


@app.cell
def _(FIG_PATH, shap, shap_model_name_no_top, shap_values_no_top):
    from mathanx.ml.helpers import plot_shap_beeswarm as _plot_shap_beeswarm

    _indices = [i for i, _name in enumerate(shap_values_no_top.feature_names) if _name != "Model"]
    shap_values_filtered_no_top = shap.Explanation(
        values=shap_values_no_top.values[:, _indices],
        base_values=shap_values_no_top.base_values,
        data=shap_values_no_top.data[:, _indices] if shap_values_no_top.data is not None else None,
        feature_names=[shap_values_no_top.feature_names[i] for i in _indices],
    )

    _fig = _plot_shap_beeswarm(
        shap_values_filtered_no_top,
        f"SHAP beeswarm for {shap_model_name_no_top} (no top performers, model excluded)",
    )
    _fig.savefig(FIG_PATH / "beeswarm_all_predictors_model_excluded_no_top.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "beeswarm_all_predictors_model_excluded_no_top.png", format="png")
    _fig
    return (shap_values_filtered_no_top,)


@app.cell
def _(shap_model_name_no_top, shap_values_filtered_no_top):
    from mathanx.ml.helpers import plot_shap_bar as _plot_shap_bar

    _plot_shap_bar(
        shap_values_filtered_no_top,
        f"Global SHAP importance for {shap_model_name_no_top} (no top performers, without Model)",
    )
    return


@app.cell
def _(shap_model_name_no_top, shap_values_no_top):
    from mathanx.ml.helpers import plot_shap_beeswarm as _plot_shap_beeswarm

    _plot_shap_beeswarm(
        shap_values_no_top,
        f"SHAP beeswarm for {shap_model_name_no_top} (no top performers)",
    )
    return


@app.cell
def _(shap_model_name_no_top, shap_values_no_top):
    from mathanx.ml.helpers import plot_shap_bar as _plot_shap_bar

    _plot_shap_bar(
        shap_values_no_top,
        f"Global SHAP importance for {shap_model_name_no_top} (no top performers)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 17b. Without Model (no top performers)

    All non-leakage features except `Model`, trained on the filtered dataset.
    """)
    return


@app.cell
def _(
    X_no_top,
    nominal_features_no_top,
    numeric_features_no_top,
    tree_nominal_features_no_top,
    y_no_top,
):
    from pathlib import Path as _Path

    from mathanx.ml.config import RANDOM_STATE as _RANDOM_STATE
    from mathanx.ml.helpers import (
        build_linear_pipeline as _build_linear_pipeline,
        build_tree_pipeline as _build_tree_pipeline,
        load_experiment as _load_experiment,
        make_model_specs as _make_model_specs,
        run_experiment as _run_experiment,
    )

    _numeric_f = [c for c in numeric_features_no_top if c != "Model"]
    _nominal_f = [c for c in nominal_features_no_top if c != "Model"]
    _tree_nominal_f = [c for c in tree_nominal_features_no_top if c != "Model"]
    model_specs_no_model_no_top = _make_model_specs(
        lambda m: _build_linear_pipeline(m, _numeric_f, _nominal_f),
        lambda m: _build_tree_pipeline(m, _numeric_f, _tree_nominal_f),
        random_state=_RANDOM_STATE,
    )

    _cache_dir = _Path("models/no_model_no_top")
    if _cache_dir.exists():
        _result = _load_experiment(_cache_dir)
    else:
        _result = _run_experiment(X_no_top, y_no_top, model_specs_no_model_no_top, random_state=_RANDOM_STATE)

    model_summary_df_no_model_no_top = _result.model_summary
    best_model_name_no_model_no_top = _result.best_model_name
    best_params_by_model_no_model_no_top = _result.best_params_by_model
    permutation_importance_df_no_model_no_top = _result.permutation_importance
    return (
        best_model_name_no_model_no_top,
        best_params_by_model_no_model_no_top,
        model_specs_no_model_no_top,
        model_summary_df_no_model_no_top,
        permutation_importance_df_no_model_no_top,
    )


@app.cell
def _(model_summary_df_no_model_no_top):
    model_summary_df_no_model_no_top
    return


@app.cell
def _(permutation_importance_df_no_model_no_top):
    if permutation_importance_df_no_model_no_top is not None:
        print(permutation_importance_df_no_model_no_top)
    return


@app.cell
def _(
    SHAP_SAMPLE_SIZE,
    TREE_MODEL_NAMES,
    X_no_top,
    best_model_name_no_model_no_top,
    best_params_by_model_no_model_no_top,
    model_specs_no_model_no_top,
    model_summary_df_no_model_no_top,
    y_no_top,
):
    from mathanx.ml.helpers import run_shap_analysis as _run_shap_analysis

    shap_values_no_model_no_top, shap_model_name_no_model_no_top = _run_shap_analysis(
        X_no_top, y_no_top, model_specs_no_model_no_top,
        best_model_name_no_model_no_top,
        best_params_by_model_no_model_no_top,
        model_summary_df_no_model_no_top,
        TREE_MODEL_NAMES,
        shap_sample_size=SHAP_SAMPLE_SIZE, shap_random_state=42,
    )
    return shap_model_name_no_model_no_top, shap_values_no_model_no_top


@app.cell
def _(FIG_PATH, shap_model_name_no_model_no_top, shap_values_no_model_no_top):
    from mathanx.ml.helpers import plot_shap_beeswarm as _plot_shap_beeswarm

    _fig = _plot_shap_beeswarm(
        shap_values_no_model_no_top,
        f"SHAP beeswarm for {shap_model_name_no_model_no_top} (no top performers, without Model)",
    )
    _fig.savefig(FIG_PATH / "beeswarm_no_model_no_top.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "beeswarm_no_model_no_top.png", format="png")
    _fig
    return


@app.cell
def _(shap_model_name_no_model_no_top, shap_values_no_model_no_top):
    from mathanx.ml.helpers import plot_shap_bar as _plot_shap_bar

    _plot_shap_bar(
        shap_values_no_model_no_top,
        f"Global SHAP importance for {shap_model_name_no_model_no_top} (no top performers, without Model)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 17c. Five predictors (no top performers)

    Only `mseaq_anx`, `amas_score`, `maes_score`, `mseaq_se`, trained on the filtered dataset.
    """)
    return


@app.cell
def _(StandardScaler, TARGET, ml_df_no_top):
    from pathlib import Path as _Path
    import pandas as _pd

    from mathanx.ml.config import FIVE_FEATURE_COLUMNS as _FIVE_FEATURE_COLUMNS, RANDOM_STATE as _RANDOM_STATE
    from mathanx.ml.helpers import (
        build_linear_pipeline as _build_linear_pipeline,
        build_tree_pipeline as _build_tree_pipeline,
        classify_columns as _classify_columns,
        load_experiment as _load_experiment,
        make_model_specs as _make_model_specs,
        run_experiment as _run_experiment,
        Pipeline as _Pipeline
    )

    X_five_no_top = ml_df_no_top.loc[:, _FIVE_FEATURE_COLUMNS].copy()
    y_five_no_top = ml_df_no_top[TARGET].copy()

    _build_linear_five = lambda m: _Pipeline([("scale", StandardScaler()), ("model", m)])
    _build_tree_five = lambda m: _Pipeline([("model", m)])
    model_specs_five_no_top = _make_model_specs(
        _build_linear_five, _build_tree_five, random_state=_RANDOM_STATE,
    )

    _cache_dir = _Path("models/five_predictors_no_top")
    if _cache_dir.exists():
        _result_five = _load_experiment(_cache_dir)
    else:
        _result_five = _run_experiment(X_five_no_top, y_five_no_top, model_specs_five_no_top, random_state=_RANDOM_STATE)

    model_summary_df_five_no_top = _result_five.model_summary
    best_model_name_five_no_top = _result_five.best_model_name
    best_model_pipeline_five_no_top = _result_five.final_estimator
    permutation_importance_df_five_no_top = _result_five.permutation_importance
    return (
        X_five_no_top,
        best_model_name_five_no_top,
        best_model_pipeline_five_no_top,
        model_summary_df_five_no_top,
        permutation_importance_df_five_no_top,
        y_five_no_top,
    )


@app.cell
def _(model_summary_df_five_no_top):
    model_summary_df_five_no_top
    return


@app.cell
def _(
    SHAP_SAMPLE_SIZE,
    TREE_MODEL_NAMES,
    X_five_no_top,
    best_model_name_five_no_top,
    best_model_pipeline_five_no_top,
    model_summary_df_five_no_top,
    y_five_no_top,
):
    from mathanx.ml.helpers import run_shap_analysis as _run_shap_analysis

    shap_values_five_no_top, shap_model_name_five_no_top = _run_shap_analysis(
        X_five_no_top, y_five_no_top, {}, best_model_name_five_no_top, {},
        model_summary_df_five_no_top, TREE_MODEL_NAMES,
        pipeline=best_model_pipeline_five_no_top,
        shap_sample_size=SHAP_SAMPLE_SIZE, shap_random_state=42,
        check_linear=True,
    )
    return shap_model_name_five_no_top, shap_values_five_no_top


@app.cell
def _(shap_model_name_five_no_top, shap_values_five_no_top):
    from mathanx.ml.helpers import plot_shap_beeswarm as _plot_shap_beeswarm

    _plot_shap_beeswarm(
        shap_values_five_no_top,
        f"SHAP beeswarm for {shap_model_name_five_no_top} (no top performers, selected predictors)",
    )
    return


@app.cell
def _(shap_model_name_five_no_top, shap_values_five_no_top):
    from mathanx.ml.helpers import plot_shap_bar as _plot_shap_bar

    _plot_shap_bar(
        shap_values_five_no_top,
        f"Global SHAP importance for {shap_model_name_five_no_top} (no top performers, selected predictors)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 17d. PCA predictors (no top performers)

    PCA-derived psychometric components (PC1, PC2) without the `Model` variable, trained on the filtered dataset.
    """)
    return


@app.cell
def _(X_pca_no_top, y_pca_no_top):
    from pathlib import Path as _Path

    from mathanx.ml.config import RANDOM_STATE as _RANDOM_STATE
    from mathanx.ml.helpers import (
        build_linear_pipeline as _build_linear_pipeline,
        build_tree_pipeline as _build_tree_pipeline,
        classify_columns as _classify_columns,
        load_experiment as _load_experiment,
        make_model_specs as _make_model_specs,
        run_experiment as _run_experiment,
    )

    _col_types_pca = _classify_columns(X_pca_no_top)
    _nominal_f = [c for c in _col_types_pca["nominal_features"] if c != "Model"]
    _tree_nominal_f = [c for c in _col_types_pca["tree_nominal_features"] if c != "Model"]
    _numeric_f = [c for c in _col_types_pca["numeric_features"] if c != "Model"]

    model_specs_pca_no_top = _make_model_specs(
        lambda m: _build_linear_pipeline(m, _numeric_f, _nominal_f),
        lambda m: _build_tree_pipeline(m, _numeric_f, _tree_nominal_f),
        random_state=_RANDOM_STATE,
    )

    _cache_dir = _Path("models/pca_predictors_no_top")
    if _cache_dir.exists():
        _result = _load_experiment(_cache_dir)
    else:
        _result = _run_experiment(X_pca_no_top, y_pca_no_top, model_specs_pca_no_top, random_state=_RANDOM_STATE)

    model_summary_df_pca_no_top = _result.model_summary
    best_model_name_pca_no_top = _result.best_model_name
    best_model_pipeline_pca_no_top = _result.final_estimator
    permutation_importance_df_pca_no_top = _result.permutation_importance
    return (
        best_model_name_pca_no_top,
        best_model_pipeline_pca_no_top,
        model_summary_df_pca_no_top,
        permutation_importance_df_pca_no_top,
    )


@app.cell
def _(model_summary_df_pca_no_top):
    model_summary_df_pca_no_top
    return


@app.cell
def _(best_model_name_pca_no_top, permutation_importance_df_pca_no_top):
    if permutation_importance_df_pca_no_top is not None and len(permutation_importance_df_pca_no_top) > 0:
        _top3 = permutation_importance_df_pca_no_top.head(3)["feature"].tolist()
        print(f"Best model: {best_model_name_pca_no_top}")
        print(f"Top 3 features: {', '.join(_top3)}")
    return


@app.cell
def _(
    SHAP_SAMPLE_SIZE,
    TREE_MODEL_NAMES,
    X_pca_no_top,
    best_model_name_pca_no_top,
    best_model_pipeline_pca_no_top,
    model_summary_df_pca_no_top,
    y_pca_no_top,
):
    from mathanx.ml.helpers import run_shap_analysis as _run_shap_analysis

    shap_values_pca_no_top, shap_model_name_pca_no_top = _run_shap_analysis(
        X_pca_no_top, y_pca_no_top, {}, best_model_name_pca_no_top, {},
        model_summary_df_pca_no_top, TREE_MODEL_NAMES,
        pipeline=best_model_pipeline_pca_no_top,
        shap_sample_size=SHAP_SAMPLE_SIZE, shap_random_state=42,
        check_linear=True,
    )
    return shap_model_name_pca_no_top, shap_values_pca_no_top


@app.cell
def _(FIG_PATH, shap_model_name_pca_no_top, shap_values_pca_no_top):
    from mathanx.ml.helpers import plot_shap_beeswarm as _plot_shap_beeswarm

    _fig = _plot_shap_beeswarm(
        shap_values_pca_no_top,
        f"SHAP beeswarm for {shap_model_name_pca_no_top} (no top performers, PCA predictors)",
    )
    _fig.savefig(FIG_PATH / "beeswarm_shap_model_name_pca_no_top.png", format="png")
    _fig.savefig(FIG_PATH / "beeswarm_shap_model_name_pca_no_top.pdf", format="pdf")
    _fig
    return


@app.cell
def _(shap_model_name_pca_no_top, shap_values_pca_no_top):
    from mathanx.ml.helpers import plot_shap_bar as _plot_shap_bar

    _plot_shap_bar(
        shap_values_pca_no_top,
        f"Global SHAP importance for {shap_model_name_pca_no_top} (no top performers, PCA predictors)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 17e. PCA + Model (no top performers)

    PCA-derived psychometric components (PC1, PC2) with `Model`, trained on the filtered dataset.
    """)
    return


@app.cell
def _(X_pca_no_top, y_pca_no_top):
    from pathlib import Path as _Path

    from mathanx.ml.config import RANDOM_STATE as _RANDOM_STATE
    from mathanx.ml.helpers import (
        build_linear_pipeline as _build_linear_pipeline,
        build_tree_pipeline as _build_tree_pipeline,
        classify_columns as _classify_columns,
        load_experiment as _load_experiment,
        make_model_specs as _make_model_specs,
        run_experiment as _run_experiment,
    )

    _col_types_pca = _classify_columns(X_pca_no_top)
    _nominal_f = _col_types_pca["nominal_features"]
    _tree_nominal_f = _col_types_pca["tree_nominal_features"]
    _numeric_f = _col_types_pca["numeric_features"]

    model_specs_pca_wm_no_top = _make_model_specs(
        lambda m: _build_linear_pipeline(m, _numeric_f, _nominal_f),
        lambda m: _build_tree_pipeline(m, _numeric_f, _tree_nominal_f),
        random_state=_RANDOM_STATE,
    )

    _cache_dir = _Path("models/pca_with_model_no_top")
    if _cache_dir.exists():
        _result = _load_experiment(_cache_dir)
    else:
        _result = _run_experiment(X_pca_no_top, y_pca_no_top, model_specs_pca_wm_no_top, random_state=_RANDOM_STATE)

    model_summary_df_pca_wm_no_top = _result.model_summary
    best_model_name_pca_wm_no_top = _result.best_model_name
    best_model_pipeline_pca_wm_no_top = _result.final_estimator
    permutation_importance_df_pca_wm_no_top = _result.permutation_importance
    return (
        best_model_name_pca_wm_no_top,
        best_model_pipeline_pca_wm_no_top,
        model_summary_df_pca_wm_no_top,
        permutation_importance_df_pca_wm_no_top,
    )


@app.cell
def _(model_summary_df_pca_wm_no_top):
    model_summary_df_pca_wm_no_top
    return


@app.cell
def _(best_model_name_pca_wm_no_top, permutation_importance_df_pca_wm_no_top):
    if permutation_importance_df_pca_wm_no_top is not None and len(permutation_importance_df_pca_wm_no_top) > 0:
        _top3 = permutation_importance_df_pca_wm_no_top.head(3)["feature"].tolist()
        print(f"Best model: {best_model_name_pca_wm_no_top}")
        print(f"Top 3 features: {', '.join(_top3)}")
    return


@app.cell
def _(
    SHAP_SAMPLE_SIZE,
    TREE_MODEL_NAMES,
    X_pca_no_top,
    best_model_name_pca_wm_no_top,
    best_model_pipeline_pca_wm_no_top,
    model_summary_df_pca_wm_no_top,
    y_pca_no_top,
):
    from mathanx.ml.helpers import run_shap_analysis as _run_shap_analysis

    shap_values_pca_wm_no_top, shap_model_name_pca_wm_no_top = _run_shap_analysis(
        X_pca_no_top, y_pca_no_top, {}, best_model_name_pca_wm_no_top, {},
        model_summary_df_pca_wm_no_top, TREE_MODEL_NAMES,
        pipeline=best_model_pipeline_pca_wm_no_top,
        shap_sample_size=SHAP_SAMPLE_SIZE, shap_random_state=42,
        check_linear=True,
    )
    return shap_model_name_pca_wm_no_top, shap_values_pca_wm_no_top


@app.cell
def _(shap_model_name_pca_wm_no_top, shap_values_pca_wm_no_top):
    from mathanx.ml.helpers import plot_shap_beeswarm as _plot_shap_beeswarm

    _plot_shap_beeswarm(
        shap_values_pca_wm_no_top,
        f"SHAP beeswarm for {shap_model_name_pca_wm_no_top} (no top performers, PCA + Model)",
    )
    return


@app.cell
def _(shap_model_name_pca_wm_no_top, shap_values_pca_wm_no_top):
    from mathanx.ml.helpers import plot_shap_bar as _plot_shap_bar

    _plot_shap_bar(
        shap_values_pca_wm_no_top,
        f"Global SHAP importance for {shap_model_name_pca_wm_no_top} (no top performers, PCA + Model)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 17f. Experiment comparison (no top performers)

    Compare best models, parameters, and top permutation importances across all five feature sets on the filtered dataset.
    """)
    return


@app.cell
def _(
    best_model_name_five_no_top,
    best_model_name_no_model_no_top,
    best_model_name_no_top,
    best_model_name_pca_no_top,
    best_model_name_pca_wm_no_top,
    model_summary_df_five_no_top,
    model_summary_df_no_model_no_top,
    model_summary_df_no_top,
    model_summary_df_pca_no_top,
    model_summary_df_pca_wm_no_top,
    pd,
    permutation_importance_df_five_no_top,
    permutation_importance_df_no_model_no_top,
    permutation_importance_df_no_top,
    permutation_importance_df_pca_no_top,
    permutation_importance_df_pca_wm_no_top,
):
    _sections = []
    for _label, _best, _summ, _top_perm in [
        ("All features", best_model_name_no_top, model_summary_df_no_top, permutation_importance_df_no_top),
        ("Without Model", best_model_name_no_model_no_top, model_summary_df_no_model_no_top, permutation_importance_df_no_model_no_top),
        ("Five predictors", best_model_name_five_no_top, model_summary_df_five_no_top, permutation_importance_df_five_no_top),
        ("PCA components", best_model_name_pca_no_top, model_summary_df_pca_no_top, permutation_importance_df_pca_no_top),
        ("PCA + Model", best_model_name_pca_wm_no_top, model_summary_df_pca_wm_no_top, permutation_importance_df_pca_wm_no_top),
    ]:
        _row = _summ[_summ["model"] == _best]
        _top3 = _top_perm.head(3)["feature"].tolist() if _top_perm is not None and len(_top_perm) > 0 else []
        _sections.append({
            "Feature set": _label,
            "Best model": _best,
            "Nested CV R²": f'{_row.iloc[0]["mean_r2"]:.4f}' if len(_row) > 0 else "",
            "Nested CV RMSE": f'{_row.iloc[0]["mean_rmse"]:.4f}' if len(_row) > 0 else "",
            "Top 3 features": ", ".join(_top3),
        })
    _experiment_comparison_df = pd.DataFrame(_sections)
    _experiment_comparison_df
    return


if __name__ == "__main__":
    app.run()
