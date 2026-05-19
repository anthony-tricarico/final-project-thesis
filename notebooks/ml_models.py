import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    from typing import Dict, List, Tuple

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.stats import pearsonr, spearmanr
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    sns.set_theme(style="whitegrid", context="notebook")

    TARGET = "accuracy"
    DATASET_PATH = Path("data/processed/ml/ml_dataset.csv").resolve().absolute()

    # Specify which columns could introduce leakage in the analysis. These include columns
    # that are linear transformation of the target variable.
    LEAKAGE_COLS = {
        "run_id",
        "accuracy",
        "confidence",
        "confidence_scaled",
        "delta_confidence",
        "total_correct",
        "n_observations",
    }
    return (
        ColumnTransformer,
        DATASET_PATH,
        Dict,
        DummyRegressor,
        ElasticNet,
        HistGradientBoostingRegressor,
        LEAKAGE_COLS,
        Pipeline,
        RandomForestRegressor,
        Ridge,
        StandardScaler,
        TARGET,
        leaves_list,
        linkage,
        mean_absolute_error,
        mean_squared_error,
        mo,
        np,
        pd,
        pearsonr,
        permutation_importance,
        plt,
        r2_score,
        sns,
        spearmanr,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ML Models Exploration

    This notebook ranks candidate predictors for the `accuracy` target and checks whether scaling is needed for the intended model families.
    """)
    return


@app.cell
def _(DATASET_PATH, mo, pd):
    if not DATASET_PATH.exists():
        mo.callout(
            mo.md(
                f"The engineered dataset was not found at `{DATASET_PATH}`. Run `ml_dataset_creation.py` first so it can materialize the modeling table."
            ),
            kind="warn",
        )
        raise FileNotFoundError(DATASET_PATH)

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
        ml_df.isna()
        .sum()
        .rename("missing_count")
        .to_frame()
        .assign(missing_rate=lambda df: df["missing_count"] / len(ml_df))
        .query("missing_count > 0")
        .sort_values(["missing_count", "missing_rate"], ascending=False)
    )
    missing_summary
    return


@app.cell
def _(TARGET, ml_df):
    target_summary = ml_df[TARGET].describe().to_frame().T
    target_summary["skew"] = ml_df[TARGET].skew()
    target_summary
    return


@app.cell
def _(TARGET, ml_df):
    ml_df[TARGET].hist()
    return


@app.cell
def _(TARGET, ml_df, pd):
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
def _(ml_df, pd):
    def infer_feature_type(name: str, series: pd.Series) -> str:
        """Categorize the type of the different features to build a feature inventory"""
        values = series.dropna().unique()
        if name in {"run_id"}:
            return "identifier"
        if name in {"accuracy", "confidence", "confidence_scaled", "delta_confidence", "total_correct", "n_observations"}:
            return "target_or_leakage"
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
    return (feature_type_counts,)


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
    feature_cols = [c for c in ml_df.columns if c not in LEAKAGE_COLS and c != TARGET]
    X = ml_df.loc[:, feature_cols].copy()
    y = ml_df[TARGET].copy()

    binary_features = [c for c in X.columns if set(X[c].dropna().unique()).issubset({0, 1})]
    numeric_features = [c for c in X.columns if c not in binary_features]
    ordinal_features = [c for c in numeric_features if c.endswith("_ord")]
    continuous_features = [c for c in numeric_features if c not in ordinal_features]

    feature_split_df = pd.DataFrame(
        {
            "group": ["continuous_or_ordinal", "binary_dummies", "all_predictors"],
            "count": [len(numeric_features), len(binary_features), len(feature_cols)],
        }
    )
    return (
        X,
        binary_features,
        continuous_features,
        feature_split_df,
        ordinal_features,
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
def _(Dict, X, np, pd, pearsonr, spearmanr, y):
    def score_numeric_feature(series: pd.Series, target: pd.Series) -> Dict[str, float]:
        valid = series.notna() & target.notna()
        x = series.loc[valid]
        t = target.loc[valid]
        if x.nunique() < 2:
            return {"pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan}
        pearson_r_value, pearson_p_value = pearsonr(x, t)
        spearman_r_value, spearman_p_value = spearmanr(x, t)
        return {
            "pearson_r": pearson_r_value,
            "pearson_p": pearson_p_value,
            "spearman_r": spearman_r_value,
            "spearman_p": spearman_p_value,
        }

    numeric_rows = []
    for column in [c for c in X.columns if c not in X.columns[X.dtypes.eq("object")]]:
        stats = score_numeric_feature(X[column], y)
        numeric_rows.append({"feature": column, **stats, "abs_spearman_r": abs(stats["spearman_r"]) if pd.notna(stats["spearman_r"]) else np.nan})

    numeric_screen_df = pd.DataFrame(numeric_rows).sort_values("abs_spearman_r", ascending=False)
    top_numeric_screen_df = numeric_screen_df.head(20)
    top_numeric_screen_df
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
    ## 8. Baseline model comparison

    Compare a mean baseline, regularized linear models, and tree-based models on a holdout split.
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    DummyRegressor,
    ElasticNet,
    HistGradientBoostingRegressor,
    Pipeline,
    RandomForestRegressor,
    Ridge,
    StandardScaler,
    X,
    binary_features,
    continuous_features,
    mean_absolute_error,
    mean_squared_error,
    np,
    ordinal_features,
    pd,
    r2_score,
    train_test_split,
    y,
):
    # TODO: include CV to compute the CV error not the simple test-error.

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Declare which features are continuous and need rescaling
    scale_features = continuous_features + [c for c in ordinal_features if c not in continuous_features]
    # Specify that the binary features are not to be scaled or preprocessed
    passthrough_features = binary_features

    def build_linear_pipeline(model):
        """Build a full scikit-learn compatible model training pipeline including feature preprocessing"""
    
        # Specify the pre-processing to be performed on columns
        transformer = ColumnTransformer(
            transformers=[
                ("scale", StandardScaler(), scale_features), # Continuous features are standardized.
                ("pass", "passthrough", passthrough_features), # Do nothing for binary variables.
            ],
            remainder="drop", # Drop remaining features. TODO: in the next iteration we could try including ordinal features.
            verbose_feature_names_out=False,
        )
    
        # Return the Pipeline object
        return Pipeline([("preprocess", transformer), ("model", model)])

    models = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": build_linear_pipeline(Ridge(alpha=1.0)),
        "elastic_net": build_linear_pipeline(ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42)),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "hist_gb": HistGradientBoostingRegressor(random_state=42),
    }

    model_rows = []
    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        fitted_models[name] = model
        model_rows.append(
            {
                "model": name,
                "r2": r2_score(y_test, pred),
                "rmse": np.sqrt(mean_squared_error(y_test, pred)),
                "mae": mean_absolute_error(y_test, pred),
            }
        )

    model_comparison_df = pd.DataFrame(model_rows).sort_values("rmse")
    model_comparison_df
    return X_test, fitted_models, y_test


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Permutation importance

    Use permutation importance on the holdout set to identify the strongest predictors.
    """)
    return


@app.cell
def _(X_test, fitted_models, pd, permutation_importance, y_test):
    importance_tables = []
    for _name in ["ridge", "random_forest", "hist_gb"]:
        _model = fitted_models[_name]
        result = permutation_importance(_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        importance_tables.append(
            pd.DataFrame(
                {
                    "model": _name,
                    "feature": X_test.columns,
                    "importance_mean": result.importances_mean,
                    "importance_std": result.importances_std,
                }
            ).sort_values("importance_mean", ascending=False)
        )

    permutation_importance_df = pd.concat(importance_tables, axis=0, ignore_index=True)
    top_permutation_importance_df = permutation_importance_df.groupby("model").head(15)
    top_permutation_importance_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Next steps

    The next steps include:
    1. setup CV to check which models perform better
    2. setup the XAI evaluation system by performing SHAP and plotting beeswarm plots
    """)
    return


if __name__ == "__main__":
    app.run()
