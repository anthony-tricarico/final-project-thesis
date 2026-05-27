import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Model Experiment Explorer
    Select an experiment directory below to explore its trained models, metrics, and SHAP analysis.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import shap
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from pathlib import Path
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib

    from mathanx.ml.config import (
        DATASET_PATH,
        FIVE_FEATURE_COLUMNS,
        LEAKAGE_COLS,
        PCA_COMPONENT_COLUMNS,
        PCA_TRANSFORM_PATH,
        PSYCH_SCORE_COLUMNS,
        RANDOM_STATE,
        TARGET,
        TOP_PERFORMERS,
        TREE_MODEL_NAMES,
        FIG_PATH,
    )
    from mathanx.ml.helpers import (
        build_linear_pipeline,
        build_tree_pipeline,
        classify_columns,
        load_experiment,
        make_model_specs,
        run_shap_analysis,
        plot_shap_beeswarm,
        plot_shap_bar,
    )

    sns.set_theme(style="whitegrid", context="notebook")
    return (
        DATASET_PATH,
        FIG_PATH,
        FIVE_FEATURE_COLUMNS,
        LEAKAGE_COLS,
        PCA_COMPONENT_COLUMNS,
        PCA_TRANSFORM_PATH,
        PSYCH_SCORE_COLUMNS,
        Path,
        Pipeline,
        RANDOM_STATE,
        StandardScaler,
        TARGET,
        TOP_PERFORMERS,
        TREE_MODEL_NAMES,
        build_linear_pipeline,
        build_tree_pipeline,
        classify_columns,
        joblib,
        load_experiment,
        make_model_specs,
        mo,
        pd,
        plot_shap_bar,
        plot_shap_beeswarm,
        plt,
        run_shap_analysis,
        shap,
        sns,
    )


@app.cell
def _(Path):
    _all_dirs = sorted(
        [d.name for d in Path("models").iterdir() if d.is_dir()]
    )
    AVAILABLE = {d: d for d in _all_dirs}
    return (AVAILABLE,)


@app.cell
def _(AVAILABLE, mo):
    EXP_SELECT = mo.ui.dropdown(
        options=AVAILABLE,
        value=list(AVAILABLE.keys())[0] if AVAILABLE else None,
        label="Select experiment directory",
    )
    EXP_SELECT
    return (EXP_SELECT,)


@app.cell
def _(
    DATASET_PATH,
    EXP_SELECT,
    FIVE_FEATURE_COLUMNS,
    LEAKAGE_COLS,
    PCA_COMPONENT_COLUMNS,
    PCA_TRANSFORM_PATH,
    PSYCH_SCORE_COLUMNS,
    Pipeline,
    RANDOM_STATE,
    StandardScaler,
    TARGET,
    TOP_PERFORMERS,
    build_linear_pipeline,
    build_tree_pipeline,
    classify_columns,
    joblib,
    make_model_specs,
    pd,
):
    _BASE_NAMES = [
        "all_features",
        "no_model",
        "five_predictors",
        "pca_predictors",
        "pca_with_model",
    ]
    dirname = EXP_SELECT.value

    if dirname is None:
        X = y = model_specs = dataset_info = base_name = dir_error = None
    else:
        dir_error = None
        base_name = None
        _has_no_top = False
        _model_slug = None
        _rest = ""
        for _bn in _BASE_NAMES:
            if dirname == _bn or dirname.startswith(_bn + "_"):
                base_name = _bn
                _rest = dirname[len(_bn):]
                break
        if _rest:
            if _rest.startswith("_no_top"):
                _has_no_top = True
                _rest = _rest[len("_no_top"):]
            if _rest.startswith("_"):
                _model_slug = _rest[1:]

        _ml_df = pd.read_csv(DATASET_PATH)

        if _has_no_top:
            _ml_df = _ml_df[~_ml_df["Model"].isin(TOP_PERFORMERS)]

        if _model_slug:
            for _m in _ml_df["Model"].unique():
                _computed = (
                    _m.lower()
                    .replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace(".", "_")
                )
                if _computed == _model_slug:
                    _ml_df = _ml_df[_ml_df["Model"] == _m].reset_index(drop=True)
                    break

        _feature_cols = [
            c for c in _ml_df.columns
            if c not in LEAKAGE_COLS
            and c not in {TARGET, "education_vs_parent_mean_gap"}
        ]
        dataset_info = f"{len(_ml_df)} rows"

        if base_name == "five_predictors":
            X = _ml_df.loc[:, FIVE_FEATURE_COLUMNS].copy()
            y = _ml_df[TARGET].copy()
            model_specs = make_model_specs(
                lambda m: Pipeline([("scale", StandardScaler()), ("model", m)]),
                lambda m: Pipeline([("model", m)]),
                random_state=RANDOM_STATE,
            )
        elif base_name in ("pca_predictors", "pca_with_model"):
            _pca_data = joblib.load(PCA_TRANSFORM_PATH)
            _pc_scores = _pca_data["pca"].transform(
                _pca_data["scaler"].transform(_ml_df[PSYCH_SCORE_COLUMNS])
            )
            X = _ml_df.loc[:, _feature_cols].copy()
            X = X.drop(columns=PSYCH_SCORE_COLUMNS)
            X[PCA_COMPONENT_COLUMNS[0]] = _pc_scores[:, 0]
            X[PCA_COMPONENT_COLUMNS[1]] = _pc_scores[:, 1]
            y = _ml_df[TARGET].copy()
            _col_types = classify_columns(X)
            _num_f = _col_types["numeric_features"]
            _nom_f = _col_types["nominal_features"]
            _tree_f = _col_types["tree_nominal_features"]
            if base_name == "pca_predictors":
                _num_f = [c for c in _num_f if c != "Model"]
                _nom_f = [c for c in _nom_f if c != "Model"]
                _tree_f = [c for c in _tree_f if c != "Model"]
            model_specs = make_model_specs(
                lambda m: build_linear_pipeline(m, _num_f, _nom_f),
                lambda m: build_tree_pipeline(m, _num_f, _tree_f),
                random_state=RANDOM_STATE,
            )
        else:
            X = _ml_df.loc[:, _feature_cols].copy()
            y = _ml_df[TARGET].copy()
            _col_types = classify_columns(X)
            _num_f = _col_types["numeric_features"]
            _nom_f = _col_types["nominal_features"]
            _tree_f = _col_types["tree_nominal_features"]
            if base_name == "no_model":
                _num_f = [c for c in _num_f if c != "Model"]
                _nom_f = [c for c in _nom_f if c != "Model"]
                _tree_f = [c for c in _tree_f if c != "Model"]
            model_specs = make_model_specs(
                lambda m: build_linear_pipeline(m, _num_f, _nom_f),
                lambda m: build_tree_pipeline(m, _num_f, _tree_f),
                random_state=RANDOM_STATE,
            )
    return X, base_name, dataset_info, dir_error, dirname, model_specs, y


@app.cell
def _(Path, dir_error, dirname, load_experiment):
    if dir_error or dirname is None:
        EXP_ERROR = dir_error or "No experiment selected."
        model_summary_df = None
        best_params_by_model = None
        best_model_name = None
        tuned_cv_results_df = None
        permutation_importance_df = None
        best_model_pipeline = None
    else:
        _cache_dir = Path("models") / dirname
        if not _cache_dir.exists():
            EXP_ERROR = f"Directory {_cache_dir} does not exist."
            model_summary_df = best_params_by_model = best_model_name = None
            tuned_cv_results_df = permutation_importance_df = best_model_pipeline = None
        else:
            EXP_ERROR = None
            _result = load_experiment(_cache_dir)
            model_summary_df = _result.model_summary
            best_params_by_model = _result.best_params_by_model
            best_model_name = _result.best_model_name
            tuned_cv_results_df = _result.tuned_cv_results
            permutation_importance_df = _result.permutation_importance
            best_model_pipeline = _result.final_estimator
    return (
        EXP_ERROR,
        best_model_name,
        best_model_pipeline,
        best_params_by_model,
        model_summary_df,
        permutation_importance_df,
        tuned_cv_results_df,
    )


@app.cell
def _(EXP_ERROR, best_model_name, dataset_info, dirname, mo, model_summary_df):
    if EXP_ERROR:
        mo.callout(EXP_ERROR, kind="danger")
    else:
        mo.md(
            f"**Experiment:** `{dirname}` — {dataset_info}  \n"
            f"**Best model:** {best_model_name}"
        )
        model_summary_df.head()
    return


@app.cell
def _(EXP_ERROR, mo):
    if EXP_ERROR:
        METRIC = None
    else:
        METRIC = mo.ui.radio(
            options=["mean_rmse", "mean_mae", "mean_r2"],
            value="mean_r2",
            label="Metric",
        )
        METRIC
    return (METRIC,)


@app.cell
def _(EXP_ERROR, METRIC, model_summary_df, plt, sns):
    if not EXP_ERROR and METRIC is not None:
        _fig, _ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=model_summary_df, x=METRIC.value, y="model", ax=_ax)
        _ax.set_title(f"Model comparison ({METRIC.value})")
        _fig.tight_layout()
        _fig
    return


@app.cell
def _(EXP_ERROR, mo, model_summary_df, tuned_cv_results_df):
    if EXP_ERROR or tuned_cv_results_df is None:
        if not EXP_ERROR:
            mo.callout("No tuned CV results available.", kind="info")
    else:
        _compared = model_summary_df.merge(
            tuned_cv_results_df,
            on="model",
            suffixes=("_nested", "_tuned"),
        )
        _cols = [
            "model",
            "mean_r2_nested",
            "mean_r2_tuned",
            "mean_rmse_nested",
            "mean_rmse_tuned",
        ]
        _display = (
            _compared[_cols]
            .sort_values("mean_r2_tuned", ascending=False)
            .reset_index(drop=True)
        )
        mo.md("### Nested CV vs tuned CV")
        _display
    return


@app.cell
def _(EXP_ERROR, mo, permutation_importance_df, plt, sns):
    if EXP_ERROR or permutation_importance_df is None:
        if not EXP_ERROR:
            mo.callout("No permutation importance available.", kind="info")
    else:
        _top = permutation_importance_df.sort_values(
            "importance_mean", ascending=False
        ).head(10)
        _fig, _ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=_top, x="importance_mean", y="feature", ax=_ax)
        _ax.set_title("Top 10 features by permutation importance")
        _fig.tight_layout()
        _fig
        plt.close(_fig)
    return


@app.cell
def _(
    EXP_ERROR,
    TREE_MODEL_NAMES,
    X,
    base_name,
    best_model_name,
    best_model_pipeline,
    best_params_by_model,
    model_specs,
    model_summary_df,
    run_shap_analysis,
    shap,
    tuned_cv_results_df,
    y,
):
    if EXP_ERROR or X is None or best_model_pipeline is None:
        shap_values = None
        shap_model_name = None
        shap_values_filtered = None
    else:
        _model_candidates = (
            tuned_cv_results_df
            if tuned_cv_results_df is not None
            else model_summary_df
        )

        shap_values, shap_model_name = run_shap_analysis(
            X,
            y,
            model_specs,
            best_model_name,
            best_params_by_model,
            _model_candidates,
            TREE_MODEL_NAMES,
            pipeline=best_model_pipeline,
            shap_sample_size=1000,
            shap_random_state=42,
            check_linear=(base_name == "five_predictors"),
        )

        _indices = [
            i
            for i, _name in enumerate(shap_values.feature_names)
            if _name != "Model"
        ]
        shap_values_filtered = shap.Explanation(
            values=shap_values.values[:, _indices],
            base_values=shap_values.base_values,
            data=(
                shap_values.data[:, _indices]
                if shap_values.data is not None
                else None
            ),
            feature_names=[shap_values.feature_names[i] for i in _indices],
        )
    return shap_model_name, shap_values, shap_values_filtered


@app.cell
def _(
    EXP_ERROR,
    EXP_SELECT,
    FIG_PATH,
    plot_shap_bar,
    plot_shap_beeswarm,
    plt,
    shap_model_name,
    shap_values_filtered,
):
    if not (EXP_ERROR or shap_values_filtered is None):
        _dirname = EXP_SELECT.value
        _title = (
            f"SHAP for {shap_model_name} ({_dirname}, Model excluded)"
        )

        _fig = plot_shap_beeswarm(shap_values_filtered, _title)
        _fig.savefig(
            FIG_PATH / f"beeswarm_{_dirname}_model_excluded.pdf",
            format="pdf",
        )
        _fig.savefig(
            FIG_PATH / f"beeswarm_{_dirname}_model_excluded.png",
            format="png",
        )
        _fig
        plt.close(_fig)

        _fig = plot_shap_bar(shap_values_filtered, _title)
        _fig
        plt.close(_fig)
    return


@app.cell
def _(
    EXP_ERROR,
    EXP_SELECT,
    plot_shap_bar,
    plot_shap_beeswarm,
    plt,
    shap_model_name,
    shap_values,
):
    if not (EXP_ERROR or shap_values is None):
        _dirname = EXP_SELECT.value
        _title = f"SHAP for {shap_model_name} ({_dirname})"

        _fig = plot_shap_beeswarm(shap_values, _title)
        _fig
        plt.close(_fig)

        _fig = plot_shap_bar(shap_values, _title)
        _fig
        plt.close(_fig)
    return


if __name__ == "__main__":
    app.run()
