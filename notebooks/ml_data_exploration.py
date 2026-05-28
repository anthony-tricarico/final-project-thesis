import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.stats import (
        f_oneway,
        kruskal,
        kurtosis,
        pearsonr,
        shapiro,
        skew,
        spearmanr,
    )
    from sklearn.ensemble import IsolationForest

    from mathanx.ml.config import DATASET_PATH, FIG_PATH, LEAKAGE_COLS, TARGET

    sns.set_theme(style="whitegrid", context="notebook")
    return (
        DATASET_PATH,
        FIG_PATH,
        IsolationForest,
        LEAKAGE_COLS,
        TARGET,
        f_oneway,
        kruskal,
        kurtosis,
        leaves_list,
        linkage,
        mo,
        np,
        pd,
        pearsonr,
        plt,
        shapiro,
        skew,
        sns,
        spearmanr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ML Dataset — Exploratory Data Analysis

    Comprehensive exploration of the engineered ML dataset to understand feature distributions, target
    characteristics, correlations, model effects, and potential data quality issues before modeling.
    """)
    return


@app.cell
def _(DATASET_PATH, mo, pd):
    if not DATASET_PATH.exists():
        mo.callout(
            mo.md(
                f"The engineered dataset was not found at `{DATASET_PATH}`. Run `ml_dataset_creation.py` first."
            ),
            kind="warn",
        )
        raise FileNotFoundError(DATASET_PATH)

    ml_df = pd.read_csv(DATASET_PATH, low_memory=False)

    for _c in ml_df.select_dtypes(include="str").columns:
        ml_df[_c] = ml_df[_c].str.strip()

    ml_df.head()
    return (ml_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Dataset Overview
    """)
    return


@app.cell
def _(LEAKAGE_COLS, TARGET, ml_df, pd):
    _n_rows = len(ml_df)
    _n_cols = ml_df.shape[1]
    _duplicate_run_ids = int(ml_df["run_id"].duplicated().sum())

    _feature_count = _n_cols - 1
    _leakage_count = len(LEAKAGE_COLS - {TARGET})

    overview_df = pd.DataFrame(
        {
            "metric": [
                "rows",
                "columns",
                "features (excl. target)",
                "leakage columns",
                "duplicate run_ids",
                "memory (MB)",
            ],
            "value": [
                f"{_n_rows:,}",
                _n_cols,
                _feature_count,
                _leakage_count,
                _duplicate_run_ids,
                f"{ml_df.memory_usage(deep=True).sum() / 1024 ** 2:.2f}",
            ],
        }
    )
    overview_df
    return


@app.cell
def _(ml_df):
    dtypes_df = (
        ml_df.dtypes.reset_index()
        .rename(columns={"index": "column", 0: "dtype"})
        .assign(dtype=lambda _df: _df["dtype"].astype(str))
        .merge(
            ml_df.nunique().reset_index().rename(columns={"index": "column", 0: "unique"}),
            on="column",
        )
        .merge(
            ml_df.isna().sum().reset_index().rename(columns={"index": "column", 0: "missing"}),
            on="column",
        )
        .assign(missing_rate=lambda _df: (_df["missing"] / len(ml_df)).round(4))
        .sort_values(["dtype", "column"])
        .reset_index(drop=True)
    )
    dtypes_df
    return


@app.cell
def _(ml_df):
    ml_df.describe(include="all").T
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Missing Value Analysis
    """)
    return


@app.cell
def _(ml_df, pd):
    missing_summary = (
        ml_df.isna()
        .sum()
        .rename("missing_count")
        .to_frame()
        .assign(missing_rate=lambda _df: _df["missing_count"] / len(ml_df))
        .query("missing_count > 0")
        .assign(
            missing_rate=lambda _df: _df["missing_rate"].map("{:.2%}".format),
        )
        .sort_values("missing_count", ascending=False)
    )

    if missing_summary.empty:
        missing_summary = pd.DataFrame(
            {"missing_count": [0], "missing_rate": ["0.00%"]},
            index=["no missing values"],
        )

    missing_summary
    return


@app.cell
def _(FIG_PATH, ml_df, plt, sns):
    if ml_df.isna().any().any():
        _fig, _ax = plt.subplots(figsize=(12, 1.5))
        sns.heatmap(
            ml_df.isna().T,
            cbar=False,
            xticklabels=False,
            yticklabels=True,
            cmap="RdBu_r",
            ax=_ax,
        )
        _ax.set_title("Missing Value Matrix", fontsize=11)
        _ax.set_ylabel("")
        _fig.tight_layout()
        _fig.savefig(FIG_PATH / "missing_value_heatmap.pdf", format="pdf")
        _fig.savefig(FIG_PATH / "missing_value_heatmap.png", format="png")
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Target Variable — `accuracy`
    """)
    return


@app.cell
def _(TARGET, kurtosis, ml_df, skew):
    target = ml_df[TARGET]
    target_summary = target.describe().to_frame().T
    target_summary["skew"] = skew(target.dropna())
    target_summary["kurtosis"] = kurtosis(target.dropna(), fisher=False)
    target_summary
    return (target,)


@app.cell
def _(FIG_PATH, plt, sns, target):
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4))

    sns.histplot(target, kde=True, bins=30, ax=_axes[0], edgecolor="white")
    _axes[0].set_title("Distribution with KDE")
    _axes[0].set_xlabel("Accuracy")

    sns.boxplot(x=target, ax=_axes[1], width=0.4)
    _axes[1].set_title("Boxplot")
    _axes[1].set_xlabel("Accuracy")

    sns.ecdfplot(target, ax=_axes[2])
    _axes[2].set_title("ECDF")
    _axes[2].set_xlabel("Accuracy")
    _axes[2].set_ylabel("Cumulative proportion")

    _fig.tight_layout()
    _fig.savefig(FIG_PATH / "accuracy_distribution.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "accuracy_distribution.png", format="png")
    plt.show()
    return


@app.cell
def _(plt, shapiro, target):
    import statsmodels.api as sm
    # Assuming 'target' is your pandas Series
    # Drop NaNs once and store it to keep the code clean and efficient
    target_clean = target.dropna()

    _n_sample = min(5000, len(target_clean))

    # Perform Shapiro-Wilk test on a random sample
    _shapiro_stat, _shapiro_p = shapiro(
        target_clean.sample(_n_sample, random_state=42)
    )

    _fig, _ax = plt.subplots(figsize=(6, 4))

    # Use statsmodels to plot directly onto the matplotlib axis (_ax)
    # line='s' adds a standardized reference line to easily check for normality
    sm.qqplot(target_clean, line='s', ax=_ax, alpha=0.5)

    _ax.set_title(f"Q-Q Plot (Shapiro-Wilk p={_shapiro_p:.2e}, n={_n_sample})")
    _fig.tight_layout()
    plt.show()
    return


@app.cell
def _(pd, target):
    _q1 = target.quantile(0.25)
    _q3 = target.quantile(0.75)
    _iqr = _q3 - _q1
    _lower = _q1 - 1.5 * _iqr
    _upper = _q3 + 1.5 * _iqr
    _outliers = target[(target < _lower) | (target > _upper)]
    _outlier_pct = 100 * len(_outliers) / len(target)

    outlier_iqr_df = pd.DataFrame(
        {
            "metric": [
                "Q1",
                "Q3",
                "IQR",
                "lower fence",
                "upper fence",
                "outlier count",
                "outlier %",
            ],
            "value": [
                f"{_q1:.4f}",
                f"{_q3:.4f}",
                f"{_iqr:.4f}",
                f"{_lower:.4f}",
                f"{_upper:.4f}",
                len(_outliers),
                f"{_outlier_pct:.2f}%",
            ],
        }
    )
    outlier_iqr_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Numerical Feature Distributions
    """)
    return


@app.cell
def _(LEAKAGE_COLS, TARGET, ml_df, pd):
    _all_numeric = ml_df.select_dtypes(include="number").columns.tolist()
    _all_numeric = [_c for _c in _all_numeric if _c not in {"run_id"}]

    leakage = [_c for _c in LEAKAGE_COLS if _c in _all_numeric and _c != TARGET]
    predictors = [_c for _c in _all_numeric if _c not in LEAKAGE_COLS and _c != TARGET]

    _numeric_profile = []
    for _col in predictors + leakage:
        _s = ml_df[_col].dropna()
        _numeric_profile.append(
            {
                "feature": _col,
                "type": "predictor" if _col in predictors else "leakage",
                "count": len(_s),
                "mean": _s.mean(),
                "std": _s.std(),
                "min": _s.min(),
                "p25": _s.quantile(0.25),
                "p50": _s.median(),
                "p75": _s.quantile(0.75),
                "max": _s.max(),
                #"skew": skew(_s),
            }
        )
    numeric_profile_df = pd.DataFrame(_numeric_profile).sort_values(["type", "feature"])
    numeric_profile_df
    return leakage, predictors


@app.cell
def _(FIG_PATH, leakage, ml_df, np, plt, predictors, sns):
    _plot_cols = predictors + leakage

    _n_cols = 4
    _n_rows = int(np.ceil(len(_plot_cols) / _n_cols))

    _fig, _axes = plt.subplots(
        _n_rows, _n_cols, figsize=(_n_cols * 3.5, _n_rows * 2.5)
    )
    _axes = _axes.flatten() if _n_rows > 1 else [_axes] if _n_cols == 1 else _axes

    for _idx, _col in enumerate(_plot_cols):
        _ax = _axes[_idx]
        _data = ml_df[_col].dropna()
        sns.histplot(_data, kde=True, bins=30, edgecolor="white", ax=_ax)
        _ax.set_title(_col, fontsize=9)
        _ax.set_xlabel("")

    for _j in range(len(_plot_cols), len(_axes)):
        _axes[_j].set_visible(False)

    _fig.suptitle("Numerical Feature Distributions", fontsize=13, y=1.01)
    _fig.tight_layout()
    _fig.savefig(FIG_PATH / "numeric_distributions.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "numeric_distributions.png", format="png")
    plt.show()
    return


@app.cell
def _(FIG_PATH, leakage, ml_df, np, plt, predictors, sns):
    _box_cols = predictors + leakage

    _n_cols = 4
    _n_rows = int(np.ceil(len(_box_cols) / _n_cols))

    _fig, _axes = plt.subplots(
        _n_rows, _n_cols, figsize=(_n_cols * 3, _n_rows * 2.5)
    )
    _axes = _axes.flatten() if _n_rows > 1 else [_axes] if _n_cols == 1 else _axes

    for _idx, _col in enumerate(_box_cols):
        _ax = _axes[_idx]
        sns.boxplot(x=ml_df[_col], ax=_ax, width=0.4)
        _ax.set_title(_col, fontsize=9)
        _ax.set_xlabel("")

    for _j in range(len(_box_cols), len(_axes)):
        _axes[_j].set_visible(False)

    _fig.suptitle("Numerical Feature Boxplots", fontsize=13, y=1.01)
    _fig.tight_layout()
    _fig.savefig(FIG_PATH / "numeric_boxplots.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "numeric_boxplots.png", format="png")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Categorical Feature Analysis
    """)
    return


@app.cell
def _(ml_df, pd):
    _nominal_cols = [
        "gender",
        "sexual_orientation",
        "employment_status",
        "marital_status",
        "migration_status",
        "religious_beliefs",
        "Model",
    ]

    _cat_rows = []
    for _col in _nominal_cols:
        _vc = ml_df[_col].value_counts(dropna=False)
        _cat_rows.append(
            {
                "feature": _col,
                "n_unique": ml_df[_col].nunique(),
                "top_category": _vc.index[0],
                "top_freq": _vc.iloc[0],
                "top_rate": _vc.iloc[0] / len(ml_df),
            }
        )
    cat_summary_df = pd.DataFrame(_cat_rows)
    nominal_cols = _nominal_cols
    return (nominal_cols,)


@app.cell
def _(FIG_PATH, ml_df, nominal_cols, plt, sns):
    for _col in nominal_cols:
        _fig, _ax = plt.subplots(figsize=(8, 4))
        _vc = ml_df[_col].value_counts()
        _colors = sns.color_palette("muted", n_colors=len(_vc))
        sns.barplot(
            x=_vc.values,
            y=_vc.index,
            hue=_vc.index,
            palette=_colors,
            ax=_ax,
            legend=False,
        )
        for _container in _ax.containers:
            _ax.bar_label(_container, fmt="%d", fontsize=8)
        _ax.set_title(f"{_col} distribution", fontsize=11)
        _ax.set_xlabel("Count")
        _ax.set_ylabel("")
        _fig.tight_layout()
        _fig.savefig(FIG_PATH / f"cat_{_col}_distribution.pdf", format="pdf")
        plt.show()
    return


@app.cell
def _(FIG_PATH, ml_df, plt, sns):
    _top_n = 15
    _vc_city = ml_df["city_of_living"].value_counts().head(_top_n)

    _fig, _ax = plt.subplots(figsize=(9, 5))
    _colors = sns.color_palette("muted", n_colors=_top_n)
    sns.barplot(
        x=_vc_city.values,
        y=_vc_city.index,
        hue=_vc_city.index,
        palette=_colors,
        ax=_ax,
        legend=False,
    )
    for _container in _ax.containers:
        _ax.bar_label(_container, fmt="%d", fontsize=7)
    _ax.set_title(f"Top {_top_n} cities of living", fontsize=11)
    _ax.set_xlabel("Count")
    _fig.tight_layout()
    _fig.savefig(FIG_PATH / "city_of_living_top15.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "city_of_living_top15.png", format="png")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Correlation Analysis
    """)
    return


@app.cell
def _(TARGET, ml_df):
    _numeric_corr = ml_df.select_dtypes(include="number").drop(
        columns=["run_id"], errors="ignore"
    )

    pearson_corr = _numeric_corr.corr(method="pearson")

    _target_pearson = (
        pearson_corr[[TARGET]]
        .drop(index=TARGET)
        .sort_values(TARGET, ascending=False)
        .reset_index()
        .rename(columns={"index": "feature", TARGET: "pearson_r"})
    )
    _target_pearson["abs_r"] = _target_pearson["pearson_r"].abs()
    target_pearson = _target_pearson.sort_values("abs_r", ascending=False)
    return pearson_corr, target_pearson


@app.cell
def _(FIG_PATH, pearson_corr, plt, sns):
    _fig, _ax = plt.subplots(figsize=(10, 8))
    _cmap = sns.diverging_palette(250, 10, as_cmap=True)
    sns.heatmap(
        pearson_corr,
        cmap=_cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.6, "label": "Pearson r"},
        ax=_ax,
    )
    _ax.set_title("Pearson Correlation Matrix", fontsize=12)
    _fig.tight_layout()
    _fig.savefig(FIG_PATH / "pearson_correlation_heatmap.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "pearson_correlation_heatmap.png", format="png")
    plt.show()
    return


@app.cell
def _(FIG_PATH, ml_df, pd, plt, sns, spearmanr):
    _n_samples = min(5000, len(ml_df))
    _corr_sample = (
        ml_df.select_dtypes(include="number")
        .drop(columns=["run_id"], errors="ignore")
        .dropna()
        .sample(_n_samples, random_state=42)
    )

    _spearman_cols = _corr_sample.columns
    spearman_matrix = pd.DataFrame(
        index=_spearman_cols, columns=_spearman_cols, dtype=float
    )
    for _c1 in _spearman_cols:
        for _c2 in _spearman_cols:
            _r, _ = spearmanr(_corr_sample[_c1], _corr_sample[_c2])
            spearman_matrix.loc[_c1, _c2] = _r
    spearman_matrix = spearman_matrix.astype(float)

    _fig, _ax = plt.subplots(figsize=(10, 8))
    _cmap = sns.diverging_palette(250, 10, as_cmap=True)
    sns.heatmap(
        spearman_matrix,
        cmap=_cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.6, "label": "Spearman \u03c1"},
        ax=_ax,
    )
    _ax.set_title("Spearman Rank Correlation Matrix", fontsize=12)
    _fig.tight_layout()
    _fig.savefig(FIG_PATH / "spearman_correlation_heatmap.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "spearman_correlation_heatmap.png", format="png")
    plt.show()
    return


@app.cell
def _(FIG_PATH, leaves_list, linkage, pearson_corr, plt, sns):
    _link = linkage(pearson_corr, method="average", metric="euclidean")
    _order = leaves_list(_link)
    _clustered = pearson_corr.iloc[_order, _order]

    _fig, _ax = plt.subplots(figsize=(10, 8))
    _cmap = sns.diverging_palette(250, 10, as_cmap=True)
    sns.heatmap(
        _clustered,
        cmap=_cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.6, "label": "Pearson r"},
        ax=_ax,
    )
    _ax.set_title("Clustered Pearson Correlation Matrix", fontsize=12)
    _fig.tight_layout()
    _fig.savefig(FIG_PATH / "clustered_corr_heatmap_eda.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "clustered_corr_heatmap_eda.png", format="png")
    plt.show()
    return


@app.cell
def _(target_pearson):
    target_pearson.head(15).drop(columns=["abs_r"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Target vs. Numerical Features
    """)
    return


@app.cell
def _(FIG_PATH, TARGET, ml_df, pearsonr, plt, sns, target_pearson):
    _top_features = target_pearson.head(8)["feature"].tolist()

    _n_cols = 4
    _n_rows = 2
    _fig, _axes = plt.subplots(
        _n_rows, _n_cols, figsize=(_n_cols * 3.2, _n_rows * 2.8)
    )
    _axes = _axes.flatten()

    for _idx, _feat in enumerate(_top_features):
        _ax = _axes[_idx]
        _data = ml_df[[_feat, TARGET]].dropna()
        sns.regplot(
            data=_data,
            x=_feat,
            y=TARGET,
            scatter_kws={"alpha": 0.15, "s": 8, "color": "steelblue"},
            line_kws={"color": "red", "linewidth": 1.5},
            lowess=False,
            ax=_ax,
        )
        _r, _p = pearsonr(_data[_feat], _data[TARGET])
        _p_str = f"p={_p:.2e}" if _p < 0.001 else f"p={_p:.4f}"
        _ax.set_title(f"{_feat}  (r={_r:.3f}, {_p_str})", fontsize=9)
        _ax.set_xlabel("")

    _fig.suptitle("Top 8 Features vs. Accuracy (with OLS fit)", fontsize=13)
    _fig.tight_layout()
    _fig.savefig(FIG_PATH / "top_features_vs_accuracy.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "top_features_vs_accuracy.png", format="png")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Accuracy by Model
    """)
    return


@app.cell
def _(FIG_PATH, TARGET, ml_df, plt, sns):
    _model_order = (
        ml_df.groupby("Model")[TARGET]
        .median()
        .sort_values()
        .index.tolist()
    )

    _fig, _ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(
        data=ml_df,
        x="Model",
        y=TARGET,
        order=_model_order,
        width=0.6,
        fliersize=2,
        ax=_ax,
    )
    sns.stripplot(
        data=ml_df,
        x="Model",
        y=TARGET,
        order=_model_order,
        color="black",
        alpha=0.15,
        size=2,
        jitter=True,
        ax=_ax,
    )
    _ax.set_title("Accuracy Distribution by Model", fontsize=12)
    _ax.set_xlabel("")
    plt.setp(_ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    _fig.tight_layout()
    _fig.savefig(FIG_PATH / "accuracy_by_model.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "accuracy_by_model.png", format="png")
    plt.show()
    return


@app.cell
def _(TARGET, f_oneway, kruskal, ml_df, pd):
    _models = ml_df["Model"].unique()
    _groups = [ml_df.loc[ml_df["Model"] == _m, TARGET].dropna() for _m in _models]

    _anova_f, _anova_p = f_oneway(*_groups)
    _kw_h, _kw_p = kruskal(*_groups)

    model_stats = (
        ml_df.groupby("Model")[TARGET]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .round(4)
        .sort_values("mean", ascending=False)
        .reset_index()
    )

    test_results = pd.DataFrame(
        {
            "test": ["ANOVA", "Kruskal-Wallis"],
            "statistic": [f"{_anova_f:.4f}", f"{_kw_h:.4f}"],
            "p_value": [
                f"{_anova_p:.2e}",
                f"{_kw_p:.2e}",
            ],
            "significant": [
                "Yes" if _anova_p < 0.05 else "No",
                "Yes" if _kw_p < 0.05 else "No",
            ],
        }
    )
    test_results
    return (model_stats,)


@app.cell
def _(model_stats):
    model_stats
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Demographic Breakdown of Accuracy
    """)
    return


@app.cell
def _(TARGET, ml_df, pd, plt, sns):
    _age_bins = [0, 25, 35, 45, 55, 65, 100]
    _age_labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
    _ml_df = ml_df.assign(
        age_group=pd.cut(
            ml_df["age"], bins=_age_bins, labels=_age_labels, right=True
        )
    )

    _demo_cols = [
        "age_group",
        "gender",
        "sexual_orientation",
        "employment_status",
        "marital_status",
    ]

    for _col in _demo_cols:
        _fig, _ax = plt.subplots(figsize=(9, 4))
        _order = (
            _ml_df.groupby(_col)[TARGET]
            .median()
            .sort_values()
            .index.tolist()
        )
        sns.boxplot(
            data=_ml_df,
            x=_col,
            y=TARGET,
            order=_order,
            width=0.5,
            fliersize=2,
            ax=_ax,
        )
        _ax.set_title(f"Accuracy by {_col}", fontsize=11)
        _ax.set_xlabel("")
        if _ml_df[_col].nunique() > 6:
            plt.setp(_ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        _fig.tight_layout()
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Psychometric Scores Analysis
    """)
    return


@app.cell
def _():
    psych_cols = ["amas_score", "maes_score", "mseaq_anx", "mseaq_se"]
    return (psych_cols,)


@app.cell
def _(FIG_PATH, ml_df, plt, psych_cols, sns):
    _g = sns.pairplot(
        data=ml_df,
        vars=psych_cols,
        diag_kind="kde",
        plot_kws={"alpha": 0.15, "s": 5},
        diag_kws={"alpha": 0.5},
    )
    _g.fig.suptitle("Psychometric Score Relationships", y=1.02, fontsize=13)
    _g.fig.savefig(FIG_PATH / "psychometric_pairplot.pdf", format="pdf")
    _g.fig.savefig(FIG_PATH / "psychometric_pairplot.png", format="png")
    plt.show()
    return


@app.cell
def _(TARGET, ml_df, psych_cols):
    _psych_with_target = ml_df[psych_cols + [TARGET]].dropna()
    psych_corr = _psych_with_target.corr(method="pearson").round(4)

    psych_target_corr = (
        psych_corr[[TARGET]]
        .drop(index=TARGET)
        .sort_values(TARGET, ascending=False)
    )
    psych_target_corr
    return (psych_corr,)


@app.cell
def _(FIG_PATH, plt, psych_corr, sns):
    _fig, _ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        psych_corr,
        annot=True,
        fmt=".3f",
        cmap=sns.diverging_palette(250, 10, as_cmap=True),
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        ax=_ax,
    )
    _ax.set_title("Psychometric Score Correlations", fontsize=11)
    _fig.tight_layout()
    _fig.savefig(FIG_PATH / "psychometric_correlation_heatmap.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "psychometric_correlation_heatmap.png", format="png")
    plt.show()
    return


@app.cell
def _(FIG_PATH, TARGET, ml_df, pearsonr, plt, psych_cols, sns):
    _fig, _axes = plt.subplots(1, len(psych_cols), figsize=(14, 3.5))

    for _idx, _col in enumerate(psych_cols):
        _ax = _axes[_idx]
        _data = ml_df[[_col, TARGET]].dropna()
        sns.regplot(
            data=_data,
            x=_col,
            y=TARGET,
            scatter_kws={"alpha": 0.1, "s": 6, "color": "steelblue"},
            line_kws={"color": "red", "linewidth": 1.5},
            ax=_ax,
        )
        _r, _ = pearsonr(_data[_col], _data[TARGET])
        _ax.set_title(f"{_col} (r={_r:.3f})", fontsize=10)
        _ax.set_xlabel("")

    _fig.suptitle("Psychometric Scores vs. Accuracy", fontsize=13)
    _fig.tight_layout()
    _fig.savefig(FIG_PATH / "psychometric_vs_accuracy.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "psychometric_vs_accuracy.png", format="png")
    plt.show()
    return


@app.cell
def _(FIG_PATH, TARGET, ml_df, pearsonr, plt, psych_cols, sns):
    from mathanx.ml.config import TOP_PERFORMERS

    _fig, _axes = plt.subplots(1, len(psych_cols), figsize=(14, 3.5))

    for _idx, _col in enumerate(psych_cols):
        _ax = _axes[_idx]
        _data = ml_df[[_col, TARGET]][~(ml_df["Model"].isin(TOP_PERFORMERS))].dropna()
        sns.regplot(
            data=_data,
            x=_col,
            y=TARGET,
            scatter_kws={"alpha": 0.1, "s": 6, "color": "steelblue"},
            line_kws={"color": "red", "linewidth": 1.5},
            ax=_ax,
        )
        _r, _ = pearsonr(_data[_col], _data[TARGET])
        _ax.set_title(f"{_col} (r={_r:.3f})", fontsize=10)
        _ax.set_xlabel("")

    _fig.suptitle("Psychometric Scores vs. Accuracy", fontsize=13)
    _fig.tight_layout()
    _fig.savefig(FIG_PATH / "psychometric_vs_accuracy_no_top.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "psychometric_vs_accuracy_no_top.png", format="png")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 10a. Psychometric Scores by Model
    """)
    return


@app.cell
def _(FIG_PATH, ml_df, np, plt, psych_cols):
    _fig, _axes = plt.subplots(2, 2, figsize=(16, 10))
    _axes = _axes.flatten()

    for _idx, _col in enumerate(psych_cols):
        _ax = _axes[_idx]
        _stats = ml_df.groupby("Model")[_col].agg(["mean", "std", "count"])
        _stats["ci"] = 1.96 * _stats["std"] / np.sqrt(_stats["count"])
        _stats = _stats.sort_values("mean", ascending=False)

        _models = _stats.index.tolist()
        _means = _stats["mean"].values
        _cis = _stats["ci"].values

        _ax.bar(
            range(len(_models)), _means, yerr=_cis, capsize=3,
            color="steelblue", edgecolor="black", linewidth=0.5,
        )
        _ax.set_xticks(range(len(_models)))
        _ax.set_xticklabels(_models, rotation=45, ha="right", fontsize=7)
        _ax.set_ylabel(f"Mean {_col}")
        _ax.set_title(f"{_col} by Model", fontsize=11)

    _fig.suptitle("Psychometric Score Means by Model", fontsize=14, y=1.02)
    _fig.tight_layout()
    _fig.savefig(FIG_PATH / "psychometric_scores_by_model.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "psychometric_scores_by_model.png", format="png")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Ordinal & Derived Feature Analysis
    """)
    return


@app.cell
def _():
    ordinal_cols = [
        "education_level_ord",
        "parent_1_education_ord",
        "parent_2_education_ord",
        "parent_education_mean_ord",
        "parent_education_gap_ord",
    ]
    ocean_cols = [
        "ocean_openness_level_ord",
        "ocean_conscientiousness_level_ord",
        "ocean_extraversion_level_ord",
        "ocean_agreeableness_level_ord",
        "ocean_neuroticism_level_ord",
    ]
    flag_cols = ["has_children_flg", "math_lover_flg", "math_hater_flg"]
    return flag_cols, ocean_cols, ordinal_cols


@app.cell
def _(FIG_PATH, TARGET, ml_df, np, ordinal_cols, plt, sns):
    _n_cols = 3
    _n_rows = int(np.ceil(len(ordinal_cols) / _n_cols))
    _fig, _axes = plt.subplots(
        _n_rows, _n_cols, figsize=(_n_cols * 3.5, _n_rows * 3)
    )
    _axes = _axes.flatten()

    for _idx, _col in enumerate(ordinal_cols):
        _ax = _axes[_idx]
        sns.boxplot(
            data=ml_df,
            x=_col,
            y=TARGET,
            ax=_ax,
            width=0.5,
            fliersize=2,
        )
        _ax.set_title(_col, fontsize=9)
        _ax.set_xlabel("")

    for _j in range(len(ordinal_cols), len(_axes)):
        _axes[_j].set_visible(False)

    _fig.suptitle("Ordinal Education Features vs. Accuracy", fontsize=13)
    _fig.tight_layout()
    _fig.savefig(FIG_PATH / "ordinal_education_vs_accuracy.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "ordinal_education_vs_accuracy.png", format="png")
    plt.show()
    return


@app.cell
def _(FIG_PATH, TARGET, ml_df, np, ocean_cols, plt, sns):
    _n_cols = 3
    _n_rows = int(np.ceil(len(ocean_cols) / _n_cols))
    _fig, _axes = plt.subplots(
        _n_rows, _n_cols, figsize=(_n_cols * 3.5, _n_rows * 3)
    )
    _axes = _axes.flatten()

    for _idx, _col in enumerate(ocean_cols):
        _ax = _axes[_idx]
        sns.boxplot(
            data=ml_df,
            x=_col,
            y=TARGET,
            ax=_ax,
            width=0.5,
            fliersize=2,
        )
        _ax.set_title(_col, fontsize=9)
        _ax.set_xlabel("")

    for _j in range(len(ocean_cols), len(_axes)):
        _axes[_j].set_visible(False)

    _fig.suptitle("OCEAN Personality Levels vs. Accuracy", fontsize=13)
    _fig.tight_layout()
    _fig.savefig(FIG_PATH / "ocean_levels_vs_accuracy.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "ocean_levels_vs_accuracy.png", format="png")
    plt.show()
    return


@app.cell
def _(FIG_PATH, TARGET, flag_cols, ml_df, plt, sns):
    _fig, _axes = plt.subplots(
        1, len(flag_cols), figsize=(len(flag_cols) * 3.5, 3.5)
    )

    for _idx, _col in enumerate(flag_cols):
        _ax = _axes[_idx]
        sns.boxplot(
            data=ml_df,
            x=_col,
            y=TARGET,
            ax=_ax,
            width=0.4,
            fliersize=2,
        )
        _ax.set_title(_col, fontsize=9)
        _ax.set_xlabel("")

    _fig.suptitle("Binary Flag Features vs. Accuracy", fontsize=13)
    _fig.tight_layout()
    _fig.savefig(FIG_PATH / "flag_features_vs_accuracy.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "flag_features_vs_accuracy.png", format="png")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. Feature Interactions
    """)
    return


@app.cell
def _(FIG_PATH, TARGET, ml_df, plt, sns):
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4.5))

    _ax = _axes[0]
    _scatter = _ax.scatter(
        ml_df["mseaq_anx"],
        ml_df["mseaq_se"],
        c=ml_df[TARGET],
        cmap="viridis",
        alpha=0.3,
        s=5,
    )
    _ax.set_xlabel("mseaq_anx")
    _ax.set_ylabel("mseaq_se")
    _ax.set_title("Anxiety vs Self-Efficacy (colored by accuracy)")
    plt.colorbar(_scatter, ax=_ax, label="accuracy")

    _ax = _axes[1]
    _top_models = ml_df["Model"].value_counts().head(4).index.tolist()
    _subset = ml_df[ml_df["Model"].isin(_top_models)]
    sns.scatterplot(
        data=_subset,
        x="amas_score",
        y="maes_score",
        hue="Model",
        alpha=0.3,
        s=8,
        ax=_ax,
    )
    _ax.set_title("AMAS vs MAES by Top 4 Models")

    _ax = _axes[2]
    sns.boxplot(
        data=ml_df,
        x="migration_status",
        y=TARGET,
        hue="gender",
        ax=_ax,
        width=0.6,
        fliersize=2,
    )
    _ax.set_title("Migration × Gender vs Accuracy")
    plt.setp(_ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)

    _fig.suptitle("Feature Interactions", fontsize=13)
    _fig.tight_layout()
    _fig.savefig(FIG_PATH / "feature_interactions.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "feature_interactions.png", format="png")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 13. Outlier & Anomaly Detection
    """)
    return


@app.cell
def _(ml_df, np, pd):
    _numeric_for_outliers = ml_df.select_dtypes(include="number").drop(
        columns=["run_id"], errors="ignore"
    )

    _z_scores = np.abs(
        (_numeric_for_outliers - _numeric_for_outliers.mean())
        / _numeric_for_outliers.std()
    )

    _any_outlier = (_z_scores > 3).any(axis=1)
    _outlier_count = _any_outlier.sum()
    _outlier_rate = _outlier_count / len(ml_df)

    most_outlier_features = (
        (_z_scores > 3)
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .rename("outlier_count")
        .to_frame()
        .assign(outlier_rate=lambda _df: _df["outlier_count"] / len(ml_df))
    )

    outlier_summary = pd.DataFrame(
        {
            "metric": [
                "rows with any |z| > 3",
                "outlier rate",
            ],
            "value": [
                f"{_outlier_count:,} ({_outlier_rate:.2%})",
                f"{_outlier_rate:.2%}",
            ],
        }
    )
    outlier_summary
    return (most_outlier_features,)


@app.cell
def _(most_outlier_features):
    most_outlier_features
    return


@app.cell
def _(FIG_PATH, IsolationForest, ml_df, plt):
    _iso_features = ["age", "amas_score", "maes_score", "mseaq_anx", "mseaq_se"]
    _iso_data = ml_df[_iso_features].dropna().sample(5000, random_state=42)

    _iso = IsolationForest(
        n_estimators=100, contamination="auto", random_state=42, n_jobs=-1
    )
    _iso_preds = _iso.fit_predict(_iso_data)
    _iso_anomaly_rate = (_iso_preds == -1).mean()

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))

    _ax = _axes[0]
    _colors = ["steelblue" if _p == 1 else "red" for _p in _iso_preds]
    _ax.scatter(
        _iso_data["age"],
        _iso_data["amas_score"],
        c=_colors,
        alpha=0.4,
        s=6,
    )
    _ax.set_xlabel("age")
    _ax.set_ylabel("amas_score")
    _ax.set_title(
        f"IsolationForest anomalies ({_iso_anomaly_rate:.1%})", fontsize=10
    )

    _ax = _axes[1]
    _ax.scatter(
        _iso_data["mseaq_anx"],
        _iso_data["mseaq_se"],
        c=_colors,
        alpha=0.4,
        s=6,
    )
    _ax.set_xlabel("mseaq_anx")
    _ax.set_ylabel("mseaq_se")
    _ax.set_title(
        f"Detected {int((_iso_preds == -1).sum())} anomalies", fontsize=10
    )

    _fig.tight_layout()
    _fig.savefig(FIG_PATH / "anomaly_detection.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "anomaly_detection.png", format="png")
    plt.show()
    return


@app.cell
def _(TARGET, ml_df, pd):
    _extreme_accuracy = ml_df[ml_df[TARGET].isin([0.0, 1.0])]
    _n_extreme = len(_extreme_accuracy)

    if _n_extreme > 0:
        extreme_summary = (
            _extreme_accuracy.groupby("Model")[TARGET]
            .agg(["count", "mean"])
            .rename(columns={"count": "n_extreme", "mean": "mean_accuracy"})
            .sort_values("n_extreme", ascending=False)
            .reset_index()
        )
    else:
        extreme_summary = pd.DataFrame(
            {"Model": ["none"], "n_extreme": [0], "mean_accuracy": [0.0]}
        )

    extreme_info = pd.DataFrame(
        {
            "metric": ["accuracy = 0.0 or 1.0 rows"],
            "value": [f"{_n_extreme:,}"],
        }
    )
    extreme_info
    return (extreme_summary,)


@app.cell
def _(extreme_summary):
    extreme_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 14. Analysis Without Extreme Models (Grok & DeepSeek Chat)
    """)
    return


@app.cell
def _(ml_df, pd):
    EXCLUDED_MODELS = ["Grok 4.1 Fast (Reasoning)", "DeepSeek Chat"]

    ml_df_filtered = ml_df[~ml_df["Model"].isin(EXCLUDED_MODELS)].copy()
    _removed = len(ml_df) - len(ml_df_filtered)

    filtered_comparison = pd.DataFrame(
        {
            "metric": ["original rows", "filtered rows", "removed rows", "removed %", "models kept"],
            "value": [
                f"{len(ml_df):,}",
                f"{len(ml_df_filtered):,}",
                f"{_removed:,}",
                f"{100 * _removed / len(ml_df):.1f}%",
                len(ml_df_filtered["Model"].unique()),
            ],
        }
    )
    filtered_comparison
    return (ml_df_filtered,)


@app.cell
def _(TARGET, ml_df, ml_df_filtered):
    _original_count = ml_df["Model"].nunique()
    _kept_count = ml_df_filtered["Model"].nunique()

    model_filter_stats = (
        ml_df_filtered.groupby("Model")[TARGET]
        .agg(["count", "mean", "std", "min", "max", "median"])
        .round(4)
        .sort_values("mean")
        .reset_index()
    )
    model_filter_stats
    return


@app.cell
def _(FIG_PATH, TARGET, ml_df, ml_df_filtered, plt, sns):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(
        ml_df[TARGET], kde=True, bins=30, alpha=0.5, label="Original",
        edgecolor="white", ax=_axes[0], color="steelblue",
    )
    sns.histplot(
        ml_df_filtered[TARGET], kde=True, bins=30, alpha=0.5, label="Filtered",
        edgecolor="white", ax=_axes[0], color="coral",
    )
    _axes[0].set_title("Accuracy Distribution: Original vs Filtered")
    _axes[0].set_xlabel("Accuracy")
    _axes[0].legend()

    sns.boxplot(
        data=ml_df_filtered, x="Model", y=TARGET,
        ax=_axes[1], width=0.6, fliersize=2,
        order=ml_df_filtered.groupby("Model")[TARGET].median().sort_values().index,
    )
    _axes[1].set_title("Accuracy Distribution (Filtered, by Model)")
    _axes[1].set_xlabel("")
    plt.setp(_axes[1].get_xticklabels(), rotation=45, ha="right", fontsize=8)

    _fig.tight_layout()
    _fig.savefig(FIG_PATH / "filtered_accuracy_distribution.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "filtered_accuracy_distribution.png", format="png")
    plt.show()
    return


@app.cell
def _(FIG_PATH, TARGET, ml_df_filtered, plt, sns):
    _model_order = (
        ml_df_filtered.groupby("Model")[TARGET]
        .mean()
        .sort_values()
        .index
    )

    _stats = (
        ml_df_filtered.groupby("Model")[TARGET]
        .agg(["mean", "std", "count"])
        .loc[_model_order]
    )
    _stats["ci"] = 1.96 * _stats["std"] / _stats["count"] ** 0.5

    _fig, _ax = plt.subplots(figsize=(9, 5))
    _ax.barh(
        y=range(len(_stats)),
        width=_stats["mean"],
        xerr=_stats["ci"],
        color=sns.color_palette("muted", n_colors=len(_stats)),
        capsize=3,
        tick_label=_stats.index,
    )
    _ax.set_xlabel("Mean Accuracy (\u00b1 95% CI)")
    _ax.set_title("Model Rankings Without Extreme Models", fontsize=12)
    _ax.axvline(
        x=_stats["mean"].mean(), color="gray", linestyle="--", alpha=0.6,
        label=f"Grand mean = {_stats['mean'].mean():.3f}",
    )
    _ax.legend(fontsize=9)
    _fig.tight_layout()
    _fig.savefig(FIG_PATH / "filtered_model_rankings.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "filtered_model_rankings.png", format="png")
    plt.show()
    return


@app.cell
def _(FIG_PATH, TARGET, ml_df, ml_df_filtered, pd, plt):
    _numeric = ml_df.select_dtypes(include="number").drop(columns=["run_id"], errors="ignore")
    _numeric_f = ml_df_filtered.select_dtypes(include="number").drop(columns=["run_id"], errors="ignore")

    _orig_corr = _numeric.corr(method="pearson")[TARGET].drop(TARGET)
    _filt_corr = _numeric_f.corr(method="pearson")[TARGET].drop(TARGET)

    corr_delta = pd.DataFrame({"original": _orig_corr, "filtered": _filt_corr}).dropna()
    corr_delta["delta"] = corr_delta["filtered"] - corr_delta["original"]
    corr_delta["abs_delta"] = corr_delta["delta"].abs()
    corr_delta = corr_delta.sort_values("abs_delta", ascending=False)

    _top_features = corr_delta.head(10)

    _fig, _ax = plt.subplots(figsize=(10, 5))
    _x = range(len(_top_features))
    _width = 0.35
    _ax.bar(
        [_i - _width / 2 for _i in _x],
        _top_features["original"],
        _width,
        label="Original",
        color="steelblue",
        alpha=0.8,
    )
    _ax.bar(
        [_i + _width / 2 for _i in _x],
        _top_features["filtered"],
        _width,
        label="Filtered",
        color="coral",
        alpha=0.8,
    )
    _ax.set_xticks(list(_x))
    _ax.set_xticklabels(_top_features.index, rotation=45, ha="right", fontsize=9)
    _ax.set_ylabel("Pearson r with Accuracy")
    _ax.set_title("Top 10 Feature Correlations: Original vs Filtered", fontsize=12)
    _ax.legend()
    _ax.axhline(y=0, color="gray", linewidth=0.5)
    _fig.tight_layout()
    _fig.savefig(FIG_PATH / "filtered_correlation_delta.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "filtered_correlation_delta.png", format="png")
    plt.show()

    corr_delta.drop(columns=["abs_delta"])
    return (corr_delta,)


@app.cell
def _(FIG_PATH, TARGET, ml_df_filtered, pearsonr, plt, psych_cols, sns):
    _fig, _axes = plt.subplots(1, len(psych_cols), figsize=(14, 3.5))

    for _idx, _col in enumerate(psych_cols):
        _ax = _axes[_idx]
        _data = ml_df_filtered[[_col, TARGET]].dropna()
        sns.regplot(
            data=_data,
            x=_col,
            y=TARGET,
            scatter_kws={"alpha": 0.1, "s": 6, "color": "steelblue"},
            line_kws={"color": "red", "linewidth": 1.5},
            ax=_ax,
        )
        _r, _p = pearsonr(_data[_col], _data[TARGET])
        _p_str = f"p={_p:.2e}" if _p < 0.001 else f"p={_p:.4f}"
        _ax.set_title(f"{_col}  (r={_r:.3f}, {_p_str})", fontsize=10)
        _ax.set_xlabel("")

    _fig.suptitle("Psychometric Scores vs. Accuracy (Filtered, No Extremes)", fontsize=13)
    _fig.tight_layout()
    _fig.savefig(FIG_PATH / "filtered_psychometric_vs_accuracy.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "filtered_psychometric_vs_accuracy.png", format="png")
    plt.show()
    return


@app.cell
def _(TARGET, corr_delta, ml_df, ml_df_filtered, mo):
    _orig_var = ml_df[TARGET].var()
    _filt_var = ml_df_filtered[TARGET].var()

    _top_delta = corr_delta.head(5)
    _delta_lines = "\n".join(
        f"- **{feat}**: original r={_top_delta.loc[feat, 'original']:.3f} \u2192 filtered r={_top_delta.loc[feat, 'filtered']:.3f} (\u0394={_top_delta.loc[feat, 'delta']:+.3f})"
        for feat in _top_delta.index
    )

    mo.md(
        f"""
        **Key findings after removing Grok and DeepSeek Chat:**

        - Target variance collapsed from **{_orig_var:.4f}** \u2192 **{_filt_var:.4f}** (\u2014{100 * (1 - _filt_var / _orig_var):.0f}% reduction)
        - Remaining models range from ~0.44 (Ministral 3B) to ~0.71 (Anita/Mistral Small 4)
        - Top 5 features whose correlation with accuracy changed most:

        {_delta_lines}

        - The near-perfect ceiling of Grok/DeepSeek was inflating apparent predictor strength;
          after removal, psychometric correlations are weaker but the rank order among the
          12 remaining models provides a more realistic signal for modeling.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 15. Per-Model Psychometric Correlations
    """)
    return


@app.cell
def _(TARGET, ml_df_filtered, pd, pearsonr, psych_cols):
    _per_model_rows = []
    for _model in sorted(ml_df_filtered["Model"].unique()):
        _sub = ml_df_filtered[ml_df_filtered["Model"] == _model]
        _row = {"Model": _model, "n": len(_sub)}
        for _col in psych_cols:
            _r, _p = pearsonr(_sub[_col], _sub[TARGET])
            _row[f"{_col}_r"] = round(_r, 4)
            _row[f"{_col}_p"] = _p
        _per_model_rows.append(_row)

    per_model_corr = pd.DataFrame(_per_model_rows)

    _display_cols = ["Model", "n"] + [f"{c}_r" for c in psych_cols]
    per_model_corr[_display_cols]
    return (per_model_corr,)


@app.cell
def _(FIG_PATH, per_model_corr, plt, psych_cols, sns):

    _r_cols = [f"{c}_r" for c in psych_cols]
    _plot_data = per_model_corr.set_index("Model")[_r_cols].rename(
        columns=lambda c: c.replace("_r", "")
    )

    _link = sns.clustermap(
        _plot_data,
        annot=True,
        fmt=".3f",
        cmap=sns.diverging_palette(250, 10, as_cmap=True),
        vmin=-0.6,
        vmax=0.6,
        center=0,
        linewidths=1.5,
        figsize=(6, 8),
        dendrogram_ratio=(0.12, 0.08),
        cbar_kws={"shrink": 0.5, "label": "Pearson r"},
    )
    _link.ax_heatmap.set_ylabel("Model", fontsize=10)
    _link.ax_heatmap.set_xlabel("Psychometric Score", fontsize=10)
    _link.ax_heatmap.set_title("Per-Model Psychometric Correlations with Accuracy", fontsize=12)
    _link.savefig(FIG_PATH / "per_model_psychometric_heatmap.pdf", format="pdf")
    _link.savefig(FIG_PATH / "per_model_psychometric_heatmap.png", format="png")
    plt.show()
    return


@app.cell
def _(FIG_PATH, TARGET, ml_df_filtered, plt, psych_cols, sns):
    for _col in psych_cols:
        _g = sns.lmplot(
            data=ml_df_filtered,
            x=_col,
            y=TARGET,
            col="Model",
            col_wrap=4,
            scatter_kws={"alpha": 0.3, "s": 8, "color": "steelblue"},
            line_kws={"color": "red", "linewidth": 1},
            lowess=False,
            height=2.5,
            aspect=1.1,
            sharex=True,
            sharey=True,
        )
        _g.fig.subplots_adjust(top=0.9)
        _g.fig.suptitle(f"{_col} vs. Accuracy — Faceted by Model", fontsize=13)
        _g.savefig(FIG_PATH / f"per_model_facet_{_col}.pdf", format="pdf")
        _g.savefig(FIG_PATH / f"per_model_facet_{_col}.png", format="png")
        plt.show()
    return


@app.cell
def _(FIG_PATH, TARGET, ml_df, plt, psych_cols, sns):
    for _col in psych_cols:
        _g = sns.lmplot(
            data=ml_df,
            x=_col,
            y=TARGET,
            col="Model",
            col_wrap=4,
            scatter_kws={"alpha": 0.3, "s": 8, "color": "steelblue"},
            line_kws={"color": "red", "linewidth": 1},
            lowess=False,
            height=2.5,
            aspect=1.1,
            sharex=True,
            sharey=True,
        )
        _g.fig.subplots_adjust(top=0.9)
        _g.fig.suptitle(f"{_col} vs. Accuracy — Faceted by Model", fontsize=13)
        _g.savefig(FIG_PATH / f"per_model_facet_all_models_{_col}.pdf", format="pdf")
        _g.savefig(FIG_PATH / f"per_model_facet_all_models_{_col}.png", format="png")
        plt.show()
    return


@app.cell
def _(mo, pd, per_model_corr, psych_cols):
    _r_cols = [f"{c}_r" for c in psych_cols]
    _p_cols = [f"{c}_p" for c in psych_cols]

    _summary_rows = []
    for _col, _r_col, _p_col in zip(psych_cols, _r_cols, _p_cols):
        _vals = per_model_corr[_r_col]
        _sig_pos = ((_vals > 0) & (per_model_corr[_p_col] < 0.05)).sum()
        _sig_neg = ((_vals < 0) & (per_model_corr[_p_col] < 0.05)).sum()
        _n_sig = _sig_pos + _sig_neg
        _non_sig = len(per_model_corr) - _n_sig
        _mean_r = _vals.mean()
        _std_r = _vals.std()
        _min_model = per_model_corr.loc[_vals.idxmin(), "Model"]
        _max_model = per_model_corr.loc[_vals.idxmax(), "Model"]
        _summary_rows.append(
            {
                "score": _col,
                "mean_r": f"{_mean_r:.3f}",
                "std_r": f"{_std_r:.3f}",
                "min_r": f"{_vals.min():.3f} ({_min_model})",
                "max_r": f"{_vals.max():.3f} ({_max_model})",
                "sig_pos": _sig_pos,
                "sig_neg": _sig_neg,
                "non_sig": _non_sig,
            }
        )

    _summary_df = pd.DataFrame(_summary_rows)

    _consistent = _summary_df.loc[_summary_df["sig_neg"].astype(int) + _summary_df["sig_pos"].astype(int) > 0, "score"].tolist()
    _inconsistent = _summary_df.loc[(_summary_df["sig_neg"].astype(int) + _summary_df["sig_pos"].astype(int)) == 0, "score"].tolist()

    _consensus_lines = []
    for _, _row in _summary_df.iterrows():
        _consensus_lines.append(
            f"- **{_row['score']}**: mean r={_row['mean_r']} \u00b1 {_row['std_r']}, "
            f"range [{_row['min_r']} to {_row['max_r']}], "
            f"{_row['sig_pos']} positive / {_row['sig_neg']} negative significant"
        )

    _het_lines = []
    for _col, _r_col in zip(psych_cols, _r_cols):
        _min_r = per_model_corr[_r_col].min()
        _max_r = per_model_corr[_r_col].max()
        if _max_r - _min_r > 0.3:
            _min_m = per_model_corr.loc[per_model_corr[_r_col].idxmin(), "Model"]
            _max_m = per_model_corr.loc[per_model_corr[_r_col].idxmax(), "Model"]
            _het_lines.append(f"- **{_col}**: wide spread ({_min_r:.2f} to {_max_r:.2f}), between '{_min_m}' and '{_max_m}'")

    mo.md(
        f"""
        **Per-model psychometric correlation summary ({len(per_model_corr)} models, excluding Grok/DeepSeek):**

        {chr(10).join(_consensus_lines)}

        **Slope heterogeneity across models:**

        {chr(10).join(_het_lines) if _het_lines else "None of the psychometric scores show extreme slope variation across models."}

        **Implication for stratified modeling:**
        The per-model correlations show {_consistent if _consistent else 'some'} score(s) with meaningful
        variation across models, suggesting that fitting separate models per model group is warranted.
        The degree of slope heterogeneity will determine whether pooled or stratified approaches
        yield better predictive performance.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 16. Psychometric Scores by OCEAN Levels
    """)
    return


@app.cell
def _(ml_df_filtered, pd):
    ocean_dims = [
        "ocean_openness_level_ord",
        "ocean_conscientiousness_level_ord",
        "ocean_extraversion_level_ord",
        "ocean_agreeableness_level_ord",
        "ocean_neuroticism_level_ord",
    ]
    ocean_labels = {
        "ocean_openness_level_ord": "Openness",
        "ocean_conscientiousness_level_ord": "Conscientiousness",
        "ocean_extraversion_level_ord": "Extraversion",
        "ocean_agreeableness_level_ord": "Agreeableness",
        "ocean_neuroticism_level_ord": "Neuroticism",
    }
    _psych_cols = ["amas_score", "maes_score", "mseaq_anx", "mseaq_se"]

    _rows = []
    for _dim in ocean_dims:
        for _level in sorted(ml_df_filtered[_dim].dropna().unique()):
            _sub = ml_df_filtered[ml_df_filtered[_dim] == _level]
            _row = {
                "ocean_trait": ocean_labels[_dim],
                "ocean_column": _dim,
                "level": int(_level),
                "n": len(_sub),
            }
            for _col in _psych_cols:
                _row[f"{_col}_mean"] = _sub[_col].mean()
                _row[f"{_col}_std"] = _sub[_col].std()
            _rows.append(_row)

    ocean_psych_stats = pd.DataFrame(_rows)
    ocean_psych_stats
    return ocean_dims, ocean_labels, ocean_psych_stats


@app.cell
def _(FIG_PATH, ocean_psych_stats, plt, psych_cols, sns):
    _long = ocean_psych_stats.melt(
        id_vars=["ocean_trait", "level", "n"],
        value_vars=[f"{c}_mean" for c in psych_cols],
        var_name="psych_metric",
        value_name="mean_score",
    )
    _long["psych_score"] = _long["psych_metric"].str.replace("_mean", "")
    _long["trait_level"] = _long["ocean_trait"] + " (L" + _long["level"].astype(str) + ")"

    _pivot = _long.pivot_table(
        index="trait_level", columns="psych_score", values="mean_score"
    )
    _pivot = _pivot.loc[
        sorted(_pivot.index, key=lambda x: (x.split(" (")[0], x.split("L")[1].rstrip(")")))
    ]

    _fig, _ax = plt.subplots(figsize=(7, 7))
    sns.heatmap(
        _pivot,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        linewidths=1,
        ax=_ax,
        cbar_kws={"shrink": 0.6, "label": "Mean Score"},
    )
    _ax.set_title("Mean Psychometric Score by OCEAN Level", fontsize=12)
    _ax.set_ylabel("OCEAN Trait × Level")
    _ax.set_xlabel("Psychometric Scale")
    _fig.tight_layout()
    _fig.savefig(FIG_PATH / "ocean_psych_heatmap.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "ocean_psych_heatmap.png", format="png")
    plt.show()
    return


@app.cell
def _(
    FIG_PATH,
    f_oneway,
    ml_df_filtered,
    ocean_dims,
    ocean_labels,
    ocean_psych_stats,
    pd,
    plt,
    psych_cols,
):
    _anova_rows = []
    for _dim in ocean_dims:
        for _col in psych_cols:
            _groups = [
                ml_df_filtered.loc[ml_df_filtered[_dim] == _l, _col].dropna()
                for _l in sorted(ml_df_filtered[_dim].dropna().unique())
            ]
            _f, _p = f_oneway(*_groups)
            _anova_rows.append({
                "ocean_trait": ocean_labels[_dim],
                "psych_score": _col,
                "F": f"{_f:.2f}",
                "p": f"{_p:.4f}",
                "sig": "***" if _p < 0.001 else "**" if _p < 0.01 else "*" if _p < 0.05 else "ns",
            })
    anova_df = pd.DataFrame(_anova_rows)

    _display = anova_df.pivot(index="ocean_trait", columns="psych_score", values="sig")
    _f_vals = anova_df.pivot(index="ocean_trait", columns="psych_score", values="F")

    _fig, _axes = plt.subplots(2, 2, figsize=(12, 8))
    _axes = _axes.flatten()

    for _idx, _col in enumerate(psych_cols):
        _ax = _axes[_idx]
        _dim_data = ocean_psych_stats[
            ["ocean_trait", "level", f"{_col}_mean", f"{_col}_std"]
        ].copy()

        _labels = sorted(_dim_data["ocean_trait"].unique())
        _x = range(len(_labels))
        _width = 0.25

        for _lev in [0, 1, 2]:
            _lev_data = _dim_data[_dim_data["level"] == _lev]
            _lev_data = _lev_data.set_index("ocean_trait").reindex(_labels)
            _offsets = [_i + (_lev - 1) * _width for _i in _x]
            _ax.bar(
                _offsets,
                _lev_data[f"{_col}_mean"],
                _width,
                yerr=_lev_data[f"{_col}_std"],
                label=f"Level {_lev}",
                capsize=3,
                alpha=0.8,
            )

        _ax.set_xticks(list(_x))
        _ax.set_xticklabels(_labels, rotation=30, ha="right", fontsize=8)
        _ax.set_ylabel("Mean Score")
        _ax.set_title(f"{_col}", fontsize=10)
        _ax.legend(fontsize=7)

    _fig.suptitle("Psychometric Scores by OCEAN Dimension and Level", fontsize=13)
    _fig.tight_layout()
    _fig.savefig(FIG_PATH / "ocean_psych_grouped_bars.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "ocean_psych_grouped_bars.png", format="png")
    plt.show()

    _display
    return (anova_df,)


@app.cell
def _(anova_df, mo, psych_cols):
    _sig_counts = {}
    for _col in psych_cols:
        _sub = anova_df[anova_df["psych_score"] == _col]
        _sig = (_sub["sig"] != "ns").sum()
        _total = len(_sub)
        _sig_counts[_col] = f"{_sig}/{_total} significant"

    _sig_lines = "\n".join(f"- **{k}**: {v}" for k, v in _sig_counts.items())

    mo.md(
        f"""
        **OCEAN × Psychometric score relationships:**

        The ANOVA tests show whether OCEAN personality traits are associated with psychometric scale scores.
        Significant associations suggest mediation pathways.

        {_sig_lines}

        **Interpretation:**
        OCEAN traits that significantly predict both accuracy (Section 11) and psychometric scores (above)
        could confound the psychometric → accuracy relationship. In the stratified modeling approach,
        OCEAN variables should be included as covariates to isolate the unique contribution of
        psychometric scales.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 17. PCA on Psychometric Scores
    """)
    return


@app.cell
def _(ml_df_filtered, pd, psych_cols):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    _X = ml_df_filtered[psych_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(_X)

    pca = PCA(n_components=len(psych_cols), random_state=42)
    pca.fit(X_scaled)

    var_exp = pd.DataFrame(
        {
            "component": [f"PC{i+1}" for i in range(len(psych_cols))],
            "eigenvalue": pca.explained_variance_,
            "variance_ratio": pca.explained_variance_ratio_,
            "cumulative": pca.explained_variance_ratio_.cumsum(),
        }
    )

    loadings = pd.DataFrame(
        pca.components_.T,
        index=psych_cols,
        columns=[f"PC{i+1}" for i in range(len(psych_cols))],
    )

    var_exp
    return StandardScaler, X_scaled, loadings, pca, scaler, var_exp


@app.cell
def _(loadings):
    loadings
    return


@app.cell
def _(FIG_PATH, plt, var_exp):
    _fig, _ax1 = plt.subplots(figsize=(8, 4.5))

    _colors = ["steelblue" if _v > 1 else "lightgray" for _v in var_exp["eigenvalue"]]
    _ax1.bar(range(1, len(var_exp) + 1), var_exp["eigenvalue"], color=_colors, edgecolor="black")
    _ax1.axhline(y=1, color="red", linestyle="--", alpha=0.6, label="Kaiser criterion (\u03bb=1)")
    _ax1.set_xlabel("Principal Component")
    _ax1.set_ylabel("Eigenvalue")
    _ax1.set_xticks(range(1, len(var_exp) + 1))
    _ax1.set_title("Scree Plot with Cumulative Variance", fontsize=12)

    _ax2 = _ax1.twinx()
    _ax2.plot(
        range(1, len(var_exp) + 1),
        var_exp["cumulative"] * 100,
        "ro-",
        markersize=6,
        label="Cumulative %",
    )
    _ax2.set_ylabel("Cumulative Variance (%)")
    for _i, _v in enumerate(var_exp["cumulative"]):
        _ax2.text(_i + 1, _v * 100 + 2, f"{_v*100:.0f}%", ha="center", fontsize=8)

    _lines1, _labels1 = _ax1.get_legend_handles_labels()
    _lines2, _labels2 = _ax2.get_legend_handles_labels()
    _ax1.legend(_lines1 + _lines2, _labels1 + _labels2, loc="center right", fontsize=8)

    _fig.tight_layout()
    _fig.savefig(FIG_PATH / "pca_scree.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "pca_scree.png", format="png")
    plt.show()
    return


@app.cell
def _(FIG_PATH, TARGET, X_scaled, loadings, ml_df_filtered, pca, pd, plt, sns):

    # Fixed variable name: ensure we consistently use 'pca' (or '_pca', but matching is key)
    _X_pca = pca.transform(X_scaled)
    pca_df = pd.DataFrame(_X_pca, columns=[f"PC{i+1}" for i in range(pca.n_components_)])

    # FIX: Since X_scaled is a numpy array, it has no index. 
    # Assuming row order is preserved from ml_df_filtered, just pass the values directly.
    pca_df[TARGET] = ml_df_filtered[TARGET].values

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 5))

    _ax = _axes[0]
    _quantiles = pd.qcut(pca_df[TARGET], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    _palette = sns.color_palette("viridis", 4)
    for _i, _q in enumerate(["Q1", "Q2", "Q3", "Q4"]):
        _sub = pca_df[_quantiles == _q]
        _ax.scatter(
            _sub["PC1"], _sub["PC2"],
            c=[_palette[_i]], alpha=0.2, s=4, label=_q,
        )
    # FIX: Matched the pca variable name here
    _ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    _ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    _ax.set_title("PC1 vs PC2 Colored by Accuracy Quartile", fontsize=11)
    _ax.legend(fontsize=8, markerscale=3)

    _ax = _axes[1]
    _loadings_2d = loadings[["PC1", "PC2"]]
    for _var in _loadings_2d.index:
        _ax.arrow(0, 0, _loadings_2d.loc[_var, "PC1"], _loadings_2d.loc[_var, "PC2"],
                  head_width=0.03, head_length=0.03, fc="red", ec="red", alpha=0.7)
        _ax.text(
            _loadings_2d.loc[_var, "PC1"] * 1.1,
            _loadings_2d.loc[_var, "PC2"] * 1.1,
            _var, fontsize=9, color="red",
        )
    _ax.axhline(y=0, color="gray", linewidth=0.5)
    _ax.axvline(x=0, color="gray", linewidth=0.5)
    _ax.set_xlim(-1.1, 1.1)
    _ax.set_ylim(-1.1, 1.1)
    # FIX: Matched the pca variable name here
    _ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    _ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    _ax.set_title("Loadings Biplot (PC1 vs PC2)", fontsize=11)
    _ax.set_aspect("equal")

    _fig.tight_layout()
    # Assuming FIG_PATH is defined earlier
    _fig.savefig(FIG_PATH / "pca_biplot.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "pca_biplot.png", format="png")
    plt.show()
    return (pca_df,)


@app.cell
def _(FIG_PATH, X_scaled, loadings, pca, plt):
    _X_pca = pca.transform(X_scaled)

    _fig = plt.figure(figsize=(12, 5))
    _ax = plt.axes()

    _loadings_2d = loadings[["PC1", "PC2"]]
    for _var in _loadings_2d.index:
        _ax.arrow(0, 0, _loadings_2d.loc[_var, "PC1"], _loadings_2d.loc[_var, "PC2"],
                  head_width=0.03, head_length=0.03, fc="red", ec="red", alpha=0.7)
        _ax.text(
            _loadings_2d.loc[_var, "PC1"] * 1.1,
            _loadings_2d.loc[_var, "PC2"] * 1.1,
            _var, fontsize=9, color="red",
        )

    _ax.axhline(y=0, color="gray", linewidth=0.5)
    _ax.axvline(x=0, color="gray", linewidth=0.5)
    _ax.set_xlim(-1.1, 1.1)
    _ax.set_ylim(-1.1, 1.1)

    _ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    _ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    _ax.set_title("Loadings Biplot (PC1 vs PC2)", fontsize=11)
    _ax.set_aspect("equal")

    _fig.tight_layout()
    # Assuming FIG_PATH is defined earlier
    _fig.savefig(FIG_PATH / "pca_biplot_loadings.pdf", format="pdf")
    _fig.savefig(FIG_PATH / "pca_biplot_loadings.png", format="png")
    plt.show()
    return


@app.cell
def _(
    StandardScaler,
    TARGET,
    ml_df,
    ml_df_filtered,
    pca,
    pca_df,
    pd,
    pearsonr,
    psych_cols,
):
    _pc_cols = [f"PC{i+1}" for i in range(pca.n_components_)]

    _comparison_rows = []
    for _col in psych_cols:
        _r_f, _ = pearsonr(ml_df_filtered[_col], ml_df_filtered[TARGET])
        _r_all, _ = pearsonr(ml_df[_col], ml_df[TARGET])
        _comparison_rows.append({"feature": _col, "type": "raw", "r_filtered": round(_r_f, 4), "r_full": round(_r_all, 4)})

    for _i, _pc in enumerate(_pc_cols):
        _r_f, _ = pearsonr(pca_df[_pc], pca_df[TARGET])
        _full_sub = ml_df[psych_cols].dropna()
        _r_all, _ = pearsonr(pca.transform(StandardScaler().fit_transform(_full_sub))[:, _i], ml_df.loc[_full_sub.index, TARGET])
        _comparison_rows.append({"feature": _pc, "type": "pca", "r_filtered": round(_r_f, 4), "r_full": round(_r_all, 4)})

    pd.DataFrame(_comparison_rows)
    return


@app.cell
def _(pca, scaler):
    import joblib
    from mathanx.ml.config import PCA_TRANSFORM_PATH

    PCA_TRANSFORM_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "pca": pca}, PCA_TRANSFORM_PATH)
    print(f"PCA saved to {PCA_TRANSFORM_PATH}")
    return


if __name__ == "__main__":
    app.run()
