import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from collections import Counter
    from pathlib import Path
    from typing import Dict, List, Literal, Tuple

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import shap
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.stats import pearsonr, spearmanr
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.feature_extraction import FeatureHasher
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, cross_validate
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor

    # Set theme for plots
    sns.set_theme(style="whitegrid", context="notebook")

    # Specify what the target of the ML models is (y).
    TARGET = "accuracy"
    # Provide the path to the cleaned dataset, as created by the ml_dataset_creation.py notebook.
    DATASET_PATH = Path("data/processed/ml/ml_dataset.csv").resolve().absolute()

    # Specify which columns could introduce leakage in the analysis. These include columns
    # that are linear transformation of the target variable.
    LEAKAGE_COLS = {
        "run_id",
        "accuracy",
        "confidence",
        # "confidence_scaled",
        "delta_confidence",
        "total_correct",
        "n_observations",
    }
    return (
        BaseEstimator,
        ColumnTransformer,
        Counter,
        DATASET_PATH,
        DecisionTreeRegressor,
        Dict,
        ElasticNet,
        FeatureHasher,
        GridSearchCV,
        KFold,
        LEAKAGE_COLS,
        List,
        Literal,
        OrdinalEncoder,
        Path,
        Pipeline,
        RandomForestRegressor,
        RandomizedSearchCV,
        Ridge,
        StandardScaler,
        TARGET,
        TransformerMixin,
        XGBRegressor,
        cross_validate,
        leaves_list,
        linkage,
        mean_squared_error,
        mo,
        np,
        pd,
        permutation_importance,
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


@app.cell
def _(BaseEstimator, FeatureHasher, TransformerMixin, np, pd):
    class FrequencyEncoder(BaseEstimator, TransformerMixin):
        """
        FrequencyEncoder is the object that handles the encoding of nominal features into numeric features by
        relying on computing their frequency

        The object inherits from BaseEstimator and TransformerMixin, two objects from the scikit-learn library that
        make this new object compatible for use in pipelines and training runs.
        """

        def fit(self, X, y=None):
            """
            This method fits the encoder on the city_of_living column in the original dataset.
            Notice that this method changes internal attributes of the object, returning a new instance
            of itself.
            """
            series = pd.Series(X.squeeze(), dtype="string").fillna("__missing__")
            # Get the name of the feature
            self.feature_name_in_ = getattr(X, "columns", ["city_of_living"])[0]
            # Build the frequency map by counting frequencies and normalizing in a [0,1] range
            self.frequency_map_ = series.value_counts(normalize=True).to_dict()
            # Set a default value of 0
            self.default_value_ = 0.0
            return self

        def transform(self, X):
            """
            This method transforms the specified series by applying the normalized frequency count
            """

            # Convert a potential matrix into a vector
            series = pd.Series(X.squeeze(), dtype="string").fillna("__missing__")
            # Relies on the frequency_map_ obtained during the fit step.
            encoded = series.map(self.frequency_map_).fillna(self.default_value_).astype(float)
            # Return the transformed array, ensuring it is a vector of dimension 1
            return encoded.to_numpy().reshape(-1, 1)

        def get_feature_names_out(self, input_features=None):
            """
            This method is used to specify the name of the feature that is used in the model.
            """
            # Get the name of the feature if it was passed from the function, else try to extract
            # it from the object itself, defaulting to a generic "feature" on failure.
            feature = input_features[0] if input_features else getattr(self, "feature_name_in_", "feature")
            # Return the name of the feature appending __frequency to it.
            return np.array([f"{feature}__frequency"])

    class CategoricalHasher(BaseEstimator, TransformerMixin):
        """
        This class is used to build a custom CategoricalHasher compatible with the sklearn APIs.
        """

        def __init__(self, n_features=64):
            """
            Initialize the object by picking a number of features.
            The feature number is the amount of elements that will be in the hashed vector.
            """
            self.n_features = n_features
            # Wrap the sklearn FeatureHasher
            self.hasher_ = FeatureHasher(
                # Pass the number of desired features in the output matrix
                n_features=n_features,
                # Specify the input type to be string
                input_type="string",
                # use alternating sign to preserve dot products in case of name collisions
                alternate_sign=True,
            )

        def fit(self, X, y=None):
            """
            Specify the features to which the CategoricalHasher should be applied.
            """
            # In case X does not contain the columns attribute, default to an empty list
            self.feature_names_in_ = list(getattr(X, "columns", []))
            return self

        def transform(self, X):
            """
            Transform the target column by applying the hashing function.
            """
            # Cast X to a DataFrame with columns of type string. 
            frame = pd.DataFrame(X, columns=self.feature_names_in_).astype("string").fillna("__missing__")
            # Build the list of documents by prepending the column name to the value, thus forcing
            # different hashes for similar values in different columns.
            docs = [
                [f"{col}={row[col]}" for col in self.feature_names_in_]
                for _, row in frame.iterrows()
            ]
            # Return the transformed series.
            return self.hasher_.transform(docs)

        def get_feature_names_out(self, input_features=None):
            return np.array([f"hash_{i}" for i in range(self.n_features)])

    return CategoricalHasher, FrequencyEncoder


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
    # Filter the features to include only those that do not introduce leakage or are the target themselves
    feature_cols = [c for c in ml_df.columns if c not in LEAKAGE_COLS and c not in {TARGET, "education_vs_parent_mean_gap"}]

    # Check which features will be used by the ML models
    print("Included features:\n", feature_cols)

    # Create copy of the data
    X = ml_df.loc[:, feature_cols].copy()
    y = ml_df[TARGET].copy()

    # Pick the nominal features making sure that they are actually present in the dataframe
    nominal_features = [
        c
        for c in [
            "gender",
            "sexual_orientation",
            "city_of_living",
            "employment_status",
            "marital_status",
            "migration_status",
            "religious_beliefs",
            "Model",
        ]
        if c in X.columns
    ]

    # Identify binary features as those containing only 0, 1 values
    binary_features = [c for c in X.columns if set(X[c].dropna().unique()).issubset({0, 1})]
    # Identify numeric features as those not present in binary or nominal features
    numeric_features = [c for c in X.columns if c not in binary_features and c not in nominal_features]
    # identify ordinal features as those ending with _ord
    ordinal_features = [c for c in numeric_features if c.endswith("_ord")]
    # continuous features are those features that are numbers but their range extends beyond that of ordinal features
    continuous_features = [c for c in numeric_features if c not in ordinal_features]
    # specify the nominal features for tree models. Do not include city of living.
    tree_nominal_features = [c for c in nominal_features if c != "city_of_living"]

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
    CategoricalHasher,
    ColumnTransformer,
    Counter,
    DecisionTreeRegressor,
    Dict,
    ElasticNet,
    FrequencyEncoder,
    GridSearchCV,
    KFold,
    List,
    Literal,
    OrdinalEncoder,
    Pipeline,
    RandomForestRegressor,
    RandomizedSearchCV,
    Ridge,
    StandardScaler,
    X,
    XGBRegressor,
    mean_squared_error,
    nominal_features,
    np,
    numeric_features,
    pd,
    tree_nominal_features,
    y,
):
    RANDOM_STATE = 42
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 1)
    post_tuning_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 2)

    # numeric_features = [c for c in X.columns if c not in nominal_features]
    scale_features = numeric_features

    def build_linear_pipeline(model, features: Dict[str, List[str]] = {"scale_features": scale_features, "nominal_features": nominal_features}):
        transformer = ColumnTransformer(
            transformers=[
                ("scale", StandardScaler(), features["scale_features"]),
                ("hash", CategoricalHasher(n_features=64), features["nominal_features"]),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        return Pipeline([("preprocess", transformer), ("model", model)])

    def build_tree_pipeline(model, features: Dict[str, List[str]] = {"numeric_features": numeric_features, "tree_nominal_features": tree_nominal_features}):
        transformer = ColumnTransformer(
            transformers=[
                ("freq_city", FrequencyEncoder(), ["city_of_living"]),
                (
                    "ordinal_nominal",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                    features["tree_nominal_features"],
                ),
                ("numeric", "passthrough", features["numeric_features"]),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        return Pipeline([("preprocess", transformer), ("model", model)])

    model_specs = {
        "ridge": {
            "estimator": build_linear_pipeline(Ridge()),
            "search_cls": GridSearchCV,
            "search_kwargs": {
                "param_grid": {
                    "model__alpha": [0.01, 0.1, 1.0, 10.0],
                },
            },
        },
        "elastic_net": {
            "estimator": build_linear_pipeline(ElasticNet(random_state=RANDOM_STATE, max_iter=5000)),
            "search_cls": GridSearchCV,
            "search_kwargs": {
                "param_grid": {
                    "model__alpha": [0.0001, 0.001, 0.01],
                    "model__l1_ratio": [0.1, 0.5, 0.9],
                },
            },
        },
        "decision_tree": {
            "estimator": build_tree_pipeline(DecisionTreeRegressor(random_state=RANDOM_STATE)),
            "search_cls": RandomizedSearchCV,
            "search_kwargs": {
                "param_distributions": {
                    "model__max_depth": [None, 4, 6, 8, 12],
                    "model__min_samples_split": [2, 5, 10],
                    "model__min_samples_leaf": [1, 3, 5],
                    "model__criterion": ["squared_error", "friedman_mse"],
                },
                "n_iter": 10,
            },
        },
        "random_forest": {
            "estimator": build_tree_pipeline(RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)),
            "search_cls": RandomizedSearchCV,
            "search_kwargs": {
                "param_distributions": {
                    "model__n_estimators": [200, 400],
                    "model__max_depth": [None, 10, 20],
                    "model__min_samples_split": [2, 5, 10],
                    "model__min_samples_leaf": [1, 3, 5],
                    "model__max_features": ["sqrt", 0.5],
                },
                "n_iter": 10,
            },
        },
        "xgboost": {
            "estimator": build_tree_pipeline(
                XGBRegressor(
                    objective="reg:squarederror",
                    random_state=RANDOM_STATE,
                    n_estimators=300,
                    n_jobs=1,
                    tree_method="hist",
                )
            ),
            "search_cls": RandomizedSearchCV,
            "search_kwargs": {
                "param_distributions": {
                    "model__n_estimators": [200, 300, 400],
                    "model__max_depth": [3, 5, 7],
                    "model__learning_rate": [0.03, 0.1],
                    "model__subsample": [0.8, 1.0],
                    "model__colsample_bytree": [0.8, 1.0],
                    "model__min_child_weight": [1, 5],
                    "model__reg_alpha": [0.0, 0.1],
                    "model__reg_lambda": [1.0, 5.0],
                },
                "n_iter": 12,
            },
        },
    }

    def build_search(model_name: str, model_specs: Dict[str, Dict[str, List]] = model_specs):
        spec = model_specs[model_name]
        search_cls = spec["search_cls"]
        search_kwargs = dict(spec["search_kwargs"])
        base_kwargs = {
            "estimator": spec["estimator"],
            "cv": inner_cv,
            "scoring": "neg_root_mean_squared_error",
            "n_jobs": -1,
            "refit": True,
        }
        if search_cls is GridSearchCV:
            return search_cls(**base_kwargs, **search_kwargs)

        return search_cls(
            **base_kwargs,
            **search_kwargs,
            random_state=RANDOM_STATE,
        )

    def run_nested_cv(model_name: str, model_specs: Dict[str, Dict[str, List]] = model_specs):
        rows = []
        for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            search = build_search(model_name, model_specs)
            search.fit(X_train, y_train)
            best_estimator = search.best_estimator_
            y_pred = best_estimator.predict(X_test)

            rows.append(
                {
                    "model": model_name,
                    "outer_fold": outer_fold,
                    "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                    "mae": float(np.mean(np.abs(y_test - y_pred))),
                    "r2": float(best_estimator.score(X_test, y_test)),
                    "best_params": search.best_params_,
                    "best_params_repr": repr(search.best_params_),
                }
            )

        return pd.DataFrame(rows)

    def build_nested_cv_results_df(model_specs: Dict[str, Dict[str, List]] = model_specs):
        nested_cv_results_df = pd.concat([run_nested_cv(model_name) for model_name in model_specs], ignore_index=True)
        return nested_cv_results_df

    def build_model_summary_df(nested_cv_results_df, metric: Literal["mean_rmse", "mean_r2"] = "mean_r2"):
        """
        Build a summary of the models where the top model is the one with the better
        performance according to a specified `metric`.

        `metric` defaults to the mean R^2 score across the different folds.
        """

        ascending = False if metric == "mean_r2" else True

        model_summary_df = (
            nested_cv_results_df.groupby("model")
            .agg(
                mean_rmse=("rmse", "mean"),
                std_rmse=("rmse", "std"),
                mean_mae=("mae", "mean"),
                mean_r2=("r2", "mean"),
            )
            .reset_index()
            .sort_values(metric, ascending = ascending)
        )
        return model_summary_df

    def build_params_summary_df(nested_cv_results_df):
        best_params_summary_rows = []
        best_params_by_model = {}
        for model_name, model_rows in nested_cv_results_df.groupby("model"):
            modal_repr, count = Counter(model_rows["best_params_repr"]).most_common(1)[0]
            modal_params = model_rows.loc[model_rows["best_params_repr"] == modal_repr, "best_params"].iloc[0]
            best_params_by_model[model_name] = modal_params
            best_params_summary_rows.append(
                {
                    "model": model_name,
                    "best_params_repr": modal_repr,
                    "support": count,
                }
            )

        return best_params_summary_rows, best_params_by_model

    return (
        RANDOM_STATE,
        build_linear_pipeline,
        build_model_summary_df,
        build_nested_cv_results_df,
        build_params_summary_df,
        build_tree_pipeline,
        model_specs,
        post_tuning_cv,
        scale_features,
    )


@app.cell
def _(
    X,
    build_model_summary_df,
    build_nested_cv_results_df,
    build_params_summary_df,
    model_specs,
    pd,
    y,
):


    nested_cv_results_df = build_nested_cv_results_df()
    model_summary_df = build_model_summary_df(nested_cv_results_df)
    best_params_summary_rows, best_params_by_model = build_params_summary_df(nested_cv_results_df)

    best_params_summary_df = pd.DataFrame(best_params_summary_rows).sort_values("model")

    best_model_name = model_summary_df.iloc[0]["model"]
    best_model_params = best_params_by_model[best_model_name]
    best_model_pipeline = model_specs[best_model_name]["estimator"].set_params(**best_model_params)
    best_model_pipeline.fit(X, y)
    return (
        best_model_name,
        best_params_by_model,
        best_params_summary_df,
        model_summary_df,
    )


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


@app.cell
def _(
    X,
    cross_validate,
    model_specs,
    pd,
    permutation_importance,
    post_tuning_cv,
    y,
):
    def build_tuned_cv_rows(best_params_by_model):
        tuned_cv_rows = []
        for model_name, params in best_params_by_model.items():
            estimator = model_specs[model_name]["estimator"].set_params(**params)
            cv_scores = cross_validate(
                estimator,
                X,
                y,
                cv=post_tuning_cv,
                scoring={
                    "rmse": "neg_root_mean_squared_error",
                    "mae": "neg_mean_absolute_error",
                    "r2": "r2",
                },
                n_jobs=-1,
                return_train_score=False,
            )
            tuned_cv_rows.append(
                {
                    "model": model_name,
                    "mean_rmse": float(-cv_scores["test_rmse"].mean()),
                    "std_rmse": float(cv_scores["test_rmse"].std()),
                    "mean_mae": float(-cv_scores["test_mae"].mean()),
                    "mean_r2": float(cv_scores["test_r2"].mean()),
                    "params": params,
                }
            )

        return tuned_cv_rows

    def build_permutation_importance_df(estimator):
        permutation_result = permutation_importance(estimator, X, y, n_repeats=10, random_state=42, n_jobs=-1)
        permutation_importance_df = (
            pd.DataFrame(
                {
                    "feature": X.columns,
                    "importance_mean": permutation_result.importances_mean,
                    "importance_std": permutation_result.importances_std,
                }
            )
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )
        return permutation_importance_df

    return build_permutation_importance_df, build_tuned_cv_rows


@app.cell
def _(
    X,
    best_model_name,
    best_params_by_model,
    build_permutation_importance_df,
    build_tuned_cv_rows,
    model_specs,
    pd,
    y,
):
    tuned_cv_rows = build_tuned_cv_rows(best_params_by_model)
    tuned_cv_results_df = pd.DataFrame(tuned_cv_rows).sort_values("mean_r2", ascending=False)

    final_estimator = model_specs[best_model_name]["estimator"].set_params(**best_params_by_model[best_model_name])
    final_estimator.fit(X, y)

    permutation_importance_df = build_permutation_importance_df(final_estimator)

    top_permutation_importance_df = permutation_importance_df.head(20)
    return top_permutation_importance_df, tuned_cv_results_df


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
    SHAP_SAMPLE_SIZE = 1000
    SHAP_RANDOM_STATE = 42
    TREE_MODEL_NAMES = ["decision_tree", "random_forest", "xgboost"]
    return SHAP_RANDOM_STATE, SHAP_SAMPLE_SIZE, TREE_MODEL_NAMES


@app.cell
def _(
    TREE_MODEL_NAMES,
    best_model_name,
    best_params_by_model,
    model_specs,
    pd,
    tuned_cv_results_df,
):
    if best_model_name in TREE_MODEL_NAMES:
        shap_model_name = best_model_name
    else:
        tree_candidates = tuned_cv_results_df[tuned_cv_results_df["model"].isin(TREE_MODEL_NAMES)].sort_values("mean_rmse")
        shap_model_name = tree_candidates.iloc[0]["model"]

    shap_model_pipeline = model_specs[shap_model_name]["estimator"].set_params(**best_params_by_model[shap_model_name])
    shap_selection_df = pd.DataFrame(
        {
            "selected_for_shap": [shap_model_name],
            "best_overall_model": [best_model_name],
        }
    )
    return shap_model_name, shap_model_pipeline, shap_selection_df


@app.cell
def _(shap_selection_df):
    shap_selection_df
    return


@app.cell
def _(SHAP_RANDOM_STATE, SHAP_SAMPLE_SIZE, X, pd):
    sample_n = min(SHAP_SAMPLE_SIZE, len(X))
    X_shap = X.sample(n=sample_n, random_state=SHAP_RANDOM_STATE).copy()
    X_background = X.sample(n=min(200, len(X)), random_state=SHAP_RANDOM_STATE + 1).copy()
    shap_sample_summary = pd.DataFrame(
        {
            "sample_size": [sample_n],
            "background_size": [len(X_background)],
            "feature_count": [X.shape[1]],
        }
    )
    return X_background, X_shap, shap_sample_summary


@app.cell
def _(shap_sample_summary):
    shap_sample_summary
    return


@app.cell
def _(X, shap_model_pipeline, y):
    shap_fitted_model = shap_model_pipeline.fit(X, y)
    shap_preprocessor = shap_fitted_model.named_steps["preprocess"]
    shap_tree_estimator = shap_fitted_model.named_steps["model"]
    return shap_preprocessor, shap_tree_estimator


@app.cell
def _(X_background, X_shap, shap, shap_preprocessor, shap_tree_estimator):
    X_background_transformed = shap_preprocessor.transform(X_background)
    X_shap_transformed = shap_preprocessor.transform(X_shap)
    shap_feature_names = shap_preprocessor.get_feature_names_out()
    shap_explainer = shap.TreeExplainer(shap_tree_estimator, data=X_background_transformed, feature_names=shap_feature_names)
    shap_values = shap_explainer(X_shap_transformed)
    return (shap_values,)


@app.cell
def _(plt, shap, shap_model_name, shap_values):
    plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title(f"SHAP beeswarm for {shap_model_name}")
    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(plt, shap, shap_model_name, shap_values):
    plt.figure(figsize=(12, 8))
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.title(f"Global SHAP importance for {shap_model_name}")
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Without Model feature

    Since the model feature seems to be explaining a lot of the variance, leaving almost no explanatory power to all other features, we can try to refit the models using the same methodology but excluding it from the pool of features.
    """)
    return


@app.cell
def _(
    DecisionTreeRegressor,
    ElasticNet,
    GridSearchCV,
    RANDOM_STATE,
    RandomForestRegressor,
    RandomizedSearchCV,
    Ridge,
    XGBRegressor,
    build_linear_pipeline,
    build_tree_pipeline,
    nominal_features,
    numeric_features,
    scale_features,
    tree_nominal_features,
):
    # Here is where Model is excluded from the set of features used to train the model
    nominal_features_no_model = nominal_features
    nominal_features_no_model.remove("Model")
    tree_nominal_features_no_model = tree_nominal_features
    tree_nominal_features_no_model.remove("Model")

    linear_pipeline_dict = {"scale_features": scale_features, "nominal_features": nominal_features_no_model}
    tree_pipeline_dict = {"numeric_features": numeric_features, "tree_nominal_features": tree_nominal_features_no_model}

    print(linear_pipeline_dict)
    print(tree_pipeline_dict)

    model_specs_no_model = {
        "ridge": {
            "estimator": build_linear_pipeline(Ridge(), linear_pipeline_dict),
            "search_cls": GridSearchCV,
            "search_kwargs": {
                "param_grid": {
                    "model__alpha": [0.01, 0.1, 1.0, 10.0],
                },
            },
        },
        "elastic_net": {
            "estimator": build_linear_pipeline(ElasticNet(random_state=RANDOM_STATE, max_iter=5000), linear_pipeline_dict),
            "search_cls": GridSearchCV,
            "search_kwargs": {
                "param_grid": {
                    "model__alpha": [0.0001, 0.001, 0.01],
                    "model__l1_ratio": [0.1, 0.5, 0.9],
                },
            },
        },
        "decision_tree": {
            "estimator": build_tree_pipeline(DecisionTreeRegressor(random_state=RANDOM_STATE), tree_pipeline_dict),
            "search_cls": RandomizedSearchCV,
            "search_kwargs": {
                "param_distributions": {
                    "model__max_depth": [None, 4, 6, 8, 12],
                    "model__min_samples_split": [2, 5, 10],
                    "model__min_samples_leaf": [1, 3, 5],
                    "model__criterion": ["squared_error", "friedman_mse"],
                },
                "n_iter": 10,
            },
        },
        "random_forest": {
            "estimator": build_tree_pipeline(RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1), tree_pipeline_dict),
            "search_cls": RandomizedSearchCV,
            "search_kwargs": {
                "param_distributions": {
                    "model__n_estimators": [200, 400],
                    "model__max_depth": [None, 10, 20],
                    "model__min_samples_split": [2, 5, 10],
                    "model__min_samples_leaf": [1, 3, 5],
                    "model__max_features": ["sqrt", 0.5],
                },
                "n_iter": 10,
            },
        },
        "xgboost": {
            "estimator": build_tree_pipeline(
                XGBRegressor(
                    objective="reg:squarederror",
                    random_state=RANDOM_STATE,
                    n_estimators=300,
                    n_jobs=1,
                    tree_method="hist",
                ),
                tree_pipeline_dict
            ),
            "search_cls": RandomizedSearchCV,
            "search_kwargs": {
                "param_distributions": {
                    "model__n_estimators": [200, 300, 400],
                    "model__max_depth": [3, 5, 7],
                    "model__learning_rate": [0.03, 0.1],
                    "model__subsample": [0.8, 1.0],
                    "model__colsample_bytree": [0.8, 1.0],
                    "model__min_child_weight": [1, 5],
                    "model__reg_alpha": [0.0, 0.1],
                    "model__reg_lambda": [1.0, 5.0],
                },
                "n_iter": 12,
            },
        },
    }
    return (model_specs_no_model,)


@app.cell
def _(
    X,
    build_model_summary_df,
    build_nested_cv_results_df,
    build_params_summary_df,
    model_specs_no_model,
    pd,
    y,
):
    nested_cv_results_df_no_model = build_nested_cv_results_df(model_specs_no_model)
    model_summary_df_no_model = build_model_summary_df(nested_cv_results_df_no_model)
    best_params_summary_rows_no_model, best_params_by_model_no_model = build_params_summary_df(nested_cv_results_df_no_model)

    best_params_summary_df_no_model = pd.DataFrame(best_params_summary_rows_no_model).sort_values("model")

    best_model_name_no_model = model_summary_df_no_model.iloc[0]["model"]
    best_model_params_no_model = best_params_by_model_no_model[best_model_name_no_model]
    best_model_pipeline_no_model = model_specs_no_model[best_model_name_no_model]["estimator"].set_params(**best_model_params_no_model)
    best_model_pipeline_no_model.fit(X, y)
    return best_model_name_no_model, best_params_by_model_no_model


@app.cell
def _(
    X,
    best_model_name_no_model,
    best_params_by_model_no_model,
    build_permutation_importance_df,
    build_tuned_cv_rows,
    model_specs_no_model,
    pd,
    y,
):
    tuned_cv_rows_no_model = build_tuned_cv_rows(best_params_by_model_no_model)
    tuned_cv_results_df_no_model = pd.DataFrame(tuned_cv_rows_no_model).sort_values("mean_r2", ascending=False)

    final_estimator_no_model = model_specs_no_model[best_model_name_no_model]["estimator"]\
        .set_params(**best_params_by_model_no_model[best_model_name_no_model])

    final_estimator_no_model.fit(X, y)

    permutation_importance_df_no_model = build_permutation_importance_df(final_estimator_no_model)

    top_permutation_importance_df_no_model = permutation_importance_df_no_model.head(20)
    return (permutation_importance_df_no_model,)


@app.cell
def _(permutation_importance_df_no_model):
    print(permutation_importance_df_no_model)
    return


if __name__ == "__main__":
    app.run()
