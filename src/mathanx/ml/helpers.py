from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

import shap
from collections import Counter
from pathlib import Path
import pickle


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        series = pd.Series(X.squeeze(), dtype="string").fillna("__missing__")
        self.feature_name_in_ = getattr(X, "columns", ["city_of_living"])[0]
        self.frequency_map_ = series.value_counts(normalize=True).to_dict()
        self.default_value_ = 0.0
        return self

    def transform(self, X):
        series = pd.Series(X.squeeze(), dtype="string").fillna("__missing__")
        encoded = series.map(self.frequency_map_).fillna(self.default_value_).astype(float)
        return encoded.to_numpy().reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        feature = input_features[0] if input_features else getattr(self, "feature_name_in_", "feature")
        return np.array([f"{feature}__frequency"])


class CategoricalHasher(BaseEstimator, TransformerMixin):
    def __init__(self, n_features: int = 64):
        self.n_features = n_features
        self.hasher_ = FeatureHasher(
            n_features=n_features,
            input_type="string",
            alternate_sign=True,
        )

    def fit(self, X, y=None):
        self.feature_names_in_ = list(getattr(X, "columns", []))
        return self

    def transform(self, X):
        frame = pd.DataFrame(X, columns=self.feature_names_in_).astype("string").fillna("__missing__")
        docs = [
            [f"{col}={row[col]}" for col in self.feature_names_in_]
            for _, row in frame.iterrows()
        ]
        return self.hasher_.transform(docs)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"hash_{i}" for i in range(self.n_features)])


def build_linear_pipeline(
    model,
    scale_features: List[str],
    nominal_features: List[str],
    n_hash_features: int = 64,
) -> Pipeline:
    transformer = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), scale_features),
            ("hash", CategoricalHasher(n_features=n_hash_features), nominal_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("preprocess", transformer), ("model", model)])


def build_tree_pipeline(
    model,
    numeric_features: List[str],
    tree_nominal_features: List[str],
) -> Pipeline:
    transformer = ColumnTransformer(
        transformers=[
            ("freq_city", FrequencyEncoder(), ["city_of_living"]),
            (
                "ordinal_nominal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                tree_nominal_features,
            ),
            ("numeric", "passthrough", numeric_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("preprocess", transformer), ("model", model)])


def build_search(
    model_name: str,
    model_specs: dict,
    inner_cv,
    random_state: int,
):
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

    return search_cls(**base_kwargs, **search_kwargs, random_state=random_state)


def run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_specs: dict,
    outer_cv,
    inner_cv,
    random_state: int,
):
    rows = []
    model_names = list(model_specs.keys())
    if len(model_names) != 1:
        raise ValueError("run_nested_cv expects model_specs for a single model")
    model_name = model_names[0]
    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        search = build_search(model_name=model_name, model_specs=model_specs, inner_cv=inner_cv, random_state=random_state)
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


def build_nested_cv_results_df(
    X: pd.DataFrame,
    y: pd.Series,
    model_specs: dict,
    outer_cv,
    inner_cv,
    random_state: int,
):
    return pd.concat(
        [run_nested_cv(X, y, {model_name: model_specs[model_name]}, outer_cv, inner_cv, random_state) for model_name in model_specs],
        ignore_index=True,
    )


def build_model_summary_df(nested_cv_results_df: pd.DataFrame, metric: Literal["mean_rmse", "mean_r2"] = "mean_r2"):
    ascending = False if metric == "mean_r2" else True
    return (
        nested_cv_results_df.groupby("model")
        .agg(
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            mean_mae=("mae", "mean"),
            mean_r2=("r2", "mean"),
        )
        .reset_index()
        .sort_values(metric, ascending=ascending)
    )


def build_params_summary_df(nested_cv_results_df: pd.DataFrame):
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


def build_tuned_cv_rows(
    X: pd.DataFrame,
    y: pd.Series,
    model_specs: dict,
    best_params_by_model: dict,
    post_tuning_cv,
):
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


def build_permutation_importance_df(estimator, X: pd.DataFrame, y: pd.Series):
    permutation_result = permutation_importance(estimator, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    return (
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


def select_shap_model(best_model_name: str, model_summary_df: pd.DataFrame, tree_model_names: List[str]):
    if best_model_name in tree_model_names:
        return best_model_name
    tree_candidates = model_summary_df[model_summary_df["model"].isin(tree_model_names)].sort_values("mean_rmse")
    return tree_candidates.iloc[0]["model"]


def sample_shap_data(X: pd.DataFrame, sample_size: int, random_state: int):
    sample_n = min(sample_size, len(X))
    return (
        X.sample(n=sample_n, random_state=random_state).copy(),
        X.sample(n=min(200, len(X)), random_state=random_state + 1).copy(),
    )


def fit_shap_model(model_pipeline: Pipeline, X: pd.DataFrame, y: pd.Series):
    fitted = model_pipeline.fit(X, y)
    model = fitted.named_steps["model"]
    preprocessor = fitted.named_steps.get("preprocess") or fitted.named_steps.get("scale")
    return fitted, preprocessor, model


def compute_shap_values_for_tree(shap_tree_estimator, shap_preprocessor, X_background: pd.DataFrame, X_shap: pd.DataFrame):
    if shap_preprocessor is not None:
        X_background_transformed = shap_preprocessor.transform(X_background)
        X_shap_transformed = shap_preprocessor.transform(X_shap)
        shap_feature_names = shap_preprocessor.get_feature_names_out()
    else:
        X_background_transformed = X_background.to_numpy() if hasattr(X_background, "to_numpy") else X_background
        X_shap_transformed = X_shap.to_numpy() if hasattr(X_shap, "to_numpy") else X_shap
        shap_feature_names = X_background.columns.tolist() if hasattr(X_background, "columns") else None
    shap_explainer = shap.TreeExplainer(shap_tree_estimator, data=X_background_transformed, feature_names=shap_feature_names)
    shap_values = shap_explainer(X_shap_transformed)
    return shap_explainer, shap_feature_names, shap_values


def compute_shap_values_for_linear(shap_linear_estimator, shap_preprocessor, X_background: pd.DataFrame, X_shap: pd.DataFrame):
    if shap_preprocessor is not None:
        X_background_transformed = pd.DataFrame(
            shap_preprocessor.transform(X_background),
            columns=X_background.columns,
            index=X_background.index,
        )
        X_shap_transformed = pd.DataFrame(
            shap_preprocessor.transform(X_shap),
            columns=X_shap.columns,
            index=X_shap.index,
        )
        shap_feature_names = shap_preprocessor.get_feature_names_out()
    else:
        X_background_transformed = X_background
        X_shap_transformed = X_shap
        shap_feature_names = X_background.columns.tolist()
    shap_explainer = shap.LinearExplainer(shap_linear_estimator, X_background_transformed)
    shap_values = shap_explainer(X_shap_transformed)
    return shap_explainer, shap_feature_names, shap_values


def plot_shap_beeswarm(shap_values, title: str):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()


def plot_shap_bar(shap_values, title: str):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()


def make_model_specs(
    build_linear: Callable,
    build_tree: Callable,
    *,
    random_state: int = 42,
    n_iter_search: int = 10,
    n_iter_xgb: int = 12,
    extra_specs: Optional[dict] = None,
) -> dict:
    specs = {
        "ridge": {
            "estimator": build_linear(Ridge()),
            "search_cls": GridSearchCV,
            "search_kwargs": {
                "param_grid": {"model__alpha": [0.01, 0.1, 1.0, 10.0]},
            },
        },
        "elastic_net": {
            "estimator": build_linear(ElasticNet(random_state=random_state, max_iter=5000)),
            "search_cls": GridSearchCV,
            "search_kwargs": {
                "param_grid": {
                    "model__alpha": [0.0001, 0.001, 0.01],
                    "model__l1_ratio": [0.1, 0.5, 0.9],
                },
            },
        },
        "decision_tree": {
            "estimator": build_tree(DecisionTreeRegressor(random_state=random_state)),
            "search_cls": RandomizedSearchCV,
            "search_kwargs": {
                "param_distributions": {
                    "model__max_depth": [None, 4, 6, 8, 12],
                    "model__min_samples_split": [2, 5, 10],
                    "model__min_samples_leaf": [1, 3, 5],
                    "model__criterion": ["squared_error", "friedman_mse"],
                },
                "n_iter": n_iter_search,
            },
        },
        "random_forest": {
            "estimator": build_tree(RandomForestRegressor(random_state=random_state, n_jobs=1)),
            "search_cls": RandomizedSearchCV,
            "search_kwargs": {
                "param_distributions": {
                    "model__n_estimators": [200, 400],
                    "model__max_depth": [None, 10, 20],
                    "model__min_samples_split": [2, 5, 10],
                    "model__min_samples_leaf": [1, 3, 5],
                    "model__max_features": ["sqrt", 0.5],
                },
                "n_iter": n_iter_search,
            },
        },
        "xgboost": {
            "estimator": build_tree(
                XGBRegressor(
                    objective="reg:squarederror",
                    random_state=random_state,
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
                "n_iter": n_iter_xgb,
            },
        },
    }
    if extra_specs:
        specs.update(extra_specs)
    return specs


@dataclass
class ExperimentResult:
    nested_cv_results: pd.DataFrame
    model_summary: pd.DataFrame
    best_params_summary_rows: list
    best_params_by_model: dict
    best_model_name: str
    best_model_params: dict
    best_model_pipeline: Pipeline
    tuned_cv_results: Optional[pd.DataFrame] = None
    final_estimator: Optional[Pipeline] = None
    permutation_importance: Optional[pd.DataFrame] = None


def run_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    model_specs: dict,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    run_tuned_cv: bool = True,
    run_permutation_importance: bool = True,
) -> ExperimentResult:
    outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    inner_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state + 1)

    nested_cv_results = build_nested_cv_results_df(X, y, model_specs, outer_cv, inner_cv, random_state)
    model_summary = build_model_summary_df(nested_cv_results)
    best_params_summary_rows, best_params_by_model = build_params_summary_df(nested_cv_results)
    best_model_name = model_summary.iloc[0]["model"]
    best_model_params = best_params_by_model[best_model_name]
    best_model_pipeline = model_specs[best_model_name]["estimator"].set_params(**best_model_params)

    tuned_cv_results = None
    final_estimator = None
    permutation_importance = None

    if run_tuned_cv:
        post_tuning_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state + 2)
        tuned = build_tuned_cv_rows(X, y, model_specs, best_params_by_model, post_tuning_cv)
        tuned_cv_results = pd.DataFrame(tuned).sort_values("mean_r2", ascending=False)

        final_estimator = model_specs[best_model_name]["estimator"].set_params(**best_model_params)
        final_estimator.fit(X, y)

        if run_permutation_importance:
            permutation_importance = build_permutation_importance_df(final_estimator, X, y)

    return ExperimentResult(
        nested_cv_results=nested_cv_results,
        model_summary=model_summary,
        best_params_summary_rows=best_params_summary_rows,
        best_params_by_model=best_params_by_model,
        best_model_name=best_model_name,
        best_model_params=best_model_params,
        best_model_pipeline=best_model_pipeline,
        tuned_cv_results=tuned_cv_results,
        final_estimator=final_estimator,
        permutation_importance=permutation_importance,
    )


def save_experiment(
    cache_dir: str | Path,
    result: ExperimentResult,
) -> None:
    """Save an ExperimentResult to disk for later reuse."""
    cache_dir = Path(cache_dir) if isinstance(cache_dir, str) else cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    result.model_summary.to_csv(cache_dir / "model_summary.csv", index=False)
    result.nested_cv_results.to_csv(cache_dir / "nested_cv_results.csv", index=False)

    if result.tuned_cv_results is not None:
        result.tuned_cv_results.to_csv(cache_dir / "tuned_cv_results.csv", index=False)
    if result.permutation_importance is not None:
        result.permutation_importance.to_csv(cache_dir / "permutation_importance.csv", index=False)

    pd.DataFrame(result.best_params_summary_rows).to_csv(cache_dir / "best_params_summary.csv", index=False)

    with open(cache_dir / "best_params_by_model.pkl", "wb") as f:
        pickle.dump(result.best_params_by_model, f)

    with open(cache_dir / "best_model_name.txt", "w") as f:
        f.write(result.best_model_name)

    if result.final_estimator is not None:
        import joblib
        joblib.dump(result.final_estimator, cache_dir / "pipeline.joblib")


def load_experiment(
    cache_dir: str | Path,
) -> ExperimentResult:
    """Load a previously saved ExperimentResult from disk."""
    import joblib

    cache_dir = Path(cache_dir) if isinstance(cache_dir, str) else cache_dir

    model_summary = pd.read_csv(cache_dir / "model_summary.csv")
    nested_cv_results = pd.read_csv(cache_dir / "nested_cv_results.csv")

    tuned_cv_path = cache_dir / "tuned_cv_results.csv"
    tuned_cv_results = pd.read_csv(tuned_cv_path) if tuned_cv_path.exists() else None

    perm_imp_path = cache_dir / "permutation_importance.csv"
    permutation_importance = pd.read_csv(perm_imp_path) if perm_imp_path.exists() else None

    best_params_summary = pd.read_csv(cache_dir / "best_params_summary.csv").to_dict("records")

    with open(cache_dir / "best_params_by_model.pkl", "rb") as f:
        best_params_by_model = pickle.load(f)

    with open(cache_dir / "best_model_name.txt") as f:
        best_model_name = f.read().strip()

    pipeline_path = cache_dir / "pipeline.joblib"
    final_estimator = joblib.load(pipeline_path) if pipeline_path.exists() else None

    return ExperimentResult(
        nested_cv_results=nested_cv_results,
        model_summary=model_summary,
        best_params_summary_rows=best_params_summary,
        best_params_by_model=best_params_by_model,
        best_model_name=best_model_name,
        best_model_params=best_params_by_model.get(best_model_name, {}),
        best_model_pipeline=final_estimator,
        tuned_cv_results=tuned_cv_results,
        final_estimator=final_estimator,
        permutation_importance=permutation_importance,
    )


def run_shap_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    model_specs: dict,
    best_model_name: str,
    best_params_by_model: dict,
    model_candidates_df: pd.DataFrame,
    tree_model_names: list[str],
    *,
    pipeline: Optional[Pipeline] = None,
    shap_sample_size: int = 1000,
    shap_random_state: int = 42,
    check_linear: bool = False,
) -> tuple:
    shap_model_name = best_model_name if pipeline is not None else select_shap_model(
        best_model_name, model_candidates_df, tree_model_names
    )
    if pipeline is not None:
        model_pipeline = pipeline
    else:
        model_pipeline = model_specs[shap_model_name]["estimator"].set_params(
            **best_params_by_model[shap_model_name]
        )
    X_shap, X_background = sample_shap_data(X, shap_sample_size, shap_random_state)
    fitted, preprocessor, tree_estimator = fit_shap_model(model_pipeline, X, y)

    if check_linear and shap_model_name in ("ridge", "elastic_net"):
        linear_estimator = fitted.named_steps["model"]
        *_, shap_values = compute_shap_values_for_linear(
            linear_estimator, preprocessor, X_background, X_shap
        )
    else:
        *_, shap_values = compute_shap_values_for_tree(
            tree_estimator, preprocessor, X_background, X_shap
        )

    return shap_values, shap_model_name


def classify_columns(X: pd.DataFrame) -> dict:
    from .config import NOMINAL_FEATURE_NAMES

    nominal_features = [c for c in NOMINAL_FEATURE_NAMES if c in X.columns]
    binary_features = [c for c in X.columns if set(X[c].dropna().unique()).issubset({0, 1})]
    numeric_features = [c for c in X.columns if c not in binary_features and c not in nominal_features]
    ordinal_features = [c for c in numeric_features if c.endswith("_ord")]
    continuous_features = [c for c in numeric_features if c not in ordinal_features]
    tree_nominal_features = [c for c in nominal_features if c != "city_of_living"]

    return {
        "nominal_features": nominal_features,
        "binary_features": binary_features,
        "numeric_features": numeric_features,
        "ordinal_features": ordinal_features,
        "continuous_features": continuous_features,
        "tree_nominal_features": tree_nominal_features,
    }
