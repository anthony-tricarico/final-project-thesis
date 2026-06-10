"""Train and cache all ML experiment models for the mathanx project.

Usage:
    uv run python scripts/train_models.py
    uv run python scripts/train_models.py --exclude-top-performers
    uv run python scripts/train_models.py --model-name "Grok 4.1 Fast (Reasoning)"
    uv run python scripts/train_models.py --model-name "Ministral 14B (Reasoning)" "Anita 24B (Uncensored)"
    uv run python scripts/train_models.py --experiments all_features no_model
    uv run python scripts/train_models.py -e all_features --model-name "Ministral 3B"

Replicates the experiments from notebooks/ml_models.py and saves
trained artifacts to models/{experiment_name}/ for later reuse.

Filtering flags (composable):
  --exclude-top-performers   Exclude TOP_PERFORMERS models from the dataset.
  --model-name <name>...     One or more model names to filter the dataset by.

Experiment selection:
  --experiments / -e         Space-separated list of experiment names to run
                              (default: all six). Choices: all_features,
                              no_model, five_predictors,
                              all_features_with_confidence_scaled,
                              pca_predictors, pca_with_model.

Author: Anthony Tricarico
Email: tricarico672@gmail.com
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mathanx.ml.config import (
    DATASET_PATH,
    FIVE_FEATURE_COLUMNS,
    LEAKAGE_COLS,
    MODEL_FAMILIES,
    PCA_COMPONENT_COLUMNS,
    PCA_TRANSFORM_PATH,
    PSYCH_SCORE_COLUMNS,
    RANDOM_STATE,
    TARGET,
    TOP_PERFORMERS,
)
from mathanx.ml.helpers import (
    build_linear_pipeline,
    build_tree_pipeline,
    classify_columns,
    make_model_specs,
    run_experiment,
    save_experiment,
)


def _make_no_model_specs(numeric, nominal, tree_nominal, random_state):
    numeric_f = [c for c in numeric if c != "Model"]
    nominal_f = [c for c in nominal if c != "Model"]
    tree_nominal_f = [c for c in tree_nominal if c != "Model"]
    return make_model_specs(
        lambda m: build_linear_pipeline(m, numeric_f, nominal_f),
        lambda m: build_tree_pipeline(m, numeric_f, tree_nominal_f),
        random_state=random_state,
    )


def _make_five_predictor_specs(random_state):
    return make_model_specs(
        lambda m: Pipeline([("scale", StandardScaler()), ("model", m)]),
        lambda m: Pipeline([("model", m)]),
        random_state=random_state,
    )


def _feature_cols_without_leakage(df, *, allowed_leakage: set[str] | None = None):
    leakage_cols = set(LEAKAGE_COLS)
    if allowed_leakage:
        leakage_cols -= allowed_leakage
    return [
        c for c in df.columns
        if c not in leakage_cols and c not in {TARGET, "education_vs_parent_mean_gap"}
    ]


def _prepare_pca_data(df, feature_cols):
    _pca_data = joblib.load(PCA_TRANSFORM_PATH)
    _pc_scores = _pca_data["pca"].transform(
        _pca_data["scaler"].transform(df[PSYCH_SCORE_COLUMNS])
    )
    X_pca = df.loc[:, feature_cols].copy()
    X_pca = X_pca.drop(columns=PSYCH_SCORE_COLUMNS)
    X_pca[PCA_COMPONENT_COLUMNS[0]] = _pc_scores[:, 0]
    X_pca[PCA_COMPONENT_COLUMNS[1]] = _pc_scores[:, 1]
    y_pca = df[TARGET].copy()
    return X_pca, y_pca


EXPERIMENT_CHOICES = [
    "all_features",
    "all_features_with_confidence_scaled",
    "no_model",
    "five_predictors",
    "pca_predictors",
    "pca_with_model",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exclude-top-performers",
        action="store_true",
        help="Exclude TOP_PERFORMERS models from the dataset",
    )
    parser.add_argument(
        "--experiments", "-e",
        nargs="+",
        choices=EXPERIMENT_CHOICES,
        default=None,
        help="Experiment(s) to run (default: all six)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        nargs="+",
        default=None,
        help="One or more model names to filter the dataset by (e.g. 'Grok 4.1 Fast (Reasoning)')",
    )
    args = parser.parse_args()

    suffix_parts = []
    if args.exclude_top_performers:
        suffix_parts.append("_no_top")
    if args.model_name:
        model_set = frozenset(args.model_name)
        matched = False
        for family_name, family_models in MODEL_FAMILIES.items():
            if model_set == frozenset(family_models):
                suffix_parts.append(f"_{family_name}")
                matched = True
                break
        if not matched:
            if len(args.model_name) == 1:
                name = args.model_name[0]
                slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(".", "_")
                suffix_parts.append(f"_{slug}")
            else:
                slug = f"_{len(args.model_name)}_models"
                suffix_parts.append(slug)
    suffix = "".join(suffix_parts)

    print("Loading dataset...")
    ml_df = pd.read_csv(DATASET_PATH)
    print(f"  Shape: {ml_df.shape}")

    if args.exclude_top_performers:
        n_before = len(ml_df)
        ml_df = ml_df[~ml_df["Model"].isin(TOP_PERFORMERS)]
        print(f"  Excluded {len(TOP_PERFORMERS)} top performer(s): "
              f"{n_before} -> {len(ml_df)} rows")

    if args.model_name:
        available = sorted(ml_df["Model"].unique())
        missing = [m for m in args.model_name if m not in available]
        if missing:
            raise ValueError(
                f"Model(s) not found in dataset: {missing}. "
                f"Available models: {available}"
            )
        n_before = len(ml_df)
        ml_df = ml_df[ml_df["Model"].isin(args.model_name)].reset_index(drop=True)
        print(f"  Filtered to {len(args.model_name)} model(s): "
              f"{n_before} -> {len(ml_df)} rows")

    feature_cols = _feature_cols_without_leakage(ml_df)
    col_types = classify_columns(ml_df.loc[:, feature_cols])
    numeric_features = col_types["numeric_features"]
    nominal_features = col_types["nominal_features"]
    tree_nominal_features = col_types["tree_nominal_features"]

    feature_cols_with_confidence_scaled = _feature_cols_without_leakage(
        ml_df,
        allowed_leakage={"confidence_scaled"},
    )
    col_types_with_confidence_scaled = classify_columns(ml_df.loc[:, feature_cols_with_confidence_scaled])
    numeric_features_with_confidence_scaled = col_types_with_confidence_scaled["numeric_features"]
    nominal_features_with_confidence_scaled = col_types_with_confidence_scaled["nominal_features"]
    tree_nominal_features_with_confidence_scaled = col_types_with_confidence_scaled["tree_nominal_features"]

    print(
        f"  Features: {len(feature_cols)} "
        f"({len(numeric_features)} numeric, {len(nominal_features)} nominal)"
    )

    print("  Preparing PCA-transformed dataset...")
    _X_pca_full, _ = _prepare_pca_data(ml_df, feature_cols)
    pca_col_types = classify_columns(_X_pca_full)

    experiments = [
        {
            "base_name": "all_features",
            "name": f"all_features{suffix}",
            "get_xy": lambda df: (
                df.loc[:, feature_cols].copy(),
                df[TARGET].copy(),
            ),
            "build_specs": lambda: make_model_specs(
                lambda m: build_linear_pipeline(m, numeric_features, nominal_features),
                lambda m: build_tree_pipeline(m, numeric_features, tree_nominal_features),
                random_state=RANDOM_STATE,
            ),
        },
        {
            "base_name": "no_model",
            "name": f"no_model{suffix}",
            "get_xy": lambda df: (
                df.loc[:, feature_cols].copy(),
                df[TARGET].copy(),
            ),
            "build_specs": lambda: _make_no_model_specs(
                numeric_features, nominal_features, tree_nominal_features, RANDOM_STATE
            ),
        },
        {
            "base_name": "five_predictors",
            "name": f"five_predictors{suffix}",
            "get_xy": lambda df: (
                df.loc[:, FIVE_FEATURE_COLUMNS].copy(),
                df[TARGET].copy(),
            ),
            "build_specs": lambda: _make_five_predictor_specs(RANDOM_STATE),
        },
        {
            "base_name": "all_features_with_confidence_scaled",
            "name": f"all_features_with_confidence_scaled{suffix}",
            "get_xy": lambda df: (
                df.loc[:, feature_cols_with_confidence_scaled].copy(),
                df[TARGET].copy(),
            ),
            "build_specs": lambda: make_model_specs(
                lambda m: build_linear_pipeline(m, numeric_features_with_confidence_scaled, nominal_features_with_confidence_scaled),
                lambda m: build_tree_pipeline(m, numeric_features_with_confidence_scaled, tree_nominal_features_with_confidence_scaled),
                random_state=RANDOM_STATE,
            ),
        },
        {
            "base_name": "pca_predictors",
            "name": f"pca_predictors{suffix}",
            "get_xy": lambda df: _prepare_pca_data(df, feature_cols),
            "build_specs": lambda: make_model_specs(
                lambda m: build_linear_pipeline(
                    m,
                    [c for c in pca_col_types["numeric_features"] if c != "Model"],
                    [c for c in pca_col_types["nominal_features"] if c != "Model"],
                ),
                lambda m: build_tree_pipeline(
                    m,
                    [c for c in pca_col_types["numeric_features"] if c != "Model"],
                    [c for c in pca_col_types["tree_nominal_features"] if c != "Model"],
                ),
                random_state=RANDOM_STATE,
            ),
        },
        {
            "base_name": "pca_with_model",
            "name": f"pca_with_model{suffix}",
            "get_xy": lambda df: _prepare_pca_data(df, feature_cols),
            "build_specs": lambda: make_model_specs(
                lambda m: build_linear_pipeline(
                    m,
                    pca_col_types["numeric_features"],
                    pca_col_types["nominal_features"],
                ),
                lambda m: build_tree_pipeline(
                    m,
                    pca_col_types["numeric_features"],
                    pca_col_types["tree_nominal_features"],
                ),
                random_state=RANDOM_STATE,
            ),
        },
    ]

    if args.experiments:
        experiments = [e for e in experiments if e["base_name"] in args.experiments]

    for exp in experiments:
        name = exp["name"]
        print(f"\n{'=' * 60}")
        print(f"  Running experiment: {name}")
        print(f"{'=' * 60}")

        X_exp, y_exp = exp["get_xy"](ml_df)
        model_specs = exp["build_specs"]()

        result = run_experiment(X_exp, y_exp, model_specs, random_state=RANDOM_STATE)

        cache_dir = Path("models") / name
        print(f"  Best model: {result.best_model_name} "
              f"(R² = {result.model_summary.iloc[0]['mean_r2']:.4f})")
        print(f"  Saving artifacts to {cache_dir}")
        save_experiment(cache_dir, result)

    print("\nDone. All models trained and saved.")


if __name__ == "__main__":
    main()
