"""Train and cache all ML experiment models for the mathanx project.

Usage:
    uv run python scripts/train_models.py

Replicates the three experiments from notebooks/ml_models.py and saves
trained artifacts to models/{experiment_name}/ for later reuse.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mathanx.ml.config import (
    DATASET_PATH,
    FIVE_FEATURE_COLUMNS,
    LEAKAGE_COLS,
    RANDOM_STATE,
    TARGET,
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


def main():
    print("Loading dataset...")
    ml_df = pd.read_csv(DATASET_PATH)
    print(f"  Shape: {ml_df.shape}")

    feature_cols = [
        c for c in ml_df.columns
        if c not in LEAKAGE_COLS and c not in {TARGET, "education_vs_parent_mean_gap"}
    ]
    col_types = classify_columns(ml_df.loc[:, feature_cols])
    numeric_features = col_types["numeric_features"]
    nominal_features = col_types["nominal_features"]
    tree_nominal_features = col_types["tree_nominal_features"]

    print(
        f"  Features: {len(feature_cols)} "
        f"({len(numeric_features)} numeric, {len(nominal_features)} nominal)"
    )

    experiments = [
        {
            "name": "all_features",
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
            "name": "no_model",
            "get_xy": lambda df: (
                df.loc[:, feature_cols].copy(),
                df[TARGET].copy(),
            ),
            "build_specs": lambda: _make_no_model_specs(
                numeric_features, nominal_features, tree_nominal_features, RANDOM_STATE
            ),
        },
        {
            "name": "five_predictors",
            "get_xy": lambda df: (
                df.loc[:, FIVE_FEATURE_COLUMNS].copy(),
                df[TARGET].copy(),
            ),
            "build_specs": lambda: _make_five_predictor_specs(RANDOM_STATE),
        },
    ]

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
