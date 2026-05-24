from __future__ import annotations

from pathlib import Path


TARGET: str = "accuracy"
RANDOM_STATE: int = 42
N_FOLDS: int = 5

DATASET_PATH: Path = Path("data/processed/ml/ml_dataset.csv").resolve()
FIG_PATH: Path = Path("figures/").resolve()

LEAKAGE_COLS: set[str] = {
    "run_id",
    "accuracy",
    "confidence",
    "delta_confidence",
    "total_correct",
    "n_observations",
}

TREE_MODEL_NAMES: list[str] = ["decision_tree", "random_forest", "xgboost"]

SHAP_SAMPLE_SIZE: int = 1000

NOMINAL_FEATURE_NAMES: set[str] = {
    "gender",
    "sexual_orientation",
    "city_of_living",
    "employment_status",
    "marital_status",
    "migration_status",
    "religious_beliefs",
    "Model",
}

FIVE_FEATURE_COLUMNS: list[str] = [
    "mseaq_anx",
    "amas_score",
    "maes_score",
    "mseaq_se",
    "confidence_scaled",
]
