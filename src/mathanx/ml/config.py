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
    # "confidence_scaled", # include confidence scaled for every model
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
]

PCA_TRANSFORM_PATH: Path = DATASET_PATH.parent / "pca_transform.joblib"

PSYCH_SCORE_COLUMNS: list[str] = [
    "amas_score",
    "maes_score",
    "mseaq_anx",
    "mseaq_se",
]

PCA_COMPONENT_COLUMNS: list[str] = [
    "PC1",
    "PC2",
]

TOP_PERFORMERS: list[str] = [
    "Grok 4.1 Fast (Reasoning)",
    "DeepSeek Chat"
]

MISTRAL_FAMILY: list[str] = [
    "Ministral 14B (Reasoning)",
    "Anita 24B (Uncensored)",
    "Magistral Small",
    "Mistral Small 4",
    "Mistral Small 3.2",
]

QWEN3_FAMILY: list[str] = [
    "Qwen3 4B (Thinking)",
    "Qwen3 4B (Uncensored)",
    "Qwen3 4B",
]

MODEL_FAMILIES: dict[str, list[str]] = {
    "mistral_family": MISTRAL_FAMILY,
    "qwen3_family": QWEN3_FAMILY,
}
