# mathanx — Math Anxiety in LLMs

Exploring whether large language models exhibit math anxiety through psychometric scales, emotion analysis, and predictive modelling.

14 LLMs each completed a structured 4-call experimental paradigm — both raw (`llm` mode) and with synthetic persona demographics (`human` mode) — producing ~28,000 experimental runs. The ML pipeline predicts math problem-solving accuracy from psychometric scores, demographics, emotional features, and fallacy scores using nested cross-validation with SHAP explainability.

---

## The Four Calls

| Call | Task | Description | Output |
|------|------|-------------|--------|
| **1** | **TFMN** (Textual Forma Mentis) | 7 open-ended questions probing the LLM's relationship with math, anxiety, AI use, and math explanations. Each answer analysed with the `emoatlas` library for 8 emotion dimensions. | Free-text answers + emotion z-scores |
| **2** | **Psychometric Scales** | Three validated Likert instruments: **MAES** (9 items), **AMAS** (9 items), **MSEAQ** (28 items). All rated 1–5. | 46 item ratings |
| **3** | **Forma Mentis Network** | 25 cue words for free association to build behavioural semantic networks. | Association pairs + valence scores |
| **4** | **MSESR Problem Solving** | 18 multiple-choice math problems. Each records the chosen option (A–E), free-text reasoning, and confidence rating (1–5). | Chosen options + reasoning + confidence |

**Target variable:** `accuracy` derived from Call 4 by comparing chosen options against `MSESR_CORRECT_ANSWERS`.

---

## What is Marimo?

[Marimo](https://marimo.io) is a reactive Python notebook — a modern alternative to Jupyter. Notebooks are stored as **plain `.py` files**, not `.ipynb` JSON blobs.

### Marimo vs. Jupyter

| Feature | Marimo | Jupyter |
|---------|--------|---------|
| File format | `.py` — diffable, mergeable, importable | `.ipynb` — large JSON diffs, hard to review |
| Execution model | **Reactive** — cells auto-update when dependencies change | **Sequential** — cells must be manually re-run in order |
| Hidden state | Impossible — the reactive DAG guarantees consistency | Common source of bugs — stale cells produce misleading results |
| IDE integration | Full `.py` files — open in any editor, review in PRs, lint with ruff | Requires Notebook viewer or `nbconvert` |
| Version control | Standard Git diffs — meaningful code reviews | Requires `nbstripout` or similar to avoid binary-like diffs |
| Importable | `from notebooks import ml_models` works | Requires `nbformat` + `nbconvert` to execute programmatically |
| Headless execution | `uv run python notebook.py` runs all cells | Requires `jupyter nbconvert --execute` |

### Why Marimo here?

Research reproducibility demands deterministic execution. Marimo's reactive model ensures results are consistent across sessions, and `.py` files enable code review, CI checks, and direct import of notebook logic into scripts. Each notebook in this repository is both an interactive exploration environment and a runnable Python module.

---

## What is `uv`?

[uv](https://docs.astral.sh/uv/) is a fast Python package and project manager written in Rust by Astral. It replaces `pip` + `venv` + `pip-tools` + `poetry` with a single binary.

### How `uv` is used in this repo

| Command / File | Purpose |
|----------------|---------|
| `uv sync` | Install exact dependencies from `uv.lock` — deterministic across machines |
| `uv run` | Execute any command in the project's virtual environment without manual activation |
| `uv add` / `uv remove` | Add or remove dependencies in `pyproject.toml` and update `uv.lock` |
| `uv.lock` | Committed lockfile — guarantees all collaborators and CI get identical trees |
| `.python-version` | uv auto-detects this and uses Python 3.12 |
| `uv_build` | PEP 517 build backend, configured in `pyproject.toml` |
| `uv run marimo edit ...` | Launch marimo notebooks in the project environment |

Notable: there is no `pip install`, no `pip freeze`, no `poetry.lock` — uv handles everything.

---

## Quick Start

**Prerequisites:** Python 3.12, [uv installed](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone <repo-url>
cd mathanx
uv sync
export PYTHONPATH=src
```

The `.env` file already contains `PYTHONPATH=src` — uv loads it automatically when running commands.

---

## Repository Structure

```
.
├── pyproject.toml          # Project metadata, dependencies, build config
├── uv.lock                 # Deterministic dependency lockfile
├── .python-version         # 3.12
├── .env                    # PYTHONPATH=src
│
├── src/mathanx/            # Python package
│   ├── constants.py        # Domain constants (cues, correct answers, model names, …)
│   ├── schemas.py          # Pydantic v2 schemas for all 4 calls
│   ├── validator.py        # DataValidator — validates raw JSON runs
│   └── ml/
│       ├── config.py       # ML experiment configuration
│       └── helpers.py      # Training pipeline: transformers, specs, experiments, SHAP
│
├── scripts/
│   └── train_models.py     # Headless ML training CLI (see scripts/README.md)
│
├── notebooks/              # Marimo notebooks (.py files)
│   ├── file_cleaning.py
│   ├── task2_distributions.py
│   ├── task4_accuracy.py
│   ├── ml_dataset_creation.py
│   ├── ml_data_exploration.py
│   ├── ml_models.py
│   └── ml_models_explorer.py
│
├── code/data_processing/   # Standalone data processing utilities
│   ├── generate_call2_dataset.py
│   ├── build_fallacies_dataset.py
│   └── random_sampler.py
│
├── misc/
│   ├── dataset_aggregations_pooling_system.py   # Pooling system notebook
│   └── README.md                                # Pooling system docs
│
├── data/                   # Datasets (gitignored)
│   ├── raw/                # Raw JSON output per model (14 model dirs)
│   ├── raw_original/       # Original MEDS dataset
│   └── processed/          # Validated tables, ML dataset, pooled table
│
├── figures/                # Generated figures (gitignored)
│   └── mermaid/            # Pipeline diagrams (.mmd source + .pdf/.png renders)
│
└── models/                 # Trained ML artifacts, 18 experiments (gitignored)
```

---

## Data Pipeline Overview

```
Raw JSON (14 models)
    │
    ▼
DataValidator (Pydantic schema validation)
    │
    ├── Processed tables (CSV, Parquet, Pickle)
    │
    ▼
Pooling System (misc/dataset_aggregations_pooling_system.py)
    │  Consolidates Call 1 + Call 2 + Call 4 + Demographics into
    │  a single wide-format table (27,987 rows × 205 columns)
    │
    ▼
ML Feature Engineering (notebooks/ml_dataset_creation.py)
    │  Ordinal encoding, derived flags (math_lover, has_children, …),
    │  fallacy aggregation, PCA on psychometric scores
    │
    ▼
ml_dataset.csv → Model Training (scripts/train_models.py)
    │  Nested CV → Model selection → Tuned CV → Permutation importance → SHAP
    │
    ▼
Model artifacts & figures
```

### Mermaid Pipeline Diagrams

The `figures/mermaid/` directory contains three pipeline diagrams in editable `.mmd` format with rendered `.pdf` and `.png` versions:

| Diagram | File | Description |
|---------|------|-------------|
| **Data Preparation** | `data_preparation.mmd` | Dataset loading → cleaning → merging → feature engineering → final `ml_dataset.csv` |
| **Feature Transform** | `feature_transform.mmd` | Two preprocessing pipelines: **Linear** (StandardScaler + CategoricalHasher) and **Tree** (FrequencyEncoder + OrdinalEncoder) routing to their respective estimators |
| **ML Pipeline** | `ml_pipeline.mmd` | Top-level training flow: data prep → nested cross-validation (inner tune + outer evaluate) → tuned CV → final fit → persisted artifacts |

---

## Marimo Notebooks — How to Run & Inspect

### Notebook Inventory

| Notebook | Purpose | Est. run time |
|----------|---------|---------------|
| `file_cleaning.py` | Validate raw JSON against Pydantic schemas, track valid/invalid runs | ~2 min |
| `task2_distributions.py` | Psychometric score distribution analysis with item reversal logic | ~1 min |
| `task4_accuracy.py` | Accuracy computation + confidence/overconfidence analysis | ~3 min |
| `ml_dataset_creation.py` | Feature engineering: ordinal encodings, derived flags, PCA → `ml_dataset.csv` | ~5 min |
| `ml_data_exploration.py` | Comprehensive EDA: 17 sections covering distributions, correlations, PCA, per-model analysis | ~10 min |
| `ml_models.py` | Main ML modelling: 6 experiments × 5 algorithms, nested CV, SHAP explainability | ~60–90 min |
| `ml_models_explorer.py` | Interactive browser for trained experiment artifacts + on-demand SHAP | real-time |

### Commands

```bash
# Edit mode — opens an interactive web UI with a reactive notebook editor
uv run marimo edit notebooks/file_cleaning.py

# App mode — read-only rendered UI (useful for presenting results)
uv run marimo run notebooks/file_cleaning.py

# Headless execution — runs all cells and prints output to terminal
uv run python notebooks/file_cleaning.py

# Export to standalone HTML
uv run marimo export html notebooks/file_cleaning.py -o output.html

# Export to Jupyter notebook (for sharing with Jupyter users)
uv run marimo export ipynb notebooks/file_cleaning.py
```

Marimo also supports a `--port` flag and `--host` for remote access:

```bash
uv run marimo edit notebooks/ml_models.py --port 8888 --host 0.0.0.0
```

---

## Scripts — Training Pipeline

The main headless entry point is `scripts/train_models.py`. It replicates the experiments from `notebooks/ml_models.py` and caches trained artifacts so the notebook can skip retraining.

### Usage

```bash
# Full run — all 6 experiments on the complete dataset
uv run python scripts/train_models.py

# Single experiment
uv run python scripts/train_models.py -e all_features

# Exclude top-performing models (DeepSeek Chat, Grok 4.1 Fast)
uv run python scripts/train_models.py --exclude-top-performers

# Train on a specific model
uv run python scripts/train_models.py --model-name "Ministral 3B"

# Combined filters
uv run python scripts/train_models.py -e all_features \
    --model-name "Granite 4 Tiny" --exclude-top-performers
```

### 6 Experiments

| Name | Description |
|------|-------------|
| `all_features` | All 35 features including `Model` |
| `no_model` | Same features, `Model` column excluded |
| `five_predictors` | 5-best-predictors baseline |
| `all_features_with_confidence_scaled` | All features with `confidence_scaled` |
| `pca_predictors` | Psychometric scores → 2 PCA components, no Model |
| `pca_with_model` | PCA components + Model |

### Model Families

`--model-name` auto-detects known families from `config.py`:

```bash
# Mistral family (5 models)
uv run python scripts/train_models.py -e no_model \
    --model-name "Ministral 14B (Reasoning)" "Anita 24B (Uncensored)" \
                 "Magistral Small" "Mistral Small 4" "Mistral Small 3.2"

# Qwen3 family (3 models)
uv run python scripts/train_models.py -e no_model \
    --model-name "Qwen3 4B (Thinking)" "Qwen3 4B (Uncensored)" "Qwen3 4B"
```

### Saved Artifacts per Experiment

```
models/{experiment_name}/
├── model_summary.csv          # Nested CV performance by model
├── nested_cv_results.csv      # Per-fold nested CV results
├── tuned_cv_results.csv       # Post-tuning CV performance
├── permutation_importance.csv # Permutation importance (best model)
├── best_params_summary.csv    # Modal best params per model
├── best_params_by_model.pkl   # Pickled {model: params} dict
├── best_model_name.txt        # Best model identifier
└── pipeline.joblib            # Fitted best-model Pipeline
```

See `scripts/README.md` for the full CLI reference and developer guide.

---

## Models & Results

### Algorithms

| Model | Type | Implementation |
|-------|------|----------------|
| **Ridge** | Linear | `sklearn.linear_model.Ridge` |
| **ElasticNet** | Linear | `sklearn.linear_model.ElasticNet` |
| **Decision Tree** | Tree | `sklearn.tree.DecisionTreeRegressor` |
| **Random Forest** | Tree | `sklearn.ensemble.RandomForestRegressor` |
| **XGBoost** | Tree | `xgboost.XGBRegressor` |

### 18 Experiments

All combinations of feature sets × model filters produce 18 experiment directories in `models/`. The target is always `accuracy` (R² regression). Full results per experiment are in `models/{name}/model_summary.csv`.

| Experiment Directory | Best Model |
|----------------------|------------|
| `all_features` | XGBoost |
| `all_features_no_top` | XGBoost |
| `all_features_with_confidence_scaled` | XGBoost |
| `five_predictors` | XGBoost |
| `five_predictors_no_top` | XGBoost |
| `no_model` | XGBoost |
| `no_model_no_top` | XGBoost |
| `no_model_best_performers` | Random Forest |
| `no_model_mistral_family` | Random Forest |
| `no_model_mistral_small_4` | ElasticNet |
| `no_model_qwen3_family` | XGBoost |
| `no_model_misc_models` | Random Forest |
| `no_model_grok_4_1_fast_reasoning` | ElasticNet |
| `pca_predictors` | XGBoost |
| `pca_predictors_no_top` | XGBoost |
| `pca_predictors_grok_4_1_fast_reasoning` | ElasticNet |
| `pca_with_model` | XGBoost |
| `pca_with_model_no_top` | XGBoost |

---

## Reproduction Instructions

### Full End-to-End Reproduction

```bash
# 1. Set up the environment
uv sync
export PYTHONPATH=src

# 2. Validate raw data against Pydantic schemas
uv run marimo run notebooks/file_cleaning.py

# 3. Analyse psychometric score distributions
uv run marimo run notebooks/task2_distributions.py

# 4. Compute accuracy scores from Call 4
uv run marimo run notebooks/task4_accuracy.py

# 5. Build the pooled wide-format dataset
uv run marimo run misc/dataset_aggregations_pooling_system.py

# 6. Engineer ML features and export ml_dataset.csv
uv run marimo run notebooks/ml_dataset_creation.py

# 7. Run exploratory data analysis
uv run marimo run notebooks/ml_data_exploration.py

# 8. Train all ML models headless (saves artifacts to models/)
uv run python scripts/train_models.py

# 9. Open the full ML modelling notebook (auto-loads cached artifacts)
uv run marimo edit notebooks/ml_models.py

# 10. Browse trained experiments interactively
uv run marimo edit notebooks/ml_models_explorer.py
```

### Partial Reproduction

- **Skip to modelling**: If `models/` already contains trained artifacts, open `notebooks/ml_models.py` directly — it detects cached results and skips retraining.
- **Single experiment**: Use `-e` to run one experiment at a time (see [Scripts](#scripts--training-pipeline)).
- **Model family only**: Use `--model-name` to train on a subset of models.

### Data Access

- Raw data lives in `data/raw/` — 14 model directories with JSON files.
- If raw directories are missing, extract `data/dataset.zip`.
- The original MEDS dataset snapshot is preserved in `data/raw_original/`.
- All processed intermediates (validated tables, pooled dataset, ML features) are in `data/processed/`.

---

## `scripts/README.md`

The `scripts/` directory has its own README with detailed documentation on:
- Adding new experiments and models
- Model family experiment results (R² values)
- Regenerating specific artifact directories

---

## `misc/README.md`

The `misc/` directory contains documentation for the **Pooling System Dataset** (`df_pooling_system.parquet`), the 205-column wide-format table that consolidates all four calls with demographics.

---

## Citation

**Author:** Anthony Tricarico — [tricarico672@gmail.com](mailto:tricarico672@gmail.com)

---

## License

This project is licensed under the terms specified in the repository license file.
