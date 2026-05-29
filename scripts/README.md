# Training pipeline

Trains the ML models used by `notebooks/ml_models.py` and caches the fitted
artifacts so the notebook can skip retraining on every run.

## Quick start

```bash
# Full run — all six experiments on complete dataset
uv run python scripts/train_models.py
```

Once artifacts exist, opening `notebooks/ml_models.py` in Marimo will load
them instead of re-running nested cross-validation.

## CLI reference

| Flag | Description |
|------|-------------|
| _(none)_ | Run all 6 experiments on the full dataset. |
| `--exclude-top-performers` | Remove `TOP_PERFORMERS` rows; appends `_no_top` to experiment dirs. |
| `--model-name NAME...` | Filter to one or more models (exact match); appends family slug or `_{N}_models` to dirs. |
| `--experiments` / `-e` | Space-separated subset of experiments to run (see below). |

All flags are composable. Filtering order: `--model-name` first, then
`--exclude-top-performers`.

## Experiment selection

Use `--experiments` / `-e` to run a subset:

```bash
# Run one experiment
uv run python scripts/train_models.py -e all_features

# Run two experiments
uv run python scripts/train_models.py -e all_features five_predictors

# Combine with other flags
uv run python scripts/train_models.py -e all_features --exclude-top-performers
uv run python scripts/train_models.py -e all_features --model-name "Ministral 3B"
```

Valid experiment names:

| Name | Description |
|------|-------------|
| `all_features` | All 35 features including `Model`. |
| `no_model` | Same features, `Model` column excluded. |
| `five_predictors` | 5-best-predictors baseline. |
| `all_features_with_confidence_scaled` | All features, with `confidence_scaled` treated as non-leakage only for this experiment. |
| `pca_predictors` | Psychometric scores replaced by 2 PCA components, `Model` excluded. |
| `pca_with_model` | Psychometric scores replaced by 2 PCA components, `Model` included. |

## Filtering dataset rows

```bash
# Exclude the two top-performing models (DeepSeek Chat, Grok 4.1 Fast)
uv run python scripts/train_models.py --exclude-top-performers

# Train only on a specific model's data
uv run python scripts/train_models.py --model-name "Ministral 3B"

# Combine: specific model, excluding top performers, single experiment
uv run python scripts/train_models.py -e all_features \
    --model-name "Granite 4 Tiny" --exclude-top-performers
```

Invalid model names raise `ValueError` listing available models.

```bash
# Filter to a subset of models (model family)
uv run python scripts/train_models.py -e no_model \
    --model-name "Ministral 14B (Reasoning)" \
                 "Anita 24B (Uncensored)" \
                 "Magistral Small" \
                 "Mistral Small 4" \
                 "Mistral Small 3.2"

# Filter to Qwen3 family
uv run python scripts/train_models.py -e no_model \
    --model-name "Qwen3 4B (Thinking)" \
                 "Qwen3 4B (Uncensored)" \
                 "Qwen3 4B"
```

When the model names exactly match a known family (e.g. `MISTRAL_FAMILY` or
`QWEN3_FAMILY` in `config.py`), the experiment directory gets a readable suffix
like `no_model_mistral_family`. Otherwise, a generic `_{N}_models` suffix is
used.

## Model family experiments

Run the `no_model` experiment on a predefined model family to measure
explanatory power within an architecture family.

### Mistral family

```bash
uv run python scripts/train_models.py -e no_model \
    --model-name "Ministral 14B (Reasoning)" \
                 "Anita 24B (Uncensored)" \
                 "Magistral Small" \
                 "Mistral Small 4" \
                 "Mistral Small 3.2"
```

| Detail | Value |
|--------|-------|
| Output dir | `models/no_model_mistral_family/` |
| Rows | 7,517 (5 models × ~1,500) |
| Best model | `random_forest` |
| Best R² | 0.0609 |

### Qwen3 family

```bash
uv run python scripts/train_models.py -e no_model \
    --model-name "Qwen3 4B (Thinking)" \
                 "Qwen3 4B (Uncensored)" \
                 "Qwen3 4B"
```

| Detail | Value |
|--------|-------|
| Output dir | `models/no_model_qwen3_family/` |
| Rows | 4,499 (3 models × ~1,500) |
| Best model | `xgboost` |
| Best R² | 0.3055 |

### Best performers

```bash
uv run python scripts/train_models.py -e no_model \
    --model-name "Grok 4.1 Fast (Reasoning)" \
                 "DeepSeek Chat"
```

| Detail | Value |
|--------|-------|
| Output dir | `models/no_model_best_performers/` |
| Rows | 3,000 (2 models × 1,500) |
| Best model | `random_forest` |
| Best R² | 0.1646 |

### Misc models

```bash
uv run python scripts/train_models.py -e no_model \
    --model-name "Qwen3.5 9B" \
                 "Ministral 3B" \
                 "Phi-4 (Reasoning+)" \
                 "Granite 4 Tiny"
```

| Detail | Value |
|--------|-------|
| Output dir | `models/no_model_misc_models/` |
| Rows | 5,982 (4 models × ~1,500) |
| Best model | `random_forest` |
| Best R² | 0.2978 |

All experiments exclude the `Model` column from features (via the `no_model`
experiment), so the remaining predictors compete to explain accuracy variance
_within_ the family. See `notebooks/ml_models.py` sections **11b–11i** for SHAP
explainability.

## Artifacts saved per experiment

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

## Adding a new experiment

Add a dict to the `experiments` list in `train_models.py`. Both `base_name`
(the CLI-friendly identifier) and `name` (with suffix applied) are required:

```python
{
    "base_name": "only_demographics",
    "name": f"only_demographics{suffix}",
    "get_xy": lambda df: (
        df[["gender", "age", "education"]].copy(),
        df[TARGET].copy(),
    ),
    "build_specs": lambda: make_model_specs(
        lambda m: build_linear_pipeline(m, numeric_f, nominal_f),
        lambda m: build_tree_pipeline(m, numeric_f, tree_nominal_f),
        random_state=RANDOM_STATE,
    ),
}
```

Also add the experiment's `base_name` to `EXPERIMENT_CHOICES` so `--experiments`
accepts it.

## Adding new models

Pass `extra_specs` to `make_model_specs()`:

```python
extra = {
    "svm": {
        "estimator": build_linear(SVR()),
        "search_cls": GridSearchCV,
        "search_kwargs": {
            "param_grid": {"model__kernel": ["rbf", "linear"]},
        },
    },
}
model_specs = make_model_specs(build_linear, build_tree, extra_specs=extra)
```

## Regenerating artifacts

Delete the subdirectory under `models/` and re-run:

```bash
rm -rf models/all_features_no_top
uv run python scripts/train_models.py -e all_features --exclude-top-performers
```
