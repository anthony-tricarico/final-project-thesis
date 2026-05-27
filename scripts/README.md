# Training pipeline

Trains the ML models used by `notebooks/ml_models.py` and caches the fitted
artifacts so the notebook can skip retraining on every run.

## Quick start

```bash
# Full run — all five experiments on complete dataset
uv run python scripts/train_models.py
```

Once artifacts exist, opening `notebooks/ml_models.py` in Marimo will load
them instead of re-running nested cross-validation.

## CLI reference

| Flag | Description |
|------|-------------|
| _(none)_ | Run all 5 experiments on the full dataset. |
| `--exclude-top-performers` | Remove `TOP_PERFORMERS` rows; appends `_no_top` to experiment dirs. |
| `--model-name NAME` | Filter to a single model (exact match); appends `_{slug}` to dirs. |
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

Also add `"only_demographics"` to `EXPERIMENT_CHOICES` so `--experiments`
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
