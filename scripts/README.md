# Training pipeline

Trains the ML models used by `notebooks/ml_models.py` and caches the fitted artifacts so the notebook can skip retraining on every run.

## Quick start

```bash
uv run python scripts/train_models.py
```

This runs three experiments (all features, without Model, five predictors)
and saves artifacts under `models/{experiment_name}/`.

Once artifacts exist, opening `notebooks/ml_models.py` in Marimo will load
them instead of re-running nested cross-validation.

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

Add a dict to the `experiments` list in `train_models.py`:

```python
{
    "name": "only_demographics",
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

Delete the relevant subdirectory under `models/` and re-run the script:

```bash
rm -rf models/all_features
uv run python scripts/train_models.py
```
