---
tags: [mlviz, extractor, random-forest]
status: current
version: v0.2.3
---

# Extractor — Random Forest (`extractors/random_forest.py`)

## What It Does

Serialises a fitted `RandomForestClassifier` or `RandomForestRegressor` into the `/api/forest` payload. Each tree in the ensemble is serialised using [[Extractor - Shared]]'s `serialize_sklearn_tree()`. The extractor also computes ensemble-level summaries: OOB score, feature importances, vote distribution (classification) or prediction distribution (regression).

## Public API

```python
serialize(model, X_train, y_train, query=None, feature_names=None, dataset_name=None) → dict
```

## JSON Data Contract

```json
{
  "model_type": "random_forest",
  "task_type": "classification",
  "classes": [0, 1, 2],
  "feature_names": ["petal length (cm)", ...],
  "dataset_name": "iris",
  "n_estimators": 100,
  "summary": {
    "n_estimators": 100,
    "max_depth": null,
    "max_features": "sqrt",
    "n_features": 4,
    "n_classes": 3
  },
  "trees": [
    {
      "tree_index": 0,
      "nodes": [...],
      "path": [...] | null
    }
  ],
  "importances": [
    {"feature_name": "petal length (cm)", "importance": 0.441, "rank": 1}
  ],
  "oob": {
    "available": true,
    "score": 0.9533,
    "error": 0.0467
  },
  "vote_distribution": { ... } | null,
  "prediction_distribution": { ... } | null,
  "prediction": { ... } | null
}
```

- `trees` — array of per-tree payloads. Each has the same node/path shape as the DT extractor.
- `vote_distribution` — classification only. Per-class vote counts and fractions across all estimators.
- `prediction_distribution` — regression only. Min/max/mean/std of per-tree predictions, plus a histogram.
- `oob` — only present when `model.oob_score=True`.
- `prediction` — null if no query supplied.

## Known Issue (code review backlog)

The vote winner in `_serialize_vote_distribution` uses hard-majority vote (most trees vote for class X). sklearn's `RandomForestClassifier.predict()` uses soft-probability averaging instead. These can disagree on borderline cases. Tracked as C1 in the code review backlog.

## Tests

File: `tests/test_random_forest_extractor.py`

| Test | Checks |
|------|--------|
| `test_random_forest_classification_*` | node count, leaf shapes, OOB block, vote distribution |
| `test_random_forest_regression_*` | regression leaf values, prediction distribution |
| `test_random_forest_feature_importances_*` | importances present, sorted, match model |

## Related Notes

- [[Extractor - Shared]] — tree serialisation helpers called per-estimator
- [[Architecture]] — where RF fits into the dispatch chain
- [[Frontend]] — `RandomForestView` renders this payload
