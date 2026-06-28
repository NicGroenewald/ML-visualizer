---
tags: [mlviz, extractor, shared]
status: current
version: v0.2.3
---

# Extractor — Shared Helpers (`extractors/shared.py`)

## What It Does

Holds the core sklearn tree-walking and serialisation logic used by both [[Extractor - DT]] and [[Extractor - RF]]. Neither extractor duplicates this code.

## Functions

### `resolve_feature_names(model, X_train, feature_names=None) → list[str]`

Priority order:
1. `feature_names` kwarg — always wins
2. `model.feature_names_in_` — set by sklearn when fitted on a DataFrame
3. `X_train.columns` — if X_train is still a pandas DataFrame
4. Auto-generate `["x0", "x1", ...]` — fallback for plain numpy arrays

### `normalize_query_vector(query, feature_names) → np.ndarray | None`

Converts a query to a 1D numpy array. Handles DataFrame rows, Series, lists, and plain arrays. Returns `None` if query is `None`.

### `model_task_type(model) → "classification" | "regression"`

Checks `is_classifier(model)` from sklearn. Raises `NotImplementedError` for unsupported types.

### `ensure_supported_tree_task(task_type)`

Raises `ValueError` for multi-output regression (not supported).

### `regression_value(tree, node_id) → float`

Extracts the leaf mean prediction value from `tree.value[node_id]` for regression trees. Rounds to 4 decimal places.

### `serialize_sklearn_tree(tree, feature_names, task_type) → list[dict]`

Walks `range(tree.node_count)` and emits one dict per node.

**Split node shape:**
```json
{
  "node_id": 0,
  "feature": "petal length (cm)",
  "threshold": 2.45,
  "n_samples": 150,
  "gini": 0.667,
  "left_child": 1,
  "right_child": 2,
  "leaf": false
}
```

**Classification leaf shape:**
```json
{
  "node_id": 1,
  "feature": null,
  "threshold": null,
  "n_samples": 50,
  "gini": 0.0,
  "left_child": null,
  "right_child": null,
  "leaf": true,
  "counts": [50, 0, 0],
  "prediction": 0
}
```

**Regression leaf shape:** same but `value: 23.45` instead of `counts` / `prediction`.

Rules:
- `nodes` is a flat list — React looks up nodes by ID in O(1)
- `prediction` is a class index — resolved to a class name via `classes` array in the view
- Thresholds and gini values rounded to 3 decimal places
- `counts` values are integers (sklearn stores them as floats internally)

### `trace_sklearn_path(tree, feature_names, task_type, query) → list[dict] | None`

Follows a single query point root-to-leaf. Returns `None` if query is `None`.

**Path step shape (split node):**
```json
{
  "node_id": 0,
  "feature": "petal length (cm)",
  "threshold": 2.45,
  "value": 1.4,
  "went_left": true,
  "leaf": false
}
```

**Path step shape (leaf):**
```json
{
  "node_id": 1,
  "leaf": true,
  "counts": [50, 0, 0],
  "prediction": 0
}
```

Regression leaf path step uses `value` instead of `counts`/`prediction`.

## Related Notes

- [[Extractor - DT]] — calls these helpers for single-tree payloads
- [[Extractor - RF]] — calls `serialize_sklearn_tree` for each tree in the ensemble
- [[Decision Tree]] — sklearn flat array internals
