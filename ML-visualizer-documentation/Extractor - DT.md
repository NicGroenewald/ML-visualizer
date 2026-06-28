---
tags: [mlviz, extractor, decision-tree]
status: current
version: v0.2.3
---

# Extractor — Decision Tree (`extractors/decision_tree.py`)

## What It Does

A thin wrapper over [[Extractor - Shared]]. Calls `serialize_sklearn_tree()` and `trace_sklearn_path()` from `shared.py` and assembles the `/api/tree` payload.

Supports both `DecisionTreeClassifier` and `DecisionTreeRegressor`.

## Public API

```python
serialize(model, X_train, y_train, query=None, feature_names=None, dataset_name=None) → dict
```

The only function the server calls. It:

1. Calls `shared.model_task_type(model)` → `"classification"` or `"regression"`
2. Calls `shared.resolve_feature_names(model, X_train, feature_names)`
3. Calls `shared.normalize_query_vector(query, feature_names)` if query supplied
4. Calls `shared.serialize_sklearn_tree(model.tree_, feature_names, task_type)` → `nodes`
5. Calls `shared.trace_sklearn_path(model.tree_, feature_names, task_type, query)` → `path`
6. Returns the assembled payload dict

## JSON Data Contract

```json
{
  "model_type": "decision_tree",
  "task_type": "classification",
  "classes": [0, 1, 2],
  "feature_names": ["petal length (cm)", "petal width (cm)"],
  "dataset_name": "iris",
  "nodes": [...],
  "path": [...]
}
```

`task_type` is `"regression"` for `DecisionTreeRegressor`. Regression payloads omit `classes` and replace per-class `counts`/`prediction` in leaf nodes with a `value` field (mean target value).

See [[Extractor - Shared]] for the full node and path shapes.

## Tests

File: `tests/test_decision_tree_extractor.py`  
Fixtures: iris dataset (classification), diabetes dataset (regression). No mocks.

| Test | Checks |
|------|--------|
| `test_extract_tree_node_count` | `len(nodes)` == `model.tree_.node_count` |
| `test_all_leaves_have_counts` | every classification leaf has `counts` |
| `test_trace_path_ends_at_leaf` | last path element has `leaf: true` |
| `test_trace_path_went_left_consistent` | `went_left` matches `value <= threshold` |
| `test_feature_name_fallback` | numpy array → `["x0", "x1", ...]` |
| `test_feature_names_from_kwarg` | explicit kwarg overrides everything |
| `test_feature_names_from_dataframe` | DataFrame columns used when no kwarg |
| `test_serialize_structure` | full payload shape correct |
| `test_serialize_path_none_when_no_query` | `path` is `null` without query |
| `test_decision_tree_regressor_*` | regression leaf value, path shape |
| `test_decision_tree_feature_importances_*` | importances present, sorted, match model |

## Related Notes

- [[Extractor - Shared]] — the actual tree-walking and serialisation logic
- [[Decision Tree]] — DT theory and sklearn flat arrays
- [[Server]] — consumes the output of `serialize()`
