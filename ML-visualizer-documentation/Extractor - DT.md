---
tags: [mlviz, python, v1, extractor]
status: complete
---

# Extractor — Decision Tree (`decision_tree.py`)

## What It Does

Reads a fitted `sklearn.tree.DecisionTreeClassifier` and converts its internal flat arrays into a JSON-serialisable dict. This dict is what the server sends to the React frontend.

## Three Functions

### `_extract_tree(tree, feature_names) → list[dict]`

Internal. Walks `range(tree.node_count)` and emits one dict per node. Split nodes get feature name, threshold, and child IDs. Leaf nodes get class counts and the predicted class index.

### `_trace_path(tree, feature_names, X) → list[dict]`

Internal. Follows a single query point (`X`, a 1D array) root-to-leaf, recording at each split: what was asked, what the query's value was, which direction it went.

### `serialize(model, X_train, y_train, query=None, feature_names=None) → dict`

Public. The only function the server calls. Resolves feature names, calls the two internal functions, and returns the complete payload.

## Feature Name Resolution (Priority Order)

1. `feature_names` kwarg — explicit override, always wins
2. `model.feature_names_in_` — set by sklearn when model was fitted on a DataFrame
3. `X_train.columns` — if X_train is still a pandas DataFrame at call time
4. Auto-generate `["x0", "x1", ...]` — fallback for plain numpy arrays

## JSON Data Contract

`serialize()` returns this shape:

```json
{
  "model_type": "decision_tree",
  "classes": [0, 1, 2],
  "feature_names": ["petal length (cm)", "petal width (cm)", "sepal length (cm)", "sepal width (cm)"],
  "nodes": [
    {
      "node_id": 0,
      "feature": "petal length (cm)",
      "threshold": 2.45,
      "n_samples": 150,
      "gini": 0.667,
      "left_child": 1,
      "right_child": 2,
      "leaf": false
    },
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
  ],
  "path": [
    {
      "node_id": 0,
      "feature": "petal length (cm)",
      "threshold": 2.45,
      "value": 1.4,
      "went_left": true
    },
    {
      "node_id": 1,
      "leaf": true,
      "counts": [50, 0, 0],
      "prediction": 0
    }
  ]
}
```

**Rules:**
- `nodes` is flat (not nested) — React looks up nodes by ID in O(1)
- `path` is `null` when no query point is passed
- `prediction` is a class index — resolved to a class name using `classes` array
- Thresholds and Gini values rounded to 3 decimal places
- `counts` values are integers (sklearn stores them as floats internally)

## Tests

File: `tests/test_decision_tree_extractor.py`
Fixture: real `DecisionTreeClassifier` on iris dataset — no mocks.

| Test | Checks |
|------|--------|
| `test_extract_tree_node_count` | `len(nodes)` == `model.tree_.node_count` |
| `test_all_leaves_have_counts` | every leaf node has `counts` field |
| `test_trace_path_ends_at_leaf` | last path element has `leaf: true` |
| `test_trace_path_went_left_consistent` | `went_left` matches `value <= threshold` |
| `test_feature_name_fallback` | numpy array → `["x0", "x1", "x2", "x3"]` |
| `test_feature_names_from_kwarg` | explicit kwarg overrides everything |
| `test_feature_names_from_dataframe` | DataFrame columns used when no kwarg/model names |
| `test_serialize_structure` | full payload shape correct with query |
| `test_serialize_path_none_when_no_query` | `path` is `null` without query |

## Related Notes

- [[Architecture]] — where this file sits in the project structure
- [[Decision Tree]] — DT theory and sklearn flat arrays
- [[Server]] — consumes the output of `serialize()`
