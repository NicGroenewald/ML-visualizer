---
tags: [mlviz, extractor, svm]
status: current
version: v0.2.3
---

# Extractor — SVM (`extractors/svm.py`)

## What It Does

Serialises a fitted `SVC` or `SVR` into the `/api/svm` payload. It exposes support vector internals (indices, feature values, class labels) and, when a query is supplied, the decision function value and predicted class/value. It also calls `projection.svm_projection()` to produce the 2D PCA canvas data rendered by `SvmProjectionPlot`.

Unlike KNN, the support vectors are **not** treated as nearest neighbours — they are the structural components of the SVM itself. The extractor does not expose raw neighbour distances.

## Public API

```python
serialize(model, X_train, y_train, query=None, feature_names=None, dataset_name=None) → dict
```

## Internal Functions

- `_serialize_support_vectors(model, X_train, y_train, feature_names, task_type)` → list of SV dicts
- `_serialize_prediction(model, query, task_type)` → predicted class/value + decision function

## JSON Data Contract

```json
{
  "model_type": "svm",
  "task_type": "classification",
  "classes": [0, 1],
  "feature_names": ["mean radius", ...],
  "dataset_name": "breast_cancer",
  "summary": {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "degree": 3,
    "coef0": 0.0,
    "epsilon": null,
    "n_features": 30,
    "support_vector_count": 120
  },
  "support_vectors": [
    {
      "support_vector_index": 0,
      "training_index": 14,
      "class_label": "malignant",
      "class_index": 0,
      "target_value": null,
      "feature_values": [
        {"feature_name": "mean radius", "value": 17.99}
      ]
    }
  ],
  "prediction": {
    "class_label": "malignant",
    "class_index": 0,
    "value": null,
    "decision_function": [-1.23]
  } | null,
  "projection": { ... } | null
}
```

- `target_value` in SVs is `null` for classification; `class_label`/`class_index` are `null` for regression.
- `prediction.decision_function` is a list for multi-class SVC (`decision_function_shape="ovr"`), a scalar for binary SVC and SVR.
- `epsilon` in summary is `null` for `SVC`; `coef0` is `null` for `SVR`.

## Projection

`projection` is always included. For `SVC`, `projection.margin_regions` provides the decision boundary and margin band for the canvas. These are always rendered in `SvmProjectionPlot` regardless of the boundary glow toggle. See [[Extractor - Projection]].

## Tests

File: `tests/test_svm_extractor.py`

| Test | Checks |
|------|--------|
| `test_svm_classification_*` | task_type, classes, support vectors, decision function, projection |
| `test_svm_regression_*` | SVR fields, numeric prediction, no decision_function |
| `test_svm_binary_margin_regions` | margin_regions present for binary SVC |

## Related Notes

- [[Extractor - Projection]] — PCA projection + margin region logic
- [[Frontend]] — `SvmView` and `SvmProjectionPlot` render this payload
- [[Architecture]] — SVM dispatch chain
