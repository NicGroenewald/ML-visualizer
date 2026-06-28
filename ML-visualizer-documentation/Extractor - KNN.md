---
tags: [mlviz, extractor, knn]
status: current
version: v0.2.3
---

# Extractor — KNN (`extractors/knn.py`)

## What It Does

Serialises a fitted `KNeighborsClassifier` or `KNeighborsRegressor` into the `/api/knn` payload. When a query is supplied it identifies the k nearest neighbours, computes distances and weights, and assembles vote/prediction details. It also calls `projection.knn_projection()` to produce the 2D PCA scatter data rendered by `KnnProjectionPlot`. For regression queries it adds a query-local `neighbor_projection` that embeds only the query and its neighbours from original feature-space distances.

## Public API

```python
serialize(model, X_train, y_train, query=None, feature_names=None, dataset_name=None) → dict
```

## Internal Functions

- `_effective_weights(model, distances)` — computes per-neighbour weights. For `weights="uniform"` all weights are `1/k`. For `weights="distance"` uses inverse-distance weighting (with a guard for zero distance).
- `_serialize_neighbors(model, X_train, y_train, query, feature_names, task_type)` → list of neighbour dicts
- `_serialize_vote_distribution(neighbors, classes)` → per-class vote counts and fractions (classification)
- `_serialize_prediction(model, query, task_type)` → predicted class or regression value
- `_serialize_neighbor_projection(model, X_train, query, neighbors, prediction)` → query-local metric MDS payload for KNN regression

## JSON Data Contract

```json
{
  "model_type": "knn",
  "task_type": "classification",
  "classes": [0, 1, 2],
  "feature_names": ["sepal length (cm)", ...],
  "dataset_name": "iris",
  "summary": {
    "n_neighbors": 5,
    "n_features": 4,
    "n_samples": 150,
    "metric": "minkowski",
    "weights": "uniform",
    "algorithm": "auto",
    "p": 2
  },
  "neighbors": [
    {
      "rank": 1,
      "training_index": 42,
      "distance": 0.141,
      "effective_weight": 0.2,
      "class_label": "setosa",
      "class_index": 0,
      "target_value": null
    }
  ] | null,
  "vote_distribution": {
    "class_votes": [
      {"class_label": "setosa", "class_index": 0, "votes": 3, "weight": 0.6, "fraction": 0.6}
    ]
  } | null,
  "prediction": {
    "class_label": "setosa",
    "class_index": 0,
    "value": null
  } | null,
  "projection": { ... } | null,
  "neighbor_projection": {
    "method": "metric_mds",
    "distance_source": "original_feature_space",
    "prediction_value": 152.0,
    "points": [
      {"point_type": "query", "wx": 0.0, "wy": 0.0},
      {"point_type": "neighbor", "rank": 1, "training_index": 42, "distance": 0.141, "target_value": 151.0}
    ],
    "links": [
      {"source": "query", "target": 42, "distance": 0.141, "effective_weight": 7.09}
    ]
  } | null
}
```

- `neighbors` is `null` when no query is supplied.
- `vote_distribution` is `null` for regression or when no query.
- `prediction.value` is `null` for classification; `prediction.class_label`/`class_index` are `null` for regression.
- `target_value` in neighbours is `null` for classification; `class_label`/`class_index` are `null` for regression.
- `neighbor_projection` is only populated for KNN regression with a query. The query point is centered at `(0, 0)` and neighbour positions come from metric MDS over original feature-space distances.

## Projection

`projection` is always included (query-independent). Shape defined in [[Extractor - Projection]].

For regression: `KnnRegressionExplanation` uses `neighbor_projection` as the primary view so the query-neighbour picture reflects true KNN distances. The PCA `KnnProjectionPlot` remains as a secondary overview and keeps its caveat because full-dataset PCA positions do not preserve the neighbour metric.

## Tests

File: `tests/test_knn_extractor.py`

| Test | Checks |
|------|--------|
| `test_knn_classification_*` | task_type, classes, summary, neighbours, vote distribution, projection shape |
| `test_knn_regression_*` | regression fields, target_value, numeric prediction, query-local neighbour projection |
| `test_knn_no_query_*` | neighbours/vote_distribution/prediction all null |

## Related Notes

- [[Extractor - Projection]] — PCA projection logic
- [[Frontend]] — `KnnView` and `KnnProjectionPlot` render this payload
- [[Architecture]] — KNN dispatch chain
