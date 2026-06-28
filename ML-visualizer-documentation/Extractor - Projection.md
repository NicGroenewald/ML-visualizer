---
tags: [mlviz, extractor, projection, pca]
status: current
version: v0.2.3
---

# Extractor — Projection (`extractors/projection.py`)

## What It Does

Reduces training data to 2D via PCA and computes an 80×80 decision-region grid for rendering in the canvas. Used by both the KNN and SVM extractors. The 2D positions are for display only — neighbour distances and SVM decisions are always computed in the original feature space.

## Functions

### `base_projection(model, X_train) → (pca, X_2d)`

Fits a 2-component PCA on `X_train` and transforms it. Returns the fitted `PCA` object and the 2D array.

### `_classification_regions(model, pca, X_2d, n_classes) → list[dict]`

Builds an 80×80 grid over the 2D PCA space, transforms each cell back to the original space, queries `model.predict()` on the grid, and returns a list of region dicts. Each dict has `wx`, `wy` (world coordinates), `cell_w`, `cell_h`, and `class_index`.

### `_margin_regions(model, pca, X_2d) → list[dict]`

Binary SVC only. Same grid as `_classification_regions` but uses `model.decision_function()` to identify cells near the decision boundary and within the margin band. Returns dicts tagged with `region_type: "boundary" | "margin"`.

### `_grid_from_projection(X_2d, resolution=80) → (xx, yy, grid_2d, cell_w, cell_h)`

Shared grid builder. Pads the data bounding box by 10% and creates a `resolution × resolution` meshgrid.

### `json_value(v)` / `_as_2d_array(q)`

Serialisation helpers — convert numpy scalars to Python types, reshape 1D arrays.

### `knn_projection(model, X_train, y_train, query, task_type, neighbors) → dict`

Builds the full KNN projection payload:
- `points` — all training points in 2D (with class/target labels and `training_index`)
- `query_2d` — query point projected into 2D
- `regions` — decision regions (classification only)

### `svm_projection(model, X_train, y_train, query, task_type) → dict`

Builds the full SVM projection payload:
- `points` — all training points in 2D, tagged `is_support_vector`
- `query_2d` — query point projected into 2D (if supplied)
- `regions` — classification regions (classification only)
- `margin_regions` — boundary/margin cells (binary SVC only)

### `svr_fit_line(model, pca, X_2d) → list[dict]`

SVR only. Samples points along the first PCA axis and computes the model's 1D regression prediction for each — used to overlay a fit line on the SVR projection canvas.

## Projection Payload Shape

```json
{
  "points": [
    {
      "training_index": 0,
      "wx": -1.23,
      "wy": 0.45,
      "class_index": 0,
      "class_label": "setosa",
      "target_value": null,
      "is_support_vector": false
    }
  ],
  "query_2d": {"wx": -0.8, "wy": 0.2} | null,
  "regions": [
    {"wx": -2.1, "wy": 0.3, "cell_w": 0.12, "cell_h": 0.09, "class_index": 1}
  ] | null,
  "margin_regions": [
    {"wx": 0.0, "wy": 0.0, "cell_w": 0.12, "cell_h": 0.09, "region_type": "boundary"}
  ] | null
}
```

- `class_index` / `class_label` are `null` for regression; `target_value` is `null` for classification.
- `is_support_vector` is only meaningful in SVM payloads (always `false` for KNN).
- `regions` is `null` for regression models.
- `margin_regions` is only present for binary `SVC`.

## Canvas Rendering

`NonTreeVisuals.jsx` renders projection payloads:
- `KnnProjectionPlot` uses `points`, `query_2d`, `regions` (when boundary glow is on), and the neighbour indices from `data.neighbors`.
- `SvmProjectionPlot` uses `points`, `query_2d`, `regions` (glow), and `margin_regions` (always on).

The 80×80 decision region grid maps directly to the BFS distance transform used for the soft boundary glow effect.

## Related Notes

- [[Extractor - KNN]] — calls `knn_projection()`
- [[Extractor - SVM]] — calls `svm_projection()`
- [[Frontend]] — `NonTreeVisuals.jsx` renders the projection payloads
