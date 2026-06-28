---
tags: [mlviz, python, overview]
status: current
version: v0.2.3
---

# mlviz — ML Model Visualizer

A Python library that takes a real trained sklearn model + real data and launches an interactive browser visualisation showing exactly how the model makes decisions.

## Usage

```python
from mlviz import visualize

model.fit(X_train, y_train)
visualize(model, X_train, y_train, query=X_test[0], feature_names=iris.feature_names, dataset_name="iris")
```

`visualize()` starts a local FastAPI server, serialises the model internals to JSON, and opens a browser tab automatically.

## Supported Models

| Model | Endpoint | Task types |
|-------|----------|-----------|
| `DecisionTreeClassifier` / `DecisionTreeRegressor` | `/api/tree` | classification, regression |
| `RandomForestClassifier` / `RandomForestRegressor` | `/api/forest` | classification, regression |
| `KNeighborsClassifier` / `KNeighborsRegressor` | `/api/knn` | classification, regression |
| `SVC` / `SVR` | `/api/svm` | classification, regression |

Multi-output regression is not supported.

## API

```python
mlviz.visualize(
    model,                   # fitted sklearn model
    X_train,                 # training features (numpy array or DataFrame)
    y_train,                 # training labels/targets
    query=None,              # optional: single sample to explain (1D array)
    feature_names=None,      # optional: list of feature name strings
    dataset_name=None,       # optional: label shown in header pill
)
```

## Notes

- [[Architecture]] — project structure, tech stack, call sequence, dispatch logic
- [[Server]] — FastAPI routes, manifest endpoint, payload caching
- [[Frontend]] — React views, design system, canvas rendering
- [[Decision Tree]] — DT theory: Gini impurity, nodes vs leaves, sklearn internals
- [[Extractor - Shared]] — shared sklearn tree helpers used by DT and RF
- [[Extractor - DT]] — `decision_tree.py` — thin wrapper over shared helpers
- [[Extractor - RF]] — `random_forest.py` — ensemble serialiser, OOB, vote/prediction distribution
- [[Extractor - KNN]] — `knn.py` — neighbour summary, vote/weight details, PCA projection
- [[Extractor - SVM]] — `svm.py` — support vector internals, decision scores, margin projection

## Build Status

| Module | Status |
|--------|--------|
| `extractors/shared.py` | ✅ complete — shared tree helpers (DT + RF) |
| `extractors/decision_tree.py` | ✅ complete |
| `extractors/random_forest.py` | ✅ complete — classification + regression |
| `extractors/knn.py` | ✅ complete — classification + regression |
| `extractors/svm.py` | ✅ complete — SVC + SVR |
| `extractors/projection.py` | ✅ complete — PCA 2D projection for KNN + SVM |
| `server/app.py` | ✅ complete — `/api/model` manifest + 4 model endpoints |
| `mlviz/__init__.py` | ✅ complete — dispatch by model type |
| `frontend/` — DecisionTreeView | ✅ complete |
| `frontend/` — RandomForestView | ✅ complete — tree browser, ensemble sidebar, OOB card |
| `frontend/` — KnnView | ✅ complete — projection canvas, neighbour table, charts |
| `frontend/` — SvmView | ✅ complete — projection canvas, support vector table, charts |
