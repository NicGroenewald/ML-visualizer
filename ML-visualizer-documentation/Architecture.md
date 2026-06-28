---
tags: [mlviz, architecture]
status: current
version: v0.2.3
---

# Architecture

## Project Structure

```
mlviz/
├── __init__.py                  ← visualize() entry point, model dispatch
├── extractors/
│   ├── shared.py                ← shared sklearn tree helpers (DT + RF)
│   ├── decision_tree.py         ← thin wrapper over shared helpers → /api/tree
│   ├── random_forest.py         ← ensemble serialiser → /api/forest
│   ├── knn.py                   ← nearest-neighbour serialiser → /api/knn
│   ├── projection.py            ← PCA 2D projection (KNN + SVM)
│   └── svm.py                   ← support-vector serialiser → /api/svm
├── server/
│   └── app.py                   ← FastAPI: /api/model manifest + model endpoints
└── frontend/
    ├── src/
    │   ├── App.jsx              ← manifest fetch, model_type routing, dark mode
    │   ├── theme.js             ← design tokens (DARK/LIGHT), class colours
    │   ├── MLVizTree.jsx        ← model-agnostic single-tree canvas renderer
    │   ├── views/
    │   │   ├── DecisionTreeView.jsx
    │   │   ├── RandomForestView.jsx
    │   │   ├── KnnView.jsx
    │   │   └── SvmView.jsx
    │   └── components/
    │       ├── AppHeader.jsx
    │       └── forest/          ← ForestHeader, ForestTreeBrowser, Panel,
    │                               FeatureImportancePanel, VoteDistributionPanel,
    │                               PredictionDistributionPanel, OobMetricCard
    └── dist/                    ← built bundle (committed, ships with package)

tests/
├── test_decision_tree_extractor.py
├── test_random_forest_extractor.py
├── test_knn_extractor.py
├── test_svm_extractor.py
├── test_server.py
└── test_visualize.py
```

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Model introspection | scikit-learn | reads `model.tree_`, `model.classes_`, `model.support_vectors_`, etc. |
| Dimensionality reduction | sklearn `PCA` | 2D canvas projection for KNN + SVM |
| Local server | FastAPI + uvicorn | starts in milliseconds, async, serves JSON |
| Frontend | React + Vite | bundled into package — no Node.js at runtime |
| Animations | Framer Motion | HTML layer only — SVG uses CSS transitions |
| Browser launch | `webbrowser.open()` | stdlib, cross-platform |

## Call Sequence

```
User: visualize(model, X_train, y_train, query=X_test[0])
        │
        ▼
mlviz/__init__.py
  → validate model is fitted
  → dispatch by model type:
      DecisionTree*  → extractors/decision_tree.serialize()
      RandomForest*  → extractors/random_forest.serialize()
      KNeighbors*    → extractors/knn.serialize()
      SVC/SVR        → extractors/svm.serialize()
        │
        ▼
extractor (model-specific)
  → resolve feature names (shared.resolve_feature_names)
  → normalize query vector (shared.normalize_query_vector)
  → build payload dict (nodes, path, projection, etc.)
  → return {model_type, endpoint, payload}
        │
        ▼
server/app.py (FastAPI)
  → GET /api/model   →  {model_type, endpoint}  (manifest)
  → GET /api/tree    →  DT payload
  → GET /api/forest  →  RF payload
  → GET /api/knn     →  KNN payload
  → GET /api/svm     →  SVM payload
  → static files     →  serve React bundle
        │
        ▼
frontend/ (React)
  → App.jsx fetches /api/model
  → routes to DecisionTreeView / RandomForestView / KnnView / SvmView
  → view fetches its own endpoint (/api/tree, /api/forest, etc.)
  → renders model-appropriate visualisation
        │
        ▼
webbrowser.open("http://localhost:{port}")
```

## Key Design Decisions

**`/api/model` manifest** — `App.jsx` fetches this first to learn the `model_type` and correct data endpoint. This lets each view fetch from its own endpoint and keeps `App.jsx` model-agnostic.

**Flat node list, not nested JSON** — tree extractors return `nodes` as a flat list. React builds a `Map<node_id, node>` in one pass, making path-highlight lookup O(1).

**`task_type` discriminant** — every payload includes `task_type: "classification" | "regression"`. Views use this to conditionally render class distributions vs numeric predictions without needing separate model-type checks.

**Shared tree helpers (`extractors/shared.py`)** — `serialize_sklearn_tree()`, `trace_sklearn_path()`, `resolve_feature_names()`, `normalize_query_vector()`, `model_task_type()` are called by both `decision_tree.py` and `random_forest.py`. Neither duplicates this logic.

**`MLVizTree.jsx` is model-agnostic** — the canvas tree renderer accepts `{nodes, path, classes, dark, taskType}` as props. It knows nothing about DT vs RF. `DecisionTreeView` and `RandomForestView` both use it; non-tree models (KNN, SVM) use `NonTreeVisuals.jsx` instead.

**Projection views** — `extractors/projection.py` reduces training data to 2D via PCA and computes an 80×80 decision-region grid for KNN classification and SVM views. KNN regression uses a separate query-local metric MDS projection over original feature-space distances, showing the query, selected neighbours, and all training rows in one primary zoomable graph with weighted-average details in the sidebar.

**Frontend bundled inside Python package** — `mlviz/frontend/dist/` is committed and ships with the package. Users run `pip install mlviz` and get everything.

**Daemon vs non-daemon thread** — `visualize()` checks `_is_interactive()` to detect Jupyter. In Jupyter, uvicorn runs in a daemon thread (non-blocking). In a script, it runs in a non-daemon thread so the process stays alive after `visualize()` returns.

## Related Notes

- [[Server]] — FastAPI routes, manifest, payload caching
- [[Frontend]] — React views, design system, canvas rendering
- [[Extractor - Shared]] — shared tree helpers
- [[Extractor - DT]], [[Extractor - RF]], [[Extractor - KNN]], [[Extractor - SVM]]
