---
tags: [mlviz, python, v1]
status: complete
---

# mlviz — ML Model Visualizer

A Python library that takes a real trained sklearn model + real data and launches an interactive browser visualisation showing exactly how the model thinks.

## Usage

```python
from mlviz import visualize

model.fit(X_train, y_train)
visualize(model, X_train, y_train, query=X_test[0])
```

`visualize()` starts a local FastAPI server, serialises the model internals to JSON, and opens a browser tab automatically. No separate server process to manage.

## V1 Scope

Decision Tree only. KNN, Random Forest, and SVM are planned for later versions.

## Notes

- [[Architecture]] — project structure, tech stack, and call sequence
- [[Decision Tree]] — DT theory: Gini impurity, nodes vs leaves, sklearn internals
- [[Extractor - DT]] — `decision_tree.py` progress and JSON data contract
- [[Server]] — FastAPI routes and API design
- [[Frontend]] — React component structure and rendering plan

## Build Status

| Module | Status |
|--------|--------|
| Project scaffold | ✅ complete |
| `extractors/decision_tree.py` | ✅ complete — 9 tests passing |
| `server/app.py` | ✅ complete — 2 tests passing |
| `mlviz/__init__.py` (visualize) | ✅ complete — 3 tests passing |
| `frontend/` | ✅ complete — dark mode, zoom controls, label truncation |
