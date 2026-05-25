---
tags: [mlviz, python, v1, architecture]
status: in-progress
---

# Architecture

## Project Structure

```
mlviz/
├── __init__.py               ← visualize() entry point, detects model type
├── extractors/
│   ├── __init__.py
│   └── decision_tree.py      ← reads model.tree_, serialises to JSON
├── server/
│   ├── __init__.py
│   └── app.py                ← FastAPI, serves JSON to frontend
└── frontend/                 ← React + Vite visualiser (bundled into package)

tests/
└── test_decision_tree_extractor.py
```

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Model introspection | scikit-learn | reads `model.tree_`, `model.classes_`, etc. |
| Local server | FastAPI + uvicorn | starts in milliseconds, async, serves JSON |
| Frontend | React + Vite | bundled into package — no Node.js at runtime |
| Browser launch | `webbrowser.open()` | stdlib, cross-platform |

## Call Sequence

```
User: visualize(model, X_train, y_train, query=X_test[0])
        │
        ▼
mlviz/__init__.py
  → validate model is fitted
  → check model type (V1: DecisionTreeClassifier only)
  → call extractors/decision_tree.serialize()
        │
        ▼
extractors/decision_tree.py
  → resolve feature names
  → _extract_tree()  →  nodes list
  → _trace_path()    →  path list
  → return JSON dict
        │
        ▼
server/app.py (FastAPI)
  → cache payload at startup
  → GET /api/tree  →  serve JSON
  → static files   →  serve React bundle
        │
        ▼
frontend/ (React)
  → fetch /api/tree
  → render tree diagram
  → highlight decision path
        │
        ▼
webbrowser.open("http://localhost:{port}")
```

## Key Design Decisions

**Flat node list, not nested JSON** — the extractor returns `nodes` as a flat list (not a recursive tree). React builds a `Map<node_id, node>` in one pass, making path-highlight lookup O(1). See [[Extractor - DT]] for the full data contract.

**Frontend bundled inside Python package** — the React build output lives in `mlviz/frontend/dist/` and ships with the package. Users run `pip install mlviz` and get everything — no Node.js required at runtime.

**Per-model extractors** — each model type lives in its own file under `extractors/`. The server stays model-agnostic. V1 has `decision_tree.py` only.

## Related Notes

- [[Extractor - DT]] — how decision_tree.py reads sklearn internals
- [[Server]] — FastAPI routes and JSON serving
- [[Frontend]] — React rendering plan
