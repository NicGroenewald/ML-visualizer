---
tags: [mlviz, server, fastapi]
status: current
version: v0.2.3
---

# Server (`server/app.py`)

## What It Does

A local FastAPI server that starts when `visualize()` is called. It serves the serialised model JSON to the React frontend, plus a manifest that tells the frontend which model type is loaded.

## Routes

```
GET /api/model          →  {model_type, endpoint}
GET /api/tree           →  Decision Tree payload
GET /api/forest         →  Random Forest payload
GET /api/knn            →  KNN payload
GET /api/svm            →  SVM payload
GET /                   →  React bundle (index.html + JS)
```

### `/api/model` — manifest

```json
{
  "model_type": "random_forest",
  "endpoint": "/api/forest"
}
```

`App.jsx` fetches this first on mount. It uses `model_type` to pick the right view component and `endpoint` to know where to fetch the model payload. This keeps `App.jsx` model-agnostic — adding a new model type only requires adding a new endpoint; `App.jsx` routes by `model_type` string.

### Model endpoints

Each endpoint returns a pre-computed payload injected at construction time. No re-computation on request.

```python
create_app(payload, model_type, endpoint)
```

- `payload` — the dict from the extractor's `serialize()` call
- `model_type` — string key used in the manifest (`"decision_tree"`, `"random_forest"`, `"knn"`, `"svm"`)
- `endpoint` — the route string (`"/api/tree"`, `"/api/forest"`, `"/api/knn"`, `"/api/svm"`)

`create_app()` wires one dynamic endpoint at `endpoint` that returns the payload. Each call produces an isolated `FastAPI` app instance — payloads don't bleed between calls.

## Port

A random available port is selected at startup using `socket.bind(('', 0))`. The port is passed to `webbrowser.open()`. Multiple `visualize()` calls run on independent ports.

## Static Files

`mlviz/frontend/dist/` is mounted at `/` as static files. FastAPI serves `index.html` for all non-API paths. The JS bundle fetches `/api/model` then the model endpoint.

## Thread Model

`visualize()` starts uvicorn in a thread:
- **Jupyter / interactive** (`_is_interactive()` returns True) → daemon thread. Process doesn't block.
- **Script mode** → non-daemon thread. Process stays alive after `visualize()` returns. Ctrl+C sets `server.should_exit = True` for clean shutdown.

`_wait_for_server()` polls the port with a short backoff until the server accepts connections before `webbrowser.open()` is called.

## Related Notes

- [[Architecture]] — how the server fits into the full call sequence
- [[Frontend]] — the React app that fetches from this server
