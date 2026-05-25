---
tags: [mlviz, python, v1, server, fastapi]
status: not-started
---

# Server (`server/app.py`)

## What It Does

A local FastAPI server that starts when `visualize()` is called. It has one job: serve the serialised model JSON to the React frontend. It shuts down when the user closes the browser tab or stops the Python process.

## Why FastAPI

FastAPI starts in milliseconds (via uvicorn), handles async requests, and lets us write the whole server in ~30 lines of Python. It also auto-generates OpenAPI docs at `/docs` for free — useful during development.

## Planned Route

```
GET /api/tree
```

Returns the full JSON payload from `extractors/decision_tree.serialize()`. The payload is computed **once at startup** (passed into the app at construction) and cached in memory. No re-computation on each request.

```python
# Rough shape
@app.get("/api/tree")
def get_tree():
    return payload  # pre-computed at startup, injected at construction
```

## Static Files

The bundled React frontend (`mlviz/frontend/dist/`) is served as static files on `/`. When the browser opens `http://localhost:{port}`, FastAPI serves `index.html`, which loads the JS bundle, which fetches `/api/tree`.

## Port

A random available port is selected at startup using `socket.bind(('', 0))`. The port is passed to `webbrowser.open()`. This avoids conflicts if the user runs `visualize()` multiple times.

## Status

Not started — implemented in the session after the extractor is complete and tested.

## Related Notes

- [[Extractor - DT]] — produces the JSON payload this server serves
- [[Frontend]] — the React app that fetches from this server
