---
tags: [mlviz, python, v1, frontend, react]
status: not-started
---

# Frontend (`frontend/`)

## What It Does

A React + Vite app that fetches the model JSON from the FastAPI server and renders an interactive decision tree diagram in the browser. The user never sees Node.js or npm — the built output ships bundled inside the Python package.

## Tech Stack

| Tool | Role |
|------|------|
| React | component rendering |
| Vite | development server + production bundler |
| CSS / SVG | tree layout and path highlighting |

## What It Renders

### Full Tree Diagram

Every node shown with:
- Feature name and threshold (e.g. `petal_length ≤ 2.45`)
- Sample count at that node
- Gini impurity
- Left = condition true, right = condition false

Leaf nodes shown with:
- Class distribution (counts per class)
- Predicted class name (resolved from `classes` array via the prediction index)

### Decision Path Highlighting

When a query point is present (`path` is not null), every node on the path from root to leaf is highlighted. At each highlighted node:
- The question asked
- The query point's value for that feature
- Which direction it went (left/right) and why

### Data Fetching

On mount, React fetches `GET /api/tree` from the local FastAPI server. The `nodes` array is converted to a `Map<node_id, node>` for O(1) lookup. The `path` array is converted to a `Set<node_id>` — checking whether a node is on the path is a single `has()` call.

## Bundle Approach

Built with `npm run build` inside `mlviz/frontend/`. Output goes to `mlviz/frontend/dist/`. This `dist/` folder is committed and ships inside the Python package. FastAPI serves it as static files.

No Node.js is needed at runtime — only at build time by the mlviz developer.

## Status

Not started — implemented after the server is working. Frontend development is session 3+.

## Related Notes

- [[Server]] — provides the JSON this frontend fetches
