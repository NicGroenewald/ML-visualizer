---
tags: [mlviz, python, v1, frontend, react]
status: complete
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

Complete — Session 3 (tree + path), Session 4 (dark mode), Session 5 (zoom + label truncation).

## What was built

`mlviz/frontend/src/MLVizTree.jsx` — presentational component:
- SVG tree rendered from flat node list using leaf-counting layout algorithm
- Apple HIG design tokens — `LIGHT` and `DARK` token objects, resolved per mode at render time
- Path highlighting via `Set<node_id>` — blue fill + border on active nodes, blue strokes on edges
- Floating tooltip on hover showing gini, samples, class counts per node
- Path bar at bottom (only when `data.path` is not null)
- **Dark mode** — `LIGHT`/`DARK` token objects, toggled via button, persisted to `localStorage`, OS preference detected on first load
- **Zoom controls** — `+` / `−` / `fit` buttons in header; SVG uses `viewBox` so all coordinates stay unscaled; auto-fits to container width on mount via `useEffect`
- **Label truncation** — `clampLabel(str, max=19)` applied to split node text; full label always visible in tooltip

`mlviz/frontend/src/App.jsx` — data owner:
- Fetches `GET /api/tree` on mount
- Three states: loading / error / ready (all three respect dark mode)
- Owns `dark` state and `toggleDark` callback, passed down to `MLVizTree`
- `initDark()` reads `localStorage` on mount, falls back to `prefers-color-scheme`

Built with `npm run build` → `dist/` served as static files by FastAPI. 14 tests passing.

## Design Token System

Two token objects replace the original single `T` constant:

```js
const LIGHT = { bg, surface, text, ... , classFill: [...] }
const DARK  = { bg, surface, text, ... , classFill: [...] }
```

Inside the component: `const T = dark ? DARK : LIGHT`. All sub-components (`Pill`, `TRow`, `Tooltip`) receive `T` as a prop so they react to mode changes.

Class accent colours (`CLASS_COLORS`) are always the same saturated values — only the fill backgrounds differ between light and dark.

## Related Notes

- [[Server]] — provides the JSON this frontend fetches
- [[Architecture]] — how the frontend fits into the full call sequence
