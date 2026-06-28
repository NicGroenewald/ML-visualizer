---
tags: [mlviz, frontend, react]
status: current
version: v0.2.3
---

# Frontend (`frontend/`)

## What It Does

A React + Vite app that fetches model JSON from the FastAPI server and renders an interactive, model-appropriate visualisation. The built output ships bundled inside the Python package — no Node.js required at runtime.

## Tech Stack

| Tool | Role |
|------|------|
| React | component rendering |
| Vite | dev server + production bundler |
| Framer Motion | HTML-layer animations only |
| SVG / HTML Canvas | tree diagrams and scatter plots |

## Routing

`App.jsx` fetches `/api/model` on mount to get `{model_type, endpoint}`. It then routes to the appropriate view:

```
model_type === "decision_tree"  →  <DecisionTreeView>
model_type === "random_forest"  →  <RandomForestView>
model_type === "knn"            →  <KnnView>
model_type === "svm"            →  <SvmView>
```

Each view fetches its own endpoint (`/api/tree`, `/api/forest`, etc.) and owns its full layout — header, main canvas, sidebar.

## Design System (`theme.js`)

Two token objects, resolved at render time via `getTheme(dark)`:

```js
const DARK  = { bg: "#09090B", canvas: "#0C0C0E", accent: "#7B86E2", ... }
const LIGHT = { bg: "#FAFAFA", canvas: "#FCFCFC", accent: "#5E6AD2", ... }
```

Key tokens: `bg`, `canvas`, `gridDot`, `surface`, `border`, `text`, `textSecondary`, `textTertiary`, `accent`, `accentFill`, `separator`, `green`, `red`.

Class accent colours are separate constants — `CLASS_COLORS` (dark) and `CLASS_COLORS_LIGHT` — 6 classes. Resolved via `getClassColors(dark)`.

All colour changes happen in `theme.js`. Nothing else holds raw hex values.

## Views

### `DecisionTreeView`

- `AppHeader` with model/task/dataset pills and class legend
- `MLVizTree` canvas (single tree)
- Feature importance sidebar panel

### `RandomForestView`

- `ForestHeader` (model pills, OOB metric card, feature importance, class legend)
- `ForestTreeBrowser` — horizontal strip of trees; click to inspect individual tree
- `MLVizTree` canvas — renders whichever tree is selected
- Ensemble sidebar: vote distribution (classification) or prediction distribution (regression)

### `KnnView`

- `AppHeader` with model/task/dataset pills
- Classification: `KnnProjectionPlot` canvas (PCA scatter with optional boundary glow)
- Regression: `KnnRegressionExplanation` primary view with one query-local true-distance neighbour map, all training rows, shared zoom/pan controls, and weighted-average breakdown
- Sidebar: `VotePanel` (classification) / prediction value (regression), weighted-average panel (regression), model summary, `KnnVoteBarChart` / `KnnRegressionTargetStrip`
- Collapsible neighbour table at bottom

### `SvmView`

- `AppHeader` with model/task/dataset pills
- `SvmProjectionPlot` canvas (PCA scatter with SVM margin lines and optional boundary glow)
- Sidebar: `PredictionPanel`, model summary, `SvmDecisionScoreChart` / `SvmRegressionValueStrip`
- Collapsible support vector table at bottom

## `MLVizTree.jsx` — Tree Canvas Renderer

Model-agnostic. Props: `{nodes, path, classes, dark, taskType}`.

- `computeLayout()` — two-pass SVG layout: leaf-count x-position pass then depth y-position pass
- Node rendering: split nodes show feature/threshold/gini/samples; leaf nodes show class distribution (classification) or regression value
- Path highlighting via `Set<node_id>` — indigo fill + border on active nodes, indigo strokes on edges
- Floating tooltip on hover
- Path breadcrumb bar at bottom
- Zoom controls (`+` / `−` / auto-fit)

Both `DecisionTreeView` and `RandomForestView` use this component. KNN and SVM use `NonTreeVisuals.jsx` instead.

## `NonTreeVisuals.jsx` — Canvas Layer for KNN and SVM

Shared canvas module for non-tree model visualisations.

**`KnnProjectionPlot`**
- PCA 2D scatter of all training points
- Non-neighbour points: radius 4, 0.6 opacity, surface stroke
- Neighbour points: radius 6, full opacity, white ring stroke (lineWidth 1.5)
- Neighbour connection lines: white at 0.2 opacity
- Query point: amber `#F5A524` crossed lines (lineWidth 2.5, size 8px)
- Classification boundary toggle (top-left pill): soft glow via BFS distance transform when ON — alpha lerped 0.22→0.04 near boundary, 0.06 farther
- KNN regression caveat label (bottom-left, 10px `textTertiary`): "positions are PCA projections — neighbour distances are computed in the original feature space"

**`KnnRegressionExplanation`**
- Primary KNN regression explanation surface
- `KnnLocalNeighborPlot` shows the query, selected neighbours, and all training rows using metric MDS over original feature-space distances
- The query marker is intentionally compact so it does not obscure nearby datapoints
- The graph uses the same interaction hook as the classifier plot, so ctrl-scroll zoom, drag pan, double-click reset, and button controls behave consistently
- Hover uses nearest-point lookup across the dense SVG plot so context rows, selected neighbours, and the query can all show tooltip details
- `KnnWeightedAveragePanel` lives in the right sidebar and owns selected-neighbour target, distance, raw weight, contribution, and weighted-average rows
- The old embedded PCA overview is no longer shown in this regression layout because it can make true neighbours look artificially far apart

**`SvmProjectionPlot`**
- PCA 2D scatter; non-SV points at 0.6 opacity r=4, SVs at full opacity r=6 with white ring
- SVM margin boundary lines (`margin_regions`) always visible
- Same soft boundary glow toggle as KNN
- Query point rendered as amber crossed lines

**Shared utilities**: `hexToRgb()`, `mixHex()`, `extent()`, `fmt()`, `ChartFrame()`, `CanvasTooltip()`

## `AppHeader.jsx`

Shared header shell used by all four views. Slots: `pills` (left), `right` (legend + toggle).

Sub-components: `Pill`, `ClassLegend`, `ThemeToggle` — all exported from `AppHeader.jsx`.

## Animation Rules

- Framer Motion for HTML elements only — never on SVG
- CSS `transition` on SVG `<rect>` fill/stroke only
- Ease-out curves, sub-300ms durations, no `transition: all`
- `useReducedMotion()` respected

## Dark Mode

`App.jsx` owns `dark` state. Initialised from `localStorage` → falls back to `prefers-color-scheme`. Persisted to `localStorage` on toggle. All components receive `dark` as a prop; theme is resolved locally via `getTheme(dark)`.

## Bundle

`npm run build` inside `mlviz/frontend/` writes to `mlviz/frontend/dist/`. This directory is committed and ships with the package. **Always rebuild before committing frontend changes.**

## Related Notes

- [[Server]] — provides the JSON this frontend fetches
- [[Architecture]] — call sequence, manifest routing
- [[mlviz Overview]] — full build status
