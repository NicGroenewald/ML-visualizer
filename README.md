# mlviz — ML Model Visualizer

`mlviz` is a Python library that takes a trained scikit-learn model, spins up a local viewer, and opens an interactive browser UI so you can inspect how the model makes decisions.

Right now it supports:

- `sklearn.tree.DecisionTreeClassifier`
- `sklearn.ensemble.RandomForestClassifier`

You get the actual model structure, not a fake mock-up. If you pass a query point, `mlviz` also shows the exact path that sample took through the model.

## Quick start

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from mlviz import visualize

iris = load_iris()
X, y = iris.data, iris.target

model = DecisionTreeClassifier(max_depth=3, random_state=0)
model.fit(X, y)

visualize(
    model,
    X,
    y,
    query=X[0],
    feature_names=iris.feature_names,
    dataset_name="iris",
)
```

For Random Forest:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from mlviz import visualize

breast = load_breast_cancer()
X, y = breast.data, breast.target

model = RandomForestClassifier(
    n_estimators=100,
    random_state=0,
    oob_score=True,
)
model.fit(X, y)

visualize(
    model,
    X,
    y,
    query=X[0],
    feature_names=breast.feature_names,
    dataset_name="breast cancer",
)
```

## What the UI shows

### Decision Tree

- Full tree layout with every split and leaf
- Highlighted decision path for the current query
- Hover tooltips for node details
- Feature-importance sidebar
- Dataset label in the header when `dataset_name` is provided
- Browser tab titles like `mlviz · iris · decision tree`

### Random Forest

- Horizontal tree browser for selecting individual trees
- Single-tree canvas with path highlighting
- Ensemble vote distribution
- Feature-importance sidebar
- OOB score card when the fitted model exposes OOB data
- Dataset label in the header when `dataset_name` is provided
- Browser tab titles like `mlviz · breast cancer · random forest`

## Current scope

Supported now:

- Decision Tree visualisation
- Random Forest visualisation
- Light/dark theme toggle with saved preference
- Dataset naming via `visualize(..., dataset_name="...")`
- Query-path tracing
- Feature-importance display for both DT and RF

Planned later:

- KNN
- SVM
- More teaching-oriented explanation layers
- Better controls for larger models and interactive probing

## API

Main entry point:

```python
visualize(
    model,
    X_train,
    y_train,
    query=None,
    feature_names=None,
    dataset_name=None,
)
```

### Parameters

- `model`: fitted `DecisionTreeClassifier` or `RandomForestClassifier`
- `X_train`: training features used to fit the model
- `y_train`: training labels used to fit the model
- `query`: optional sample to trace through the model
- `feature_names`: optional feature names; DataFrame column names are also supported
- `dataset_name`: optional UI label used in the header and browser tab title

### Current behaviour

- Starts a local FastAPI server on a random free port
- Serves a small model manifest at `/api/model`
- Serves model payloads at `/api/tree` or `/api/forest`
- Opens the browser automatically with `webbrowser.open(...)`

## Project structure

```text
mlviz/
├── __init__.py
├── extractors/
│   ├── decision_tree.py
│   ├── random_forest.py
│   └── shared.py
├── server/
│   └── app.py
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── MLVizTree.jsx
    │   ├── theme.js
    │   ├── components/
    │   │   ├── AppHeader.jsx
    │   │   └── forest/
    │   └── views/
    │       ├── DecisionTreeView.jsx
    │       └── RandomForestView.jsx
    └── dist/

tests/
├── test_decision_tree_extractor.py
├── test_random_forest_extractor.py
├── test_server.py
└── test_visualize.py
```

## How it works

`visualize()` does three things:

1. Detects the supported model type.
2. Serializes the fitted sklearn model into a frontend-friendly JSON payload.
3. Starts a local FastAPI app and opens the browser viewer.

The frontend first fetches `/api/model`, then loads the matching payload endpoint:

- Decision Tree: `/api/tree`
- Random Forest: `/api/forest`

The tree canvas renderer is shared between both views, while the surrounding UI differs by model type.

## Development setup

### Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest tests/ -v
```

If you are using the existing project environment instead:

```bash
/opt/anaconda3/envs/NN-crew/bin/python -m pytest tests/ -v
```

### Frontend

```bash
cd mlviz/frontend
npm install
npm run dev
```

Build the shipped frontend bundle after source changes:

```bash
cd mlviz/frontend
npm run build
```

## Demo notebook

`demo.ipynb` includes ready-to-run examples for:

- iris
- wine
- breast cancer

The notebook now passes dataset names into the viewer so sessions are easier to distinguish in both the UI and browser tabs.

## Testing

The repo currently has regression coverage for:

- Decision Tree extractor shape and path tracing
- Random Forest extractor payload structure
- FastAPI endpoint behaviour
- `visualize()` dispatch and server startup flow
- Dataset-name payload propagation

Recent targeted verification in this repo:

- `npm run build` passed
- `tests/test_decision_tree_extractor.py`, `tests/test_server.py`, and `tests/test_visualize.py` passed together

## Known limitations

- Only `DecisionTreeClassifier` and `RandomForestClassifier` are supported today.
- Random Forest class labels are still driven by the serialized payload shape; richer class-name customisation is still worth doing later.
- Very large forests will need a better tree-browser strategy than the current flat strip.
- Light mode needs a dedicated contrast pass in the tree canvas and chrome.

## License

MIT
