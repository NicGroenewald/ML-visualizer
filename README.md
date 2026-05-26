# mlviz — ML Model Visualizer

A Python library that takes a real trained scikit-learn model and your real training data, then opens an interactive browser visualisation showing exactly how the model thinks.

```python
from mlviz import visualize

model.fit(X_train, y_train)
visualize(model, X_train, y_train, query=X_test[0])
```

One call. A browser tab opens. You see the full model — and, if you pass a query point, the exact path it took to reach its prediction.

---

## What you see in the browser

For a Decision Tree:

- **Full tree diagram** — every node showing the feature it splits on, the threshold, how many training samples reached it, and the Gini impurity
- **Leaf nodes** — the class distribution and final predicted class
- **Decision path** (when a query is passed) — the root-to-leaf journey highlighted, with the query point's value shown at each split alongside the question being asked

---

## V1 scope

Decision Tree only (`sklearn.tree.DecisionTreeClassifier`).

KNN, Random Forest, and SVM are planned for later versions.

---

## Requirements

- Python 3.10+
- scikit-learn
- FastAPI + uvicorn

---

## Development setup

```bash
# Clone the repo
git clone <repo-url>
cd ML-visualizer

# Create and activate a virtual environment (or use your existing ML env)
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

# Install dependencies
pip install fastapi uvicorn scikit-learn httpx pytest

# Install mlviz in editable mode so imports resolve from anywhere
pip install -e .

# Run the test suite
pytest tests/ -v
```

All 14 tests should pass.

---

## Try it out

Open `demo.ipynb` in VS Code or Jupyter. It has ready-to-run examples using iris, wine, and breast-cancer datasets, plus an empty cell at the bottom for your own model.

---

## Project structure

```
mlviz/
├── __init__.py               ← visualize() entry point
├── extractors/
│   └── decision_tree.py      ← reads model.tree_, serialises to JSON
├── server/
│   └── app.py                ← FastAPI server, serves JSON + React app
└── frontend/
    ├── src/                  ← React + Vite source
    └── dist/                 ← built bundle (ships with package)

tests/                        ← 14 tests, all passing
demo.ipynb                    ← interactive examples notebook
```

---

## How it works

`visualize()` does three things in sequence:

1. **Extracts** — reads the model's internal arrays (`model.tree_`) and converts every node to a plain JSON dict
2. **Serves** — starts a local FastAPI server on a random port and serves the JSON
3. **Opens** — calls `webbrowser.open()` so the React frontend loads automatically

The frontend fetches the JSON, renders the full tree diagram, and highlights the decision path if a query point was provided.

The JSON the extractor produces looks like this:

```json
{
  "model_type": "decision_tree",
  "classes": ["setosa", "versicolor", "virginica"],
  "feature_names": ["petal length (cm)", "petal width (cm)"],
  "nodes": [
    { "node_id": 0, "feature": "petal length (cm)", "threshold": 2.45,
      "n_samples": 150, "gini": 0.667, "left_child": 1, "right_child": 2, "leaf": false },
    { "node_id": 1, "leaf": true, "counts": [50, 0, 0], "prediction": 0 }
  ],
  "path": [
    { "node_id": 0, "feature": "petal length (cm)", "threshold": 2.45,
      "value": 1.4, "went_left": true },
    { "node_id": 1, "leaf": true, "counts": [50, 0, 0], "prediction": 0 }
  ]
}
```

---

## Build progress

| Component | Status |
|-----------|--------|
| Package scaffold | ✅ done |
| `extractors/decision_tree.py` | ✅ done — 14 tests passing |
| `server/app.py` (FastAPI) | ✅ done |
| `mlviz/__init__.py` (`visualize()`) | ✅ done |
| `frontend/` (React tree diagram) | ✅ done |
| `frontend/` (path highlighting) | ✅ done |
| Installable package (`pip install`) | 🔲 next |

---

## License

MIT
