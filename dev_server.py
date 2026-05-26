"""Frontend dev helper — starts FastAPI on :8765 with iris test data."""
import uvicorn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from mlviz.extractors.decision_tree import serialize
from mlviz.server.app import create_app

if __name__ == "__main__":
    iris = load_iris()
    model = DecisionTreeClassifier(max_depth=3, random_state=0)
    model.fit(iris.data, iris.target)
    payload = serialize(
        model, iris.data, iris.target,
        query=iris.data[0],
        feature_names=iris.feature_names,
    )

    app = create_app(payload)
    print("FastAPI dev server → http://127.0.0.1:8765/api/tree")
    print("Start  npm run dev  in mlviz/frontend/ to develop the frontend.")
    print("Ctrl+C to stop.\n")
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
