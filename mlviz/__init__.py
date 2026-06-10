import socket
import threading
import time
import webbrowser

import uvicorn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from mlviz.extractors.decision_tree import serialize as serialize_decision_tree
from mlviz.extractors.random_forest import serialize as serialize_random_forest
from mlviz.server.app import create_app


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_server(port: int, timeout: float = 5.0) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.05)


def visualize(
    model,
    X_train,
    y_train,
    query=None,
    feature_names=None,
    dataset_name=None,
):
    if isinstance(model, DecisionTreeClassifier):
        if not hasattr(model, "tree_"):
            raise ValueError("Model must be fitted before calling visualize()")
        payload = serialize_decision_tree(model, X_train, y_train, query=query, feature_names=feature_names)
        endpoint = "/api/tree"
    elif isinstance(model, RandomForestClassifier):
        if not hasattr(model, "estimators_"):
            raise ValueError("Model must be fitted before calling visualize()")
        payload = serialize_random_forest(model, X_train, y_train, query=query, feature_names=feature_names)
        endpoint = "/api/forest"
    else:
        raise NotImplementedError(
            "Only DecisionTreeClassifier and RandomForestClassifier are supported"
        )

    payload["dataset_name"] = dataset_name

    port = _find_free_port()
    app = create_app(payload, endpoint=endpoint)

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    _wait_for_server(port)
    webbrowser.open(f"http://127.0.0.1:{port}")
