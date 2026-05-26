import socket
import threading
import time
import webbrowser

import uvicorn
from sklearn.tree import DecisionTreeClassifier

from mlviz.extractors.decision_tree import serialize
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


def visualize(model, X_train, y_train, query=None, feature_names=None):
    if not isinstance(model, DecisionTreeClassifier):
        raise NotImplementedError("Only DecisionTreeClassifier is supported in V1")
    if not hasattr(model, "tree_"):
        raise ValueError("Model must be fitted before calling visualize()")

    payload = serialize(model, X_train, y_train, query=query, feature_names=feature_names)
    port = _find_free_port()
    app = create_app(payload)

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    _wait_for_server(port)
    webbrowser.open(f"http://127.0.0.1:{port}")
