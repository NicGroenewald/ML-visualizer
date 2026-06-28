import socket
import sys
import threading
import time
import webbrowser

import uvicorn
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from mlviz.extractors.decision_tree import serialize as serialize_decision_tree
from mlviz.extractors.knn import serialize as serialize_knn
from mlviz.extractors.random_forest import serialize as serialize_random_forest
from mlviz.extractors.svm import serialize as serialize_svm
from mlviz.server.app import create_app


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _is_interactive() -> bool:
    return "ipykernel" in sys.modules


def _wait_for_server(port: int, timeout: float = 10.0) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(
        f"mlviz server did not start on port {port} within {timeout:.0f}s. "
        "Check that uvicorn installed correctly and no firewall is blocking loopback connections."
    )


def visualize(
    model,
    X_train,
    y_train,
    query=None,
    feature_names=None,
    dataset_name=None,
):
    if isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
        if not hasattr(model, "tree_"):
            raise ValueError("Model must be fitted before calling visualize()")
        payload = serialize_decision_tree(model, X_train, y_train, query=query, feature_names=feature_names)
        endpoint = "/api/tree"
    elif isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
        if not hasattr(model, "estimators_"):
            raise ValueError("Model must be fitted before calling visualize()")
        payload = serialize_random_forest(model, X_train, y_train, query=query, feature_names=feature_names)
        endpoint = "/api/forest"
    elif isinstance(model, (KNeighborsClassifier, KNeighborsRegressor)):
        if not hasattr(model, "n_features_in_"):
            raise ValueError("Model must be fitted before calling visualize()")
        payload = serialize_knn(model, X_train, y_train, query=query, feature_names=feature_names)
        endpoint = "/api/knn"
    elif isinstance(model, (SVC, SVR)):
        if not hasattr(model, "support_vectors_"):
            raise ValueError("Model must be fitted before calling visualize()")
        payload = serialize_svm(model, X_train, y_train, query=query, feature_names=feature_names)
        endpoint = "/api/svm"
    else:
        raise NotImplementedError(
            "Only DecisionTreeClassifier, DecisionTreeRegressor, "
            "RandomForestClassifier, RandomForestRegressor, "
            "KNeighborsClassifier, KNeighborsRegressor, SVC, and SVR are supported"
        )

    payload["dataset_name"] = dataset_name

    port = _find_free_port()
    app = create_app(payload, endpoint=endpoint)

    daemon = _is_interactive()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=daemon)
    thread.start()

    _wait_for_server(port)
    url = f"http://127.0.0.1:{port}"
    webbrowser.open(url)

    if not daemon:
        print(f"mlviz serving at {url} — press Ctrl+C to stop")
        try:
            thread.join()
        except KeyboardInterrupt:
            server.should_exit = True
            try:
                thread.join()
            except KeyboardInterrupt:
                pass
