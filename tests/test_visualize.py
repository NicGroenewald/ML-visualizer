import time
import types

import pytest
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def test_visualize_raises_if_not_fitted():
    from mlviz import visualize
    model = DecisionTreeClassifier()
    iris = load_iris()
    with pytest.raises(ValueError, match="Model must be fitted"):
        visualize(model, iris.data, iris.target)


def test_visualize_raises_for_unsupported_model():
    from mlviz import visualize
    iris = load_iris()
    model = KNeighborsClassifier()
    model.fit(iris.data, iris.target)
    with pytest.raises(NotImplementedError, match="Only DecisionTreeClassifier"):
        visualize(model, iris.data, iris.target)


def test_visualize_supports_random_forest(monkeypatch):
    import sys
    import httpx
    from sklearn.ensemble import RandomForestClassifier
    from mlviz import visualize

    # Run in daemon/interactive mode so visualize() returns without blocking
    monkeypatch.setitem(sys.modules, "ipykernel", types.ModuleType("ipykernel"))

    opened_urls = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened_urls.append(url))

    iris = load_iris()
    model = RandomForestClassifier(n_estimators=5, random_state=0)
    model.fit(iris.data, iris.target)

    visualize(model, iris.data, iris.target, query=iris.data[0])

    assert len(opened_urls) == 1
    url = opened_urls[0]
    manifest = httpx.get(f"{url}/api/model").json()
    assert manifest == {"model_type": "random_forest", "endpoint": "/api/forest"}
    forest = httpx.get(f"{url}/api/forest").json()
    assert forest["model_type"] == "random_forest"
    assert forest["vote_distribution"] is not None


def test_visualize_random_forest_includes_dataset_name(monkeypatch):
    import sys
    import httpx
    from sklearn.ensemble import RandomForestClassifier
    from mlviz import visualize

    # Run in daemon/interactive mode so visualize() returns without blocking
    monkeypatch.setitem(sys.modules, "ipykernel", types.ModuleType("ipykernel"))

    opened_urls = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened_urls.append(url))

    iris = load_iris()
    model = RandomForestClassifier(n_estimators=5, random_state=0)
    model.fit(iris.data, iris.target)

    visualize(model, iris.data, iris.target, dataset_name="iris")

    url = opened_urls[0]
    payload = httpx.get(f"{url}/api/forest").json()
    assert payload["dataset_name"] == "iris"


def test_visualize_starts_server_and_serves_tree(monkeypatch):
    import sys
    import httpx
    from mlviz import visualize

    # Run in daemon/interactive mode so visualize() returns without blocking
    monkeypatch.setitem(sys.modules, "ipykernel", types.ModuleType("ipykernel"))

    opened_urls = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened_urls.append(url))

    iris = load_iris()
    model = DecisionTreeClassifier(random_state=0)
    model.fit(iris.data, iris.target)

    visualize(model, iris.data, iris.target)

    assert len(opened_urls) == 1
    url = opened_urls[0]

    response = httpx.get(f"{url}/api/tree")
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "decision_tree"
    assert len(data["nodes"]) > 0
    assert data["path"] is None


def test_visualize_raises_if_server_never_starts(monkeypatch):
    import socket as _socket
    from mlviz import visualize

    # Patch create_connection to always fail
    def always_refuse(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(_socket, "create_connection", always_refuse)

    # Fast-forward time so the timeout expires immediately (avoids a 10s real-time wait)
    real_time = time.time()
    call_count = [0]

    def fake_time():
        call_count[0] += 1
        # First call sets `start`; every subsequent call is already past the timeout
        return real_time if call_count[0] == 1 else real_time + 11.0

    monkeypatch.setattr(time, "time", fake_time)

    # Prevent the browser from opening and the server thread from actually binding
    monkeypatch.setattr("webbrowser.open", lambda url: None)
    monkeypatch.setattr("threading.Thread.start", lambda self: None)

    iris = load_iris()
    model = DecisionTreeClassifier(random_state=0)
    model.fit(iris.data, iris.target)

    with pytest.raises(RuntimeError, match="mlviz server did not start"):
        visualize(model, iris.data, iris.target)


def test_visualize_decision_tree_includes_dataset_name(monkeypatch):
    import sys
    import httpx
    from mlviz import visualize

    # Run in daemon/interactive mode so visualize() returns without blocking
    monkeypatch.setitem(sys.modules, "ipykernel", types.ModuleType("ipykernel"))

    opened_urls = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened_urls.append(url))

    iris = load_iris()
    model = DecisionTreeClassifier(random_state=0)
    model.fit(iris.data, iris.target)

    visualize(
        model,
        iris.data,
        iris.target,
        feature_names=iris.feature_names,
        dataset_name="iris",
    )

    url = opened_urls[0]
    payload = httpx.get(f"{url}/api/tree").json()
    assert payload["dataset_name"] == "iris"


def test_is_interactive_false_without_ipykernel(monkeypatch):
    import sys
    import mlviz

    monkeypatch.delitem(sys.modules, "ipykernel", raising=False)

    assert mlviz._is_interactive() is False


def test_is_interactive_true_with_ipykernel(monkeypatch):
    import sys
    import mlviz

    monkeypatch.setitem(sys.modules, "ipykernel", types.ModuleType("ipykernel"))

    assert mlviz._is_interactive() is True


def test_visualize_non_daemon_thread_in_script_mode(monkeypatch):
    import builtins
    import sys
    import threading
    import mlviz

    monkeypatch.delitem(sys.modules, "ipykernel", raising=False)

    monkeypatch.setattr("webbrowser.open", lambda url: None)
    monkeypatch.setattr("threading.Thread.start", lambda self: None)
    monkeypatch.setattr("threading.Thread.join", lambda self: None)
    monkeypatch.setattr(mlviz, "_find_free_port", lambda: 8765)
    monkeypatch.setattr(mlviz, "_wait_for_server", lambda port, timeout=10.0: None)
    monkeypatch.setattr(builtins, "print", lambda *args, **kwargs: None)

    captured_daemon = []
    original_init = threading.Thread.__init__

    def capturing_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        captured_daemon.append(self.daemon)

    monkeypatch.setattr(threading.Thread, "__init__", capturing_init)

    iris = load_iris()
    model = DecisionTreeClassifier(random_state=0)
    model.fit(iris.data, iris.target)

    mlviz.visualize(model, iris.data, iris.target)

    assert captured_daemon == [False], "Thread must be non-daemon in script mode"


def test_visualize_daemon_thread_in_jupyter_mode(monkeypatch):
    import sys
    import threading
    import mlviz

    monkeypatch.setitem(sys.modules, "ipykernel", types.ModuleType("ipykernel"))

    monkeypatch.setattr("webbrowser.open", lambda url: None)
    monkeypatch.setattr("threading.Thread.start", lambda self: None)
    monkeypatch.setattr(mlviz, "_find_free_port", lambda: 8765)
    monkeypatch.setattr(mlviz, "_wait_for_server", lambda port, timeout=10.0: None)

    captured_daemon = []
    original_init = threading.Thread.__init__

    def capturing_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        captured_daemon.append(self.daemon)

    monkeypatch.setattr(threading.Thread, "__init__", capturing_init)

    iris = load_iris()
    model = DecisionTreeClassifier(random_state=0)
    model.fit(iris.data, iris.target)

    mlviz.visualize(model, iris.data, iris.target)

    assert captured_daemon == [True], "Thread must be daemon in Jupyter mode"


def test_ctrl_c_sets_server_should_exit(monkeypatch):
    import builtins
    import sys
    import threading
    import uvicorn
    import mlviz
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier

    # Script mode — no ipykernel
    monkeypatch.delitem(sys.modules, "ipykernel", raising=False)

    # Suppress side effects
    monkeypatch.setattr("webbrowser.open", lambda url: None)
    monkeypatch.setattr(mlviz, "_find_free_port", lambda: 8765)
    monkeypatch.setattr(mlviz, "_wait_for_server", lambda port, timeout=10.0: None)
    monkeypatch.setattr(builtins, "print", lambda *a, **kw: None)

    # Capture the server instance so we can inspect should_exit afterwards
    captured_server = []

    class MockServer:
        def __init__(self, config):
            self.should_exit = False
            captured_server.append(self)

        def run(self):
            pass

    monkeypatch.setattr(uvicorn, "Server", MockServer)

    # First join() raises KeyboardInterrupt; second join() returns normally
    join_call_count = [0]

    class MockThread:
        def __init__(self, target=None, daemon=None, **kwargs):
            self.daemon = daemon

        def start(self):
            pass

        def join(self):
            join_call_count[0] += 1
            if join_call_count[0] == 1:
                raise KeyboardInterrupt()

    monkeypatch.setattr(threading, "Thread", MockThread)

    iris = load_iris()
    model = DecisionTreeClassifier(random_state=0)
    model.fit(iris.data, iris.target)

    mlviz.visualize(model, iris.data, iris.target)

    assert len(captured_server) == 1, "Expected one server instance"
    assert captured_server[0].should_exit is True, (
        "server.should_exit must be True after Ctrl+C so uvicorn shuts down gracefully"
    )
    assert join_call_count[0] == 2, (
        "join() must be called twice: once to wait, once to confirm clean exit after shutdown"
    )
