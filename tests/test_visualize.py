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


def test_visualize_starts_server_and_serves_tree(monkeypatch):
    import httpx
    from mlviz import visualize

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
