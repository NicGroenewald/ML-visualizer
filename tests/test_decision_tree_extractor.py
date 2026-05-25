import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


@pytest.fixture
def iris_model():
    iris = load_iris()
    model = DecisionTreeClassifier(random_state=0)
    model.fit(iris.data, iris.target)
    return model, iris.data, iris.target, list(iris.feature_names)


def test_extract_tree_node_count(iris_model):
    model, X, y, feature_names = iris_model
    from mlviz.extractors.decision_tree import _extract_tree
    nodes = _extract_tree(model.tree_, feature_names)
    assert len(nodes) == model.tree_.node_count


def test_all_leaves_have_counts(iris_model):
    model, X, y, feature_names = iris_model
    from mlviz.extractors.decision_tree import _extract_tree
    nodes = _extract_tree(model.tree_, feature_names)
    for node in nodes:
        if node["leaf"]:
            assert "counts" in node
            assert node["counts"] is not None


def test_trace_path_ends_at_leaf(iris_model):
    model, X, y, feature_names = iris_model
    from mlviz.extractors.decision_tree import _trace_path
    path = _trace_path(model.tree_, feature_names, X[0])
    assert path[-1]["leaf"] is True


def test_trace_path_went_left_consistent(iris_model):
    model, X, y, feature_names = iris_model
    from mlviz.extractors.decision_tree import _trace_path
    path = _trace_path(model.tree_, feature_names, X[0])
    for step in path:
        if not step.get("leaf"):
            assert step["went_left"] == (step["value"] <= step["threshold"])
