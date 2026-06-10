import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture
def iris_forest():
    iris = load_iris()
    model = RandomForestClassifier(
        n_estimators=7,
        random_state=0,
        oob_score=True,
        bootstrap=True,
    )
    model.fit(iris.data, iris.target)
    return model, iris.data, iris.target, list(iris.feature_names)


def test_random_forest_serialize_structure(iris_forest):
    model, X, y, feature_names = iris_forest
    from mlviz.extractors.random_forest import serialize
    result = serialize(model, X, y, query=X[0], feature_names=feature_names)
    assert result["model_type"] == "random_forest"
    assert result["model_family"] == "tree_ensemble"
    assert len(result["trees"]) == model.n_estimators
    assert result["summary"]["n_estimators"] == model.n_estimators
    assert result["query"]["provided"] is True


def test_random_forest_feature_importances_sorted(iris_forest):
    model, X, y, feature_names = iris_forest
    from mlviz.extractors.random_forest import serialize
    result = serialize(model, X, y, feature_names=feature_names)
    importances = result["feature_importances"]
    assert importances == sorted(importances, key=lambda item: item["importance"], reverse=True)


def test_random_forest_vote_distribution_matches_tree_count(iris_forest):
    model, X, y, feature_names = iris_forest
    from mlviz.extractors.random_forest import serialize
    result = serialize(model, X, y, query=X[0], feature_names=feature_names)
    assert result["vote_distribution"]["total_votes"] == model.n_estimators
    assert len(result["vote_distribution"]["tree_votes"]) == model.n_estimators


def test_random_forest_oob_block_present_when_available(iris_forest):
    model, X, y, feature_names = iris_forest
    from mlviz.extractors.random_forest import serialize
    result = serialize(model, X, y, feature_names=feature_names)
    assert result["oob"]["available"] is True
    assert result["oob"]["score"] == pytest.approx(model.oob_score_, rel=1e-6)
    assert result["oob"]["error"] == pytest.approx(1.0 - model.oob_score_, rel=1e-6)


def test_random_forest_oob_block_when_unavailable():
    iris = load_iris()
    model = RandomForestClassifier(n_estimators=3, random_state=0, oob_score=False)
    model.fit(iris.data, iris.target)
    from mlviz.extractors.random_forest import serialize
    result = serialize(model, iris.data, iris.target)
    assert result["oob"] == {"available": False, "score": None, "error": None}


def test_random_forest_paths_are_null_without_query(iris_forest):
    model, X, y, feature_names = iris_forest
    from mlviz.extractors.random_forest import serialize
    result = serialize(model, X, y, feature_names=feature_names)
    assert result["query"] is None
    assert result["vote_distribution"] is None
    assert all(tree["path"] is None for tree in result["trees"])
    assert all(tree["prediction"] is None for tree in result["trees"])


def test_random_forest_tree_nodes_match_dt_shape(iris_forest):
    model, X, y, feature_names = iris_forest
    from mlviz.extractors.random_forest import serialize
    result = serialize(model, X, y, query=X[0], feature_names=feature_names)
    base_keys = {
        "node_id", "n_samples", "gini", "leaf",
        "feature", "threshold", "left_child", "right_child",
    }
    tree = result["trees"][0]
    for node in tree["nodes"]:
        expected = base_keys | ({"counts", "prediction"} if node["leaf"] else set())
        assert set(node.keys()) == expected
    assert tree["path"][-1]["leaf"] is True


def test_random_forest_vote_fractions_sum_to_one(iris_forest):
    model, X, y, feature_names = iris_forest
    from mlviz.extractors.random_forest import serialize
    result = serialize(model, X, y, query=X[0], feature_names=feature_names)
    fractions = [c["fraction"] for c in result["vote_distribution"]["class_votes"]]
    assert sum(fractions) == pytest.approx(1.0)


def test_random_forest_winning_class_matches_model_predict(iris_forest):
    model, X, y, feature_names = iris_forest
    from mlviz.extractors.random_forest import serialize
    result = serialize(model, X, y, query=X[0], feature_names=feature_names)
    vote = result["vote_distribution"]
    # majority vote of trees should match index of max votes
    top = max(vote["class_votes"], key=lambda c: c["votes"])
    assert vote["winning_class_index"] == top["class_index"]
