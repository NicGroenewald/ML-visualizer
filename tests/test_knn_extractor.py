import pytest
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor


def test_knn_classifier_query_matches_model_prediction():
    iris = load_iris()
    model = KNeighborsClassifier(n_neighbors=5, weights="distance")
    model.fit(iris.data, iris.target)

    from mlviz.extractors.knn import serialize

    result = serialize(
        model,
        iris.data,
        iris.target,
        query=iris.data[0],
        feature_names=iris.feature_names,
    )

    expected = model.predict([iris.data[0]])[0]
    assert result["model_type"] == "knn"
    assert result["task_type"] == "classification"
    assert result["summary"]["n_neighbors"] == 5
    assert result["classes"] == model.classes_.tolist()
    assert result["query"]["provided"] is True
    assert result["prediction"]["class_label"] == expected
    assert len(result["neighbors"]) == model.n_neighbors
    assert result["vote_distribution"]["winning_class_label"] == expected
    assert result["projection"]["task_type"] == "classification"
    assert len(result["projection"]["points"]) == len(iris.data)
    assert "wx" in result["projection"]["points"][0]
    assert "wy" in result["projection"]["points"][0]
    assert "is_support" not in result["projection"]["points"][0]
    assert len(result["projection"]["regions"]) == 80 * 80
    assert "wx" in result["projection"]["regions"][0]
    assert "cell_w" in result["projection"]["regions"][0]
    assert "cell_h" in result["projection"]["regions"][0]
    assert result["projection"]["query_point"] is not None
    assert "wx" in result["projection"]["query_point"]
    assert result["projection"]["neighbor_indices"] == [
        neighbor["training_index"]
        for neighbor in result["neighbors"]
    ]
    assert result["neighbor_projection"] is None


def test_knn_classifier_includes_probabilities_when_available():
    iris = load_iris()
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(iris.data, iris.target)

    from mlviz.extractors.knn import serialize

    result = serialize(model, iris.data, iris.target, query=iris.data[0])

    probabilities = model.predict_proba([iris.data[0]])[0]
    assert result["prediction"]["probabilities"] == pytest.approx(probabilities.tolist())


def test_knn_regressor_query_matches_model_prediction():
    diabetes = load_diabetes()
    model = KNeighborsRegressor(n_neighbors=4, weights="distance")
    model.fit(diabetes.data, diabetes.target)

    from mlviz.extractors.knn import serialize

    result = serialize(
        model,
        diabetes.data,
        diabetes.target,
        query=diabetes.data[0],
        feature_names=diabetes.feature_names,
    )

    expected = model.predict([diabetes.data[0]])[0]
    assert result["model_type"] == "knn"
    assert result["task_type"] == "regression"
    assert result["prediction"]["value"] == pytest.approx(expected, rel=1e-6)
    assert len(result["neighbors"]) == model.n_neighbors
    assert all("target_value" in neighbor for neighbor in result["neighbors"])
    assert all("effective_weight" in neighbor for neighbor in result["neighbors"])
    assert result["projection"]["task_type"] == "regression"
    assert len(result["projection"]["points"]) == len(diabetes.data)
    assert "wx" in result["projection"]["points"][0]
    assert "is_support" not in result["projection"]["points"][0]
    assert result["projection"]["regions"] is None
    assert result["projection"]["query_point"] is not None
    assert result["projection"]["neighbor_indices"] == [
        neighbor["training_index"]
        for neighbor in result["neighbors"]
    ]
    assert result["neighbor_projection"]["task_type"] == "regression"
    assert result["neighbor_projection"]["method"] == "metric_mds"
    assert result["neighbor_projection"]["distance_source"] == "original_feature_space"
    assert result["neighbor_projection"]["prediction_value"] == pytest.approx(
        result["prediction"]["value"], rel=1e-6
    )
    assert len(result["neighbor_projection"]["points"]) == len(diabetes.data) + 1
    assert len(result["neighbor_projection"]["links"]) == model.n_neighbors
    assert result["neighbor_projection"]["context_count"] == len(diabetes.data) - model.n_neighbors

    query_point = result["neighbor_projection"]["points"][0]
    assert query_point["point_type"] == "query"
    assert query_point["wx"] == pytest.approx(0.0, abs=1e-9)
    assert query_point["wy"] == pytest.approx(0.0, abs=1e-9)

    projected_neighbors = [
        point
        for point in result["neighbor_projection"]["points"]
        if point["point_type"] == "neighbor"
    ]
    projected_context = [
        point
        for point in result["neighbor_projection"]["points"]
        if point["point_type"] == "context"
    ]
    assert len(projected_context) == result["neighbor_projection"]["context_count"]
    assert all(point["target_value"] is not None for point in projected_context)
    assert [point["training_index"] for point in projected_neighbors] == [
        neighbor["training_index"]
        for neighbor in result["neighbors"]
    ]
    assert [point["rank"] for point in projected_neighbors] == [
        neighbor["rank"]
        for neighbor in result["neighbors"]
    ]
    assert [link["training_index"] for link in result["neighbor_projection"]["links"]] == [
        neighbor["training_index"]
        for neighbor in result["neighbors"]
    ]
    assert [link["distance"] for link in result["neighbor_projection"]["links"]] == pytest.approx(
        [neighbor["distance"] for neighbor in result["neighbors"]]
    )


def test_knn_without_query_returns_summary_only():
    iris = load_iris()
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(iris.data, iris.target)

    from mlviz.extractors.knn import serialize

    result = serialize(model, iris.data, iris.target, feature_names=iris.feature_names)

    assert result["summary"]["n_features"] == iris.data.shape[1]
    assert result["query"] is None
    assert result["prediction"] is None
    assert result["neighbors"] is None
    assert result["vote_distribution"] is None
    assert result["neighbor_projection"] is None
    assert result["projection"]["query_point"] is None
    assert len(result["projection"]["points"]) == len(iris.data)


def test_knn_projection_is_null_for_one_feature_data():
    iris = load_iris()
    X = iris.data[:, :1]
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, iris.target)

    from mlviz.extractors.knn import serialize

    result = serialize(model, X, iris.target, query=X[0], feature_names=["sepal length"])

    assert result["summary"]["n_features"] == 1
    assert result["projection"] is None
