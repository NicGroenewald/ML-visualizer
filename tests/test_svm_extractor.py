import pytest
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.svm import SVR


def test_svc_query_includes_support_vectors_and_decision_scores():
    iris = load_iris()
    model = SVC(kernel="rbf", gamma="scale", probability=True, random_state=0)
    model.fit(iris.data, iris.target)

    from mlviz.extractors.svm import serialize

    result = serialize(
        model,
        iris.data,
        iris.target,
        query=iris.data[0],
        feature_names=iris.feature_names,
    )

    expected = model.predict([iris.data[0]])[0]
    assert result["model_type"] == "svm"
    assert result["task_type"] == "classification"
    assert result["summary"]["kernel"] == "rbf"
    assert result["summary"]["support_vector_count"] == len(model.support_)
    assert result["classes"] == model.classes_.tolist()
    assert result["prediction"]["class_label"] == expected
    assert result["prediction"]["decision_function"] is not None
    assert len(result["prediction"]["decision_scores"]) == len(model.classes_)
    assert any(score["is_winner"] for score in result["prediction"]["decision_scores"])
    assert result["prediction"]["probabilities"] == pytest.approx(
        model.predict_proba([iris.data[0]])[0].tolist()
    )
    assert len(result["support_vectors"]) == len(model.support_)
    assert "distance" not in result["support_vectors"][0]
    assert result["projection"]["task_type"] == "classification"
    assert len(result["projection"]["points"]) == len(iris.data)
    assert len(result["projection"]["regions"]) == 80 * 80
    assert "boundary_points" not in result["projection"]
    assert "margin_points" not in result["projection"]
    assert result["projection"]["margin_regions"] is None
    assert result["projection"]["is_binary"] is False
    assert "wx" in result["projection"]["points"][0]
    assert "wx" in result["projection"]["regions"][0]
    assert "cell_w" in result["projection"]["regions"][0]


def test_svc_without_probability_omits_probabilities():
    iris = load_iris()
    model = SVC(kernel="linear", probability=False)
    model.fit(iris.data, iris.target)

    from mlviz.extractors.svm import serialize

    result = serialize(model, iris.data, iris.target, query=iris.data[0])

    assert result["prediction"]["probabilities"] is None


def test_svr_query_includes_prediction_and_support_vector_summary():
    diabetes = load_diabetes()
    model = SVR(kernel="rbf", C=10.0, epsilon=0.2)
    model.fit(diabetes.data, diabetes.target)

    from mlviz.extractors.svm import serialize

    result = serialize(
        model,
        diabetes.data,
        diabetes.target,
        query=diabetes.data[0],
        feature_names=diabetes.feature_names,
    )

    expected = model.predict([diabetes.data[0]])[0]
    assert result["model_type"] == "svm"
    assert result["task_type"] == "regression"
    assert result["summary"]["epsilon"] == pytest.approx(0.2)
    assert result["prediction"]["value"] == pytest.approx(expected, rel=1e-6)
    assert len(result["support_vectors"]) == len(model.support_)
    assert "rank" not in result["support_vectors"][0]
    assert "distance" not in result["support_vectors"][0]
    assert result["support_vector_target_range"] == {
        "min": pytest.approx(diabetes.target[model.support_].min()),
        "max": pytest.approx(diabetes.target[model.support_].max()),
    }
    assert result["projection"]["task_type"] == "regression"
    assert len(result["projection"]["points"]) == len(diabetes.data)
    assert result["projection"]["regions"] is None
    assert len(result["projection"]["fit_line"]) == 200
    assert "wx" in result["projection"]["fit_line"][0]


def test_svm_without_query_returns_summary_only():
    iris = load_iris()
    model = SVC(kernel="linear")
    model.fit(iris.data, iris.target)

    from mlviz.extractors.svm import serialize

    result = serialize(model, iris.data, iris.target, feature_names=iris.feature_names)

    assert result["summary"]["n_features"] == iris.data.shape[1]
    assert result["query"] is None
    assert result["prediction"] is None
    assert result["support_vectors"]
    assert result["projection"]["query_point"] is None
    assert "wx" in result["projection"]["points"][0]


def test_binary_svc_projection_includes_margin_regions():
    cancer = load_breast_cancer()
    model = SVC(kernel="linear")
    model.fit(cancer.data, cancer.target)

    from mlviz.extractors.svm import serialize

    result = serialize(model, cancer.data, cancer.target, query=cancer.data[0],
                       feature_names=cancer.feature_names)

    proj = result["projection"]
    assert proj["is_binary"] is True
    assert "boundary_points" not in proj
    assert "margin_points" not in proj
    assert len(proj["margin_regions"]) == 80 * 80
    assert "wx" in proj["margin_regions"][0]
    assert "wy" in proj["margin_regions"][0]
    assert "margin_sign" in proj["margin_regions"][0]
    assert "cell_w" in proj["margin_regions"][0]
    assert "cell_h" in proj["margin_regions"][0]
    assert any(r["margin_sign"] != 0 for r in proj["margin_regions"])


def test_svm_projection_is_null_for_one_feature_data():
    iris = load_iris()
    X = iris.data[:, :1]
    model = SVC(kernel="linear")
    model.fit(X, iris.target)

    from mlviz.extractors.svm import serialize

    result = serialize(model, X, iris.target, query=X[0], feature_names=["sepal length"])

    assert result["summary"]["n_features"] == 1
    assert result["projection"] is None
