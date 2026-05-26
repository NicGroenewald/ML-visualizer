import pytest
from fastapi.testclient import TestClient
from mlviz.server.app import create_app


@pytest.fixture
def sample_payload():
    return {
        "model_type": "decision_tree",
        "classes": [0, 1, 2],
        "feature_names": ["x0", "x1"],
        "nodes": [
            {
                "node_id": 0,
                "leaf": True,
                "counts": [5, 3, 2],
                "prediction": 0,
                "n_samples": 10,
                "gini": 0.5,
                "feature": None,
                "threshold": None,
                "left_child": None,
                "right_child": None,
            }
        ],
        "path": None,
    }


def test_get_tree_returns_payload(sample_payload):
    app = create_app(sample_payload)
    client = TestClient(app)
    response = client.get("/api/tree")
    assert response.status_code == 200
    assert response.json() == sample_payload


def test_two_apps_do_not_share_payload():
    payload1 = {"model_type": "decision_tree", "tag": "first",
                 "nodes": [], "path": None, "classes": [], "feature_names": []}
    payload2 = {"model_type": "decision_tree", "tag": "second",
                 "nodes": [], "path": None, "classes": [], "feature_names": []}
    client1 = TestClient(create_app(payload1))
    client2 = TestClient(create_app(payload2))
    assert client1.get("/api/tree").json()["tag"] == "first"
    assert client2.get("/api/tree").json()["tag"] == "second"
