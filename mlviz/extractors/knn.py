"""KNN serializer for nearest-neighbor summary and query explanation payloads."""

from collections import defaultdict

import numpy as np
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

from mlviz.extractors.shared import (
    as_2d_array,
    class_index,
    json_value,
    model_task_type,
    normalize_query_vector,
    resolve_feature_names,
    serialize_query,
)
from mlviz.extractors.projection import knn_projection


def _serialize_summary(model, n_features):
    return {
        "n_neighbors": int(model.n_neighbors),
        "weights": model.weights if isinstance(model.weights, str) else "callable",
        "algorithm": model.algorithm,
        "metric": model.metric,
        "p": None if model.p is None else int(model.p),
        "n_features": int(n_features),
    }


def _effective_weights(model, distances):
    if callable(model.weights):
        values = np.asarray(model.weights(distances), dtype=float)
        return values.reshape(-1)
    if model.weights == "distance":
        distances = np.asarray(distances, dtype=float).reshape(-1)
        zero_mask = distances == 0
        if zero_mask.any():
            return zero_mask.astype(float)
        return 1.0 / distances
    return np.ones_like(np.asarray(distances, dtype=float).reshape(-1), dtype=float)


def _serialize_neighbors(model, y_train, distances, indices, weights, task_type):
    targets = np.asarray(y_train)
    neighbors = []
    for rank, (distance, index, weight) in enumerate(zip(distances, indices, weights), start=1):
        target = json_value(targets[int(index)])
        neighbor = {
            "rank": rank,
            "training_index": int(index),
            "distance": float(distance),
            "effective_weight": float(weight),
        }
        if task_type == "classification":
            neighbor["class_index"] = class_index(model, target)
            neighbor["class_label"] = target
        else:
            neighbor["target_value"] = float(target)
        neighbors.append(neighbor)
    return neighbors


def _serialize_vote_distribution(model, neighbors):
    totals = defaultdict(float)
    for neighbor in neighbors:
        totals[neighbor["class_label"]] += neighbor["effective_weight"]
    winning_label = max(totals, key=lambda label: (totals[label], -class_index(model, label)))
    total_weight = sum(totals.values())
    return {
        "winning_class_index": class_index(model, winning_label),
        "winning_class_label": winning_label,
        "total_weight": float(total_weight),
        "class_votes": [
            {
                "class_index": int(class_index),
                "class_label": json_value(label),
                "weight": float(totals.get(json_value(label), 0.0)),
                "fraction": (
                    0.0
                    if total_weight == 0
                    else round(float(totals.get(json_value(label), 0.0) / total_weight), 4)
                ),
            }
            for class_index, label in enumerate(model.classes_)
        ],
    }


def _serialize_prediction(model, query_vector, task_type):
    prediction = model.predict([query_vector])[0]
    if task_type == "classification":
        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = [float(value) for value in model.predict_proba([query_vector])[0]]
        label = json_value(prediction)
        return {
            "class_index": class_index(model, label),
            "class_label": label,
            "probabilities": probabilities,
        }
    return {"value": float(prediction)}


def _metric_distances(model, points):
    metric = getattr(model, "effective_metric_", None) or getattr(model, "metric", "euclidean")
    params = dict(getattr(model, "effective_metric_params_", None) or {})
    return pairwise_distances(points, metric=metric, **params)


def _serialize_neighbor_projection(model, X_train, y_train, query_vector, neighbors, prediction):
    if not neighbors or prediction is None:
        return None

    try:
        X = as_2d_array(X_train)
        targets = np.asarray(y_train)
        neighbor_indices = [neighbor["training_index"] for neighbor in neighbors]
        _, local_indices_raw = model.kneighbors([query_vector], n_neighbors=len(X), return_distance=True)
        local_indices = [int(index) for index in local_indices_raw[0]]
        neighbor_index_set = set(neighbor_indices)
        ordered_indices = neighbor_indices + [
            index for index in local_indices if index not in neighbor_index_set
        ]
        points = np.vstack([query_vector, X[ordered_indices]])
        distances = _metric_distances(model, points)
        embedding = MDS(
            n_components=2,
            metric=True,
            n_init=1,
            max_iter=120,
            random_state=0,
            dissimilarity="precomputed",
            normalized_stress="auto",
        ).fit_transform(distances)
    except (TypeError, ValueError, AttributeError):
        return None

    embedding = embedding - embedding[0]
    max_radius = max(np.linalg.norm(embedding[1:], axis=1).max(initial=0.0), 1e-9)
    embedding = embedding / max_radius * 3.0

    projected_points = [
        {
            "point_type": "query",
            "training_index": None,
            "rank": None,
            "distance": 0.0,
            "effective_weight": None,
            "target_value": None,
            "wx": float(embedding[0][0]),
            "wy": float(embedding[0][1]),
        }
    ]
    neighbor_by_index = {neighbor["training_index"]: neighbor for neighbor in neighbors}
    context_count = 0
    for coords, training_index in zip(embedding[1:], ordered_indices):
        neighbor = neighbor_by_index.get(training_index)
        if neighbor is None:
            context_count += 1
            projected_points.append(
                {
                    "point_type": "context",
                    "training_index": int(training_index),
                    "rank": None,
                    "distance": None,
                    "effective_weight": None,
                    "target_value": float(targets[int(training_index)]),
                    "wx": float(coords[0]),
                    "wy": float(coords[1]),
                }
            )
            continue
        projected_points.append(
            {
                "point_type": "neighbor",
                "training_index": neighbor["training_index"],
                "rank": neighbor["rank"],
                "distance": neighbor["distance"],
                "effective_weight": neighbor["effective_weight"],
                "target_value": neighbor["target_value"],
                "wx": float(coords[0]),
                "wy": float(coords[1]),
            }
        )

    return {
        "task_type": "regression",
        "method": "metric_mds",
        "distance_source": "original_feature_space",
        "prediction_value": prediction["value"],
        "context_count": context_count,
        "points": projected_points,
        "links": [
            {
                "source": "query",
                "target": neighbor["training_index"],
                "training_index": neighbor["training_index"],
                "rank": neighbor["rank"],
                "distance": neighbor["distance"],
                "effective_weight": neighbor["effective_weight"],
            }
            for neighbor in neighbors
        ],
    }


def serialize(model, X_train, y_train, query=None, feature_names=None):
    task_type = model_task_type(model)
    names = resolve_feature_names(model, X_train, feature_names)
    query_vector = None if query is None else normalize_query_vector(query, len(names))

    payload = {
        "model_type": "knn",
        "task_type": task_type,
        "model_family": "nearest_neighbors",
        "feature_names": names,
        "summary": _serialize_summary(model, len(names)),
        "query": None if query_vector is None else serialize_query(names, query_vector),
        "prediction": None,
        "neighbors": None,
        "vote_distribution": None,
        "neighbor_projection": None,
        "projection": None,
    }
    if task_type == "classification":
        payload["classes"] = [json_value(label) for label in model.classes_]

    if query_vector is None:
        payload["projection"] = knn_projection(model, X_train, y_train, None, task_type, [])
        return payload

    distances, indices = model.kneighbors([query_vector], return_distance=True)
    distances = distances[0]
    indices = indices[0]
    weights = _effective_weights(model, distances)
    neighbors = _serialize_neighbors(model, y_train, distances, indices, weights, task_type)

    payload["prediction"] = _serialize_prediction(model, query_vector, task_type)
    payload["neighbors"] = neighbors
    if task_type == "classification":
        payload["vote_distribution"] = _serialize_vote_distribution(model, neighbors)
    else:
        payload["neighbor_projection"] = _serialize_neighbor_projection(
            model, X_train, y_train, query_vector, neighbors, payload["prediction"]
        )
    payload["projection"] = knn_projection(model, X_train, y_train, query_vector, task_type, indices)
    return payload
