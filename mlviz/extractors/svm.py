"""SVM serializer for support-vector summaries and query prediction payloads."""

import numpy as np

from mlviz.extractors.shared import (
    json_value,
    model_task_type,
    normalize_query_vector,
    resolve_feature_names,
    serialize_query,
)
from mlviz.extractors.projection import svm_projection


def _json_decision(value):
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return array.astype(float).tolist()


def _serialize_summary(model, task_type, n_features):
    summary = {
        "kernel": model.kernel,
        "C": float(model.C),
        "gamma": json_value(model.gamma),
        "degree": int(model.degree),
        "coef0": float(model.coef0),
        "support_vector_count": int(len(model.support_)),
        "n_features": int(n_features),
    }
    if task_type == "classification":
        summary["n_support"] = [int(value) for value in model.n_support_]
    else:
        summary["epsilon"] = float(model.epsilon)
    return summary


def _support_vector_target(model, y_train, training_index, task_type):
    target = json_value(np.asarray(y_train)[int(training_index)])
    if task_type == "classification":
        return {
            "class_index": int(np.where(model.classes_ == target)[0][0]),
            "class_label": target,
        }
    return {"target_value": float(target)}


def _serialize_support_vectors(model, y_train, names, task_type):
    support_vectors = []
    for order, (training_index, vector) in enumerate(zip(model.support_, model.support_vectors_)):
        item = {
            "support_vector_index": order,
            "training_index": int(training_index),
            "feature_values": [
                {
                    "feature_index": i,
                    "feature_name": names[i],
                    "value": float(value),
                }
                for i, value in enumerate(vector)
            ],
        }
        item.update(_support_vector_target(model, y_train, training_index, task_type))
        support_vectors.append(item)
    return support_vectors


def _serialize_prediction(model, query_vector, task_type):
    prediction = model.predict([query_vector])[0]
    if task_type == "classification":
        decision = _json_decision(model.decision_function([query_vector])[0])
        label = json_value(prediction)
        decision_values = np.asarray(model.decision_function([query_vector])[0], dtype=float).reshape(-1)
        if len(model.classes_) == 2 and len(decision_values) == 1:
            class_scores = [-float(decision_values[0]), float(decision_values[0])]
        elif len(decision_values) == len(model.classes_):
            class_scores = [float(value) for value in decision_values]
        else:
            class_scores = None
        probabilities = None
        if getattr(model, "probability", False):
            probabilities = [float(value) for value in model.predict_proba([query_vector])[0]]
        return {
            "class_index": int(np.where(model.classes_ == label)[0][0]),
            "class_label": label,
            "decision_function": decision,
            "decision_scores": (
                None if class_scores is None else [
                    {
                        "class_index": int(class_index),
                        "class_label": json_value(class_label),
                        "score": score,
                        "is_winner": json_value(class_label) == label,
                    }
                    for class_index, (class_label, score) in enumerate(zip(model.classes_, class_scores))
                ]
            ),
            "probabilities": probabilities,
        }
    return {"value": float(prediction)}


def serialize(model, X_train, y_train, query=None, feature_names=None):
    task_type = model_task_type(model)
    names = resolve_feature_names(model, X_train, feature_names)
    query_vector = None if query is None else normalize_query_vector(query, len(names))

    payload = {
        "model_type": "svm",
        "task_type": task_type,
        "model_family": "support_vector_machine",
        "feature_names": names,
        "summary": _serialize_summary(model, task_type, len(names)),
        "support_vectors": _serialize_support_vectors(model, y_train, names, task_type),
        "support_vector_target_range": None,
        "projection": svm_projection(model, X_train, y_train, query_vector, task_type),
        "query": None if query_vector is None else serialize_query(names, query_vector),
        "prediction": None if query_vector is None else _serialize_prediction(model, query_vector, task_type),
    }
    if task_type == "classification":
        payload["classes"] = [json_value(label) for label in model.classes_]
    else:
        support_targets = np.asarray(y_train, dtype=float)[model.support_]
        payload["support_vector_target_range"] = {
            "min": float(support_targets.min()),
            "max": float(support_targets.max()),
        }
    return payload
