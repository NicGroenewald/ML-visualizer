"""Random Forest serializer — emits the /api/forest ensemble payload.

Each tree inside the payload reuses the exact node/path shapes produced by
the shared sklearn tree helpers, so the frontend single-tree renderer can
display any estimator without special-casing.
"""

from collections import Counter

from mlviz.extractors.shared import (
    normalize_query_vector,
    resolve_feature_names,
    serialize_sklearn_tree,
    trace_sklearn_path,
)


def _class_label(model, class_index):
    label = model.classes_[class_index]
    return label.item() if hasattr(label, "item") else label


def _serialize_summary(model):
    max_features = model.max_features
    if hasattr(max_features, "item"):
        max_features = max_features.item()
    return {
        "n_estimators": int(model.n_estimators),
        "n_features": int(model.n_features_in_),
        "n_classes": int(len(model.classes_)),
        "criterion": model.criterion,
        "bootstrap": bool(model.bootstrap),
        "max_features": max_features,
        "default_tree_id": 0,
    }


def _serialize_importances(model, names):
    items = [
        {
            "feature_index": i,
            "feature_name": names[i],
            "importance": round(float(v), 4),
        }
        for i, v in enumerate(model.feature_importances_)
    ]
    return sorted(items, key=lambda item: item["importance"], reverse=True)


def _serialize_oob(model):
    if getattr(model, "oob_score_", None) is None:
        return {"available": False, "score": None, "error": None}
    score = float(model.oob_score_)
    return {"available": True, "score": score, "error": 1.0 - score}


def _serialize_query(names, query_vector):
    return {
        "provided": True,
        "feature_values": [
            {
                "feature_index": i,
                "feature_name": names[i],
                "value": float(query_vector[i]),
            }
            for i in range(len(names))
        ],
    }


def _serialize_vote_distribution(model, tree_predictions):
    counts = Counter(tree_predictions)
    winning_index = max(counts, key=lambda idx: (counts[idx], -idx))
    total = len(tree_predictions)
    return {
        "winning_class_index": int(winning_index),
        "winning_class_label": _class_label(model, winning_index),
        "total_votes": total,
        "class_votes": [
            {
                "class_index": int(i),
                "class_label": _class_label(model, i),
                "votes": int(counts.get(i, 0)),
                "fraction": round(counts.get(i, 0) / total, 4),
            }
            for i in range(len(model.classes_))
        ],
        "tree_votes": [
            {
                "tree_id": tree_id,
                "class_index": int(pred),
                "class_label": _class_label(model, pred),
            }
            for tree_id, pred in enumerate(tree_predictions)
        ],
    }


def serialize(model, X_train, y_train, query=None, feature_names=None):
    names = resolve_feature_names(model, X_train, feature_names)
    query_vector = None if query is None else normalize_query_vector(query, len(names))

    trees = []
    tree_predictions = []
    for tree_id, estimator in enumerate(model.estimators_):
        tree = estimator.tree_
        path = None
        prediction = None
        if query_vector is not None:
            path = trace_sklearn_path(tree, names, query_vector)
            leaf = path[-1]
            class_index = leaf["prediction"]
            prediction = {
                "class_index": int(class_index),
                "class_label": _class_label(model, class_index),
                "leaf_counts": leaf["counts"],
            }
            tree_predictions.append(class_index)
        trees.append({
            "tree_id": tree_id,
            "node_count": int(tree.node_count),
            "n_leaves": int(tree.n_leaves),
            "max_depth": int(tree.max_depth),
            "prediction": prediction,
            "nodes": serialize_sklearn_tree(tree, names),
            "path": path,
        })

    return {
        "model_type": "random_forest",
        "model_family": "tree_ensemble",
        "classes": model.classes_.tolist(),
        "feature_names": names,
        "summary": _serialize_summary(model),
        "feature_importances": _serialize_importances(model, names),
        "oob": _serialize_oob(model),
        "query": None if query_vector is None else _serialize_query(names, query_vector),
        "vote_distribution": (
            None if query_vector is None
            else _serialize_vote_distribution(model, tree_predictions)
        ),
        "trees": trees,
    }
