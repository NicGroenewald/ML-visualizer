"""Shared sklearn tree extraction helpers.

Used by both the Decision Tree and Random Forest serializers so every
tree-based model emits the exact same node and path shapes.
"""

import numpy as np


def resolve_feature_names(model, X_train, feature_names):
    if feature_names is not None:
        return list(feature_names)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(X_train, "columns"):
        return list(X_train.columns)
    return [f"x{i}" for i in range(X_train.shape[1])]


def normalize_query_vector(query, n_features):
    if hasattr(query, "values"):  # pandas Series / single-row DataFrame
        query = query.values
    vector = np.asarray(query, dtype=float).reshape(-1)
    if vector.shape[0] != n_features:
        raise ValueError(
            f"Query has {vector.shape[0]} features, model expects {n_features}"
        )
    return vector


def model_task_type(model):
    return "classification" if hasattr(model, "classes_") else "regression"


def json_value(value):
    return value.item() if hasattr(value, "item") else value


def as_2d_array(X_train):
    if hasattr(X_train, "to_numpy"):
        X_train = X_train.to_numpy()
    return np.asarray(X_train, dtype=float)


def class_index(model, label):
    matches = np.where(model.classes_ == label)[0]
    return int(matches[0]) if len(matches) else None


def serialize_query(names, query_vector):
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


def ensure_supported_tree_task(tree, task_type):
    if task_type == "regression" and tree.value.shape[1] != 1:
        raise NotImplementedError("Multi-output regression is not supported")


def regression_value(tree, node):
    return float(tree.value[node][0][0])


def serialize_sklearn_tree(tree, feature_names, task_type="classification"):
    ensure_supported_tree_task(tree, task_type)
    nodes = []
    for i in range(tree.node_count):
        is_leaf = bool(tree.children_left[i] == -1)
        node = {
            "node_id": i,
            "n_samples": int(tree.n_node_samples[i]),
            "impurity": round(float(tree.impurity[i]), 3),
            "leaf": is_leaf,
        }
        if task_type == "classification":
            node["gini"] = node["impurity"]
        if is_leaf:
            node.update(
                {
                    "feature": None,
                    "threshold": None,
                    "left_child": None,
                    "right_child": None,
                }
            )
            if task_type == "classification":
                node.update({
                    "counts": tree.value[i][0].astype(int).tolist(),
                    "prediction": int(tree.value[i][0].argmax()),
                })
            else:
                node["prediction_value"] = regression_value(tree, i)
        else:
            node.update({
                "feature": feature_names[tree.feature[i]],
                "threshold": round(float(tree.threshold[i]), 3),
                "left_child": int(tree.children_left[i]),
                "right_child": int(tree.children_right[i]),
            })
        nodes.append(node)
    return nodes


def trace_sklearn_path(tree, feature_names, X, task_type="classification"):
    ensure_supported_tree_task(tree, task_type)
    node = 0
    path = []
    while tree.children_left[node] != -1:
        feat = tree.feature[node]
        thr = tree.threshold[node]
        went_left = bool(X[feat] <= thr)
        path.append({
            "node_id": node,
            "feature": feature_names[feat],
            "threshold": round(float(thr), 3),
            "value": float(X[feat]),
            "went_left": went_left,
        })
        node = int(tree.children_left[node] if went_left else tree.children_right[node])
    leaf = {
        "node_id": node,
        "leaf": True,
    }
    if task_type == "classification":
        leaf.update({
            "counts": tree.value[node][0].astype(int).tolist(),
            "prediction": int(tree.value[node][0].argmax()),
        })
    else:
        leaf["prediction_value"] = regression_value(tree, node)
    path.append(leaf)
    return path
