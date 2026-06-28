"""PCA projection helpers for non-tree model visualizations."""

import numpy as np
from sklearn.decomposition import PCA

from mlviz.extractors.shared import as_2d_array, class_index, json_value


GRID_SIZE = 80
FIT_LINE_SAMPLES = 200


def _grid_from_projection(projected):
    pad_x = max((projected[:, 0].max() - projected[:, 0].min()) * 0.08, 1e-6)
    pad_y = max((projected[:, 1].max() - projected[:, 1].min()) * 0.08, 1e-6)
    xs = np.linspace(projected[:, 0].min() - pad_x, projected[:, 0].max() + pad_x, GRID_SIZE)
    ys = np.linspace(projected[:, 1].min() - pad_y, projected[:, 1].max() + pad_y, GRID_SIZE)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    return xs, ys, grid


def _classification_regions(model, pca, projected_norm, scale_factor):
    xs, ys, grid_norm = _grid_from_projection(projected_norm)
    predictions = model.predict(pca.inverse_transform(grid_norm / scale_factor))
    cell_w = float((xs[-1] - xs[0]) / (GRID_SIZE - 1))
    cell_h = float((ys[-1] - ys[0]) / (GRID_SIZE - 1))
    return [
        {
            "wx": float(wx),
            "wy": float(wy),
            "predicted_class": json_value(label),
            "class_index": class_index(model, label),
            "cell_w": cell_w,
            "cell_h": cell_h,
        }
        for (wx, wy), label in zip(grid_norm, predictions)
    ]


def _margin_regions(model, pca, projected_norm, scale_factor):
    xs, ys, grid_norm = _grid_from_projection(projected_norm)
    grid_pca = grid_norm / scale_factor
    original = pca.inverse_transform(grid_pca)
    decision = model.decision_function(original)
    cell_w = float((xs[-1] - xs[0]) / (GRID_SIZE - 1))
    cell_h = float((ys[-1] - ys[0]) / (GRID_SIZE - 1))
    return [
        {
            "wx": float(wx),
            "wy": float(wy),
            "margin_sign": int(np.sign(d)),
            "cell_w": cell_w,
            "cell_h": cell_h,
        }
        for (wx, wy), d in zip(grid_norm, decision)
    ]


def base_projection(model, X_train, y_train, query, task_type):
    X = as_2d_array(X_train)
    if X.shape[1] < 2:
        return None, None, None, None, None

    pca = PCA(n_components=2)
    projected = pca.fit_transform(X)

    scale_factor = 3.0 / max(abs(projected[:, 0]).max(), 1e-6)
    projected_norm = projected * scale_factor

    query_vector = None if query is None else np.asarray(query, dtype=float).reshape(1, -1)
    query_point = None if query_vector is None else {
        "wx": float(pca.transform(query_vector)[0][0] * scale_factor),
        "wy": float(pca.transform(query_vector)[0][1] * scale_factor),
    }
    targets = np.asarray(y_train)
    return pca, projected_norm, query_point, targets, scale_factor


def knn_projection(model, X_train, y_train, query, task_type, neighbor_indices):
    pca, projected_norm, query_point, targets, scale_factor = base_projection(
        model, X_train, y_train, query, task_type
    )
    if pca is None:
        return None

    return {
        "task_type": task_type,
        "points": [
            {
                "training_index": int(i),
                "wx": float(wx),
                "wy": float(wy),
                "label_or_target": json_value(targets[i]) if task_type == "classification" else float(targets[i]),
                "class_index": class_index(model, targets[i]) if task_type == "classification" else None,
            }
            for i, (wx, wy) in enumerate(projected_norm)
        ],
        "query_point": query_point,
        "neighbor_indices": [int(index) for index in neighbor_indices],
        "regions": (
            _classification_regions(model, pca, projected_norm, scale_factor)
            if task_type == "classification"
            else None
        ),
    }


def svr_fit_line(model, pca, projected_norm, scale_factor):
    xs = np.linspace(projected_norm[:, 0].min(), projected_norm[:, 0].max(), FIT_LINE_SAMPLES)
    grid_norm = np.column_stack([xs, np.zeros_like(xs)])
    predictions = model.predict(pca.inverse_transform(grid_norm / scale_factor))
    return [
        {
            "wx": float(wx),
            "wy": float(prediction),
        }
        for wx, prediction in zip(xs, predictions)
    ]


def svm_projection(model, X_train, y_train, query, task_type):
    pca, projected_norm, query_point, targets, scale_factor = base_projection(
        model, X_train, y_train, query, task_type
    )
    if pca is None:
        return None

    support_indices = {int(index) for index in model.support_}
    projection = {
        "task_type": task_type,
        "points": [
            {
                "training_index": int(i),
                "wx": float(wx),
                "wy": float(wy),
                "label_or_target": json_value(targets[i]) if task_type == "classification" else float(targets[i]),
                "class_index": class_index(model, targets[i]) if task_type == "classification" else None,
                "is_support_vector": int(i) in support_indices,
            }
            for i, (wx, wy) in enumerate(projected_norm)
        ],
        "query_point": query_point,
        "regions": (
            _classification_regions(model, pca, projected_norm, scale_factor)
            if task_type == "classification"
            else None
        ),
        "fit_line": None,
        "is_binary": task_type == "classification" and len(model.classes_) == 2,
        "margin_regions": (
            _margin_regions(model, pca, projected_norm, scale_factor)
            if task_type == "classification" and len(model.classes_) == 2
            else None
        ),
    }
    if task_type == "regression":
        projection["fit_line"] = svr_fit_line(model, pca, projected_norm, scale_factor)
    return projection
