from mlviz.extractors.shared import (
    normalize_query_vector,
    resolve_feature_names,
    serialize_sklearn_tree,
    trace_sklearn_path,
)

# Kept as module-level names: existing tests import these directly.
_extract_tree = serialize_sklearn_tree
_trace_path = trace_sklearn_path


def serialize(model, X_train, y_train, query=None, feature_names=None):
    names = resolve_feature_names(model, X_train, feature_names)
    query_vector = None if query is None else normalize_query_vector(query, len(names))
    nodes = serialize_sklearn_tree(model.tree_, names)
    path = None if query_vector is None else trace_sklearn_path(model.tree_, names, query_vector)
    feature_importances = sorted(
        [
            {
                "feature_index": i,
                "feature_name": names[i],
                "importance": float(v),
            }
            for i, v in enumerate(model.feature_importances_)
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )
    return {
        "model_type": "decision_tree",
        "classes": model.classes_.tolist(),
        "feature_names": names,
        "feature_importances": feature_importances,
        "nodes": nodes,
        "path": path,
    }
