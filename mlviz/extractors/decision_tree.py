def _extract_tree(tree, feature_names):
    nodes = []
    for i in range(tree.node_count):
        is_leaf = tree.children_left[i] == -1
        node = {
            "node_id": i,
            "n_samples": int(tree.n_node_samples[i]),
            "gini": round(float(tree.impurity[i]), 3),
            "leaf": is_leaf,
        }
        if is_leaf:
            node.update({
                "feature": None,
                "threshold": None,
                "left_child": None,
                "right_child": None,
                "counts": tree.value[i][0].astype(int).tolist(),
                "prediction": int(tree.value[i][0].argmax()),
            })
        else:
            node.update({
                "feature": feature_names[tree.feature[i]],
                "threshold": round(float(tree.threshold[i]), 3),
                "left_child": int(tree.children_left[i]),
                "right_child": int(tree.children_right[i]),
            })
        nodes.append(node)
    return nodes


def _trace_path(tree, feature_names, X):
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
    path.append({
        "node_id": node,
        "leaf": True,
        "counts": tree.value[node][0].astype(int).tolist(),
        "prediction": int(tree.value[node][0].argmax()),
    })
    return path
