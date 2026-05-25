---
tags: [mlviz, python, v1, theory, decision-tree]
status: complete
---

# Decision Tree — Theory & sklearn Internals

## What a Decision Tree Does

A decision tree classifies a data point by asking a series of yes/no questions about its features. At each node it checks one feature against a threshold. The point goes left if the condition is true (`value <= threshold`), right if false. It keeps going until it hits a leaf, which holds the final prediction.

## Gini Impurity

Measures how mixed a group is. **0 = perfectly pure. 0.5 = totally mixed** (binary case).

```
Gini = 1 - (proportion of class A)² - (proportion of class B)²
```

| Example | Calculation | Gini |
|---------|------------|------|
| 5 cheap, 5 expensive out of 10 | 1 − 0.25 − 0.25 | **0.5** |
| 9 cheap, 1 expensive out of 10 | 1 − 0.81 − 0.01 | **0.18** |
| 10 cheap, 0 expensive | 1 − 1.0 − 0.0 | **0.0** |

## Information Gain

How much a split improved things. The tree tries every feature at every threshold and picks the one with the highest IG — that becomes the node.

```
IG = Gini(parent) − Weighted Gini(children)
```

Weighted Gini = each child's Gini weighted by its share of the parent's samples.

**Example — 10 houses, split on size ≤ 100m²:**

```
Parent: 5 cheap, 5 expensive → Gini = 0.5

Left  (size ≤ 100): 7 samples — 6 cheap, 1 expensive → Gini = 0.245
Right (size > 100): 3 samples — 0 cheap, 3 expensive → Gini = 0.0

Weighted Gini = (7/10) × 0.245 + (3/10) × 0.0 = 0.172

IG = 0.5 − 0.172 = 0.328
```

IG range: **0 to 0.5**. Higher = better split.

## Nodes vs Leaf Nodes

- **Node** — asks a question (`feature <= threshold`), splits data left or right.
- **Leaf node** — end of the line. No more splits. Holds the final class prediction.

```
        [petal_length ≤ 2.45?]        ← node
             /           \
      [leaf: setosa]   [petal_width ≤ 1.75?]   ← node
                          /         \
                 [versicolor]    [virginica]     ← leaves
```

## sklearn Internals — Flat Parallel Arrays

sklearn stores the entire tree as flat parallel arrays, one entry per node. Index 0 is always the root.

```python
tree = model.tree_

tree.node_count        # total number of nodes
tree.feature[i]        # feature index split on at node i  (-2 = leaf)
tree.threshold[i]      # threshold at node i               (-2.0 = leaf)
tree.children_left[i]  # node ID of left child             (-1 = leaf)
tree.children_right[i] # node ID of right child            (-1 = leaf)
tree.n_node_samples[i] # training samples that reached node i
tree.impurity[i]       # Gini impurity at node i
tree.value[i]          # class counts, shape [1, n_classes]
                       # e.g. [[6., 1., 0.]] = 6 class-0, 1 class-1, 0 class-2
```

**Leaf identification:** `children_left[i] == -1`

## Feature Importance

`model.feature_importances_` sums the total information gain each feature contributed across every node it appeared at, weighted by sample count. Features that produce the biggest gains end up near the root.

## Overfitting

Deeper tree → more splits → memorises training data → overfits. `max_depth` is the main lever. The mlviz visualiser makes this visible — you can see how the tree gets busier with depth.

## Related Notes

- [[Extractor - DT]] — how the extractor reads these arrays and serialises to JSON
