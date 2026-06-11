# Decision Tree from Scratch

A Decision Tree is a non-parametric supervised learning algorithm that learns
a hierarchy of binary rules from training data. It can be applied to both
classification and regression tasks.

---

## The Core Idea

A decision tree recursively splits the training data into progressively purer
subsets. At each internal node, the algorithm asks a question of the form:

```
Is feature[j] <= threshold?
```

Samples satisfying the condition go left; the rest go right. This continues
until a stopping criterion is met, at which point a leaf node is created that
stores a prediction.

---

## The CART Algorithm

The implementation here follows CART (Classification and Regression Trees),
which always produces binary trees. It uses a greedy, top-down approach:
at each node, try every possible feature and every possible threshold, and
pick the split that minimises a chosen impurity measure.

---

## Decision Tree Classifier

### Impurity Measure: Gini Impurity

Gini Impurity quantifies how often a randomly chosen element would be
incorrectly classified if it were labelled according to the distribution of
labels in the node.

```
Gini(y) = 1 - sum( p_k^2 )   for each class k
```

- Gini = 0   : Node is perfectly pure (contains only one class).
- Gini = 0.5 : Maximum impurity for a two-class problem (50/50 split).

### Weighted Gini After a Split

When we split a node into left (L) and right (R) children:

```
Gini_split = (|L| / |parent|) * Gini(L) + (|R| / |parent|) * Gini(R)
```

We select the (feature, threshold) pair that minimises `Gini_split`.

### Leaf Prediction

At a leaf, the predicted class is the **majority class** among all training
samples that fell into that node.

---

## Decision Tree Regressor

### Impurity Measure: Mean Squared Error (Variance)

For regression, the goal is to reduce the variance of the target values
within each node.

```
MSE(y) = (1/m) * sum( (y_i - mean(y))^2 )  = Var(y)
```

We pick the split that minimises the weighted average of the children's variances:

```
MSE_split = (|L| / |parent|) * Var(L) + (|R| / |parent|) * Var(R)
```

### Leaf Prediction

At a leaf, the predicted value is the **mean** of all target values that
reached that node.

---

## Tree Growth and Stopping Conditions

The recursive splitting stops when any of the following are true:

| Condition                    | Reason                                      |
|------------------------------|---------------------------------------------|
| Depth reaches max_depth      | Prevents overfitting by limiting tree size  |
| Node has fewer than min_samples_split samples | Node is too small to split reliably |
| Node is perfectly pure (classification) | Nothing to gain by splitting further |
| No valid split exists        | All feature values in the node are identical|

---

## Overfitting and Depth Control

An unconstrained decision tree memorises the training data perfectly
(zero training error) but generalises poorly to new data. This is the
classic **variance problem** (overfitting).

**Controls:**
- `max_depth`: Hard limit on tree depth.
- `min_samples_split`: Requires a minimum node size before splitting.

Deeper trees = higher variance. Shallower trees = higher bias.

---

## Complexity

| Operation   | Complexity            |
|-------------|-----------------------|
| Training    | O(m * n * m * log m)  |
| Prediction  | O(depth)  per sample  |

where `m` = samples, `n` = features.

---

## Advantages and Limitations

### Advantages
- No feature scaling required (splits are invariant to monotone transforms).
- Handles both numeric and categorical features.
- Highly interpretable: the tree can be visualised and reasoned about.
- Captures non-linear decision boundaries naturally.

### Limitations
- High variance: small changes in data can lead to a completely different tree.
- Greedy splitting does not guarantee a globally optimal tree.
- Prone to overfitting without proper regularisation.

---

## Files

| File                | Description                                          |
|---------------------|------------------------------------------------------|
| `decision_tree.py`  | Classifier and Regressor implementations using NumPy |

---

## How to Run

```bash
pip install numpy scikit-learn
python decision_tree.py
```

---

## References

- Breiman, L. et al. (1984). *Classification and Regression Trees*. Wadsworth.
- [CART Algorithm - Wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning)
- [Gini Impurity - Wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity)
