# Random Forest from Scratch

Random Forest is an ensemble learning algorithm that builds a large collection
of decision trees and combines their predictions. It is one of the most
powerful and widely used algorithms in classical machine learning.

---

## The Problem with a Single Decision Tree

A single decision tree suffers from **high variance**: it tends to memorise
the training data (overfitting) and performs poorly on unseen data. Small
changes in the training set can produce drastically different trees.

Random Forest solves this by building many trees and averaging their
predictions, which dramatically reduces variance while keeping bias low.

---

## The Two Sources of Randomness

Random Forest introduces randomness at two levels to ensure the trees are
decorrelated from one another. Decorrelated trees reduce variance more than
correlated ones when averaged.

### 1. Bootstrap Sampling (Bagging)

Each tree is trained on a **bootstrap sample**: a random sample of the
training data drawn with replacement.

- The bootstrap sample has the same size as the original dataset.
- On average, approximately 63.2% of training samples appear at least once.
- The remaining ~36.8% of samples (not selected) are called
  **Out-of-Bag (OOB) samples** and can be used for validation.

```
For each tree i:
    Draw a bootstrap sample B_i from the training data (with replacement)
    Train tree_i on B_i
```

### 2. Random Feature Subsets at Each Split

At every node of every tree, instead of searching all features, only a
random subset of features is considered for splitting.

- For classification: default is `sqrt(n_features)` features.
- For regression:     default is `n_features / 3` features.

This ensures that different trees focus on different aspects of the data,
reducing correlation between them.

---

## Ensemble Prediction

### Classification: Majority Vote

Each tree independently predicts a class for the test sample. The final
prediction is the class that receives the most votes:

```
y_pred = mode( tree_1(x), tree_2(x), ..., tree_T(x) )
```

### Regression: Averaging

Each tree predicts a continuous value. The final prediction is the mean
across all trees:

```
y_pred = (1/T) * sum( tree_t(x) )  for t in 1..T
```

Averaging reduces variance without increasing bias, which is the core
benefit of the bagging approach.

---

## Why Averaging Reduces Variance

Let each tree have variance sigma^2 and all trees be uncorrelated.
The average of T uncorrelated trees has variance:

```
Var(average) = sigma^2 / T
```

In practice, trees are not perfectly uncorrelated (rho > 0), so the actual
variance is:

```
Var(average) = rho * sigma^2 + (1 - rho) * sigma^2 / T
```

The two randomisation steps (bootstrap + random features) reduce `rho`,
which reduces the overall ensemble variance.

---

## Key Hyperparameters

| Hyperparameter    | Effect                                                     |
|-------------------|------------------------------------------------------------|
| n_estimators      | More trees = lower variance, but slower. Use 100 - 500.    |
| max_depth         | Controls each tree's depth. Deeper = lower bias, more cost.|
| max_features      | Fewer features = more diverse trees, less variance.        |
| min_samples_split | Minimum node size before splitting. Higher = more pruning. |

---

## Feature Importance

Random Forests can estimate the importance of each feature by measuring how
much the impurity (Gini or MSE) decreases on average when a feature is used
for splitting, weighted by the fraction of samples at each node. This is not
implemented here but is a powerful built-in feature of the algorithm.

---

## Out-of-Bag (OOB) Error Estimation

Because each tree is trained on only ~63% of the data, the remaining 37%
(OOB samples) can be used to estimate test error without a separate validation
set. This gives a nearly unbiased estimate of the generalisation error and
is a significant practical advantage of the Random Forest algorithm.

---

## Complexity

| Phase      | Complexity                          |
|------------|-------------------------------------|
| Training   | O(T * m * sqrt(n) * log(m))         |
| Prediction | O(T * depth)  per sample            |

where `T` = number of trees, `m` = samples, `n` = features.

---

## Advantages and Limitations

### Advantages
- Extremely robust to overfitting compared to a single decision tree.
- Handles high-dimensional data well.
- No feature scaling required.
- Provides feature importance estimates.
- Parallelisable: each tree can be built independently.
- OOB error estimate avoids the need for a separate validation set.

### Limitations
- Less interpretable than a single decision tree.
- Slower to train and predict than simpler models.
- Not well-suited for very sparse data (e.g. text).
- Memory intensive: must store all T trees.

---

## Files

| File                | Description                                               |
|---------------------|-----------------------------------------------------------|
| `random_forest.py`  | Classifier and Regressor built on top of custom Decision Trees |

---

## How to Run

```bash
pip install numpy scikit-learn
python random_forest.py
```

---

## References

- Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.
- [Random Forests - Wikipedia](https://en.wikipedia.org/wiki/Random_forest)
- [Bagging - Wikipedia](https://en.wikipedia.org/wiki/Bootstrap_aggregating)
- [Out-of-Bag Error Estimation](https://en.wikipedia.org/wiki/Out-of-bag_error)
