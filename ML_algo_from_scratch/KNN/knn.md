# K-Nearest Neighbors (KNN) from Scratch

K-Nearest Neighbors is one of the simplest and most intuitive machine learning
algorithms. It is a non-parametric, instance-based (lazy) learning method that
makes predictions by finding the K most similar training examples to a new
query point.

---

## The Core Idea

KNN is based on a single assumption:

> Samples that are close to each other in feature space tend to have similar outputs.

Given a new test sample, the algorithm looks at the K training samples closest
to it and uses their labels to make a prediction.

---

## Algorithm Steps

### Training Phase

KNN has no explicit training phase. The entire training dataset is stored
in memory. This is why KNN is called a **lazy learner**: it defers all
computation to prediction time.

### Prediction Phase

For each test sample `x`:

1. Compute the distance between `x` and every training sample.
2. Sort the training samples by distance (ascending).
3. Select the K samples with the smallest distances (the K nearest neighbors).
4. For classification: return the **majority vote** class among the K neighbors.
5. For regression: return the **mean** of the K neighbors' target values.

---

## Distance Metrics

The choice of distance metric significantly affects which neighbors are selected.

### Euclidean Distance (default)

The straight-line distance between two points in n-dimensional space:

```
d(a, b) = sqrt( sum( (a_i - b_i)^2 ) )
```

### Manhattan Distance

The sum of absolute differences along each dimension (also called L1 distance):

```
d(a, b) = sum( |a_i - b_i| )
```

Manhattan distance is more robust to outliers in individual dimensions.

---

## Choosing K

The value of K is the most important hyperparameter:

| K value     | Effect                                                        |
|-------------|---------------------------------------------------------------|
| Small K (1) | Very flexible boundary, high variance, prone to overfitting   |
| Large K     | Smooth boundary, high bias, may underfit                      |
| Odd K       | Preferred for binary classification to break ties             |

A common heuristic is to start with `K = sqrt(m)` where m is the number of
training samples, then tune using cross-validation.

---

## Why Feature Scaling is Critical

KNN relies entirely on distances. If one feature has a much larger range
than others, it will dominate the distance calculation. For example:

- Feature A: ranges from 0 to 1
- Feature B: ranges from 0 to 10,000

Without scaling, Feature B would almost entirely determine which neighbors
are selected, making Feature A irrelevant.

**Solution**: Always standardise features to zero mean and unit variance before
applying KNN:

```
x_scaled = (x - mean(x)) / std(x)
```

---

## Computational Complexity

| Phase      | Complexity             | Notes                            |
|------------|------------------------|----------------------------------|
| Training   | O(1)                   | Just stores the dataset          |
| Prediction | O(m * n) per sample    | m samples, n features            |
| Total test | O(q * m * n)           | q = number of test samples       |

KNN is slow at prediction time on large datasets. For production use,
data structures such as KD-Trees or Ball Trees are used to speed up the
nearest-neighbor search.

---

## Advantages and Limitations

### Advantages
- Trivial to understand and implement.
- No assumptions about the data distribution (non-parametric).
- Naturally handles multi-class classification.
- Adapts well to complex, non-linear decision boundaries.

### Limitations
- Extremely slow at prediction time for large datasets.
- High memory usage: the entire training set must be stored.
- Performance degrades significantly in high-dimensional spaces
  (the Curse of Dimensionality).
- Sensitive to irrelevant and redundant features.
- Requires feature scaling.

---

## Curse of Dimensionality

As the number of features increases, the volume of the feature space grows
exponentially. In very high-dimensional spaces:

- All points tend to become roughly equidistant from any query point.
- The concept of "nearest neighbor" loses meaning.
- A much larger number of training samples is needed to cover the space densely.

---

## Files

| File     | Description                                                  |
|----------|--------------------------------------------------------------|
| `knn.py` | Classifier and Regressor implementations using NumPy         |

---

## How to Run

```bash
pip install numpy scikit-learn
python knn.py
```

---

## References

- Cover, T. M., & Hart, P. E. (1967). *Nearest neighbor pattern classification*. IEEE Transactions on Information Theory.
- [KNN - Wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- [Curse of Dimensionality - Wikipedia](https://en.wikipedia.org/wiki/Curse_of_dimensionality)
