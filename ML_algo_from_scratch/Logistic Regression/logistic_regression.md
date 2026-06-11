# Logistic Regression from Scratch

Logistic Regression is a supervised learning algorithm used for binary
classification. Despite its name, it is a classification algorithm, not a
regression one. It models the probability that a sample belongs to a
particular class.

---

## The Core Idea

Linear Regression predicts a continuous output. For classification we need
the output bounded between 0 and 1 so it can be interpreted as a probability.
We achieve this by passing the linear output through a **sigmoid function**.

---

## Mathematical Foundation

### Step 1 - Linear Combination

Just like Linear Regression, we compute a weighted sum of the input features:

```
z = w1*x1 + w2*x2 + ... + wn*xn + b
z = X . w + b
```

where `w` is the weight vector and `b` is the bias term.

### Step 2 - Sigmoid Activation

The linear output `z` is passed through the sigmoid function to squash it
into the range (0, 1):

```
sigmoid(z) = 1 / (1 + e^(-z))
```

The output `y_hat = sigmoid(z)` represents P(y = 1 | X).

### Step 3 - Decision Boundary

A threshold (default 0.5) converts the probability to a hard label:

```
y_pred = 1  if y_hat >= 0.5
y_pred = 0  if y_hat < 0.5
```

---

## Loss Function - Binary Cross-Entropy

Mean Squared Error does not work well here because the sigmoid makes the
loss surface non-convex. Instead we use **Binary Cross-Entropy (Log Loss)**:

```
Loss = -(1/m) * sum[ y * log(y_hat) + (1 - y) * log(1 - y_hat) ]
```

- When `y = 1`:  Loss = `-log(y_hat)`. Penalises heavily if y_hat is near 0.
- When `y = 0`:  Loss = `-log(1 - y_hat)`. Penalises heavily if y_hat is near 1.

This function is convex, so gradient descent is guaranteed to find the
global minimum.

---

## Training - Gradient Descent

We minimise the loss by iteratively updating the weights in the direction of
the negative gradient.

### Gradients

```
dL/dw = (1/m) * X^T . (y_hat - y)
dL/db = (1/m) * sum(y_hat - y)
```

### Update Rules

```
w = w - learning_rate * dw
b = b - learning_rate * db
```

This process repeats for a fixed number of epochs until the loss converges.

---

## Feature Scaling

Gradient descent converges significantly faster when all features are on a
similar scale. It is standard practice to normalise input features to have
zero mean and unit variance before training:

```
x_scaled = (x - mean(x)) / std(x)
```

---

## Key Hyperparameters

| Hyperparameter  | Role                                                        |
|-----------------|-------------------------------------------------------------|
| learning_rate   | Controls the size of each gradient descent step             |
| epochs          | Number of complete passes over the training data            |
| threshold       | Probability cutoff for assigning the positive class label   |

---

## Complexity

| Operation | Complexity     |
|-----------|----------------|
| Training  | O(m * n * e)   |
| Prediction| O(m * n)       |

where `m` = number of samples, `n` = number of features, `e` = number of epochs.

---

## Advantages and Limitations

### Advantages
- Probabilistic output (not just a class label).
- Computationally inexpensive and interpretable.
- Works well when the decision boundary is approximately linear.

### Limitations
- Assumes a linear relationship between features and the log-odds.
- Cannot learn complex, non-linear decision boundaries without feature engineering.
- Sensitive to outliers in the training data.

---

## Files

| File                      | Description                                      |
|---------------------------|--------------------------------------------------|
| `logistic_regression.py`  | Full implementation using NumPy and gradient descent |

---

## How to Run

```bash
pip install numpy scikit-learn
python logistic_regression.py
```

---

## References

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- [Sigmoid Function - Wikipedia](https://en.wikipedia.org/wiki/Sigmoid_function)
- [Cross Entropy - Wikipedia](https://en.wikipedia.org/wiki/Cross_entropy)
- [Scikit-learn Breast Cancer Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset)
