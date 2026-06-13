# ML Internals

A collection of machine learning algorithms and deep learning architectures implemented **completely from scratch** using Python and NumPy.

No high-level ML frameworks are used for the core logic. Every algorithm is built from mathematical first principles, with each implementation accompanied by a detailed explanation of the underlying theory.

---

## Purpose

This repository is intended for anyone who wants to go beyond calling `model.fit()` and understand what actually happens inside a machine learning algorithm. The implementations are written to be readable and educational, not optimised for production performance.

---

## Repository Structure

```
ML_INTERNALS/
|
|-- ML_algo_from_scratch/
|   |-- Linear Regression/
|   |   |-- Simple_LinearRegression.py
|   |   |-- Multi_LinearRegression.py
|   |   `-- linear_regression.md
|   |
|   |-- Logistic Regression/
|   |   |-- logistic_regression.py
|   |   `-- logistic_regression.md
|   |
|   |-- Decision Tree/
|   |   |-- decision_tree.py
|   |   `-- decision_tree.md
|   |
|   |-- KNN/
|   |   |-- knn.py
|   |   `-- knn.md
|   |
|   `-- Random Forest/
|       |-- random_forest.py
|       `-- random_forest.md
|
|-- Deep_Learning_from_scratch/
|   |-- Perceptron/
|   |   |-- simple_perceptron.py
|   |   |-- percepton_with_loss_function.py
|   |   `-- perceptron.md
|   |
|   `-- Neural_Network/
|       |-- neural_netwok_numpy.py
|       |-- neural_netwok_pytorch.py
|       `-- neural_netwok.md
|
`-- LLM_from_scratch/
    |-- gpt.py
    `-- gpt.md
```

---

## Classical ML Algorithms

Each algorithm is self-contained in its own folder with a Python implementation and a Markdown explanation.

### Linear Regression

| File | Description |
|------|-------------|
| `Simple_LinearRegression.py` | Closed-form solution for single-feature regression |
| `Multi_LinearRegression.py` | Normal equation for n-dimensional features |

**Core concept:** Finds the line (or hyperplane) that minimises the sum of squared residuals using the formula `beta = (X^T X)^-1 X^T y`.

**Read more:** [linear_regression.md](ML_algo_from_scratch/Linear%20Regression/linear_regression.md)

---

### Logistic Regression

| File | Description |
|------|-------------|
| `logistic_regression.py` | Binary classifier using gradient descent and sigmoid activation |

**Core concept:** Maps a linear combination of features through a sigmoid function to produce a probability. Trained by minimising binary cross-entropy loss via gradient descent.

**Key equations:**
```
sigmoid(z) = 1 / (1 + e^(-z))
Loss = -(1/m) * sum[ y*log(y_hat) + (1-y)*log(1-y_hat) ]
```

**Read more:** [logistic_regression.md](ML_algo_from_scratch/Logistic%20Regression/logistic_regression.md)

---

### Decision Tree

| File | Description |
|------|-------------|
| `decision_tree.py` | Classifier (Gini Impurity) and Regressor (MSE) using the CART algorithm |

**Core concept:** Recursively partitions the feature space by finding the split that minimises impurity. The tree is grown top-down using a greedy search over all features and thresholds.

**Key equations:**
```
Gini(y) = 1 - sum( p_k^2 )          (classification)
MSE(y)  = Var(y)                     (regression)
```

**Read more:** [decision_tree.md](ML_algo_from_scratch/Decision%20Tree/decision_tree.md)

---

### K-Nearest Neighbors (KNN)

| File | Description |
|------|-------------|
| `knn.py` | Classifier (majority vote) and Regressor (mean) using Euclidean/Manhattan distance |

**Core concept:** A lazy learning algorithm that defers all computation to prediction time. For each test sample, it finds the K closest training points and uses their labels to make a prediction.

**Key equations:**
```
Euclidean: d(a, b) = sqrt( sum( (a_i - b_i)^2 ) )
Manhattan: d(a, b) = sum( |a_i - b_i| )
```

**Read more:** [knn.md](ML_algo_from_scratch/KNN/knn.md)

---

### Random Forest

| File | Description |
|------|-------------|
| `random_forest.py` | Ensemble of decision trees with bootstrap sampling and random feature selection |

**Core concept:** Builds many decorrelated decision trees using two sources of randomness (bootstrap sampling + random feature subsets) and aggregates their predictions. Reduces the high variance of a single decision tree.

**Key mechanism:**
```
Training : n_estimators trees, each on a bootstrap sample, each split uses sqrt(n_features)
Classify : majority vote across all trees
Regress  : average output across all trees
```

**Read more:** [random_forest.md](ML_algo_from_scratch/Random%20Forest/random_forest.md)

---

## Deep Learning Architectures

### Perceptron

| File | Description |
|------|-------------|
| `simple_perceptron.py` | Classic Perceptron Learning Rule with step function activation |
| `percepton_with_loss_function.py` | Margin-based perceptron using Hinge Loss for a better decision boundary |

**Read more:** [perceptron.md](Deep_Learning_from_scratch/Perceptron/perceptron.md)

---

### Neural Network

| File | Description |
|------|-------------|
| `neural_netwok_numpy.py` | 2-layer network for regression built with NumPy, manual forward/backward pass |
| `neural_netwok_pytorch.py` | 2-layer binary classifier using raw PyTorch tensors, manual gradient updates, tested on Titanic dataset |

**Core concept:** Manually implements forward propagation, the chain rule for backpropagation, and weight updates without relying on any autograd engine or optimiser library.

**Read more:** [neural_netwok.md](Deep_Learning_from_scratch/Neural_Network/neural_netwok.md)

---

## Large Language Models (Transformer)

### GPT From Scratch

| File | Description |
|------|-------------|
| `gpt.py` | Decoder-only GPT Transformer trained on Tiny Shakespeare with PyTorch |

**Core concept:** A stack of Transformer decoder blocks, each containing Multi-Head Causal Self-Attention and a Feed-Forward Network. Trained with next-token prediction (cross-entropy loss) using AdamW.

**Key equations:**
```
Attention(Q, K, V) = softmax( QKᵀ / √d_k ) · V
h_t = o_t ⊙ tanh(c_t)
```

**Achieves:** Validation loss ~1.5, Perplexity ~4.5 (vs random baseline of 65)

**Read more:** [gpt.md](LLM_from_scratch/gpt.md)

---

## Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib scikit-learn
```

### Running an Algorithm

Navigate into any algorithm's folder and run the Python file directly:

```bash
# Example: Logistic Regression
python "ML_algo_from_scratch/Logistic Regression/logistic_regression.py"

# Example: Random Forest
python "ML_algo_from_scratch/Random Forest/random_forest.py"

# Example: Decision Tree
python "ML_algo_from_scratch/Decision Tree/decision_tree.py"

# Example: KNN
python "ML_algo_from_scratch/KNN/knn.py"
```

All scripts use built-in datasets from scikit-learn so there is no manual data download required.

---

## Algorithm Summary

| Algorithm | Type | Criterion | Dataset Used |
|-----------|------|-----------|-------------|
| Simple Linear Regression | Regression | Normal Equation | Placement CSV |
| Multiple Linear Regression | Regression | Normal Equation | Diabetes |
| Logistic Regression | Classification | Binary Cross-Entropy | Breast Cancer |
| Decision Tree Classifier | Classification | Gini Impurity | Iris |
| Decision Tree Regressor | Regression | MSE / Variance | Diabetes |
| KNN Classifier | Classification | Euclidean Distance | Iris |
| KNN Regressor | Regression | Euclidean Distance | Diabetes |
| Random Forest Classifier | Classification | Gini + Bootstrap | Breast Cancer |
| Random Forest Regressor | Regression | MSE + Bootstrap | Diabetes |
| Perceptron | Classification | Hinge Loss / Step | Synthetic Blobs |
| Neural Network (NumPy) | Regression | MSE, Manual Backprop | Synthetic |
| Neural Network (PyTorch) | Classification | Binary Cross-Entropy | Titanic |
| GPT (Transformer) | Language Model | Cross-Entropy, AdamW | Tiny Shakespeare |

---

## What is Implemented from Scratch

- No `sklearn` classes are used for model logic (only for datasets and evaluation metrics).
- No `torch`, `tensorflow`, or `keras`.
- All forward passes, loss computations, gradient calculations, weight updates, tree construction, distance calculations, and bootstrap procedures are written using only **NumPy** and **Python**.

---

## Design Philosophy

Each algorithm folder follows the same structure:

```
AlgorithmName/
|-- algorithm_name.py     # Clean, well-commented implementation
`-- algorithm_name.md     # Theory, math, and explanation
```

The Python files are structured as classes with a `fit` / `predict` API
consistent with the scikit-learn interface, making them easy to understand
for anyone familiar with the Python ML ecosystem.

---

## References

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.
- Breiman, L. et al. (1984). *Classification and Regression Trees*. Wadsworth.
- Cover, T. M., & Hart, P. E. (1967). *Nearest neighbor pattern classification*. IEEE Transactions on Information Theory.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

## Contributing

Contributions are welcome. If you would like to add a new algorithm, please follow the existing structure:

1. Create a new folder under the appropriate section:
   - `ML_algo_from_scratch/` for classical ML algorithms
   - `Deep_Learning_from_scratch/` for neural network architectures
   - `LLM_from_scratch/` for Transformer-based large language models
2. Add a clean Python implementation with docstrings and comments.
3. Add a Markdown explanation covering the theory, mathematics, and key concepts.
4. Update this README.
