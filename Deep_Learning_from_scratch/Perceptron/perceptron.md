# Perceptron

The Perceptron is a fundamental algorithm in machine learning, serving as the building block for neural networks. It is a binary classifier that learns a linear decision boundary to separate two classes.

## Basic Perceptron

The basic perceptron algorithm iteratively updates weights and bias to correctly classify training examples. It uses a step function as the activation.

### Implementation

The `simple_perceptron.py` file implements the basic perceptron:

- **Initialization**: Weights and bias are initialized to zero.
- **Training**: For each epoch, iterate through samples. Compute linear output, apply sign function. If prediction differs from true label, update weights and bias.
- **Prediction**: Compute linear output and apply sign function.

Key points:
- Labels are converted to -1 and 1 for consistency.
- Learning rate controls update step size.
- Epochs determine training iterations.

## Perceptron with Loss Function

The `percepton_with_loss_function.py` implements a variant of the perceptron that incorporates a margin-based update, similar to hinge loss minimization.

### Implementation

- **Initialization**: Similar to basic perceptron.
- **Training**: Compute margin = y_i * (x_i @ w + b). If margin < 1, update weights and bias.
- **Prediction**: Use sign function on linear output, but return 1 or -1 based on >=0.

This variant updates even when the prediction is correct but the margin is small, leading to a more robust decision boundary.

## Testing

Both implementations are tested using scikit-learn's `make_blobs` to generate synthetic binary classification data. The basic perceptron fits and predicts, while the loss function variant also computes accuracy.

## Mathematical Background

The perceptron learns weights w and bias b such that for input x, the prediction is sign(w · x + b).

Update rule for basic perceptron:
- If y_pred ≠ y_true, then w += lr * y_true * x, b += lr * y_true

For the loss function variant, it's a soft-margin approach updating when margin < 1.