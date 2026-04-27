# Perceptron Implementations from Scratch

The Perceptron is the simplest form of a neural network, used for binary classification tasks. This folder contains two variations of the Perceptron algorithm implemented using NumPy.

## Implementations

### 1. Simple Perceptron (`simple_perceptron.py`)
A classic implementation of the Perceptron Learning Rule.
- **Goal**: Find a linear decision boundary to separate two classes.
- **Activation**: Step function (Sign function).
- **Labels**: Converts binary labels to -1 and 1 to simplify the update rule.
- **Update Rule**: If a point is misclassified ($y_{pred} \neq y_{true}$), weights and bias are updated:
  - $w = w + \eta \cdot y_{true} \cdot x$
  - $b = b + \eta \cdot y_{true}$
- **Dataset**: Tested on synthetic data generated using `make_blobs`.

### 2. Perceptron with Loss Function (`percepton_with_loss_function.py`)
A more robust implementation using a margin-based approach (similar to Hinge Loss).
- **Goal**: Ensure that data points are not just correctly classified, but also satisfy a minimum margin requirement.
- **Condition**: Updates occur if $y_i \cdot (x_i \cdot w + b) < 1$.
- **Robustness**: This approach tends to find a better decision boundary by continuing to update even when points are technically on the correct side of the line but too close to it.
- **Dataset**: Tested on synthetic data generated using `make_blobs` with a higher number of epochs.

## Key Concepts
- Linear separability.
- Feature scaling and label transformation.
- Impact of learning rate and epochs on convergence.
- Difference between basic mistake-driven updates and margin-driven loss minimization.

## How to Run
1. Ensure dependencies are installed: `pip install numpy scikit-learn`
2. Run the simple version: `python simple_perceptron.py`
3. Run the loss-based version: `python percepton_with_loss_function.py`