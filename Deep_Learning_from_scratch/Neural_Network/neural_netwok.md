# Neural Network Implementations from Scratch

This folder contains manual implementations of Neural Networks using both NumPy and PyTorch to understand the underlying mechanics of forward propagation, backpropagation, and gradient updates.

## Implementations

### 1. NumPy Implementation (`neural_netwok_numpy.py`)
A 2-layer Neural Network built for **Regression** problems using only `numpy`.
- **Architecture**: Input Layer -> Hidden Layer (with Sigmoid) -> Output Layer (Linear).
- **Activation**: Sigmoid function and its derivative.
- **Loss Function**: Mean Squared Error (MSE).
- **Optimization**: Manual Gradient Descent.
- **Task**: Predicts a simple sum relationship ($y = x_1 + x_2$).

### 2. PyTorch Implementation (`neural_netwok_pytorch.py`)
A 2-layer Neural Network built for **Binary Classification** using `torch` tensors.
- **Architecture**: Input Layer -> Hidden Layer (with ReLU) -> Output Layer (Sigmoid).
- **Activation**: ReLU for the hidden layer and Sigmoid for the output.
- **Loss Function**: Binary Cross-Entropy (BCE) with manual implementation.
- **Optimization**: Manual weight updates using `.grad` without using `torch.optim`.
- **Dataset**: Tested on the **Titanic dataset** to predict survival.
- **Preprocessing**: Includes handling missing values, encoding categorical features, and standard scaling.

## Key Concepts Covered
- Manual Forward and Backward Propagation.
- Chain rule application for gradient calculation.
- Weight and bias initialization.
- Handling real-world data (Titanic) in a deep learning context.
- Differences between NumPy-based and Tensor-based (PyTorch) implementations.

## How to Run
1. Ensure you have the required dependencies: `pip install numpy torch pandas scikit-learn`
2. Run the NumPy version: `python neural_netwok_numpy.py`
3. Run the PyTorch version: `python neural_netwok_pytorch.py`