import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class MyLogisticRegression:
    """
    Binary Logistic Regression implemented from scratch using gradient descent.

    The model learns a decision boundary by optimizing the binary
    cross-entropy (log loss) via iterative gradient descent.

    Parameters
    ----------
    learning_rate : float
        Step size for each gradient descent update. Default is 0.01.
    epochs : int
        Number of full passes over the training dataset. Default is 1000.
    """

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None   # Feature weights (w)
        self.bias = None      # Bias term (b)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sigmoid(self, z):
        """
        Applies the sigmoid activation function element-wise.

        sigmoid(z) = 1 / (1 + e^(-z))

        Maps any real number to the open interval (0, 1), which we
        interpret as a probability.
        """
        return 1 / (1 + np.exp(-z))

    def _compute_loss(self, y_true, y_pred):
        """
        Computes binary cross-entropy loss (log loss).

        Loss = -(1/m) * sum[ y*log(y_hat) + (1-y)*log(1-y_hat) ]

        A small epsilon is added for numerical stability to avoid log(0).
        """
        m = len(y_true)
        epsilon = 1e-9
        loss = -(1 / m) * np.sum(
            y_true * np.log(y_pred + epsilon)
            + (1 - y_true) * np.log(1 - y_pred + epsilon)
        )
        return loss

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_train, y_train):
        """
        Trains the logistic regression model using gradient descent.

        Steps per epoch:
          1. Compute linear combination:  z = X·w + b
          2. Apply sigmoid:               y_hat = sigmoid(z)
          3. Compute gradients:
                dw = (1/m) * X^T · (y_hat - y)
                db = (1/m) * sum(y_hat - y)
          4. Update parameters:
                w = w - lr * dw
                b = b - lr * db

        Parameters
        ----------
        X_train : np.ndarray, shape (m, n)
        y_train : np.ndarray, shape (m,)  -- binary labels {0, 1}
        """
        m, n = X_train.shape

        # Initialise weights to zero and bias to zero
        self.weights = np.zeros(n)
        self.bias = 0.0

        for epoch in range(self.epochs):
            # Forward pass
            z = X_train @ self.weights + self.bias
            y_hat = self._sigmoid(z)

            # Compute gradients
            error = y_hat - y_train
            dw = (1 / m) * (X_train.T @ error)
            db = (1 / m) * np.sum(error)

            # Parameter update
            self.weights -= self.learning_rate * dw
            self.bias    -= self.learning_rate * db

            # Log progress every 100 epochs
            if (epoch + 1) % 100 == 0:
                loss = self._compute_loss(y_train, y_hat)
                print(f"Epoch {epoch + 1:>5} | Loss: {loss:.6f}")

    def predict_proba(self, X):
        """
        Returns the predicted probability of class 1 for each sample.

        Parameters
        ----------
        X : np.ndarray, shape (m, n)

        Returns
        -------
        np.ndarray, shape (m,) with values in (0, 1)
        """
        z = X @ self.weights + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        """
        Returns hard binary predictions using a probability threshold.

        Samples with predict_proba >= threshold are labelled 1, else 0.

        Parameters
        ----------
        X         : np.ndarray, shape (m, n)
        threshold : float, default 0.5

        Returns
        -------
        np.ndarray of int, shape (m,)
        """
        return (self.predict_proba(X) >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Demo: Breast Cancer dataset (binary classification)
# ---------------------------------------------------------------------------

X, y = load_breast_cancer(return_X_y=True)

# Normalise features to zero mean and unit variance for stable gradient descent
X = (X - X.mean(axis=0)) / X.std(axis=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MyLogisticRegression(learning_rate=0.1, epochs=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
