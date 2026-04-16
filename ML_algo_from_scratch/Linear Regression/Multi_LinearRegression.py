import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Define our custom Multiple Linear Regression class
class MyMultiLinearRegression:

    def __init__(self):
        self.coef_ = None         # Stores weights for features (B1 to Bn)
        self.intercept_ = None    # Stores bias/intercept (B0)

    def fit(self, X_train, y_train):
        # Step 1: Add a column of 1s to X to handle the intercept (B0)
        # This converts X shape from (m x n) to (m x n+1)
        X_train = np.insert(X_train, 0, 1, axis=1)

        # Step 2: Apply the closed-form solution
        # beta = (X^T X)^-1 X^T Y
        # @ is used for matrix multiplication
        betas = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

        # Step 3: Separate intercept and coefficients
        self.intercept_ = betas[0]      # B0
        self.coef_ = betas[1:]          # B1 to Bn

    def predict(self, X_test):
        # Make predictions using: Y = X·B + B0
        return X_test @ self.coef_ + self.intercept_

# Load a real-world dataset (Diabetes dataset from scikit-learn)
X, y = load_diabetes(return_X_y=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Initialize and train our custom regression model
lr = MyMultiLinearRegression()
lr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lr.predict(X_test)

# Evaluate model performance using R² Score
print("R² Score:", r2_score(y_test, y_pred))