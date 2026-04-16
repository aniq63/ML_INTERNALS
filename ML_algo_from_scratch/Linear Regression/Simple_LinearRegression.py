import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing_extensions import Self  # Optional for return typing

# Define a custom Linear Regression class
class MyLinearRegression:
    def __init__(self):
        self.coef_ = None      # Slope (m)
        self.intercept_ = None # Intercept (b)

    def fit(self, X_train, y_train):
        # Calculate slope (m) and intercept (b) using closed-form solution
        numerator = np.sum((X_train - np.mean(X_train)) * (y_train - np.mean(y_train)))
        denominator = np.sum((X_train - np.mean(X_train)) ** 2)

        self.coef_ = numerator / denominator
        self.intercept_ = np.mean(y_train) - (self.coef_ * np.mean(X_train))

        print(f"Slope (m): {self.coef_}")
        print(f"Intercept (b): {self.intercept_}")

    def predict(self, X_test):
        # Predict using the learned line equation y = mx + b
        return self.coef_ * X_test + self.intercept_

# Load dataset
df = pd.read_csv("/content/placement.csv")

# Separate feature and target
X = df["cgpa"]
Y = df["package"]

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Initialize and train our custom Linear Regression model
lr = MyLinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
r2_score(y_test, y_pred))