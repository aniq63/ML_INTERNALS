# Linear Regression Implementations from Scratch

This repository contains educational implementations of **Simple Linear Regression** and **Multiple Linear Regression** from scratch using Python and NumPy. It demonstrates how linear models can be trained using the **closed-form solution (Normal Equation)**, without relying on external machine learning libraries for the core logic.

## 📌 Features

- **Custom Implementations**:
  - `MyLinearRegression`: For simple linear regression (single feature). Uses the mathematical formulas for slope (m) and intercept (b).
  - `MyMultiLinearRegression`: For multiple linear regression (n-dimensional features). Uses the **Normal Equation**: $\beta = (X^T X)^{-1} X^T y$.
- **Real-world Testing**:
  - `placement.csv`: Used for single-variable regression (predicting job package from CGPA).
  - `Diabetes` dataset: Used for multivariable regression from `sklearn.datasets`.
- **Performance Evaluation**: Uses R² Score to measure the goodness of fit.
- **Visualization**: Includes `matplotlib` integration for plotting regression lines in the 2D case.

---

## 🧠 Code Structure

### 1. Simple Linear Regression (`Simple_LinearRegression.py`)
Implementation for one feature using the formula:
$$m = \frac{\sum{(x - \bar{x})(y - \bar{y})}}{\sum{(x - \bar{x})^2}}, \quad b = \bar{y} - m \cdot \bar{x}$$

### 2. Multiple Linear Regression (`Multi_LinearRegression.py`)
General implementation for $n$ features using matrix operations:
$$\beta = (X^T X)^{-1} X^T y$$
This implementation automatically handles the intercept by adding a column of ones to the feature matrix.

---

## 🧪 How to Run

1. **Install Dependencies**:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```

2. **Run Simple Linear Regression**:
   ```bash
   python Simple_LinearRegression.py
   ```
   *Note: Ensure `placement.csv` is available in the expected path.*

3. **Run Multiple Linear Regression**:
   ```bash
   python Multi_LinearRegression.py
   ```

---

## 📊 Expected Results

### Simple Linear Regression
- **Feature**: CGPA
- **Target**: Placement Package (LPA)
- **Output**: Calculated Slope and Intercept for the best-fit line.

### Multiple Linear Regression (Diabetes Dataset)
- **Features**: 10 clinical variables (age, sex, bmi, etc.)
- **Target**: Quantitative measure of disease progression.
- **Metric**: R² Score (typically around 0.518 for this dataset).

---

## 📈 Visualization
For the 2D case, the code generates a scatter plot of the test data points and overlays the predicted regression line to visualize the fit.

```python
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted Line')
plt.show()
```

---

## 🔗 References
- [NumPy Matrix Multiplication](https://numpy.org/doc/stable/reference/generated/numpy.dot.html)
- [Normal Equation - Wikipedia](https://en.wikipedia.org/wiki/Linear_least_squares#The_normal_equations)
- [Scikit-learn Diabetes Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)
