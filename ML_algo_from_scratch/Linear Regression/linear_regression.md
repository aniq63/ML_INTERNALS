# Custom Linear and Multiple Linear Regression from Scratch in Python

This repository contains an educational implementation of **Simple Linear Regression** and **Multiple Linear Regression** from scratch using Python and NumPy. It demonstrates how linear models can be trained using the **closed-form solution (Normal Equation)**, without relying on external machine learning libraries like `scikit-learn`.

## 📌 Features

- Implemented two models:
  - `MyLinearRegression`: For **2D datasets** (single feature).
  - `MyMultiLinearRegression`: For **n-dimensional datasets** (multiple features).
- Uses real-world datasets:
  - `placement.csv` for single-variable regression (predicting salary from CGPA).
  - `Diabetes` dataset from `sklearn.datasets` for multivariable regression.
- Uses **closed-form solution**:  
  \[
  \beta = (X^T X)^{-1} X^T y
  \]
- Evaluates model performance using **R² Score**.
- Includes visualizations (in 2D case) using `matplotlib`.

---

## 🧠 Code Structure

### ✅ `MyLinearRegression` (for 2D data)

```python
class MyLinearRegression:
    def fit(self, X_train, y_train): ...
    def predict(self, X_test): ...
````

* Calculates the slope (m) and intercept (b) using:

  $$
  m = \frac{\sum{(x - \bar{x})(y - \bar{y})}}{\sum{(x - \bar{x})^2}}, \quad b = \bar{y} - m \cdot \bar{x}
  $$

### ✅ `MyMultiLinearRegression` (for nD data)

```python
class MyMultiLinearRegression:
    def fit(self, X_train, y_train): ...
    def predict(self, X_test): ...
```

* Adds a bias column to the feature matrix and solves using:

  $$
  \beta = (X^T X)^{-1} X^T y
  $$

---

## 🧪 How to Run

1. Clone this repository:

```bash
git clone https://github.com/aniq63/Linear-Regression-From-Scratch.git
cd Linear-regression-From-Scratch
```

2. Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn
```

3. Run the script:

```bash
python main.py
```

---

## 📊 Example Results

### Simple Linear Regression (placement.csv)

* **Feature**: CGPA
* **Target**: Placement Package
* **Sample Output**:

  ```
  Slope (m): 0.558
  Intercept (b): 0.697
  R² Score: 0.801
  ```

### Multiple Linear Regression (Diabetes dataset)

* **Features**: 10 clinical variables
* **Target**: Disease progression after one year
* **Sample Output**:

  ```
  R² Score: 0.518
  ```

---

## 📈 Visualization

For the 2D model:

```python
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted Line')
plt.xlabel("CGPA")
plt.ylabel("Package")
plt.legend()
plt.show()
```

---

## 📄 Medium Article

For a detailed explanation and breakdown of this implementation, read the companion blog post on Medium:

👉 [**Understanding Linear Regression from Scratch in Python**](https://medium.com/gopenai/build-linear-regression-from-scratch-a-hands-on-guide-with-math-and-python-0b9a9cfedbd6)

---

## 🔗 References

* [NumPy Documentation](https://numpy.org/doc/)
* [Pandas Documentation](https://pandas.pydata.org/docs/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [Matplotlib](https://matplotlib.org/)

