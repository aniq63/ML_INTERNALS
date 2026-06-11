import numpy as np
from collections import Counter
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score


class MyKNNClassifier:
    """
    K-Nearest Neighbors Classifier implemented from scratch.

    For each test sample, the algorithm:
      1. Computes the distance to every training sample.
      2. Selects the K nearest neighbors.
      3. Returns the majority class among those K neighbors.

    Parameters
    ----------
    k : int
        Number of nearest neighbors to consider. Default is 3.
    metric : str
        Distance metric to use. Options: 'euclidean', 'manhattan'. Default is 'euclidean'.
    """

    def __init__(self, k=3, metric="euclidean"):
        if k < 1:
            raise ValueError("k must be a positive integer.")
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None

    # ------------------------------------------------------------------
    # Distance metrics
    # ------------------------------------------------------------------

    def _distance(self, a, b):
        """
        Computes the distance between two vectors.

        Euclidean distance:
            d(a, b) = sqrt( sum( (a_i - b_i)^2 ) )

        Manhattan distance:
            d(a, b) = sum( |a_i - b_i| )
        """
        if self.metric == "euclidean":
            return np.sqrt(np.sum((a - b) ** 2))
        elif self.metric == "manhattan":
            return np.sum(np.abs(a - b))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_train, y_train):
        """
        KNN has no explicit training phase.

        The training data is simply stored in memory. All computation
        is deferred to prediction time (lazy learning).

        Parameters
        ----------
        X_train : np.ndarray, shape (m, n)
        y_train : np.ndarray, shape (m,)
        """
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def _predict_single(self, x):
        """
        Predicts the class label for a single test sample.

        Steps:
          1. Compute distances from x to all training points.
          2. Sort by distance ascending and take the first K indices.
          3. Gather the labels of the K nearest neighbors.
          4. Return the majority-vote label.
        """
        distances = np.array([self._distance(x, x_tr) for x_tr in self.X_train])
        k_nearest_indices = np.argsort(distances)[: self.k]
        k_nearest_labels  = self.y_train[k_nearest_indices]

        # Majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X):
        """
        Predicts class labels for all samples in X.

        Parameters
        ----------
        X : np.ndarray, shape (m, n)

        Returns
        -------
        np.ndarray of predicted labels, shape (m,)
        """
        return np.array([self._predict_single(x) for x in X])


# ---------------------------------------------------------------------------

class MyKNNRegressor:
    """
    K-Nearest Neighbors Regressor implemented from scratch.

    Identical to the classifier except the prediction step returns the
    mean of the K nearest neighbors' target values instead of a majority vote.

    Parameters
    ----------
    k      : int   -- Number of nearest neighbors. Default is 3.
    metric : str   -- 'euclidean' or 'manhattan'. Default is 'euclidean'.
    """

    def __init__(self, k=3, metric="euclidean"):
        if k < 1:
            raise ValueError("k must be a positive integer.")
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def _distance(self, a, b):
        if self.metric == "euclidean":
            return np.sqrt(np.sum((a - b) ** 2))
        elif self.metric == "manhattan":
            return np.sum(np.abs(a - b))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def fit(self, X_train, y_train):
        """Stores training data (no actual model fitting)."""
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def _predict_single(self, x):
        """
        Predicts the target value for a single test sample.

        Returns the mean of the K nearest neighbors' target values.
        """
        distances = np.array([self._distance(x, x_tr) for x_tr in self.X_train])
        k_nearest_indices = np.argsort(distances)[: self.k]
        return np.mean(self.y_train[k_nearest_indices])

    def predict(self, X):
        """
        Predicts continuous values for all samples in X.

        Parameters
        ----------
        X : np.ndarray, shape (m, n)

        Returns
        -------
        np.ndarray of predicted values, shape (m,)
        """
        return np.array([self._predict_single(x) for x in X])


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

print("=" * 50)
print("KNN Classifier  --  Iris Dataset")
print("=" * 50)

X_cls, y_cls = load_iris(return_X_y=True)

# Feature normalisation is important for distance-based methods
X_cls = (X_cls - X_cls.mean(axis=0)) / X_cls.std(axis=0)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

knn_clf = MyKNNClassifier(k=5)
knn_clf.fit(X_train_c, y_train_c)
y_pred_c = knn_clf.predict(X_test_c)
print(f"Accuracy (k=5): {accuracy_score(y_test_c, y_pred_c):.4f}\n")

# -------------------------

print("=" * 50)
print("KNN Regressor   --  Diabetes Dataset")
print("=" * 50)

X_reg, y_reg = load_diabetes(return_X_y=True)
X_reg = (X_reg - X_reg.mean(axis=0)) / X_reg.std(axis=0)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

knn_reg = MyKNNRegressor(k=5)
knn_reg.fit(X_train_r, y_train_r)
y_pred_r = knn_reg.predict(X_test_r)
print(f"R2 Score (k=5): {r2_score(y_test_r, y_pred_r):.4f}")
