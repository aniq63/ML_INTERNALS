import numpy as np
from collections import Counter
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score


# ---------------------------------------------------------------------------
# Internal helper: a single decision tree used inside the forest
# ---------------------------------------------------------------------------

class _Node:
    """Single node of a decision tree (internal or leaf)."""

    def __init__(self, feature_index=None, threshold=None,
                 left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class _DecisionTreeClassifier:
    """
    Lightweight Decision Tree Classifier used as a base estimator
    inside the Random Forest.

    Key difference from a standalone tree: during _best_split, only a
    random subset of features is evaluated at each node (controlled by
    the max_features parameter passed from the forest).
    """

    def __init__(self, max_depth=None, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None

    def _gini(self, y):
        m = len(y)
        if m == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / m
        return 1.0 - np.sum(probs ** 2)

    def _best_split(self, X, y):
        """
        Evaluates a RANDOM subset of features to find the best split.

        This is the key ingredient of Random Forests: by limiting the
        features considered at each node, the trees become decorrelated
        from one another, which reduces the variance of the ensemble.
        """
        m, n = X.shape

        # Select random subset of features
        num_features = self.max_features if self.max_features else n
        num_features = min(num_features, n)
        feature_indices = np.random.choice(n, size=num_features, replace=False)

        best_gini = float("inf")
        best_feature, best_threshold = None, None

        for feature_idx in feature_indices:
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                left_mask  = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                weighted_gini = (
                    (left_mask.sum()  / m) * self._gini(y[left_mask])
                    + (right_mask.sum() / m) * self._gini(y[right_mask])
                )

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        m = len(y)
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or m < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            return _Node(value=int(np.bincount(y).argmax()))

        feature_idx, threshold = self._best_split(X, y)
        if feature_idx is None:
            return _Node(value=int(np.bincount(y).argmax()))

        left_mask  = X[:, feature_idx] <= threshold
        return _Node(
            feature_index=feature_idx,
            threshold=threshold,
            left=self._build_tree(X[left_mask],  y[left_mask],  depth + 1),
            right=self._build_tree(X[~left_mask], y[~left_mask], depth + 1),
        )

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])


# ---

class _DecisionTreeRegressor:
    """
    Lightweight Decision Tree Regressor used as a base estimator
    inside the Random Forest Regressor.
    """

    def __init__(self, max_depth=None, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None

    def _mse(self, y):
        return np.var(y) if len(y) > 0 else 0.0

    def _best_split(self, X, y):
        m, n = X.shape
        num_features = self.max_features if self.max_features else n
        num_features = min(num_features, n)
        feature_indices = np.random.choice(n, size=num_features, replace=False)

        best_mse = float("inf")
        best_feature, best_threshold = None, None

        for feature_idx in feature_indices:
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                left_mask  = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                weighted_mse = (
                    (left_mask.sum()  / m) * self._mse(y[left_mask])
                    + (right_mask.sum() / m) * self._mse(y[right_mask])
                )

                if weighted_mse < best_mse:
                    best_mse = weighted_mse
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        m = len(y)
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or m < self.min_samples_split
        ):
            return _Node(value=np.mean(y))

        feature_idx, threshold = self._best_split(X, y)
        if feature_idx is None:
            return _Node(value=np.mean(y))

        left_mask = X[:, feature_idx] <= threshold
        return _Node(
            feature_index=feature_idx,
            threshold=threshold,
            left=self._build_tree(X[left_mask],  y[left_mask],  depth + 1),
            right=self._build_tree(X[~left_mask], y[~left_mask], depth + 1),
        )

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])


# ---------------------------------------------------------------------------
# Random Forest Classifier
# ---------------------------------------------------------------------------

class MyRandomForestClassifier:
    """
    Random Forest Classifier implemented from scratch.

    An ensemble of decision trees trained on bootstrapped subsets of the
    data, each using a random subset of features at every split.
    Predictions are made by majority vote across all trees.

    Parameters
    ----------
    n_estimators    : int   -- Number of trees in the forest. Default 100.
    max_depth       : int   -- Maximum depth of each tree. Default None.
    min_samples_split: int  -- Minimum samples required to split. Default 2.
    max_features    : int   -- Number of features to consider at each split.
                               Default is sqrt(n_features).
    random_state    : int   -- Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        max_features=None,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def _bootstrap_sample(self, X, y):
        """
        Draws a bootstrap sample (sampling with replacement) of the same
        size as the original dataset.

        Approximately 63.2% of original samples appear at least once.
        The remaining ~36.8% are called out-of-bag (OOB) samples.
        """
        m = X.shape[0]
        indices = np.random.choice(m, size=m, replace=True)
        return X[indices], y[indices]

    def fit(self, X_train, y_train):
        """
        Builds the forest by training n_estimators trees on bootstrapped
        subsets of the training data.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_features = X_train.shape[1]
        max_features = self.max_features or int(np.sqrt(n_features))

        self.trees = []
        for _ in range(self.n_estimators):
            # 1. Bootstrap the training data
            X_boot, y_boot = self._bootstrap_sample(X_train, y_train)

            # 2. Train a tree on the bootstrap sample
            tree = _DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features,
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

    def predict(self, X):
        """
        Returns class predictions via majority vote across all trees.

        Steps:
          1. Collect predictions from all n_estimators trees.
          2. For each sample, take the majority-voted class.
        """
        # Shape: (n_estimators, m)
        all_predictions = np.array([tree.predict(X) for tree in self.trees])

        # Transpose to (m, n_estimators) and vote for each sample
        majority_votes = [
            Counter(all_predictions[:, i]).most_common(1)[0][0]
            for i in range(X.shape[0])
        ]
        return np.array(majority_votes)


# ---------------------------------------------------------------------------
# Random Forest Regressor
# ---------------------------------------------------------------------------

class MyRandomForestRegressor:
    """
    Random Forest Regressor implemented from scratch.

    Same ensemble strategy as the classifier, but predictions are made
    by averaging the outputs of all trees.

    Parameters
    ----------
    n_estimators    : int   -- Number of trees. Default 100.
    max_depth       : int   -- Maximum depth per tree. Default None.
    min_samples_split: int  -- Minimum samples to split. Default 2.
    max_features    : int   -- Features to consider at each split.
                               Default is n_features // 3.
    random_state    : int   -- Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        max_features=None,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def _bootstrap_sample(self, X, y):
        m = X.shape[0]
        indices = np.random.choice(m, size=m, replace=True)
        return X[indices], y[indices]

    def fit(self, X_train, y_train):
        """Builds the forest by training n_estimators regression trees."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_features = X_train.shape[1]
        max_features = self.max_features or max(1, n_features // 3)

        self.trees = []
        for _ in range(self.n_estimators):
            X_boot, y_boot = self._bootstrap_sample(X_train, y_train)
            tree = _DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features,
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

    def predict(self, X):
        """
        Returns continuous predictions by averaging all tree outputs.
        """
        all_predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(all_predictions, axis=0)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

print("=" * 55)
print("Random Forest Classifier  --  Breast Cancer Dataset")
print("=" * 55)

X_cls, y_cls = load_breast_cancer(return_X_y=True)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

rf_clf = MyRandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
rf_clf.fit(X_train_c, y_train_c)
y_pred_c = rf_clf.predict(X_test_c)
print(f"Accuracy: {accuracy_score(y_test_c, y_pred_c):.4f}\n")

# -------------------------

print("=" * 55)
print("Random Forest Regressor   --  Diabetes Dataset")
print("=" * 55)

X_reg, y_reg = load_diabetes(return_X_y=True)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

rf_reg = MyRandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
rf_reg.fit(X_train_r, y_train_r)
y_pred_r = rf_reg.predict(X_test_r)
print(f"R2 Score: {r2_score(y_test_r, y_pred_r):.4f}")
