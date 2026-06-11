import numpy as np
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score


# ---------------------------------------------------------------------------
# Internal node / leaf representation
# ---------------------------------------------------------------------------

class _Node:
    """
    Represents a single node in the decision tree.

    A node is either:
      - An internal node: stores the best split (feature index + threshold)
                         and pointers to left/right children.
      - A leaf node    : stores the predicted value (class or real number).
    """

    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        value=None,
    ):
        self.feature_index = feature_index  # Index of the feature to split on
        self.threshold = threshold          # Split threshold
        self.left = left                    # Left child node  (X[:, f] <= threshold)
        self.right = right                  # Right child node (X[:, f] >  threshold)
        self.value = value                  # Prediction value (leaf nodes only)

    def is_leaf(self):
        return self.value is not None


# ---------------------------------------------------------------------------
# Decision Tree Classifier
# ---------------------------------------------------------------------------

class MyDecisionTreeClassifier:
    """
    Binary Decision Tree Classifier built using the CART algorithm.

    Splitting criterion : Gini Impurity
    Tree growth strategy: Recursive binary splitting (greedy, depth-first)

    Parameters
    ----------
    max_depth   : int or None
        Maximum depth of the tree. None means the tree grows until all
        leaves are pure or contain fewer than min_samples_split samples.
    min_samples_split : int
        Minimum number of samples required at a node to attempt a split.
    """

    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    # ------------------------------------------------------------------
    # Impurity and split scoring
    # ------------------------------------------------------------------

    def _gini(self, y):
        """
        Computes Gini Impurity for a set of labels.

        Gini(y) = 1 - sum( p_k^2 )  for each class k

        A Gini of 0 means the node is perfectly pure (one class only).
        A Gini of 0.5 (for binary) is maximum impurity (50/50 split).
        """
        m = len(y)
        if m == 0:
            return 0.0
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / m
        return 1.0 - np.sum(probabilities ** 2)

    def _best_split(self, X, y):
        """
        Searches all features and all thresholds for the split that
        produces the greatest reduction in weighted Gini Impurity.

        Returns the feature index and threshold of the best split found,
        or (None, None) if no improvement is possible.
        """
        m, n = X.shape
        best_gini = float("inf")
        best_feature, best_threshold = None, None

        for feature_idx in range(n):
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                left_mask  = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                # Weighted Gini of the two children
                gini_left  = self._gini(y[left_mask])
                gini_right = self._gini(y[right_mask])

                weighted_gini = (
                    (left_mask.sum()  / m) * gini_left
                    + (right_mask.sum() / m) * gini_right
                )

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    # ------------------------------------------------------------------
    # Tree construction (recursive)
    # ------------------------------------------------------------------

    def _build_tree(self, X, y, depth=0):
        """
        Recursively builds the tree by greedily choosing the best split
        at each node.

        Stopping conditions (leaf node):
          1. Current depth equals max_depth.
          2. Node has fewer samples than min_samples_split.
          3. All samples in the node belong to the same class (pure node).
          4. No valid split exists (no feature separates the data).
        """
        m = len(y)
        num_classes = len(np.unique(y))

        # --- Stopping conditions ---
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or m < self.min_samples_split
            or num_classes == 1
        ):
            # Leaf value: majority class
            leaf_value = int(np.bincount(y).argmax())
            return _Node(value=leaf_value)

        # --- Find best split ---
        feature_idx, threshold = self._best_split(X, y)
        if feature_idx is None:
            leaf_value = int(np.bincount(y).argmax())
            return _Node(value=leaf_value)

        # --- Partition and recurse ---
        left_mask  = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        left_child  = self._build_tree(X[left_mask],  y[left_mask],  depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return _Node(
            feature_index=feature_idx,
            threshold=threshold,
            left=left_child,
            right=right_child,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_train, y_train):
        """Builds the decision tree from training data."""
        self.root = self._build_tree(X_train, y_train)

    def _traverse(self, x, node):
        """Traverses the tree for a single sample and returns a leaf value."""
        if node.is_leaf():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict(self, X):
        """Returns class predictions for all samples in X."""
        return np.array([self._traverse(x, self.root) for x in X])


# ---------------------------------------------------------------------------
# Decision Tree Regressor
# ---------------------------------------------------------------------------

class MyDecisionTreeRegressor:
    """
    Binary Decision Tree Regressor built using the CART algorithm.

    Splitting criterion : Mean Squared Error (variance reduction)
    Leaf prediction     : Mean of target values in the leaf

    Parameters
    ----------
    max_depth         : int or None
    min_samples_split : int
    """

    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    # ------------------------------------------------------------------
    # Variance reduction
    # ------------------------------------------------------------------

    def _mse(self, y):
        """Mean Squared Error of a node, used as the impurity measure."""
        if len(y) == 0:
            return 0.0
        return np.var(y)

    def _best_split(self, X, y):
        """
        Finds the split that maximises variance reduction (minimises weighted MSE).
        """
        m, n = X.shape
        best_mse = float("inf")
        best_feature, best_threshold = None, None

        for feature_idx in range(n):
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                left_mask  = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                mse_left  = self._mse(y[left_mask])
                mse_right = self._mse(y[right_mask])

                weighted_mse = (
                    (left_mask.sum()  / m) * mse_left
                    + (right_mask.sum() / m) * mse_right
                )

                if weighted_mse < best_mse:
                    best_mse = weighted_mse
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

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

        left_mask  = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        left_child  = self._build_tree(X[left_mask],  y[left_mask],  depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return _Node(
            feature_index=feature_idx,
            threshold=threshold,
            left=left_child,
            right=right_child,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_train, y_train):
        """Builds the regression tree from training data."""
        self.root = self._build_tree(X_train, y_train)

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict(self, X):
        """Returns continuous predictions for all samples in X."""
        return np.array([self._traverse(x, self.root) for x in X])


# ---------------------------------------------------------------------------
# Demo: Classification (Iris) and Regression (Diabetes)
# ---------------------------------------------------------------------------

print("=" * 50)
print("Decision Tree Classifier  --  Iris Dataset")
print("=" * 50)

X_cls, y_cls = load_iris(return_X_y=True)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

clf = MyDecisionTreeClassifier(max_depth=5)
clf.fit(X_train_c, y_train_c)
y_pred_c = clf.predict(X_test_c)
print(f"Accuracy: {accuracy_score(y_test_c, y_pred_c):.4f}\n")

# -------------------------

print("=" * 50)
print("Decision Tree Regressor   --  Diabetes Dataset")
print("=" * 50)

X_reg, y_reg = load_diabetes(return_X_y=True)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg = MyDecisionTreeRegressor(max_depth=5)
reg.fit(X_train_r, y_train_r)
y_pred_r = reg.predict(X_test_r)
print(f"R2 Score: {r2_score(y_test_r, y_pred_r):.4f}")
