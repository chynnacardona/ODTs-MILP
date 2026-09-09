# Original 1984 Breiman Algorithm

import numpy as np

class Node:
    # Represents a single node in the greedy tree
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature              # Feature index to split on
        self.threshold = threshold          # Cutoff value (s)
        self.left = left
        self.right = right
        self.value = value

def gini_impurity(y):
        # Calculates Gini Impurity: I_G(m) = 1 - sum(p^2_mk)
        if len(y) == 0:
            return 0.0
        probabilities = np.bincount(y) / len(y)
        return 1.0 - np.sum(probabilities ** 2)

def greedy_cart(X, y, current_depth = 0, max_depth = 2):
    # Original CART recursive loop
    n_samples, n_features = X.shape
    num_classes = len(np.unique(y))

    if current_depth >= max_depth or num_classes == 1 or n_samples == 0:
        most_common_class = np.argmax(np.bincount(y)) if len(y) > 0 else 0
        return Node(value = most_common_class)

    best_gini = float("inf")
    best_split = None

    for j in range(n_features):
        thresholds = np.unique(X[:, j])
        for s in thresholds:
            left_mask = X[:, j] <= s
            right_mask = ~left_mask

            y_left, y_right = y[left_mask], y[right_mask]
            if len(y_left) == 0 or len(y_right) == 0:
                continue

            w_l, w_r = len(y_left) / n_samples, len(y_right) / n_samples
            g_theta = (w_l * gini_impurity(y_left)) + (w_r * gini_impurity(y_right))

            if g_theta < best_gini:
                best_gini = g_theta
                best_split = (j, s, left_mask, right_mask)

    if best_gini == float("inf"):
        return Node(value=np.argmax(np.bincount(y)))

    # Recursive step: Lock in choices irreversibly and move down
    best_j, best_s, left_mask, right_mask = best_split

    left_child = greedy_cart(X[left_mask, :], y[left_mask], current_depth + 1, max_depth)
    right_child = greedy_cart(X[right_mask, :], y[right_mask], current_depth + 1, max_depth)

    return Node(feature=best_j, threshold=best_s, left=left_child, right=right_child)

