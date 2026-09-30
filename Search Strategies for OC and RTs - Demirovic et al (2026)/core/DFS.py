# Depth-First Search (DFS) implementation for ODTs
# This code implements both standard DFS Branch-and-Bound and DFS B&B w/ AND-OR Search (AOS) -- comparative analysis of the two approaches is provided in the paper by Demirovic et al. (2026).

import time
import numpy as np

import time
import numpy as np


class StandardDFSSolver:
    """
    Standard Depth-First Search Solver (without AND-OR Search Graph logic).
    Evaluates whole subtrees sequentially without formal OR/AND node decomposition.
    """
    def __init__(self, max_depth=2):
        self.max_depth = max_depth
        self.nodes_expanded = 0 # Initializes counter tracking total tree states evaluated.

    """
    When a decision tree stops splitting and creates a leaf node, it must assign a 
    single target class to all samples in that leaf. It does this using majority voting—picking 
    whichever class appears most frequently in that subset of data.
    """
    def _get_leaf_loss(self, y): # Helper function calculating majority-voting classification error.
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True) # Counts target class frequencies in label array y
        return len(y) - np.max(counts) # Returns misclassified sample count (total samples minus most frequent class count).
   
    def solve(self, X, y):
        self.nodes_expanded = 0
        best_loss, _ = self._dfs(X, y, depth=self.max_depth, current_ub=float('inf'))
        return best_loss, self.nodes_expanded

    """
    This function serves as the core recursive Depth-First Search (DFS) engine for the standard 
    (non-AOS) Branch-and-Bound solver. Its goal is to explore feature split decisions down a branch 
    to find the tree structure that yields the minimum classification error (loss) for dataset partition 
    (X, y) within a maximum allowed depth
    """
    def _dfs(self, X, y, depth, current_ub):
        self.nodes_expanded += 1
        leaf_loss = self._get_leaf_loss(y) # Calculates the classification error if the algorithm decides to stop splitting here and convert this node into a terminal leaf node (using majority voting).

        if depth == 0 or len(y) == 0 or leaf_loss == 0:
            return leaf_loss, None

        best_loss = leaf_loss
        best_split = None

        n_samples, n_features = X.shape
        for feat in range(n_features):
            unique_vals = np.unique(X[:, feat])
            if len(unique_vals) <= 1:
                continue
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask

                if not np.any(left_mask) or not np.any(right_mask):
                    continue

                # Lower bound calculation (Trivial leaf sum)
                lb_left = self._get_leaf_loss(y[left_mask])
                lb_right = self._get_leaf_loss(y[right_mask])

                if lb_left + lb_right >= min(best_loss, current_ub):
                    continue

                # Sequential search without AND-OR subproblem decoupling
                left_loss, _ = self._dfs(X[left_mask], y[left_mask], depth - 1, current_ub=min(best_loss, current_ub) - lb_right)
                
                if left_loss + lb_right < min(best_loss, current_ub):
                    right_loss, _ = self._dfs(X[right_mask], y[right_mask], depth - 1, current_ub=min(best_loss, current_ub) - left_loss)
                    total_loss = left_loss + right_loss

                    if total_loss < best_loss:
                        best_loss = total_loss
                        best_split = (feat, thresh)
                        if best_loss == 0:
                            return 0, best_split

        return best_loss, best_split


class AOSDFSSolver:
    """
    AND-OR Search Graph (AOS) DFS Solver.
    Utilizes explicit OR nodes (choices) and AND nodes (parallel split decompositions)
    with bounds propagation back to the root.
    """
    def __init__(self, max_depth=2):
        self.max_depth = max_depth
        self.nodes_expanded = 0

    def _get_leaf_loss(self, y):
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        return len(y) - np.max(counts)

    def solve(self, X, y):
        self.nodes_expanded = 0
        root_ub = self._get_leaf_loss(y)
        final_loss = self._aos_dfs_or(X, y, depth=self.max_depth, parent_ub=root_ub)
        return final_loss, self.nodes_expanded

    def _aos_dfs_or(self, X, y, depth, parent_ub):
        """OR Node: Selects the minimum loss among feature/threshold options."""
        self.nodes_expanded += 1
        leaf_loss = self._get_leaf_loss(y)

        if depth == 0 or len(y) == 0 or leaf_loss == 0:
            return leaf_loss

        or_ub = leaf_loss
        n_samples, n_features = X.shape

        for feat in range(n_features):
            unique_vals = np.unique(X[:, feat])
            if len(unique_vals) <= 1:
                continue
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask

                if not np.any(left_mask) or not np.any(right_mask):
                    continue

                # Evaluate AND node split decomposition
                and_loss = self._aos_dfs_and(
                    X[left_mask], y[left_mask],
                    X[right_mask], y[right_mask],
                    depth=depth - 1,
                    current_or_ub=min(or_ub, parent_ub)
                )

                # BACKPROPAGATE: Update OR node upper bound
                if and_loss < or_ub:
                    or_ub = and_loss
                    if or_ub == 0:
                        return 0

        return or_ub

    def _aos_dfs_and(self, X_left, y_left, X_right, y_right, depth, current_or_ub):
        """AND Node: Decomposes into independent left and right OR subproblems."""
        lb_left = self._get_leaf_loss(y_left)
        lb_right = self._get_leaf_loss(y_right)

        # Pruning check before exploring child OR nodes
        if lb_left + lb_right >= current_or_ub:
            return float('inf')

        # Recurse down Left child OR Node
        left_loss = self._aos_dfs_or(X_left, y_left, depth, parent_ub=current_or_ub - lb_right)

        # Pruning check before exploring Right child OR Node
        if left_loss + lb_right >= current_or_ub:
            return float('inf')

        # Recurse down Right child OR Node
        right_loss = self._aos_dfs_or(X_right, y_right, depth, parent_ub=current_or_ub - left_loss)

        return left_loss + right_loss


# --- Comparison Benchmark Function ---
def run_benchmark():
    np.random.seed(42)
    # Synthetic Multi-feature dataset
    N_SAMPLES = 120
    N_FEATURES = 6
    X = np.random.randn(N_SAMPLES, N_FEATURES)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int) # XOR relationship

    MAX_DEPTH = 3

    print("=" * 65)
    print(f" OPTIMAL DECISION TREE BENCHMARK (Depth={MAX_DEPTH}, Samples={N_SAMPLES}, Feats={N_FEATURES})")
    print("=" * 65)

    # 1. Standard DFS Solver (w/o AOS)
    std_solver = StandardDFSSolver(max_depth=MAX_DEPTH)
    t0 = time.perf_counter()
    std_loss, std_nodes = std_solver.solve(X, y)
    std_time = (time.perf_counter() - t0) * 1000.0

    # 2. AOS DFS Solver (w/ AOS)
    aos_solver = AOSDFSSolver(max_depth=MAX_DEPTH)
    t0 = time.perf_counter()
    aos_loss, aos_nodes = aos_solver.solve(X, y)
    aos_time = (time.perf_counter() - t0) * 1000.0

    # Print Results Table
    print(f"{'Metric':<25} | {'Standard DFS (w/o AOS)':<20} | {'AOS DFS (w/ AOS)':<20}")
    print("-" * 71)
    print(f"{'Optimal Loss (Error)':<25} | {std_loss:<20} | {aos_loss:<20}")
    print(f"{'Nodes Expanded':<25} | {std_nodes:<20} | {aos_nodes:<20}")
    print(f"{'Execution Time (ms)':<25} | {std_time:<20.2f} | {aos_time:<20.2f}")
    print("=" * 65)

    # Verification Assertion
    assert std_loss == aos_loss, "Both solvers must arrive at identical optimal loss!"


if __name__ == "__main__":
    run_benchmark()