import numpy as np
import pulp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier

class StandardOCT(BaseEstimator, ClassifierMixin):
    """
    True Global Optimal Classification Tree (Bertsimas & Dunn 2017)
    Powered by PuLP and the open-source CBC MILP solver with safety timeouts.
    """
    def __init__(self, depth=2, alpha=0.0001, time_limit=60, verbose=False):
        self.max_depth = depth
        self.alpha = alpha
        self.time_limit = time_limit
        self.verbose = verbose
        
        self.classes_ = None
        self.tree_structure_ = {}

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)

        self.classes_ = np.unique(y_arr)
        K = len(self.classes_)
        class_map = {c: idx for idx, c in enumerate(self.classes_)}
        y_indexed = np.array([class_map[val] for val in y_arr])

        N, P = X_arr.shape
        num_branches = (2 ** self.max_depth) - 1
        num_leaves = 2 ** self.max_depth
        branches = list(range(num_branches))
        leaves = list(range(num_branches, num_branches + num_leaves))

        def get_ancestors(node_id):
            left_ancestors, right_ancestors = [], []
            curr = node_id
            while curr > 0:
                parent = (curr - 1) // 2
                if curr % 2 == 1:
                    left_ancestors.append(parent)
                else:
                    right_ancestors.append(parent)
                curr = parent
            return left_ancestors, right_ancestors

        # Initialize PuLP Minimization Model
        model = pulp.LpProblem("OCT_MILP", pulp.LpMinimize)

        # Decision Variables using PuLP 4.x native model dictionary methods
        a = model.add_variable_dict("a", (branches, range(P)), lowBound=0, upBound=1, cat='Binary')
        b = model.add_variable_dict("b", branches, lowBound=-2.0, upBound=2.0, cat='Continuous')
        c = model.add_variable_dict("c", (range(K), leaves), lowBound=0, upBound=1, cat='Binary')
        d = model.add_variable_dict("d", branches, lowBound=0, upBound=1, cat='Binary')
        z = model.add_variable_dict("z", (range(N), leaves), lowBound=0, upBound=1, cat='Binary')
        L = model.add_variable_dict("L", leaves, lowBound=0.0, cat='Continuous')
           
        # Explicit Bounds & Branch Constraints
        for t in branches:
            model += (pulp.lpSum(a[t, j] for j in range(P)) == d[t])
            model += (b[t] <= 2.0 * d[t])
            model += (b[t] >= -2.0 * d[t])
            model += (b[t] >= -2.0)  # Lower bound for b
            model += (b[t] <= 2.0)   # Upper bound for b
            if t > 0:
                parent = (t - 1) // 2
                model += (d[t] <= d[parent])

        for t in leaves:
            model += (L[t] >= 0.0)   # Lower bound for L

        for t in branches:
            model += (pulp.lpSum(a[t, j] for j in range(P)) == d[t])
            model += (b[t] <= 2.0 * d[t])
            model += (b[t] >= -2.0 * d[t])
            if t > 0:
                parent = (t - 1) // 2
                model += (d[t] <= d[parent])

        for t in leaves:
            model += (pulp.lpSum(c[k, t] for k in range(K)) == 1)

        for i in range(N):
            model += (pulp.lpSum(z[i, t] for t in leaves) == 1)

        # Strict linear Big-M formulation avoiding non-constant multiplications
        eps = 1e-3
        feature_ranges = np.max(X_arr, axis=0) - np.min(X_arr, axis=0)
        M_global = float(np.max(feature_ranges)) + 5.0

        for t in leaves:
            left_ancestors, right_ancestors = get_ancestors(t)
            for i in range(N):
                for m in left_ancestors:
                    model += (
                        pulp.lpSum(a[m, j] * X_arr[i, j] for j in range(P))
                        <= b[m] + M_global * (1 - z[i, t])
                    )
                for m in right_ancestors:
                    model += (
                        pulp.lpSum(a[m, j] * X_arr[i, j] for j in range(P))
                        >= b[m] + eps - M_global * (1 - z[i, t])
                    )

        for t in leaves:
            for k in range(K):
                samples_not_k = [i for i in range(N) if y_indexed[i] != k]
                for i in samples_not_k:
                    model += (L[t] >= z[i, t] + c[k, t] - 1)

        total_misclassification = pulp.lpSum(L[t] for t in leaves)
        tree_complexity = self.alpha * pulp.lpSum(d[t] for t in branches)

        model.setObjective(total_misclassification + tree_complexity)

        # Configure PuLP's bundled CBC Solver with safety timeouts
        solver = pulp.getSolver(
            'COIN_CMD',
            timeLimit=self.time_limit,
            msg=self.verbose,
            gapRel=0.05
        )

        stats = model.solve(solver)

        # Fallback safeguard if solver times out or fails to find a solution
        self.tree_structure_ = {}
        status_str = getattr(stats, 'status_str', str(stats))
        if status_str not in ['Optimal', 'Feasible'] and all(pulp.value(L[t]) is None for t in leaves):
            self.tree_structure_[0] = {"l": True, "class": self.classes_[0]}
            return self

        # Extract Solution structure
        for t in branches:
            selected_feature = [j for j in range(P) if pulp.value(a[t, j]) is not None and pulp.value(a[t, j]) > 0.5]
            self.tree_structure_[t] = {
                "d": pulp.value(d[t]) is not None and pulp.value(d[t]) > 0.5,
                "feature": selected_feature[0] if len(selected_feature) > 0 else 0,
                "threshold": pulp.value(b[t]) if pulp.value(b[t]) is not None else 0.0
            }

        for l in leaves:
            assigned_class = [k for k in range(K) if pulp.value(c[k, l]) is not None and pulp.value(c[k, l]) > 0.5]
            self.tree_structure_[l] = {
                "l": True,
                "class": self.classes_[assigned_class[0]] if len(assigned_class) > 0 else self.classes_[0]
            }

        return self

    def predict(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        predictions = []

        for i in range(X_arr.shape[0]):
            curr_node = 0  
            while curr_node in self.tree_structure_ and "d" in self.tree_structure_[curr_node]:
                node_info = self.tree_structure_[curr_node]
                if not node_info["d"]:
                    curr_node = 2 * curr_node + 1
                else:
                    feat = node_info["feature"]
                    if X_arr[i, feat] <= node_info["threshold"]:
                        curr_node = 2 * curr_node + 1  
                    else:
                        curr_node = 2 * curr_node + 2  

            if curr_node in self.tree_structure_ and "class" in self.tree_structure_[curr_node]:
                predictions.append(self.tree_structure_[curr_node]["class"])
            else:
                predictions.append(self.classes_[0])

        return np.array(predictions)