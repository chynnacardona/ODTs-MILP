import numpy as np
import gurobipy as gp
from gurobipy import GRB
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

class OCTH(BaseEstimator, ClassifierMixin):
    def __init__(self, depth=2, aplha=0.01, time_limit=60, verbose=False):
        self.max_depth = depth
        self.alpha = aplha
        self.time_limit = time_limit
        self.verbose = verbose

        self.model = None
        self.classes = None
        self.n_features_in_ = 0
        self.tree_structure_ = {}

    def _normalize(self, X):
        # Normalizes input features to [0, 1] range required for Big-M bounds.
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        range_val = np.where((X_max - X_min) == 0, 1.0, X_max - X_min)
        return (X - X_min) / range_val

    def _generate_warmstart(self, X, y_idx, K):
        # Generated initial warm start values (top-down greedy multivariate split (sect 4.2))
        num_branch = 2 ** self.depth - 1
        num_leaf = 2 ** self.depth
        n_samples, n_features = X.shape

        # Decision Variables init
        a_init = {}
        b_init = {}
        d_init = {t: 0 for t in range(1, num_leaf + 1)}
        z_init = np.zeros((n_samples, num_branch + num_leaf + 1))
        l_init = {t: 0 for t in range(num_branch + 1, num_branch + num_leaf + 1)}
        c_init = {
            (k, t): 0
            for k in range(K)
            for t in range(num_branch + 1, num_branch + num_leaf + 1)
        }

        # Recursive node assignments: node_index -> list of sample indicies
        node_samples = {1: np.arange(n_samples)}

        # Top-down recursive init
        for t in range(1, num_branch + 1):
            indices = node_samples.get(t, np.array([], dtype=int))

            # Need at least two distinct classes to form a meaningful split
            if len(indices) > 0 and len(np.unique(y_idx[indices])) > 1:
                # Fit a sparse linear split via L1-regularized Logistic Regression
                clf = LogisticRegression(
                    penalty='l1', solver="liblinear", C=1.0, random_state=42
                )
                clf.fit(X[indices], y_idx[indices])

                weight = clf.coef_[0]
                intercept = clf.intercept_[0]

                # Normalize weights and intercept to satisfy L1 norm <= 1
                norm_factor = np.sum(np.abs(weight) + np.abs(intercept))
                if norm_factor > 0:
                    weights = weights / norm_factor
                    intercept = intercept / norm_factor
                else:
                    weights[0] = 0.5
                    intercept[0] = 0.5

                d_init[t] = 1
                for j in range(n_features):
                    a_init[(t, j)] = weights[j]
                b_init[t] = intercept

                # Route samples to child nodes
                left_mask = (X[indices] @ weights + intercept) < 0
                node_samples[2 * t] = indices[left_mask]
                node_samples[2 * t + 1] = indices[~left_mask]
            else:
                d_init[t] = 0
                for j in range(n_features):
                    a_init[(t, j)] = 0.0
                b_init[t] = 0.0
                # Pass samples down left branch by default if node is inactive
                node_samples[2 * t] = indices
                node_samples[2 * t + 1] = np.array([], dtype=int)

        # Populate leaf variables (z, l, c)
        for t in range(num_branch + 1, num_branch + num_leaf + 1):
            indices = node_samples.get(t, np.array([], dtype=int))
            if len(indices) > 0:
                l_init[t] = 1
                for i in indices:
                    z_init[t, i] = 1

                # Majority class assignment
                counts = np.bincount(y_idx[indices], minlength=K)
                maj_class = np.argmax(counts)
                c_init[(t, maj_class)] = 1
            else:
                l_init[t] = 0
                c_init[(0, t)] = 1

            return a_init, b_init, d_init, l_init, z_init, c_init

    def fit(self, X, y):
        # Build an OCT-H model from the training set (X, y)
        X_norm = self.normalize(X, dtype=np.float64)
        y_arr = np.asarray(y)

        self.classes_ = np.unique(y_arr)
        K = len(self.classes_)
        class_map = {c: idx for idx, c in enumerate(self.classes_)}
        y_idx = np.array([class_map[val] for val in y_arr])

        n_samples, n_features = X_norm.shape
        self.n_features_in_ = n_features

        num_branch = 2 ** self.depth - 1
        num_leaf = 2 ** self.depth

        T_branch = list(range(1, num_branch + 1))
        T_leaf = list(
            range(num_branch + 1, num_branch + num_leaf + 1)
        )

        # Node navigation ancestor dicts
        A_L, A_R = {t: [] for t in T_leaf}, {t: [] for t in T_leaf}
        for t in T_leaf:
            curr = t
            while curr > 1:
                parent = curr // 2
                if curr % 2 == 0:
                    A_L[t].append(parent)
                else:
                    A_R[t].append(parent)
                curr = parent

        # Calculate smallest non-zero feature gap for epsilon boundary tolerance
        epsilons = np.zeros(n_features)
        for j in range(n_features):
            sorted_f = np.sort(np.unique(X_norm[:, j]))
            if len(sorted_f) > 1:
                diffs = np.diff(sorted_f)
                epsilons[j] = np.min(diffs[diffs > 0])
            else:
                epsilons[j] = 0.001
        eps_min = (
            np.min(epsilons[epsilons > 0])
            if np.any(epsilons > 0)
            else 0.001
        )

        # Initialize Gurobi model
        self.model = gp.Model("OCT_H")
        if not self.verbose:
            self.model.Params.OutputFlag = 0
        self.model.Params.TimeLimit = self.time_limit

        # 1. Variables
        a = self.model.addVars(
            T_branch,
            range(n_features),
            lb=-1.0,
            ub=1.0,
            vtype=GRB.CONTINUOUS,
            name="a",
        )
        b = self.model.addVars(
            T_branch, lb=-1.0, ub=1.0, vtype=GRB.CONTINUOUS, name="b"
        )

        hat_a = self.model.addVars(
            T_branch,
            range(n_features),
            lb=0.0,
            ub=1.0,
            vtype=GRB.CONTINUOUS,
            name="hat_a",
        )
        hat_b = self.model.addVars(
            T_branch, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="hat_b"
        )

        d = self.model.addVars(T_branch, vtype=GRB.BINARY, name="d")
        l_leaf = self.model.addVars(T_leaf, vtype=GRB.BINARY, name="l")
        c = self.model.addVars(range(K), T_leaf, vtype=GRB.BINARY, name="c")
        z = self.model.addVars(
            range(n_samples), T_leaf, vtype=GRB.BINARY, name="z"
        )
        L = self.model.addVars(
            T_leaf, lb=0.0, vtype=GRB.CONTINUOUS, name="L"
        )

        s_a = self.model.addVars(
            T_branch, range(n_features), vtype=GRB.BINARY, name="s_a"
        )
        s_b = self.model.addVars(T_branch, vtype=GRB.BINARY, name="s_b")

        # 2. Inject Warm Start Variables (Sec 4.2)
        (
            a_w,
            b_w,
            d_w,
            z_w,
            l_w,
            c_w,
        ) = self._generate_greedy_warmstart(X_norm, y_idx, K)

        for (t, j), val in a_w.items():
            a[t, j].Start = val
            hat_a[t, j].Start = abs(val)
        for t, val in b_w.items():
            b[t].Start = val
            hat_b[t].Start = abs(val)
        for t, val in d_w.items():
            d[t].Start = val
        for t in T_leaf:
            l_leaf[t].Start = l_w[t]
            for i in range(n_samples):
                z[i, t].Start = z_w[i, t]
            for k_idx in range(K):
                c[k_idx, t].Start = c_w[(k_idx, t)]

        # 3. Objective Function
        self.model.setObjective(
            (1.0 / n_samples) * gp.quicksum(L[t] for t in T_leaf)
            + self.alpha * gp.quicksum(d[t] for t in T_branch),
            GRB.MINIMIZE,
        )

        # 4. Constraints
        # Loss Bounds
        for t in T_leaf:
            N_t = gp.quicksum(z[i, t] for i in range(n_samples))
            for k_idx in range(K):
                N_kt = gp.quicksum(
                    z[i, t]
                    for i in range(n_samples)
                    if y_idx[i] == k_idx
                )
                self.model.addConstr(
                    L[t] >= N_t - N_kt - n_samples * (1 - c[k_idx, t])
                )
            self.model.addConstr(
                gp.quicksum(c[k_idx, t] for k_idx in range(K))
                == l_leaf[t]
            )

        # Routing Constraints
        for i in range(n_samples):
            self.model.addConstr(
                gp.quicksum(z[i, t] for t in T_leaf) == 1
            )
        for t in T_leaf:
            for i in range(n_samples):
                self.model.addConstr(z[i, t] <= l_leaf[t])

        # Big-M Decision Splits
        M_1, M_2 = 2.0, 1.0
        for t in T_leaf:
            for i in range(n_samples):
                for m in A_L[t]:
                    left_expr = (
                            gp.quicksum(
                                a[m, j] * X_norm[i, j]
                                for j in range(n_features)
                            )
                            + b[m]
                    )
                    self.model.addConstr(
                        left_expr + eps_min <= M_1 * (1 - z[i, t])
                    )
                for m in A_R[t]:
                    right_expr = (
                            gp.quicksum(
                                a[m, j] * X_norm[i, j]
                                for j in range(n_features)
                            )
                            + b[m]
                    )
                    self.model.addConstr(
                        right_expr >= -M_2 * (1 - z[i, t])
                    )

        # Tree Topology & L1 Norm Constraints
        for t in T_branch:
            if t > 1:
                self.model.addConstr(d[t] <= d[t // 2])

            for j in range(n_features):
                self.model.addConstr(hat_a[t, j] >= a[t, j])
                self.model.addConstr(hat_a[t, j] >= -a[t, j])
                self.model.addConstr(
                    hat_a[t, j] <= a[t, j] + 2 * (1 - s_a[t, j])
                )
                self.model.addConstr(
                    hat_a[t, j] <= -a[t, j] + 2 * s_a[t, j]
                )

            self.model.addConstr(hat_b[t] >= b[t])
            self.model.addConstr(hat_b[t] >= -b[t])
            self.model.addConstr(hat_b[t] <= b[t] + 2 * (1 - s_b[t]))
            self.model.addConstr(hat_b[t] <= -b[t] + 2 * s_b[t])

            self.model.addConstr(
                gp.quicksum(hat_a[t, j] for j in range(n_features))
                + hat_b[t]
                <= d[t]
            )
            self.model.addConstr(
                gp.quicksum(hat_a[t, j] for j in range(n_features))
                >= eps_min * d[t]
            )

        # Optimize
        self.model.optimize()

        # Extract Optimal Tree Structure Parameters
        for t in T_branch:
            self.tree_structure_[t] = {
                "d": d[t].X > 0.5,
                "a": np.array([a[t, j].X for j in range(n_features)]),
                "b": b[t].X,
            }
        for t in T_leaf:
            assigned_class = [
                k_idx for k_idx in range(K) if c[k_idx, t].X > 0.5
            ]
            self.tree_structure_[t] = {
                "l": l_leaf[t].X > 0.5,
                "class": (
                    self.classes_[assigned_class[0]]
                    if len(assigned_class) > 0
                    else self.classes_[0]
                ),
            }

        return self

    def predict(self, X):
        """Predict class labels for samples in X."""
        X_norm = self._normalize(np.asarray(X, dtype=np.float64))
        n_samples = X_norm.shape[0]
        predictions = []

        for i in range(n_samples):
            curr_node = 1
            # Traverse branch nodes down to leaf
            while curr_node in self.tree_structure_ and "d" in self.tree_structure_[curr_node]:
                node_info = self.tree_structure_[curr_node]
                if not node_info["d"]:
                    # Inactive branch defaults to left path
                    curr_node = 2 * curr_node
                else:
                    split_val = (
                            np.dot(node_info["a"], X_norm[i]) + node_info["b"]
                    )
                    if split_val < 0:
                        curr_node = 2 * curr_node  # Go left
                    else:
                        curr_node = 2 * curr_node + 1  # Go right

            # Leaf prediction assignment
            predictions.append(self.tree_structure_[curr_node]["class"])

        return np.array(predictions)

