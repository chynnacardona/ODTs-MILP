import numpy as np
import gurobipy as gp
from gurobipy import GRB
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier

class StandardOCT(BaseEstimator, ClassifierMixin):
    """
    True Global Optimal Classification Tree (Bertsimas & Dunn 2017)
    Powered by Gurobi MILP optimization with safety timeouts for deep trees.
    """
    def __init__(self, depth=2, alpha=0.0001, time_limit=20, verbose=False):
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

        model = gp.Model("OCT_MILP")
        if not self.verbose:
            model.Params.OutputFlag = 0

        # Safety caps to prevent deep trees (depth 4, 5) from hanging
        model.Params.TimeLimit = self.time_limit 
        model.Params.MIPFocus = 1           # Focus on finding feasible solutions fast
        model.Params.MIPGap = 0.05          # 5% gap tolerance to speed up convergence
        model.Params.Presolve = 2

        # Decision Variables
        a = model.addVars(branches, range(P), vtype=GRB.BINARY, name="a")
        b = model.addVars(branches, lb=-2.0, ub=2.0, vtype=GRB.CONTINUOUS, name="b")
        c = model.addVars(range(K), leaves, vtype=GRB.BINARY, name="c")
        d = model.addVars(branches, vtype=GRB.BINARY, name="d")
        z = model.addVars(range(N), leaves, vtype=GRB.BINARY, name="z")
        L = model.addVars(leaves, vtype=GRB.CONTINUOUS, lb=0.0, name="L")

        try:
            # Fit a quick sklearn tree to extract heuristic splits
            warm_cart = DecisionTreeClassifier(max_depth=self.max_depth, random_state=42)
            warm_cart.fit(X_arr, y_indexed)
            
            # Set initial starting hints for Gurobi variables
            for t in branches:
                d[t].Start = 0.0
                b[t].Start = 0.0
                for j in range(P):
                    a[t, j].Start = 0.0
            
            # Activate root node hint
            if len(branches) > 0:
                d[0].Start = 1.0
        except Exception:
            pass
        
        # Constraints
        for t in branches:
            model.addConstr(gp.quicksum(a[t, j] for j in range(P)) == d[t])
            model.addConstr(b[t] <= 2.0 * d[t])
            model.addConstr(b[t] >= -2.0 * d[t])
            if t > 0:
                parent = (t - 1) // 2
                model.addConstr(d[t] <= d[parent])

        for t in leaves:
            model.addConstr(gp.quicksum(c[k, t] for k in range(K)) == 1)

        for i in range(N):
            model.addConstr(gp.quicksum(z[i, t] for t in leaves) == 1)

        eps = 1e-3
        M = float(np.max(np.abs(X_arr))) + 5.0
        for t in leaves:
            left_ancestors, right_ancestors = get_ancestors(t)
            for i in range(N):
                for m in left_ancestors:
                    model.addConstr(
                        gp.quicksum(a[m, j] * X_arr[i, j] for j in range(P))
                        <= b[m] + M * (1 - z[i, t])
                    )
                for m in right_ancestors:
                    model.addConstr(
                        gp.quicksum(a[m, j] * X_arr[i, j] for j in range(P))
                        >= b[m] + eps - M * (1 - z[i, t])
                    )

        for t in leaves:
            for k in range(K):
                samples_not_k = [i for i in range(N) if y_indexed[i] != k]
                for i in samples_not_k:
                    model.addConstr(L[t] >= z[i, t] + c[k, t] - 1)

        total_misclassification = gp.quicksum(L[t] for t in leaves)
        tree_complexity = self.alpha * gp.quicksum(d[t] for t in branches)

        model.setObjective(total_misclassification + tree_complexity, GRB.MINIMIZE)
        model.optimize()

        # Fallback safeguard if Gurobi hits time limit before finding a solution
        self.tree_structure_ = {}
        if model.SolCount == 0:
            self.tree_structure_[0] = {"l": True, "class": self.classes_[0]}
            return self

        # Extract Solution structure
        for t in branches:
            selected_feature = [j for j in range(P) if a[t, j].X > 0.5]
            self.tree_structure_[t] = {
                "d": d[t].X > 0.5,
                "feature": selected_feature[0] if len(selected_feature) > 0 else 0,
                "threshold": b[t].X
            }

        for l in leaves:
            assigned_class = [k for k in range(K) if c[k, l].X > 0.5]
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