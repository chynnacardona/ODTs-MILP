import numpy as np
import pyomo.environ as pyo
import gurobipy as gp
from gurobipy import GRB
from sklearn.tree import DecisionTreeClassifier
from WarmStartMapper import WarmStartMapper

def solve_oct(X_train: np.ndarray,
              y_train: np.ndarray,
              max_depth: int = 3,
              alpha: float = 0.01,
              use_warm_start: bool = True):
    n_samples, n_features = X_train.shape
    classes = np.unique(y_train)
    n_classes = len(classes)

    num_branches = (2 ** max_depth) - 1
    num_leaves = 2 ** max_depth
    branches = list(range(num_branches))
    leaves = list(range(num_branches, num_branches + num_leaves))

    model = pyo.ConcreteModel()

    # Sets
    model.N = pyo.Set(initialize=range(n_samples))
    model.P = pyo.Set(initialize=range(n_features))
    model.Branches = pyo.Set(initialize=branches)
    model.Leaves = pyo.Set(initialize=leaves)
    model.Classes = pyo.Set(initialize=range(n_classes))

    # Decision Variables
    model.a = pyo.Var(model.Branches, model.P, domain=pyo.Binary)
    model.b = pyo.Var(model.Branches, domain=pyo.Reals, bounds=(-2.0, 2.0))
    model.d = pyo.Var(model.Branches, domain=pyo.Binary)
    model.c = pyo.Var(model.Classes, model.Leaves, domain=pyo.Binary)
    model.z = pyo.Var(model.N, model.Leaves, domain=pyo.Binary)
    model.L_error = pyo.Var(model.N, model.Leaves, domain=pyo.NonNegativeReals)

    # Objective Function
    def objective_rule(m):
        misclass_error = sum(m.L_error[i, l] for i in m.N for l in m.Leaves)
        complexity_penalty = alpha * sum(m.d[t] for t in m.Branches)
        return misclass_error + complexity_penalty

    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Constraints
    def active_split_rule(m, t):
        return sum(m.a[t, j] for j in m.P) == m.d[t]
    model.active_split_con = pyo.Constraint(model.Branches, rule=active_split_rule)

    def hierarchy_rule(m, t):
        if t == 0:
            return pyo.Constraint.Skip
        parent = (t - 1) // 2
        return m.d[t] <= m.d[parent]
    model.hierarchy_con = pyo.Constraint(model.Branches, rule=hierarchy_rule)

    def sample_routing_rule(m, i):
        return sum(m.z[i, l] for l in m.Leaves) == 1
    model.sample_routing_con = pyo.Constraint(model.N, rule=sample_routing_rule)

    def leaf_class_rule(m, l):
        return sum(m.c[k, l] for k in m.Classes) == 1
    model.leaf_class_con = pyo.Constraint(model.Leaves, rule=leaf_class_rule)

    def error_tracking_rule(m, i, l):
        actual_class = y_train[i]
        return m.L_error[i, l] >= m.z[i, l] - m.c[actual_class, l]
    model.error_tracking_con = pyo.Constraint(model.N, model.Leaves, rule=error_tracking_rule)

    # Big-M Ancestor Helper (0-based)
    M = float(np.max(np.abs(X_train))) + 5.0
    eps = 0.001

    def get_ancestor_routing(leaf_idx):
        left_anc, right_anc = [], []
        curr = leaf_idx
        while curr > 0:
            parent = (curr - 1) // 2
            if curr % 2 == 1:
                left_anc.append(parent)
            else:
                right_anc.append(parent)
            curr = parent
        return left_anc, right_anc

    def bigm_left_rule(m, i, l, t):
        return sum(m.a[t, j] * X_train[i, j] for j in m.P) <= m.b[t] + M * (1 - m.z[i, l])

    def bigm_right_rule(m, i, l, t):
        return sum(m.a[t, j] * X_train[i, j] for j in m.P) >= m.b[t] + eps - M * (1 - m.z[i, l])

    model.left_ancestor_cons = pyo.ConstraintList()
    model.right_ancestor_cons = pyo.ConstraintList()

    for l in leaves:
        left_anc, right_anc = get_ancestor_routing(l)
        for i in range(n_samples):
            for t in left_anc:
                model.left_ancestor_cons.add(bigm_left_rule(model, i, l, t))
            for t in right_anc:
                model.right_ancestor_cons.add(bigm_right_rule(model, i, l, t))

    # Warm Start Injection
    if use_warm_start:
        cart = DecisionTreeClassifier(max_depth=max_depth)
        cart.fit(X_train, y_train)

        mapper = WarmStartMapper(max_depth=max_depth, n_features=n_features, n_classes=n_classes)
        warm_start = mapper.extract_cart_parameters(cart, X_train)

        for (t_1based, j), val in warm_start.get("a", {}).items():
            t_0based = t_1based - 1
            if t_0based in branches and j < n_features:
                model.a[t_0based, j].value = val

        for t_1based, val in warm_start.get("b", {}).items():
            t_0based = t_1based - 1
            if t_0based in branches:
                model.b[t_0based].value = val

        for t_1based, val in warm_start.get("d", {}).items():
            t_0based = t_1based - 1
            if t_0based in branches:
                model.d[t_0based].value = val

        for (k, t_1based), val in warm_start.get("c", {}).items():
            t_0based = t_1based - 1
            if t_0based in leaves:
                model.c[k, t_0based].value = val

        for (i, t_1based), val in warm_start.get("z", {}).items():
            t_0based = t_1based - 1
            if t_0based in leaves:
                model.z[i, t_0based].value = val

    solver = pyo.SolverFactory("gurobi")
    results = solver.solve(model, warmstart=use_warm_start, tee=True)

    return model, results


def optimal_tree_milp(X: np.ndarray,
                      y: np.ndarray,
                      max_depth: int = 2,
                      alpha: float = 0.0001,
                      time_limit_sec: float = 300,
                      use_warm_start: bool = True):
    N, P = X.shape
    classes = np.unique(y)
    K = len(classes)

    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    y_indexed = np.array([class_to_idx[val] for val in y])

    num_branches = (2 ** max_depth) - 1
    num_leaves = 2 ** max_depth
    branches = list(range(num_branches))
    leaves = list(range(num_branches, num_branches + num_leaves))

    # Fixed 0-based ancestor helper
    def get_ancestors(node_id):
        left_ancestors, right_ancestors = [], []
        curr = node_id
        while curr > 0:
            parent = (curr - 1) // 2
            if curr % 2 == 1:  # Left child in 0-based indexing
                left_ancestors.append(parent)
            else:  # Right child in 0-based indexing
                right_ancestors.append(parent)
            curr = parent
        return left_ancestors, right_ancestors

    model = gp.Model("OCT_MILP")
    model.Params.TimeLimit = time_limit_sec

    # Decision Variables
    a = model.addVars(branches, range(P), vtype=GRB.BINARY, name="a")
    b = model.addVars(branches, lb=-2.0, ub=2.0, vtype=GRB.CONTINUOUS, name="b")
    c = model.addVars(range(K), leaves, vtype=GRB.BINARY, name="c")
    d = model.addVars(branches, vtype=GRB.BINARY, name="d")
    z = model.addVars(range(N), leaves, vtype=GRB.BINARY, name="z")
    L = model.addVars(leaves, vtype=GRB.CONTINUOUS, lb=0.0, name="L")

    # Warm Start Injection
    if use_warm_start:
        cart = DecisionTreeClassifier(max_depth=max_depth)
        cart.fit(X, y_indexed)

        mapper = WarmStartMapper(max_depth=max_depth, n_features=P, n_classes=K)
        warm_start = mapper.extract_cart_parameters(cart, X)

        for (t_1based, j), val in warm_start.get("a", {}).items():
            t_0based = t_1based - 1
            if t_0based in branches and j < P:
                a[t_0based, j].Start = val

        for t_1based, val in warm_start.get("b", {}).items():
            t_0based = t_1based - 1
            if t_0based in branches:
                b[t_0based].Start = val

        for t_1based, val in warm_start.get("d", {}).items():
            t_0based = t_1based - 1
            if t_0based in branches:
                d[t_0based].Start = val

        for (k, t_1based), val in warm_start.get("c", {}).items():
            t_0based = t_1based - 1
            if t_0based in leaves:
                c[k, t_0based].Start = val

        for (i, t_1based), val in warm_start.get("z", {}).items():
            t_0based = t_1based - 1
            if t_0based in leaves:
                z[i, t_0based].Start = val

    # Constraints
    for t in branches:
        model.addConstr(gp.quicksum(a[t, j] for j in range(P)) == d[t])
        # Fixed threshold bounds scaling with d[t]
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
    M = float(np.max(np.abs(X))) + 5.0
    for t in leaves:
        left_ancestors, right_ancestors = get_ancestors(t)
        for i in range(N):
            for m in left_ancestors:
                model.addConstr(
                    gp.quicksum(a[m, j] * X[i, j] for j in range(P))
                    <= b[m] + M * (1 - z[i, t])
                )
            for m in right_ancestors:
                model.addConstr(
                    gp.quicksum(a[m, j] * X[i, j] for j in range(P))
                    >= b[m] + eps - M * (1 - z[i, t])
                )

    for t in leaves:
        for k in range(K):
            samples_not_k = [i for i in range(N) if y_indexed[i] != k]
            for i in samples_not_k:
                model.addConstr(L[t] >= z[i, t] + c[k, t] - 1)

    total_misclassification = gp.quicksum(L[t] for t in leaves)
    tree_complexity = alpha * gp.quicksum(d[t] for t in branches)

    model.setObjective(total_misclassification + tree_complexity, GRB.MINIMIZE)
    model.optimize()

    print("\n" + "=" * 45)
    print("      TREE OPTIMIZATION RESULTS")
    print("=" * 45)

    for t in branches:
        if d[t].X > 0.5:
            selected_feature = None
            for j in range(P):
                if a[t, j].X > 0.5:
                    selected_feature = j
                    break
            print(f"Branch Node {t}: Split on Feature {selected_feature} at threshold {b[t].X:.4f}")
        else:
            print(f"Branch Node {t}: Inactive (Pruned)")

    print("\nLeaf Predictions:")
    for l in leaves:
        for k in range(K):
            if c[k, l].X > 0.5:
                print(f"  Leaf {l} -> Predicts Class {k}")
                break
    print("=" * 45 + "\n")

    return model

if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.uniform(-1, 1, (40, 2))
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, 0)

    optimal_tree_milp(X, y, max_depth=2, alpha=0.0)