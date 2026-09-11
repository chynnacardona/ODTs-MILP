import numpy as np
import pyomo.environ as pyo
import gurobipy as gp
from gurobipy import GRB
from sklearn.tree import DecisionTreeClassifier
from WarmStartMapper import WarmStartMapper


def solve_oct(X_train: np.ndarray, y_train: np.ndarray, max_depth: int = 3, alpha: float = 0.01,
              use_warm_start: bool = True):
    n_samples, n_features = X_train.shape
    n_classes = len(np.unique(y_train))

    num_branches = (2 ** max_depth) - 1
    num_leaves = 2 ** max_depth
    branches = list(range(num_branches))
    leaves = list(range(num_branches, num_branches + num_leaves))

    # --- 1. Build Pyomo MIO Model ---
    model = pyo.ConcreteModel()

    # Sets
    model.N = pyo.Set(initialize=range(n_samples))
    model.P = pyo.Set(initialize=range(n_features))
    model.Branches = pyo.Set(initialize=range((2 ** max_depth) - 1))
    model.Leaves = pyo.Set(initialize=range((2 ** max_depth) - 1, (2 ** (max_depth + 1)) - 1))
    model.Classes = pyo.Set(initialize=range(n_classes))

    # Decision Variables matching WarmStartMapper keys
    model.a = pyo.Var(model.Branches, model.P, domain=pyo.Binary)
    model.b = pyo.Var(model.Branches, domain=pyo.Reals, bounds=(-2.0, 2.0))
    model.d = pyo.Var(model.Branches, domain=pyo.Binary)
    model.c = pyo.Var(model.Classes, model.Leaves, domain=pyo.Binary)
    model.z = pyo.Var(model.N, model.Leaves, domain=pyo.Binary)

    # Objective Function: Misclassification Error + Alpha Complexity Penalty
    def objective_rule(m):
        misclass_error = sum(m.L_error[i, l] for i in m.I for l in m.TL)
        complexity_penalty = alpha * sum(m.d[t] for t in m.TB)
        return misclass_error + complexity_penalty

    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Constraints
    def active_split_rule(m, t):
        return sum(m.a[t, j] for j in m.J) == m.d[t]

    model.active_split_con = pyo.Constraint(model.TB, rule=active_split_rule)

    def hierarchy_rule(m, t):
        if t == 0:
            return pyo.Constraint.Skip
        parent = (t - 1) // 2
        return m.d[t] <= m.d[parent]

    model.hierarchy_con = pyo.Constraint(model.TB, rule=hierarchy_rule)

    def sample_routing_rule(m, i):
        return sum(m.z[i, l] for l in m.TL) == 1

    model.sample_routing_con = pyo.Constraint(model.I, rule=sample_routing_rule)

    def leaf_class_rule(m, l):
        return sum(m.c[k, l] for k in m.K) == m.L[l]

    model.leaf_class_con = pyo.Constraint(model.TL, rule=leaf_class_rule)

    def error_tracking_rule(m, i, l):
        actual_class = y_train[i]
        return m.L_error[i, l] >= m.z[i, l] - m.c[actual_class, l]

    model.error_tracking_con = pyo.Constraint(model.I, model.TL, rule=error_tracking_rule)

    # Big-M Routing Logic
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
        return sum(m.a[t, j] * X_train[i, j] for j in m.J) <= m.b[t] + M * (1 - m.z[i, l])

    def bigm_right_rule(m, i, l, t):
        return sum(m.a[t, j] * X_train[i, j] for j in m.J) >= m.b[t] + eps - M * (1 - m.z[i, l])

    model.left_ancestor_cons = pyo.ConstraintList()
    model.right_ancestor_cons = pyo.ConstraintList()

    for l in leaves:
        left_anc, right_anc = get_ancestor_routing(l)
        for i in range(n_samples):
            for t in left_anc:
                model.left_ancestor_cons.add(bigm_left_rule(model, i, l, t))
            for t in right_anc:
                model.right_ancestor_cons.add(bigm_right_rule(model, i, l, t))

    # --- 2. Instantiate Solver ---
    solver = pyo.SolverFactory('gurobi')

    # --- 3. Optional Warm Start Injection ---
    if use_warm_start:
        # Fit greedy CART baseline
        cart = DecisionTreeClassifier(max_depth=max_depth)
        cart.fit(X_train, y_train)

        # Extract initial solution parameters
        mapper = WarmStartMapper(max_depth=max_depth, n_features=n_features, n_classes=n_classes)
        warm_start = mapper.extract_cart_parameters(cart, X_train)

        # Inject warm start values into Pyomo model variables
        for (t, j), val in warm_start['a'].items():
            model.a[t, j].value = val

        for t, val in warm_start['b'].items():
            model.b[t].value = val

        for t, val in warm_start['d'].items():
            model.d[t].value = val

        for (k, t), val in warm_start['c'].items():
            model.c[k, t].value = val

        for (i, t), val in warm_start['z'].items():
            model.z[i, t].value = val

    # --- 4. Execute Solver ---
    results = solver.solve(model, warmstart=use_warm_start, tee=True)

    return model, results

def optimal_tree_milp(X, y, max_depth=2, alpha=0.0001, time_limit_sec=60):
    """
        Builds a Global Optimal Decision Tree of arbitrary depth D using Gurobi.
        Tree Architecture:
          - Branch nodes: 0 to (2^D - 2)
          - Leaf nodes: (2^D - 1) to (2^(D+1) - 2)
    """

    #Solves EQ (1): min R_xy(T) + a*|T|
    N, P = X.shape # N for no. of rows, P for no. of columns

    # DYNAMICALLY GENERATE NODE INDEXES
    num_branches = (2 ** max_depth) - 1 # Calculates total internal decision nodes 2^D -1
    num_leaves = 2 ** max_depth # Calculates total leaf nodes 2^D
    branches = list(range(num_branches)) # Generates 0-based index list for branch nodes: [0,...,2^D - 2]
    leaves = list(range(num_branches, num_branches + num_leaves)) # Generates index list for leaf nodes: [2^D - 1,...,2^D+1 - 2]

    model = gp.Model("Global_Optimal_DT")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit_sec

    # DECISION VARIABLES
    # z -- binary variables, where z[i, l] = 1 if sample i is routed to leaf l (SAMPLE ASSIGNMENT)
    z = model.addVars(N, leaves, vtype=GRB.BINARY, name = "z")

    # c[l] = 1 if leaf l predicts Class 1, 0 if Class 0 (ex. Class 1 = healthy, Class 0 = sick, LEAF ACTIVATION)
    c = model.addVars(leaves, vtype=GRB.BINARY, name="c")

    # Feature selector: d[t, j] = 1 if node t splits on feature j
    d = model.addVars(branches, P, vtype=GRB.BINARY, name="d")

    # Threshold variable for node t (SPLIT THRESHOLDS)
    b = model.addVars(branches, lb=-2.0, ub=2.0, vtype=GRB.CONTINUOUS, name="b")

    # Error variable to keep objective purely linear.  Creates binary error-tracking helper variables for sample i at leaf l.
    e = model.addVars(N, leaves, vtype=GRB.BINARY, name="e")

    # LINEARIZED OBJECTIVE CONSTRAINTS
    # Sets linear lower bounds on e_i,l to track misclassifications w/o multiplying variables (z_i,l * c_l)
    for i in range(N):
        for l in leaves:
            if y[i] == 1:
                model.addConstr(e[i, l] >= z[i, l] - c[l])
            else:
                model.addConstr(e[i, l] >= c[l] + z[i, l] - 1)

    # OBJECTIVE FUNCTION
    # Constructs and sets the overall objective to minimize classification errors
    misclass_error = gp.quicksum(e[i, l] for i in range(N) for l in leaves)
    # + Complexity penalty (count active splits)
    tree_complexity = alpha * gp.quicksum(d[t, j] for t in branches for j in range (P))
    model.setObjective(misclass_error + tree_complexity, GRB.MINIMIZE)

    # GLOBAL CONSTRAINTS
    # Exactly 1 feature per active branch node
    for t in branches:
        model.addConstr(gp.quicksum(d[t, j] for j in range(P)) <= 1, name=f"Max1Feat_{t}")

        if t > 0:
            parent = (t - 1) // 2
            # Branch t can only split if its parent also split
            model.addConstr(
                gp.quicksum(d[t, j] for j in range(P)) <= gp.quicksum(d[parent, j] for j in range(P)),
                name=f"Parent_Hierarchy_{t}"
            )

    # Every sample i must end up exactly one leaf
    for i in range(N):
        model.addConstr(gp.quicksum(z[i, l] for l in leaves) == 1)

    # DYNAMIC ANCESTOR ROUTING SET CALCULATOR
    def get_ancestor_routing(leaf_idx):
        #Traverses upward from leaf to root to find left (A_L) and right (A_R) ancestors.
        left_anc = []
        right_anc = []

        curr = leaf_idx
        while curr > 0:
            parent = (curr - 1) // 2
            if curr % 2 == 1:
                left_anc.append(parent)
            else:
                right_anc.append(parent)
            curr = parent

        return left_anc, right_anc

    # DYNAMIC BIG-M ROUTING CONSTRAINTS
    M = float(np.max(np.abs(X))) + 5.0
    #c Defines a small scalar offset (epsilon) to represent strict inequality (>) in right ancestor splits using standard non-strict (>=) MILP constraints.
    eps = 0.001

    # Iterates over each leaf l and calls get_ancestor_routing(l) to construct its specific sets of left & right anc
    for l in leaves:
        left_anc, right_anc = get_ancestor_routing(l)

        for i in range(N):
            # Enforce X_i * d_t <= b_t for all left ancestors
            for t in left_anc:
                model.addConstr(
                    gp.quicksum(d[t, j] * X[i, j] for j in range(P)) <= b[t] + M * (1 - z[i, l]),
                    name =f"BigM-Left_{i}_{l}_{t}"
                )

            # Enforce X_i * d_t >= b_t for all right ancestors
            for t in right_anc:
                model.addConstr(
                    gp.quicksum(d[t, j] * X[i, j] for j in range(P)) >= b[t] + eps - M * (1 - z[i, l]),
                    name =f"BigM-Right_{i}_{l}_{t}"
                )
    # RUN SOLVER
    model.optimize()

    # EXTRACT OUTPUT
    if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
        print("\n" + "=" * 40)
        print("       TREE OPTIMIZATION RESULTS")
        print("=" * 40)

        # 1. Branch Splits
        for t in branches:
            # Query the solved values from variable 'd' using .X
            chosen_feat = [j for j in range(P) if d[t, j].X > 0.5]
            if chosen_feat:
                feature_idx = chosen_feat[0]
                threshold_val = b[t].X
                print(f"Branch Node {t}: Split on Feature {feature_idx} at threshold {threshold_val:.4f}")
            else:
                print(f"Branch Node {t}: Inactive (Pruned)")

        # 2. Leaf Predictions
        print("\nLeaf Predictions:")
        for l in leaves:
            print(f"  Leaf {l} -> Predicts Class {int(round(c[l].X))}")

        return model
    else:
        print("No optimal solution found.")
        return None

# DUMMY DATA
if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.uniform(-1, 1, (40, 2))
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, 0)

    optimal_tree_milp(X, y, max_depth=2, alpha=0.0)