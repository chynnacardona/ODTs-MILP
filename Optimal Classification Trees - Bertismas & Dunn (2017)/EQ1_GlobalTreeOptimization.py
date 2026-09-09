import gurobipy as gp
from gurobipy import GRB
import numpy as np

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