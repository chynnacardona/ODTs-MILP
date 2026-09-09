# Optimal Classification Trees (OCT) via MILP in Gurobi -- GLOBAL TREE OPTIMIZATION

Python Implementation -- "EQ1_GlobalTreeOptimization" ()

Based on the foundational work by Bertsimas & Dunn (2017), this formulation finds globally optimal decision trees by simultaneously optimizing split selection, thresholds, and leaf class assignments rather than relying on greedy top-down heuristics (like CART).

---

## 📐 Mathematical Formulation

### Objective Function

The objective minimizes the total misclassification error $R_{xy}(T)$ plus a complexity penalty $\alpha \vert{}T\vert{}$ on active splits:

$$\min_{\mathbf{z}, \mathbf{c}, \mathbf{d}, \mathbf{b}, \mathbf{e}} \quad \sum_{i=1}^{N} \sum_{l \in L} e_{i, l} \;+\; \alpha \sum_{t \in B} \sum_{j=1}^{P} d_{t, j}$$

Where:
* $N$ is the total number of training samples.
* $P$ is the total number of features.
* $B$ is the set of branch (internal decision) nodes: $B = \{0, 1, \dots, 2^D - 2\}$.
* $L$ is the set of leaf nodes: $L = \{2^D - 1, 2^D, \dots, 2^{D+1} - 2\}$.
* $\alpha \ge 0$ is the regularization parameter controlling tree depth and pruning.

---

## 🛠️ Variables & Constraints

### 1. Decision Variables

| Variable | Domain | Definition / Role |
| :--- | :--- | :--- |
| $z_{i, l}$ | $\{0, 1\}$ | Sample routing: $1$ if sample $i$ is assigned to leaf $l$, $0$ otherwise |
| $c_l$ | $\{0, 1\}$ | Leaf prediction: Predicted class label for leaf node $l$ |
| $d_{t, j}$ | $\{0, 1\}$ | Feature selector: $1$ if branch node $t$ splits on feature $j$, $0$ otherwise |
| $b_t$ | $\mathbb{R}$ | Continuous cutoff threshold for branch node $t$ |
| $e_{i, l}$ | $\{0, 1\}$ | Misclassification tracker: $1$ if sample $i$ is misclassified in leaf $l$ |

---

### 2. Linearized Objective Constraints

To model misclassification without non-linear variable products ($z_{i, l} \cdot c_l$), linear lower bounds are enforced on $e_{i, l}$:

$$\begin{cases}  e_{i, l} \ge z_{i, l} - c_l & \text{if } y_i = 1 \\  e_{i, l} \ge c_l + z_{i, l} - 1 & \text{if } y_i = 0  \end{cases} \quad \forall i \in \{1, \dots, N\}, \, \forall l \in L$$

---

### 3. Structural Constraints

#### Feature Capacity
Each active decision node $t$ can split on at most one feature $j$:

$$\sum_{j=1}^{P} d_{t, j} \le 1 \quad \forall t \in B$$

#### Parent-Child Hierarchy
A child node $t$ can execute a split only if its parent node $p = \lfloor \frac{t - 1}{2} \rfloor$ also splits:

$$\sum_{j=1}^{P} d_{t, j} \le \sum_{j=1}^{P} d_{p, j} \quad \forall t \in B \setminus \{0\}$$

#### Unique Sample Allocation
Every sample $i$ must end up in exactly one leaf node $l$:

$$\sum_{l \in L} z_{i, l} = 1 \quad \forall i \in \{1, \dots, N\}$$

---

### 4. Dynamic Big-M Routing Constraints

Let $A_L(l)$ be the set of left ancestors and $A_R(l)$ be the set of right ancestors along the tree path to leaf $l$.

#### Left Ancestor Bounds ($t \in A_L(l)$)
Enforces $X_{i, j} \le b_t$ when sample $i$ lands in leaf $l$:

$$\sum_{j=1}^{P} d_{t, j} X_{i, j} \le b_t + M(1 - z_{i, l}) \quad \forall i, \, \forall l \in L, \, \forall t \in A_L(l)$$

#### Right Ancestor Bounds ($t \in A_R(l)$)
Enforces $X_{i, j} > b_t$ (via strictly positive offset $\epsilon$) when sample $i$ lands in leaf $l$:

$$\sum_{j=1}^{P} d_{t, j} X_{i, j} \ge b_t + \epsilon - M(1 - z_{i, l}) \quad \forall i, \, \forall l \in L, \, \forall t \in A_R(l)$$

* **Big-M Calculation:** $M = \max_{i, j} \vert{}X_{i, j}\vert{} + 5.0$
* **Epsilon Offset:** $\epsilon = 0.001$ (prevents boundary overlap ambiguity)
  
## ⚠️ Code vs. Theory Comparison

This table highlights the structural differences between naive Gurobi MILP implementations and the formal mathematical equations of Optimal Classification Trees (OCT).

| Code Element / Feature | Python Implementation | Formal Mathematical Equation | Impact & Explanation |
| :--- | :--- | :--- | :--- |
| **Root Node Feature Constraint** | `if t > 0:` loop skips `t = 0` for `gp.quicksum(d[t, j])` | $\sum_{j=1}^{P} d_{t, j} \le 1 \quad \forall t \in B$ | **Implementation Bug:** Omitted the feature selection constraint on the root node ($t=0$), allowing $d_{0,j}$ variables to remain unconstrained. |
| **Right Ancestor Loop Nesting** | `for t in right_anc:` placed inside `for t in left_anc:` | Separate sets $A_L(l)$ and $A_R(l)$ evaluated independently | **Implementation Bug:** Nesting the loops caused right ancestor constraints to execute inside the left ancestor loop, generating corrupted Big-M constraints. |
| **Right Branch Strict Inequality** | Uses non-strict lower bound: $\ge b_t - M(1 - z_{i, l})$ | Uses strict bound with offset: $\ge b_t + \epsilon - M(1 - z_{i, l})$ | **Mathematical Omission:** Without $\epsilon = 0.001$, samples on the exact boundary ($X_{i, j} = b_t$) satisfy both left ($\le$) and right ($\ge$) paths simultaneously. |
| **Big-M Value Scaling** | $M = \max(\|X\|) + 5.0$ | $M > \max_{i, j, t} \|X_{i, j} - b_t\|$ | **Correct Alignment:** Dynamically scaling $M$ relative to feature values prevents artificial solver over-pruning caused by static Big-M choices. |
| **Linearized Error Formulation** | `e >= z - c` $(y_i = 1)$<br>`e >= c + z - 1` $(y_i = 0)$ | $e_{i, l} \ge z_{i, l} - c_l \quad (y_i = 1)$<br>$e_{i, l} \ge c_l + z_{i, l} - 1 \quad (y_i = 0)$ | **Exact Alignment:** Algebraically replaces non-linear error products $e_{i, l} = \|y_i - c_l\| \cdot z_{i, l}$ with linear lower bounds. |
| **Parent-Child Hierarchy** | `quicksum(d[t]) <= quicksum(d[parent])` | $\sum_{j=1}^{P} d_{t, j} \le \sum_{j=1}^{P} d_{p, j} \quad \forall t \in B \setminus \{0\}$ | **Exact Alignment:** Enforces tree structural integrity by deactivating child node splits ($d_{t, j} = 0$) if the parent branch is inactive. |
| **Sample Leaf Allocation** | `gp.quicksum(z[i, l] for l in leaves) == 1` | $\sum_{l \in L} z_{i, l} = 1 \quad \forall i \in \{1, \dots, N\}$ | **Exact Alignment:** Guarantees every observation $i$ is routed to exactly one leaf node $l$. |
    
## 📄 Code vs. Paper Formulation (Bertsimas & Dunn, 2017)

### 📊 Structural & Constraint Comparison

| Mathematical Concept | Paper Equation | Python Code Equivalent | Implementation Notes & Differences |
| :--- | :--- | :--- | :--- |
| **Node Indexing** | Branch nodes: t = 1 to floor(T/2)<br>Leaf nodes: t = floor(T/2) + 1 to T | `branches = list(range(num_branches))` <br> `leaves = list(range(...))` | The paper utilizes 1-based indexing (1 to T), whereas the Python implementation uses standard 0-based indexing (0 to 2^(D+1) - 2). |
| **Feature Selection** | sum(a_jt for j=1 to p) = d_t for all t in T_B<br>*(Eq. 2)* | `d = model.addVars(branches, P, vtype=GRB.BINARY)`<br>`gp.quicksum(d[t, j] for j in range(P)) <= 1` | The paper explicitly separates feature choice binary a_jt and split flag binary d_t. The code collapses both into a single 2D binary variable d[t, j]. |
| **Parent Hierarchy** | d_t <= d_parent(t) for all t in T_B except root<br>*(Eq. 5)* | `gp.quicksum(d[t, j]) <= gp.quicksum(d[parent, j])` | **Exact Alignment:** Enforces that a child node cannot execute a split unless its parent node has already split. |
| **Leaf Allocation** | sum(z_it for t in T_L) = 1 for all sample i<br>*(Eq. 8)* | `gp.quicksum(z[i, l] for l in leaves) == 1` | **Exact Alignment:** Guarantees every observation i is routed to exactly one leaf node. |
| **Left Ancestor Routing** | a_m^T * (x_i + epsilon) <= b_m + (1 + epsilon_max) * (1 - z_it)<br>*(Eq. 13)* | `gp.quicksum(d[t, j] * X[i, j]) <= b[t] + M * (1 - z[i, l])` | The paper derives feature-specific tolerances epsilon_j and tight Big-M bounds (1 + epsilon_max). The code applies a dynamic scalar M = max(|X|) + 5.0. |
| **Right Ancestor Routing** | a_m^T * x_i >= b_m - (1 - z_it)<br>*(Eq. 14)* | `gp.quicksum(d[t, j] * X[i, j]) >= b[t] - M * (1 - z[i, l])` | The paper assumes normalized features x_i in [0, 1] (setting M = 1). The code uses dynamic M scaling to handle unnormalized feature values. |
| **Misclassification Loss** | L_t = N_t - max_k(N_kt)<br>*(Eq. 19–22)* | `e[i, l] >= z[i, l] - c[l]` (for y_i = 1)<br>`e[i, l] >= c[l] + z[i, l] - 1` (for y_i = 0) | The paper computes class distribution counts N_kt per leaf. The code uses sample-level binary error variables e_il tailored for binary classification (K = 2). |
| **Objective Function** | min (1 / L_hat) * sum(L_t for t in T_L) + alpha * sum(d_t for t in T_B)<br>*(Eq. 24)* | `model.setObjective(misclass_error + tree_complexity, GRB.MINIMIZE)` | The paper scales classification loss by baseline error L_hat. The code minimizes raw total misclassifications plus complexity penalty alpha * sum(d_tj). |
| **Minimum Leaf Size** | sum(z_it for i=1 to n) >= N_min * l_t for all t in T_L<br>*(Eq. 7)* | *Omitted in current Python code* | **Paper Constraint Omitted:** The paper enforces a minimum sample count threshold N_min per active leaf node using binary leaf indicators l_t. |