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



