# Optimal Decision Trees via Mixed-Integer Linear Programming (MILP)

This repository provides Python implementations using **Gurobi** to build **Optimal Classification Trees with Hyperplane Splits (OCT-H)** by formulating decision tree learning as Mixed-Integer Linear Programs (MILP) and Mixed-Integer Quadratically Constrained Programs (MIQCP).

Based on research in inherently interpretable machine learning for credit scoring and imbalanced datasets (Tu & Wu, 2025), this repository includes exact MIP formulations, cost-sensitive variants, F1-score maximization models, and scalable heuristics.

---

## 📌 Key Features

* **OCT-H Formulation**: Solves for global tree structure using MIP rather than greedy recursive splitting (like CART).
* **Cost-Sensitive Learning (CSOCT-H)**: Addresses class imbalance by directly penalizing False Positives and False Negatives with customizable class weights or financial costs.
* **F1-Score Maximization (OCT-H-F1)**: Directly maximizes the harmonic mean of Precision and Recall via MIQCP formulation.
* **Scalable Solution Methods**:
  * **Data Sample Reduction**: Discretizes features and groups identical rows to reduce variable count.
  * **Heuristic Splitting (LBC)**: Fast recursive partition solver leveraging $D=1$ linear binary classification models.
  * **Warm Start Acceleration**: Uses decision trees (e.g., CART or $D-1$ depth OCT-H) as starting bounds for Gurobi to speed up convergence.

---

## 🛠️ Prerequisites & Installation

### Requirements
* **Python**: 3.9 or higher
* **Gurobi Optimizer**: Version 10.0+ with a valid Gurobi license (academic or commercial)

### 1. Set Up Gurobi
Make sure Gurobi is installed on your system and your license is activated via `grbgetkey`:
```bash
grbgetkey YOUR-LICENSE-KEY-HERE