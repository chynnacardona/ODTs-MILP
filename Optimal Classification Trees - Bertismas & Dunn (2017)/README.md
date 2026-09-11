# Optimal Classification Trees (OCT) — Bertsimas & Dunn (2017)

An implementation of the **Optimal Classification Trees (OCT)** and **OCT with Hyperplanes (OCT-H)** algorithms introduced by Dimitris Bertsimas and Jack Dunn in their 2017 paper, *"[Optimal Classification Trees](https://dspace.mit.edu/handle/1721.1/111440)"* (Operations Research, 65(5), 1352-1386).

---

## 📌 Executive Summary

Traditional decision tree algorithms like **CART** (Classification and Regression Trees) rely on greedy top-down heuristics. While computationally fast, CART makes local decisions at each node without considering downstream impacts, often leading to sub-optimal global decision structures.

This project implements **Mixed-Integer Linear Optimization (MILO)** models that optimize the entire decision tree structure simultaneously. By treating tree creation as a single global optimization problem, OCT achieves optimal or near-optimal classification performance, superior generalization on noisy datasets, and better interpretability.

---

## 💡 Key Contributions of the Paper

1. **Global Optimality via MILO**: Formulates the tree-building process as a unified Mixed-Integer Optimization (MIO) problem, jointly optimizing split features, thresholds, leaf assignments, and class predictions.
2. **Warm Start Acceleration**: Demonstrates that passing a pre-trained CART model as an initial feasible solution (`warm start`) accelerates solver convergence by $2.5\times \text{ to } 5\times$.
3. **Multivariate Hyperplane Splits (OCT-H)**: Extends standard axis-aligned univariate splits ($a^T x \le b$) to multivariate linear combinations, offering superior flexibility without sacrificing interpretability.
4. **Synthetic Ground Truth Recovery**: Proves empirically that OCT recovers true underlying data distributions far better than greedy heuristics under label/feature noise.

---

## 📂 Repository Directory Structure

```text
Optimal Classification Trees - Bertsimas & Dunn (2017)/
│
├── .idea/                             # IDE configurations
├── Optimal classification trees.pdf  # Reference paper (Bertsimas & Dunn, 2017)
├── README.md                          # Main repository documentation
│
├── core/                              # Core algorithms & formulations
│   ├── README.md                      # Core module documentation
│   ├── GreedyCART.py                  # Baseline CART implementation / scikit-learn wrapper
│   ├── EQ1_GlobalTreeOptimization.py  # Base OCT MIO formulation (Section 2.2)
│   ├── WarmStartMapper.py              # Translates CART trees into MIO initial solutions (Section 2.3)
│   └── OCT_Hyperplane.py              # OCT-H multivariate split model (Section 4.1)
│
├── utils/                             # Helper scripts
│   ├── data_generator.py              # Synthetic tree & sample generation routines (Section 5)
│   └── metrics.py                     # Evaluation utilities (Out-of-sample accuracy, tree depth)
│
└── experiments/                       # Experimental benchmarks
    ├── synthetic_benchmark.py         # Ground truth recovery under noise (Section 5)
    └── uci_benchmark.py               # Real-world benchmark comparisons vs. CART/RF (Section 6 & 7)