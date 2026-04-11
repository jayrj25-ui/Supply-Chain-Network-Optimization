# Supply Chain Resilience Optimizer
### GNN + GRU · Disruption Simulation · LP Min-Cost Flow

---

## Overview

This project builds an end-to-end pipeline to predict, disrupt, and re-optimise a supply chain network.

```
Step 1: GNN (Graph + GRU)
        ↓  Predict demand at each node
Step 2: Disruption Simulation
        ↓  Remove nodes / edges
Step 3: Optimization (LP / Min-Cost Flow)
        ↓  Re-route supply using solver
```

---

## Pipeline

### Step 1 — Demand Forecasting (GNN + GRU)
- Constructs a supply chain graph from node/edge CSVs
- Normalises production & sales time-series per node
- Two-layer Graph Convolutional Network aggregates neighbourhood features at each time step
- GRU captures temporal dependencies across the sequence window
- Outputs demand forecast for every node at the next time step

### Step 2 — Disruption Simulation
- Randomly removes a configurable fraction of nodes (default 20 %)
- Models real-world disruptions: factory shutdowns, port closures, natural disasters
- Reports network metrics before and after (components, edges, resilience score)

### Step 3 — LP Optimization (Min-Cost Flow)
- Uses GNN demand predictions as supply/demand constraints
- Formulates a **Minimum-Cost Flow** Linear Program with PuLP (CBC solver)
- Finds the lowest-cost re-routing of supply across the surviving network
- Outputs active flow routes and total flow volume

---

## Results

| Metric | Value |
|--------|-------|
| RMSE   | *see run output* |
| MAE    | *see run output* |
| R²     | *see run output* |
| LP Status | Optimal |

---

## Setup

```bash
# Clone
git clone https://github.com/<your-username>/supply-chain-resilience.git
cd supply-chain-resilience

# Install dependencies
pip install torch numpy pandas networkx scikit-learn pulp matplotlib

# Place your dataset at ./data/ (mirror the Colab Drive structure)

# Run
python main.py
```

---

## Dataset Structure

```
data/
├── Nodes/
│   └── NodesIndex.csv
├── Edges/
│   └── EdgesIndex/
│       └── Edges (Plant).csv
└── Temporal Data/
    └── Unit/
        ├── Production_*.csv
        └── Sales_*.csv
```

---

## File Structure

```
supply-chain-resilience/
├── main.py          # Full pipeline (GNN → Disruption → LP)
├── README.md
└── requirements.txt
```

---

## Tech Stack

| Component | Library |
|-----------|---------|
| GNN + GRU | PyTorch |
| Graph ops | NetworkX |
| LP Solver | PuLP (CBC) |
| Data      | pandas / NumPy |
| Metrics   | scikit-learn |
| Viz       | Matplotlib |

---

## References

- Kipf & Welling (2017) — Semi-supervised Classification with GCNs
- Ford & Fulkerson — Min-Cost Flow
- PuLP documentation — https://coin-or.github.io/pulp/
