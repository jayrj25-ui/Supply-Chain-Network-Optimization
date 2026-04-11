"""
Supply Chain Resilience Optimizer
===================================
Pipeline:
  Step 1  → GNN (Graph + GRU)  : Predict demand at each node
  Step 2  → Disruption Simulation : Remove nodes/edges
  Step 3  → Optimization (LP / Min-Cost Flow) : Re-route supply

Author : [Your Name]
"""

# ─────────────────────────────────────────────
# 0. IMPORTS & SETUP
# ─────────────────────────────────────────────
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F

import pulp

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
DATA_ROOT   = "data"          # <-- point to your dataset folder
SEQ_LEN     = 10
EPOCHS      = 100
LR          = 0.001
TRAIN_RATIO = 0.8
DISRUPT_RATE = 0.2            # fraction of nodes to remove in disruption


# ─────────────────────────────────────────────
# 2. DATA LOADING
# ─────────────────────────────────────────────
def load_nodes(data_root: str):
    """Load node index mapping."""
    nodes_index = pd.read_csv(f"{data_root}/Nodes/NodesIndex.csv")
    node2idx = dict(zip(nodes_index["Node"], nodes_index["NodeIndex"]))
    idx2node = {v: k for k, v in node2idx.items()}
    return node2idx, idx2node


def load_temporal_data(data_root: str, keyword: str) -> pd.DataFrame:
    """Find and load a temporal CSV file by keyword."""
    path = f"{data_root}/Temporal Data/Unit"
    for f in os.listdir(path):
        if keyword.lower() in f.lower():
            return pd.read_csv(f"{path}/{f}")
    raise FileNotFoundError(f"File with keyword '{keyword}' not found in {path}")


def align_to_nodes(df: pd.DataFrame, node2idx: dict, num_nodes: int) -> np.ndarray:
    """Align dataframe columns to node indices."""
    df.columns = df.columns.str.strip()
    arr = np.zeros((len(df), num_nodes))
    for col in df.columns:
        if col in node2idx:
            arr[:, node2idx[col]] = df[col].values
    return arr


def load_graph(data_root: str, device: torch.device):
    """Build NetworkX graph and adjacency matrix from edge file."""
    edges_df = pd.read_csv(f"{data_root}/Edges/EdgesIndex/Edges (Plant).csv")
    src = edges_df.iloc[:, 1].values
    dst = edges_df.iloc[:, 2].values

    edges = [(int(s), int(d)) for s, d in zip(src, dst) if s != d]
    G = nx.Graph()
    G.add_edges_from(edges)

    A = nx.to_numpy_array(G)
    A = torch.tensor(A, dtype=torch.float32).to(device)
    return G, A


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def build_node_features(prod: np.ndarray, sales: np.ndarray, num_nodes: int) -> np.ndarray:
    """Normalize per node and stack production + sales as two-channel features."""
    prod_norm  = np.zeros_like(prod)
    sales_norm = np.zeros_like(sales)

    for i in range(num_nodes):
        prod_norm[:, i]  = MinMaxScaler().fit_transform(prod[:, i].reshape(-1, 1)).flatten()
        sales_norm[:, i] = MinMaxScaler().fit_transform(sales[:, i].reshape(-1, 1)).flatten()

    return np.stack([prod_norm, sales_norm], axis=-1)   # shape: (T, N, 2)


def create_sequences(data: np.ndarray, seq_len: int):
    """Create (X, Y) sliding window sequences. Y = production at t+1."""
    X, Y = [], []
    for t in range(len(data) - seq_len):
        X.append(data[t:t + seq_len])
        Y.append(data[t + seq_len, :, 0])
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.float32))


# ─────────────────────────────────────────────
# 4. MODEL  –  GNN + GRU
# ─────────────────────────────────────────────
class SupplyChainGNN(nn.Module):
    """
    Two-layer Graph Convolutional Network (GCN) followed by a GRU.

    Architecture
    ─────────────
    For each time step t in the input sequence:
        h = A · x_t                   (neighbourhood aggregation)
        h = ReLU(GCN1(h))
        h = A · h
        h = ReLU(GCN2(h))
    Flattened sequence → GRU → Dropout → Linear → demand forecast
    """

    def __init__(self, num_nodes: int, in_features: int = 2,
                 hidden: int = 32, gru_hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.gcn1    = nn.Linear(in_features, hidden)
        self.gcn2    = nn.Linear(hidden, hidden)
        self.gru     = nn.GRU(num_nodes * hidden, gru_hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(gru_hidden, num_nodes)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, num_nodes, in_features)
        A : (num_nodes, num_nodes)
        """
        b, t, n, f = x.shape
        out_seq = []
        for i in range(t):
            h = torch.matmul(A, x[:, i, :, :])          # (B, N, F)
            h = torch.relu(self.gcn1(h))
            h = torch.matmul(A, h)
            h = torch.relu(self.gcn2(h))
            out_seq.append(h)

        out_seq = torch.stack(out_seq, dim=1)            # (B, T, N, H)
        out_seq = out_seq.reshape(b, t, -1)              # (B, T, N*H)
        out, _  = self.gru(out_seq)
        out     = self.dropout(out[:, -1])               # last step
        return self.fc(out)                              # (B, num_nodes)


# ─────────────────────────────────────────────
# 5. TRAINING
# ─────────────────────────────────────────────
def train(model: nn.Module, X_train: torch.Tensor, Y_train: torch.Tensor,
          A: torch.Tensor, epochs: int, lr: float, device: torch.device):
    """Train the GNN+GRU model; returns loss history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train.to(device), A)
        loss = torch.sqrt(F.mse_loss(pred, Y_train.to(device)))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}  RMSE: {loss.item():.4f}")

    return losses


def evaluate(model: nn.Module, X_test: torch.Tensor, Y_test: torch.Tensor,
             A: torch.Tensor, device: torch.device):
    """Return predictions and print RMSE / MAE / R²."""
    model.eval()
    with torch.no_grad():
        pred = model(X_test.to(device), A).cpu().numpy()

    truth = Y_test.numpy()
    rmse  = np.sqrt(mean_squared_error(truth.flatten(), pred.flatten()))
    mae   = mean_absolute_error(truth.flatten(), pred.flatten())
    r2    = r2_score(truth.flatten(), pred.flatten())
    print(f"\n  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")
    return pred, truth


# ─────────────────────────────────────────────
# 6. DISRUPTION SIMULATION
# ─────────────────────────────────────────────
def simulate_disruption(G: nx.Graph, rate: float = 0.2):
    """
    Randomly remove `rate` fraction of nodes to simulate supply chain disruptions
    (e.g. factory shutdowns, port closures, natural disasters).
    """
    Gd     = G.copy()
    nodes  = list(Gd.nodes())
    failed = np.random.choice(nodes, int(len(nodes) * rate), replace=False)
    Gd.remove_nodes_from(failed)
    return Gd, failed


def plot_disruption(G: nx.Graph, failed_nodes, pos=None):
    pos = pos or nx.spring_layout(G, seed=42)
    colors = ["red" if n in failed_nodes else "green" for n in G.nodes()]
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, node_color=colors, node_size=60)
    plt.title("Disruption Simulation  (Red = Failed Nodes)")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# 7. OPTIMIZATION  –  LP / MIN-COST FLOW
# ─────────────────────────────────────────────
def build_supply_demand(pred_sample: np.ndarray, nodes: list) -> dict:
    """
    Convert GNN demand predictions into a balanced supply/demand vector.
    Nodes with production above mean  →  suppliers  (+)
    Nodes with production below mean  →  consumers  (-)
    """
    mean_val = np.mean(pred_sample)
    demand   = {n: float(pred_sample[n] - mean_val) for n in nodes}
    total    = sum(demand.values())
    for n in nodes:
        demand[n] -= total / len(nodes)
    return demand


def optimize_flow(G: nx.Graph, demand: dict, nodes: list, capacity: float = 1000.0):
    """
    Minimum-cost flow LP using PuLP.
    Returns flow dictionary and solver status.
    """
    # Build directed edges
    edges_opt = []
    for (u, v) in G.edges():
        if u in nodes and v in nodes:
            edges_opt.append((u, v))
            edges_opt.append((v, u))

    model_opt = pulp.LpProblem("SupplyChainOptimization", pulp.LpMinimize)
    flow      = pulp.LpVariable.dicts("flow", edges_opt, lowBound=0)
    cost      = {e: np.random.uniform(1, 5) for e in edges_opt}

    # Objective: minimise total transport cost
    model_opt += pulp.lpSum(cost[e] * flow[e] for e in edges_opt)

    # Flow conservation at each node
    for node in nodes:
        inflow  = [flow[(i, j)] for (i, j) in edges_opt if j == node]
        outflow = [flow[(i, j)] for (i, j) in edges_opt if i == node]
        model_opt += pulp.lpSum(outflow) - pulp.lpSum(inflow) == demand.get(node, 0)

    # Capacity constraints
    for e in edges_opt:
        model_opt += flow[e] <= capacity

    model_opt.solve(pulp.PULP_CBC_CMD(msg=0))
    status       = pulp.LpStatus[model_opt.status]
    flow_values  = {e: flow[e].value() for e in edges_opt if flow[e].value() is not None}
    return flow_values, status, edges_opt


# ─────────────────────────────────────────────
# 8. REPORTING
# ─────────────────────────────────────────────
def print_comparison(G_orig: nx.Graph, G_dis: nx.Graph, flow_values: dict):
    sig_flows = {e: v for e, v in flow_values.items() if v > 1e-2}

    print("\n" + "=" * 45)
    print("NETWORK COMPARISON")
    print("=" * 45)
    print(f"  Original  → Nodes: {len(G_orig.nodes()):4d}  Edges: {len(G_orig.edges()):4d}"
          f"  Components: {nx.number_connected_components(G_orig)}")
    print(f"  Disrupted → Nodes: {len(G_dis.nodes()):4d}  Edges: {len(G_dis.edges()):4d}"
          f"  Components: {nx.number_connected_components(G_dis)}")
    print(f"  Optimized → Total Flow: {sum(sig_flows.values()):.2f}"
          f"  Active Routes: {len(sig_flows)}")
    print(f"  Resilience Score: {len(G_dis.nodes()) / len(G_orig.nodes()):.2f}")


# ─────────────────────────────────────────────
# 9. MAIN PIPELINE
# ─────────────────────────────────────────────
def main():
    print("\n========================================")
    print("  Supply Chain Resilience Optimizer")
    print("========================================\n")

    # ── Load data ────────────────────────────
    print("[1/7] Loading data …")
    node2idx, idx2node = load_nodes(DATA_ROOT)
    num_nodes = len(node2idx)
    print(f"      Nodes: {num_nodes}")

    prod_raw  = load_temporal_data(DATA_ROOT, "production").iloc[:, 1:]
    sales_raw = load_temporal_data(DATA_ROOT, "sales").iloc[:, 1:]

    prod  = align_to_nodes(prod_raw,  node2idx, num_nodes)
    sales = align_to_nodes(sales_raw, node2idx, num_nodes)
    T     = min(len(prod), len(sales))
    prod, sales = prod[:T], sales[:T]

    G, A = load_graph(DATA_ROOT, DEVICE)
    print(f"      Graph edges: {len(G.edges())}")

    # ── Feature engineering ──────────────────
    print("\n[2/7] Building node features …")
    node_features = build_node_features(prod, sales, num_nodes)
    X_seq, Y_seq  = create_sequences(node_features, SEQ_LEN)

    split      = int(len(X_seq) * TRAIN_RATIO)
    X_train, Y_train = X_seq[:split], Y_seq[:split]
    X_test,  Y_test  = X_seq[split:], Y_seq[split:]
    print(f"      Train: {X_train.shape}  Test: {X_test.shape}")

    # ── Step 1: GNN+GRU training ─────────────
    print("\n[3/7] Step 1 — Training GNN + GRU …")
    model = SupplyChainGNN(num_nodes).to(DEVICE)
    losses = train(model, X_train, Y_train, A, EPOCHS, LR, DEVICE)

    print("\n[4/7] Evaluating model …")
    pred_test, truth = evaluate(model, X_test, Y_test, A, DEVICE)

    # ── Step 2: Disruption simulation ────────
    print("\n[5/7] Step 2 — Simulating disruption …")
    G_disrupted, failed_nodes = simulate_disruption(G, DISRUPT_RATE)
    print(f"      Removed {len(failed_nodes)} nodes  "
          f"({len(G_disrupted.nodes())} / {len(G.nodes())} remain)")

    # ── Step 3: LP optimization ───────────────
    print("\n[6/7] Step 3 — LP Optimization (Min-Cost Flow) …")
    nodes      = list(range(num_nodes))
    demand     = build_supply_demand(pred_test[0], nodes)
    flow_vals, status, _ = optimize_flow(G, demand, nodes)
    print(f"      Solver status: {status}")

    # ── Report ────────────────────────────────
    print("\n[7/7] Results …")
    print_comparison(G, G_disrupted, flow_vals)


if __name__ == "__main__":
    main()
