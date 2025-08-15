# app.py
# Streamlit demo: QKD topology + REAL PyTorch GNN edge classifier + risk-aware routing
# Run: streamlit run app.py

import math
import random
import numpy as np
import networkx as nx
import streamlit as st
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Utils
# -----------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))

# -----------------------------
# Graph generation & features
# -----------------------------
def make_topology(num_nodes=16, radius=0.35, seed=42):
    set_seed(seed)
    G = nx.random_geometric_graph(num_nodes, radius, seed=seed)
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        reps = [random.choice(list(c)) for c in comps]
        for i in range(len(reps)-1):
            u, v = reps[i], reps[i+1]
            G.add_edge(u, v)
    pos = nx.get_node_attributes(G, "pos")
    return G, pos

def assign_edge_features(G, pos, noise_range=(0.005, 0.03)):
    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        length = math.dist((x1, y1), (x2, y2))
        noise = np.random.uniform(*noise_range)
        # base qber shaped so most links are decent but not trivial
        base_qber = sigmoid_np(2.0*noise + 1.2*length - 2.0)
        G[u][v].update({
            "length": float(length),
            "noise": float(noise),
            "base_qber": float(base_qber),
        })

def clear_eavesdropping(G):
    for u, v in G.edges():
        G[u][v]["attacked"] = False
        G[u][v]["qber"] = G[u][v]["base_qber"]

def apply_eavesdropping(G, fraction=0.2, severity=0.6, seed=0):
    set_seed(seed)
    edges = list(G.edges())
    k = max(1, int(fraction * len(edges)))
    attacked = set(random.sample(edges, k)) if k > 0 else set()
    for u, v in edges:
        bump = severity if ((u, v) in attacked or (v, u) in attacked) else 0.0
        G[u][v]["attacked"] = bump > 0
        G[u][v]["qber"] = float(np.clip(G[u][v]["base_qber"] + bump, 0, 1))
    return attacked

# -----------------------------
# Data prep for PyTorch GNN
# -----------------------------
def build_node_features(G, pos):
    # Node features: [deg, mean_qber, pos_x, pos_y]
    deg = dict(G.degree())
    feats = []
    index_of = {n:i for i, n in enumerate(G.nodes())}
    for n in G.nodes():
        qlist = [G[n][nbr].get("qber", G[n][nbr]["base_qber"]) for nbr in G.neighbors(n)]
        mean_q = float(np.mean(qlist)) if qlist else 0.0
        px, py = pos[n]
        feats.append([deg[n], mean_q, px, py])
    X = torch.tensor(feats, dtype=torch.float32)
    return X, index_of

def build_adjacency(G, index_of):
    N = G.number_of_nodes()
    A = torch.zeros((N, N), dtype=torch.float32)
    for u, v in G.edges():
        i, j = index_of[u], index_of[v]
        A[i, j] = 1.0
        A[j, i] = 1.0
    # Add self loops
    A += torch.eye(N)
    # Symmetric degree norm: D^{-1/2} A D^{-1/2}
    deg = torch.sum(A, dim=1)
    D_inv_sqrt = torch.diag(torch.pow(deg, -0.5))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    return A_hat

def build_edge_dataset(G, index_of, secure_threshold=0.5):
    """
    Edge features will be produced from node embeddings later.
    Here we just prep edge index list and labels based on ground truth qber.
    """
    edges = list(G.edges())
    E = len(edges)
    y = torch.zeros(E, dtype=torch.float32)
    for k, (u, v) in enumerate(edges):
        q = G[u][v].get("qber", G[u][v]["base_qber"])
        # ground-truth label for training
        y[k] = 1.0 if q < secure_threshold else 0.0
    edge_index = torch.tensor([[index_of[u] for (u, v) in edges],
                               [index_of[v] for (u, v) in edges]], dtype=torch.long)
    return edges, edge_index, y

# -----------------------------
# Tiny Message-Passing GNN (from scratch)
# -----------------------------
class MPNN(nn.Module):
    def __init__(self, in_dim=4, hid_dim=32, K=2):
        super().__init__()
        self.K = K
        self.lin_in = nn.Linear(in_dim, hid_dim)
        self.lin_upd = nn.Linear(hid_dim, hid_dim)
        self.relu = nn.ReLU()
        # Edge scorer: uses both endpoints' embeddings + raw link scalars
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*hid_dim + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # logit
        )

    def forward(self, X, A_hat, edge_index, link_scalars):
        """
        X: [N, in_dim]
        A_hat: [N, N] normalized adjacency with self-loops
        edge_index: [2, E]
        link_scalars: [E, 3]  (qber, length, noise)
        """
        H = self.relu(self.lin_in(X))
        for _ in range(self.K):
            H = self.relu(self.lin_upd(A_hat @ H))
        # Edge embeddings: concat H[u], H[v], link scalars
        u_idx, v_idx = edge_index[0], edge_index[1]
        h_u = H[u_idx]
        h_v = H[v_idx]
        E_in = torch.cat([h_u, h_v, link_scalars], dim=1)
        logit = self.edge_mlp(E_in).squeeze(-1)
        return logit  # [E]

# -----------------------------
# Train / Predict helpers
# -----------------------------
def collect_link_scalars(G, edges):
    arr = []
    for u, v in edges:
        q = G[u][v].get("qber", G[u][v]["base_qber"])
        arr.append([q, G[u][v]["length"], G[u][v]["noise"]])
    return torch.tensor(arr, dtype=torch.float32)

@torch.no_grad()
def predict_edge_probs(model, X, A_hat, edge_index, link_scalars):
    model.eval()
    logits = model(X, A_hat, edge_index, link_scalars)
    probs = torch.sigmoid(logits)
    return probs.cpu().numpy()

def train_gnn_on_graph(G, pos, K=2, lr=1e-2, epochs=150, secure_threshold=0.5, device="cpu"):
    X, index_of = build_node_features(G, pos)
    A_hat = build_adjacency(G, index_of)
    edges, edge_index, y = build_edge_dataset(G, index_of, secure_threshold=secure_threshold)

    link_scalars = collect_link_scalars(G, edges)

    X = X.to(device)
    A_hat = A_hat.to(device)
    edge_index = edge_index.to(device)
    y = y.to(device)
    link_scalars = link_scalars.to(device)

    model = MPNN(in_dim=X.shape[1], hid_dim=32, K=K).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        logits = model(X, A_hat, edge_index, link_scalars)
        loss = bce(logits, y)
        loss.backward()
        opt.step()

    # Predict probs
    probs = predict_edge_probs(model, X, A_hat, edge_index, link_scalars)
    # Attach to graph
    for k, (u, v) in enumerate(edges):
        G[u][v]["risk"] = 1.0 - float(probs[k])  # risk = 1 - P(secure)
        G[u][v]["secure"] = bool(probs[k] >= 0.5)
    return model, (X, A_hat, edge_index, link_scalars), edges

# -----------------------------
# Risk-aware routing
# -----------------------------
def compute_edge_cost(G, u, v, alpha=1.0, beta=2.0, penalty_insecure=0.5):
    length = G[u][v]["length"]
    risk = G[u][v]["risk"]
    pen = penalty_insecure if not G[u][v]["secure"] else 0.0
    return alpha*length + beta*risk + pen

def add_weight_and_shortest_path(G, src, dst, alpha=1.0, beta=2.0, penalty_insecure=0.5):
    for u, v in G.edges():
        G[u][v]["weight"] = compute_edge_cost(G, u, v, alpha, beta, penalty_insecure)
    try:
        return nx.shortest_path(G, source=src, target=dst, weight="weight")
    except nx.NetworkXNoPath:
        return None

def path_metrics(G, path):
    if not path or len(path) < 2:
        return {}
    edges = list(zip(path[:-1], path[1:]))
    total_length = sum(G[u][v]["length"] for u, v in edges)
    avg_risk = float(np.mean([G[u][v]["risk"] for u, v in edges]))
    insecure_edges = sum(1 for u, v in edges if not G[u][v]["secure"])
    attacked_edges = sum(1 for u, v in edges if G[u][v].get("attacked", False))
    total_weight = sum(G[u][v]["weight"] for u, v in edges)
    return {
        "Hops": len(edges),
        "Total length": round(total_length, 3),
        "Avg risk": round(avg_risk, 3),
        "Insecure edges": int(insecure_edges),
        "Attacked edges": int(attacked_edges),
        "Path cost (weight)": round(total_weight, 3),
        "Edges": edges
    }

# -----------------------------
# Plotting
# -----------------------------
def draw_topology(G, pos, path=None, title=""):
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color="#f2f6ff", edgecolors="#222", linewidths=1)
    colors = []
    widths = []
    for u, v in G.edges():
        colors.append("#2ecc71" if G[u][v]["secure"] else "#e74c3c")
        widths.append(2.6 if G[u][v].get("attacked", False) else 1.6)
    nx.draw_networkx_edges(G, pos, edge_color=colors, width=widths, alpha=0.9)
    if path and len(path) > 1:
        pe = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=pe, width=4.5, edge_color="#3498db")
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_size=360, node_color="#aee1ff", edgecolors="#1f4c75", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.title(title)
    plt.axis("off")
    st.pyplot(plt.gcf())
    plt.close()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="QKD + PyTorch GNN Routing Demo", layout="wide")
st.title("🔐 QKD Network with Trainable PyTorch GNN (Edge Security) + Safest Path")

with st.sidebar:
    st.header("⚙️ Graph")
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
    num_nodes = st.slider("Number of nodes", 6, 50, 16, 1)
    radius = st.slider("Connectivity radius", 0.15, 0.6, 0.35, 0.01)

    st.markdown("---")
    st.header("🧠 GNN")
    K = st.slider("Message-passing steps (K)", 1, 4, 2, 1)
    epochs = st.slider("Training epochs", 20, 500, 150, 10)
    lr = st.select_slider("Learning rate", options=[1e-3, 3e-3, 5e-3, 1e-2, 3e-2], value=1e-2)
    truth_thresh = st.slider("Ground-truth secure threshold (QBER)", 0.05, 0.5, 0.2, 0.01)

    st.markdown("---")
    st.header("🕵️ Eavesdropping")
    enable_eve = st.checkbox("Enable eavesdropping (right panel)", value=False)
    eve_fraction = st.slider("Fraction of edges attacked", 0.05, 0.8, 0.25, 0.05)
    eve_severity = st.slider("QBER bump severity", 0.05, 1.0, 0.6, 0.05)

    st.markdown("---")
    st.header("🗺️ Routing weights")
    alpha = st.slider("α: distance weight", 0.0, 3.0, 1.0, 0.1)
    beta  = st.slider("β: risk weight", 0.0, 5.0, 2.0, 0.1)
    pen_insec = st.slider("Penalty (insecure edge)", 0.0, 2.0, 0.5, 0.1)

# -----------------------------
# Build base graph
# -----------------------------
device = "cpu"
set_seed(seed)
G, pos = make_topology(num_nodes=num_nodes, radius=radius, seed=seed)
assign_edge_features(G, pos)

# --- Scenario A: No eavesdropping (train + route) ---
clear_eavesdropping(G)
# attach qber = base
for u, v in G.edges():
    G[u][v]["qber"] = G[u][v]["base_qber"]

model_A, tensors_A, edges_A = train_gnn_on_graph(
    G, pos, K=K, lr=lr, epochs=epochs, secure_threshold=truth_thresh, device=device
)
# pick src/dst after graph is ready
nodes_sorted = sorted(G.nodes())
col_src, col_dst = st.columns(2)
with col_src:
    src = st.selectbox("Source", nodes_sorted, index=0, key="src")
with col_dst:
    dst = st.selectbox("Destination", nodes_sorted, index=min(1, len(nodes_sorted)-1), key="dst")

path_A = add_weight_and_shortest_path(G, src, dst, alpha, beta, pen_insec)
metrics_A = path_metrics(G, path_A)
G_A = G.copy()

# --- Scenario B: With eavesdropping (train + route on changed qber) ---
clear_eavesdropping(G)
attacked = set()
if enable_eve:
    attacked = apply_eavesdropping(G, fraction=eve_fraction, severity=eve_severity, seed=seed+7)
else:
    apply_eavesdropping(G, fraction=0.0, severity=0.0, seed=seed+7)

model_B, tensors_B, edges_B = train_gnn_on_graph(
    G, pos, K=K, lr=lr, epochs=epochs, secure_threshold=truth_thresh, device=device
)
path_B = add_weight_and_shortest_path(G, src, dst, alpha, beta, pen_insec)
metrics_B = path_metrics(G, path_B)

# -----------------------------
# Show side-by-side
# -----------------------------
left, right = st.columns(2, gap="large")

with left:
    st.subheader("🟢 Without Eavesdropping (trained on base QBER)")
    draw_topology(G_A, pos, path=path_A, title="Green=Secure, Red=Insecure; safest path in blue")
    if path_A:
        st.write("**Path:**", " → ".join(map(str, path_A)))
        st.json(metrics_A)
    else:
        st.warning("No path found under current weights/graph.")

with right:
    st.subheader("🔴 With Eavesdropping (retrained on attacked QBER)")
    draw_topology(G, pos, path=path_B, title="Attacked edges thicker; Green=Secure, Red=Insecure; path in blue")
    if enable_eve and attacked:
        st.caption(f"Attacked edges: {len(attacked)} (drawn thicker)")
    if path_B:
        st.write("**Path:**", " → ".join(map(str, path_B)))
        st.json(metrics_B)
    else:
        st.warning("No path found under current weights/graph.")

# -----------------------------
# Inspect edges
# -----------------------------
with st.expander("Inspect edge features & GNN predictions (Scenario B shown)"):
    rows = []
    for u, v, d in G.edges(data=True):
        rows.append({
            "u": u, "v": v,
            "length": round(d["length"], 3),
            "noise": round(d["noise"], 4),
            "qber": round(d["qber"], 3),
            "risk (1-Psecure)": round(d["risk"], 3),
            "secure_pred": d["secure"],
            "attacked": d.get("attacked", False),
            "weight": round(d["weight"], 3) if "weight" in d else None
        })
    st.dataframe(rows, hide_index=True)

st.markdown("""
**Notes**
- This is a true trainable **PyTorch GNN** (message passing + edge scorer) but kept tiny so it trains in a few hundred epochs quickly on CPU.
- The **ground-truth labels** come from QBER vs **“Ground-truth secure threshold (QBER)”**. The GNN learns to map node context + link scalars → edge security.
- We **retrain** for the eavesdropping case because QBER changes; you can compare how the predicted secure map and safest path respond.
- Tune **K (message-passing steps)**, **epochs**, **learning rate**, and **routing weights** to see different behaviors.
""")
