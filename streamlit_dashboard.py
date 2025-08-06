import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pickle
import torch.nn.functional as F

class GCNClassifier(torch.nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 2)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Aggregate node embeddings for each edge
        edge_src = edge_index[0]
        edge_dst = edge_index[1]
        edge_repr = x[edge_src] + x[edge_dst]  # Symmetric aggregation

        out = self.edge_mlp(edge_repr)
        return F.log_softmax(out, dim=1)
# --- Load Data ---
edge_index = torch.tensor(np.load("edge_index.npy"), dtype=torch.long)
edge_attr = torch.tensor(np.load("edge_features.npy"), dtype=torch.float)
x = torch.tensor(np.load("node_features.npy"), dtype=torch.float)
y = torch.tensor(np.load("labels.npy"), dtype=torch.long)
data = Data(x=x, edge_index=edge_index)

# --- Load GNN model ---
model = GCNClassifier(input_dim=4, hidden_dim=64)
model.load_state_dict(torch.load("edge_classifier_gnn.pth"))
model.eval()

# --- Inference ---
with torch.no_grad():
    logits = model(data)
    pred = logits.argmax(dim=1)

# --- Load Graph ---
with open("data/qkd_topology.gpickle", "rb") as f:
    G = pickle.load(f)

# --- Visualization ---
st.title("🔐 QKD GNN Edge Classifier")
st.write("Green = Secure | Red = Insecure")

pos = nx.spring_layout(G, seed=42)
edge_colors = ['green' if pred[i] == 1 else 'red' for i in range(len(G.edges()))]

plt.figure(figsize=(10, 8))
nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color=edge_colors, node_size=500)
st.pyplot(plt)
