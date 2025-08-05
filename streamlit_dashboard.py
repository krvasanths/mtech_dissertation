import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pickle

# --- GNN Model ---
class GCNClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return torch.nn.functional.log_softmax(x, dim=1)

# --- Load Data ---
edge_index = torch.tensor(np.load("edge_index.npy"), dtype=torch.long)
edge_attr = torch.tensor(np.load("edge_features.npy"), dtype=torch.float)
x = torch.tensor(np.load("node_features.npy"), dtype=torch.float)
y = torch.tensor(np.load("labels.npy"), dtype=torch.long)
data = Data(x=x, edge_index=edge_index)

# --- Load GNN model ---
model = GCNClassifier(input_dim=x.shape[1])
model.load_state_dict(torch.load("edge_classifier_gnn.pth", map_location="cpu"))
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
