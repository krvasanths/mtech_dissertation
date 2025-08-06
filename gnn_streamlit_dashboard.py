import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class EdgeClassifierGNN(torch.nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim + edge_feat_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 2)
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        edge_embed = torch.cat([x[edge_index[0]], x[edge_index[1]], edge_attr], dim=1)
        return self.edge_mlp(edge_embed)

def load_data():
    edge_index = torch.tensor(np.load("data/edge_index.npy"), dtype=torch.long)
    edge_attr = torch.tensor(np.load("data/edge_features.npy"), dtype=torch.float32)
    node_features = torch.tensor(np.load("data/node_features.npy"), dtype=torch.float32)
    edge_labels = torch.tensor(np.load("data/edge_labels.npy"), dtype=torch.long)
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=edge_labels)

def visualize_predictions(edge_index, predictions):
    G = nx.Graph()
    edge_index_np = edge_index.numpy()
    for i in range(edge_index_np.shape[1]):
        u, v = edge_index_np[:, i]
        label = predictions[i]
        G.add_edge(int(u), int(v), color='green' if label == 1 else 'red')
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, edge_color=edge_colors, node_color='skyblue', node_size=500)
    st.pyplot(plt.gcf())
    plt.clf()

st.title(" QKD Edge Classification with GNN")
st.markdown("Visualizing Secure (Green) vs Insecure (Red) Quantum Channels")

data = load_data()
model = EdgeClassifierGNN(data.x.shape[1], data.edge_attr.shape[1])
model.load_state_dict(torch.load("edge_classifier_gnn.pth", map_location=torch.device('cpu')))
model.eval()

with torch.no_grad():
    preds = model(data.x, data.edge_index, data.edge_attr).argmax(dim=1)

visualize_predictions(data.edge_index, preds.numpy())
