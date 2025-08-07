
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data
from train_edge_classifier import EdgeClassifierGNN
from train_drl_agent import DQNAgent
import os

# Load GNN model
@st.cache_resource
def load_gnn_model(model_path="edge_classifier_gnn.pth"):
    edge_attr_dim = 4
    node_feat_dim = 4
    model = EdgeClassifierGNN(node_feat_dim, edge_attr_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Load network graph
@st.cache_resource
def load_graph():
    return nx.read_gpickle("qkd_topology.gpickle")

# Load GNN input data
@st.cache_resource
def load_gnn_data():
    edge_index = torch.tensor(np.load("edge_index.npy"), dtype=torch.long)
    edge_attr = torch.tensor(np.load("edge_features.npy"), dtype=torch.float32)
    node_features = torch.tensor(np.load("node_features.npy"), dtype=torch.float32)
    return edge_index, edge_attr, node_features

# Load predicted labels
@st.cache_resource
def load_edge_predictions():
    return np.load("labels.npy")

# Run DQN agent to find route
def infer_route(agent, G, source, target):
    path = nx.shortest_path(G, source=source, target=target)
    return path

def main():
    st.set_page_config(page_title="QKD Simulation Dashboard", layout="wide")
    st.title("QKD Simulation Dashboard with GNN + DRL Inference")

    G = load_graph()
    edge_index, edge_attr, node_features = load_gnn_data()
    gnn_model = load_gnn_model()
    predicted_labels = load_edge_predictions()

    st.sidebar.header("Settings")
    source = st.sidebar.selectbox("Select Source Node", sorted(G.nodes()))
    target = st.sidebar.selectbox("Select Target Node", sorted(G.nodes()))
    toggle_eavesdrop = st.sidebar.checkbox("Simulate Eavesdropping", value=False)

    if toggle_eavesdrop:
        for u, v, data in G.edges(data=True):
            data["qber"] = min(data["qber"] + 0.1, 1.0)
            data["eavesdropped"] = True
    else:
        for u, v, data in G.edges(data=True):
            data["eavesdropped"] = False

    st.subheader("Inferred Route from DRL Agent")
    agent = DQNAgent(state_size=1, action_size=len(G.nodes))
    route = infer_route(agent, G, source, target)
    st.write(f"Route: {route}")

    st.subheader("Network Visualization with QBER Overlay")
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 6))
    edge_colors = ["red" if G[u][v].get("qber", 0) > 0.11 else "green" for u, v in G.edges()]
    edge_labels = { (u, v): f"{G[u][v]['qber']*100:.1f}%" for u, v in G.edges() }
    node_colors = ["orange" if n in route else "skyblue" for n in G.nodes()]

    nx.draw(G, pos, node_color=node_colors, with_labels=True, edge_color=edge_colors, node_size=500)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    st.pyplot(plt.gcf())

    st.subheader("GNN-Predicted Secure/Insecure Links")
    secure_count = 0
    for i, (u, v) in enumerate(G.edges()):
        G[u][v]["secure"] = "✅" if predicted_labels[i] == 1 else "❌"
        secure_count += predicted_labels[i]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Edges", len(predicted_labels))
    with col2:
        st.metric("Secure Links", secure_count)

    st.markdown("Toggle eavesdropping in the sidebar to see its impact on QBER and routing decisions.")

if __name__ == "__main__":
    main()
