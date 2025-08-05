import streamlit as st
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

# --- QKD Simulation Core ---

def calculate_transmittance(distance_km, alpha_db_per_km=0.2):
    return 10 ** (-alpha_db_per_km * distance_km / 10)

def estimate_qber(transmittance, detector_efficiency=0.1, dark_count_prob=1e-5, intrinsic_error=0.01):
    eta_total = transmittance * detector_efficiency
    numerator = intrinsic_error * eta_total + 0.5 * dark_count_prob
    denominator = eta_total + dark_count_prob
    return numerator / denominator if denominator > 0 else 1.0

def calculate_key_rate(transmittance, qber, simulation_mode=True):
    pulse_rate = 1e9 if simulation_mode else 1e6
    sifting_ratio = 0.5
    return pulse_rate * transmittance * sifting_ratio * (1 - 2 * qber)

def generate_qkd_network(num_nodes=30, connection_radius=0.3, seed=42, simulation_mode=True):
    np.random.seed(seed)
    random.seed(seed)
    pos = {i: np.random.rand(2) for i in range(num_nodes)}
    G = nx.random_geometric_graph(num_nodes, connection_radius, pos=pos)

    for u, v in G.edges():
        distance = np.linalg.norm(np.array(pos[u]) - np.array(pos[v])) * 100
        eta_ch = calculate_transmittance(distance)
        qber = estimate_qber(eta_ch)
        key_rate = calculate_key_rate(eta_ch, qber, simulation_mode)

        G[u][v]['distance_km'] = round(distance, 2)
        G[u][v]['eta_ch'] = round(eta_ch, 4)
        G[u][v]['qber'] = round(qber, 4)
        G[u][v]['key_rate'] = round(key_rate, 4)

    nx.set_node_attributes(G, pos, name='pos')
    return G

def draw_qber_overlay(G, title):
    pos = nx.get_node_attributes(G, 'pos')
    edge_colors = ['red' if d['qber'] > 0.11 else 'green' for _, _, d in G.edges(data=True)]
    edge_labels = {(u, v): f"{d['qber']*100:.1f}%" for u, v, d in G.edges(data=True)}

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=500)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2)
    nx.draw_networkx_labels(G, pos, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='darkred', font_size=8)
    plt.title(title)
    plt.axis('off')
    st.pyplot(plt)
    plt.clf()

# --- Streamlit UI ---

st.title("🔐 QKD Network Visualization with QBER Overlay")

mode = st.radio("Select Simulation Mode", ['Real-World', 'Simulation'])
is_simulation = mode == 'Simulation'

num_nodes = st.slider("Number of QKD Nodes", min_value=10, max_value=100, value=30)
radius = st.slider("Connection Radius", min_value=0.1, max_value=0.8, value=0.3)

if st.button("Generate QKD Network"):
    G = generate_qkd_network(num_nodes=num_nodes, connection_radius=radius, simulation_mode=is_simulation)
    title = f"QKD Network - {'Simulation' if is_simulation else 'Real-World'} Mode (QBER Overlay)"
    draw_qber_overlay(G, title)
