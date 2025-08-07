import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# --- QKD network generation ---
def calculate_transmittance(distance_km, alpha_db_per_km=0.2):
    return 10 ** (-alpha_db_per_km * distance_km / 10)

def estimate_qber(transmittance, detector_efficiency=0.1, dark_count_prob=1e-5, intrinsic_error=0.01):
    eta_total = transmittance * detector_efficiency
    numerator = intrinsic_error * eta_total + 0.5 * dark_count_prob
    denominator = eta_total + dark_count_prob
    return numerator / denominator if denominator > 0 else 1.0

def calculate_key_rate(transmittance, qber, simulation_mode=True):
    pulse_rate = 1e9 if simulation_mode else 1e4
    sifting_ratio = 0.5
    return pulse_rate * transmittance * sifting_ratio * (1 - 2 * qber)

def generate_qkd_network(num_nodes=25, connection_radius=0.4, seed=42, simulation_mode=True):
    np.random.seed(seed)
    pos = {i: np.random.rand(2) for i in range(num_nodes)}
    G = nx.random_geometric_graph(num_nodes, connection_radius, pos=pos)

    for u, v in G.edges():
        distance = np.linalg.norm(np.array(pos[u]) - np.array(pos[v])) * 100
        eta_ch = calculate_transmittance(distance)
        qber = estimate_qber(eta_ch)
        key_rate = calculate_key_rate(eta_ch, qber, simulation_mode)
        G[u][v]["distance_km"] = round(distance, 2)
        G[u][v]["eta_ch"] = round(eta_ch, 4)
        G[u][v]["qber"] = round(qber, 4)
        G[u][v]["key_rate"] = round(key_rate, 2)
    return G

# --- Streamlit UI ---
st.set_page_config(page_title="QKD Routing Dashboard", layout="wide")
st.title("Quantum Key Distribution Network Visualizer")

G = generate_qkd_network()

# Sidebar: source and target selection
st.sidebar.header("Routing Parameters")
source_node = st.sidebar.selectbox("Select Source Node", list(G.nodes), index=0)
target_node = st.sidebar.selectbox("Select Target Node", list(G.nodes), index=1)

if source_node == target_node:
    st.warning("Please select different source and target nodes.")
else:
    try:
        path = nx.shortest_path(G, source=source_node, target=target_node, weight="key_rate")
        st.success(f"Shortest Path (Min Key Rate): {' → '.join(map(str, path))}")

        pos = nx.get_node_attributes(G, "pos")
        if not pos:
            pos = nx.spring_layout(G)

        edge_colors = []
        for u, v in G.edges():
            if (u, v) in zip(path, path[1:]) or (v, u) in zip(path, path[1:]):
                edge_colors.append("blue")
            else:
                edge_colors.append("gray")

        edge_labels = {(u, v): f"{G[u][v]['key_rate']:.1f}" for u, v in G.edges()}

        fig, ax = plt.subplots(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_color="skyblue", edge_color=edge_colors, node_size=500, ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)
        st.pyplot(fig)

    except nx.NetworkXNoPath:
        st.error("No path found between the selected nodes.")
