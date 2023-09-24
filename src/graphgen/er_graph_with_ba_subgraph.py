import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import logging

def create_er_replace_with_ba_adjacency(n, p, m,
                                        min_motif_size, max_motif_size,
                                        min_num_motifs, max_num_motifs,
                                        motif_overlap=False,
                                        base_edge_weight=1.0,
                                        motif_edge_weight=1.0,
                                        plot=False,
                                        log=True):
    """
    Create a large sparse graph using the Erdős-Rényi (ER) model and replace
    a specified number of nodes with Barabási-Albert (BA) model motifs.

    Parameters:
    - n (int): Number of nodes in the Erdős-Rényi (ER) graph.
    - p (float): Probability of an edge between nodes in the ER graph (0 <= p <= 1).
    - m (int): Number of edges to attach from a new node to existing nodes in the BA model (m >= 1).
    - min_motif_size (int): Minimum number of nodes in a BA model motif.
    - max_motif_size (int): Maximum number of nodes in a BA model motif.
    - min_num_motifs (int): Minimum number of motifs to embed.
    - max_num_motifs (int): Maximum number of motifs to embed.
    - motif_overlap (bool): Whether motifs can overlap in nodes (default is False).
    - base_edge_weight (float): Scalar edge weight for edges in the base graph (ER graph).
    - motif_edge_weight (float): Scalar edge weight for edges in the motif graph (BA model).
    - plot (bool): Whether to plot the graphs (default is False).
    - log (bool): Whether to log variable values for debugging (default is True).

    Returns:
    - np.ndarray or None: The resulting adjacency matrix of the modified graph, or None if invalid parameters.
    """
    if log:  # Check if logging is enabled
        logging.debug(f"n = {n}, p = {p}, m = {m}, min_motif_size = {min_motif_size}, max_motif_size = {max_motif_size}")
        logging.debug(f"min_num_motifs = {min_num_motifs}, max_num_motifs = {max_num_motifs}, motif_overlap = {motif_overlap}")
        logging.debug(f"base_edge_weight = {base_edge_weight}, motif_edge_weight = {motif_edge_weight}, plot = {plot}")

    # Create the ER graph adjacency matrix
    er_adjacency = nx.to_numpy_array(nx.erdos_renyi_graph(n, p))
    # scale edge weight
    er_adjacency = er_adjacency * base_edge_weight

    # Ensure that none of the nodes have degree 0 in the original ER graph
    if np.sum(er_adjacency.sum(axis=1) == 0) > 0:
        logging.warning("Original ER graph contains nodes with degree 0.")

    # plot original ER graph
    if plot:
        plt.figure(figsize=(14, 6))
        plt.subplot(121)
        nx.draw_networkx(nx.Graph(er_adjacency), with_labels=True, node_color='skyblue', node_size=300)
        plt.title('Original ER graph')

    # Randomly determine the number of motifs to embed
    num_motifs = random.randint(min_num_motifs, max_num_motifs)

    node_colors = ['lightgreen'] * n

    for motif_i in range(num_motifs):
        # Generate a random motif size within the specified range
        ba_n = random.randint(min_motif_size, max_motif_size)

        # Select nodes to replace
        if motif_overlap:
            er_nodes_to_replace = random.sample(range(n), ba_n)
        else:
            er_nodes_to_replace = random.sample(range(n), ba_n)
            n -= ba_n  # Ensure non-overlapping motifs

        if log:  # Check if logging is enabled
            logging.debug(f'motif_{motif_i}: {er_nodes_to_replace}')

        # change BA nodes' color
        for er_node in er_nodes_to_replace:
            node_colors[er_node] = 'lightcoral'

        # Create the BA model adjacency matrix for the motif
        if ba_n >= m:
            ba_motif = nx.to_numpy_array(nx.barabasi_albert_graph(ba_n, m))
            # scale edge weight
            ba_motif = ba_motif * motif_edge_weight
            if np.sum(ba_motif.sum(axis=1) == 0) > 0:
                logging.warning("BA graph contains nodes with degree 0.")

        else:
            logging.warning("BA model requires ba_n >= m")
            return None

        # Replace selected nodes in ER adjacency matrix with BA model motif
        for i, idx in enumerate(er_nodes_to_replace):
            er_adjacency[idx, er_nodes_to_replace] = ba_motif[i]

    # Plot ER graph with BA subgraph
    if plot:
        plt.subplot(122)
        nx.draw_networkx(nx.Graph(er_adjacency), with_labels=True, node_color=node_colors, node_size=300)
        plt.title('ER graph with BA subgraphs')

    return er_adjacency