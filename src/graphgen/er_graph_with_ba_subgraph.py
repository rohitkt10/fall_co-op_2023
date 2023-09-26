import networkx as nx
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import logging

def create_er_replace_with_ba_adjacency(n=300, p=0.2, m=3,
                                        min_motif_size=10, max_motif_size=30,
                                        min_num_motifs=3, max_num_motifs=5,
                                        motif_overlap=False,
                                        min_base_edge_weight=1.0, max_base_edge_weight=1.0,
                                        min_motif_edge_weight=1.0, max_motif_edge_weight=1.0,
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
    - min_base_edge_weight (float): Minimum edge weight for edges in the base graph (ER graph)
    - max_base_edge_weight (float): Maximum edge weight for edges in the base graph (ER graph)
    - min_motif_edge_weight (float): Minimum edge weight for edges in the motif graph (BA model)
    - max_motif_edge_weight (float): Maximum edge weight for edges in the motif graph (BA model)
    - plot (bool): Whether to plot the graphs (default is False).
    - log (bool): Whether to log variable values for debugging (default is True).

    Returns:
    - np.ndarray or None: The resulting adjacency matrix of the modified graph, or None if invalid parameters.
    """
    if log:  # Check if logging is enabled
        logging.info(f"n = {n}, p = {p}, m = {m}, min_motif_size = {min_motif_size}, max_motif_size = {max_motif_size}")
        logging.info(f"min_num_motifs = {min_num_motifs}, max_num_motifs = {max_num_motifs}, motif_overlap = {motif_overlap}")
        logging.info(f"min_base_edge_weight = {min_base_edge_weight}, max_base_edge_weight = {max_base_edge_weight}")
        logging.info(f"min_motif_edge_weight = {min_motif_edge_weight}, max_motif_edge_weight = {max_motif_edge_weight}")
        logging.info(f"plot = {plot}")

    # Create the ER graph adjacency matrix
    er_graph = nx.erdos_renyi_graph(n, p)
    er_adjacency = nx.to_numpy_array(er_graph)
    # replace edge weights
    if min_base_edge_weight != max_base_edge_weight:
        assert min_base_edge_weight < max_base_edge_weight
        num_edges = int(er_adjacency.sum()/2)
        # sample `num_edges`
        edge_weights = np.random.uniform(min_base_edge_weight, max_base_edge_weight, num_edges)
        #[1.0, 2.0, 3.0, 1.0, 2.0, ...]
        I, J = np.where(er_adjacency) 
        # get upper triangular matrix
        idx = np.where(J > I)[0]
        I = I[idx] 
        J = J[idx]
        num_edges = len(I) 
        edge_weights = np.random.uniform(0., 1., (num_edges,))
        er_adjacency[I, J] = edge_weights
        er_adjacency[J, I] = edge_weights
        
    # check degree 0 nodes in the original ER graph
    if np.sum(er_adjacency.sum(axis=1) == 0) > 0:
        logging.info("Original ER graph contains nodes with degree 0.")

    # plot original ER graph
    if plot:
        plt.figure(figsize=(14, 8))
        plt.subplot(221)
        nx.draw_networkx(nx.Graph(er_adjacency), with_labels=True, node_color='skyblue', node_size=300)
        plt.title('Original ER graph')

    # Randomly determine the number of motifs to embed
    num_motifs = random.randint(min_num_motifs, max_num_motifs)

    node_colors = ['skyblue'] * n
    edge_colors = {}  # Initialize edge colors as a dictionary
    # Randomly select 'n' colors from the named_colors list 
    edge_colors_list = random.sample(list(mcolors.CSS4_COLORS.keys()), num_motifs)

    # nodes available to create subgraphs: for the first motif, all nodes are available
    nodes_available = list(np.arange(er_adjacency.shape[0]))

    for motif_i in range(num_motifs):
        # Generate a random motif size within the specified range
        ba_n = random.randint(min_motif_size, max_motif_size)

        # Select nodes to replace
        er_nodes_to_replace = np.random.choice(nodes_available, size=(ba_n,), replace=False)
        # if motif overlapping not allowed
        if not motif_overlap:    
            nodes_available = [node for node in nodes_available if node not in er_nodes_to_replace]

        if log:  # Check if logging is enabled
            logging.info(f'motif_{motif_i}: {er_nodes_to_replace}')

        # change BA nodes' color
        for er_node in er_nodes_to_replace:
            # if overlapping motifs not allowed, color nodes of each motif differently
            if not motif_overlap:
                node_colors[er_node] = edge_colors_list[motif_i]
            # color all nodes of motifs same, but edge colors are decided motif-wise later
            else:
                node_colors[er_node] = 'lightcoral'

        # Create the BA model adjacency matrix for the motif
        if ba_n >= m:
            ba_motif = nx.to_numpy_array(nx.barabasi_albert_graph(ba_n, m))

            # replace edge weights
            if min_motif_edge_weight != max_motif_edge_weight:
                assert min_motif_edge_weight < max_motif_edge_weight
                num_edges = int(ba_motif.sum()/2)
                # sample `num_edges`
                edge_weights = np.random.uniform(min_motif_edge_weight, max_motif_edge_weight, num_edges)
                #[1.0, 2.0, 3.0, 1.0, 2.0, ...]
                I, J = np.where(ba_motif) 
                # get upper triangular matrix
                idx = np.where(J > I)[0]
                I = I[idx] 
                J = J[idx]
                num_edges = len(I) 
                edge_weights = np.random.uniform(0., 1., (num_edges,))
                ba_motif[I, J] = edge_weights
                ba_motif[J, I] = edge_weights

            if np.sum(ba_motif.sum(axis=1) == 0) > 0:
                logging.info("BA graph contains nodes with degree 0.")

            if plot:
                # Set edge colors within the subgraph to distinguish from others
                for i, idx in enumerate(er_nodes_to_replace):
                    for j, jdx in enumerate(er_nodes_to_replace):
                        if i != j:
                            edge_colors[(idx, jdx)] = edge_colors_list[motif_i]
        else:
            logging.warning("BA model requires ba_n >= m")
            return None

        # Replace selected nodes in ER adjacency matrix with BA model motif
        for i, idx in enumerate(er_nodes_to_replace):
            er_adjacency[idx, er_nodes_to_replace] = ba_motif[i]

    # Ensure that none of the nodes have degree 0 in the original ER graph
    if np.sum(er_adjacency.sum(axis=1) == 0) > 0:
        logging.warning(f"ER with BA subgraph contains nodes with degree 0: {np.sum(er_adjacency.sum(axis=1) == 0)}")

    # Connect zero-degree nodes to random other nodes
    g = nx.Graph(er_adjacency)
    degree_dict = dict(g.degree())
    zero_degree_nodes = [k for k in degree_dict if degree_dict[k] == 0]
    logging.info(f'Nodes with zero degree in ER with BA subgraph: {zero_degree_nodes}')
    nonzero_degrees = np.sort([v for k, v in degree_dict.items() if k not in zero_degree_nodes] )
    nonzero_degree_nodes = [k for k, v in degree_dict.items() if v != 0]
    degrees, counts = np.unique(nonzero_degrees, return_counts=True) 
    probs = counts / np.sum(counts)
    for node1 in zero_degree_nodes:
        sample_degree = np.random.choice(degrees, p=probs) 
        sample_connections = np.random.choice(nonzero_degree_nodes, size=(sample_degree,), replace=False)
        for node2 in sample_connections: 
            g.add_edge(node1, node2)

    degree_dict = dict(g.degree())
    zero_degree_nodes = [k for k in degree_dict if degree_dict[k] == 0]
    if len(zero_degree_nodes) == 0:
        logging.info("Nodes with degree 0 connected!")

    # Plot ER graph with BA subgraph
    if plot:
        plt.subplot(222)
        pos = nx.spring_layout(g)  # Position nodes using a layout algorithm
        edge_color_l = [edge_colors.get((u, v), 'gray') for u, v in g.edges()]
        nx.draw_networkx(g, pos, with_labels=True, node_color=node_colors, node_size=300, edge_color=edge_color_l)
        plt.title(f'ER graph with {num_motifs} BA subgraphs')

        # Plot degree distribution
        plt.subplot(223)
        for g, l in zip([er_graph, nx.Graph(er_adjacency)], ['ER', 'ER with BA subgraphs']):
            plt.plot(np.sort([j for _, j in g.degree()])[::-1], marker='.', label=l)
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.title('Degree Distribution')
        plt.legend()

    return er_adjacency