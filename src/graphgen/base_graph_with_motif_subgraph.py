import networkx as nx
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import logging

def create_graph_with_motif_adjacency(base_graph_model='ER',
                                      motif_graph_model='BA',
                                      n=300, p=0.2, m=3, k=4,
                                      min_motif_size=10, max_motif_size=30,
                                      min_num_motifs=3, max_num_motifs=5,
                                      motif_overlap=False,
                                      min_base_edge_weight=1.0, max_base_edge_weight=1.0,
                                      min_motif_edge_weight=1.0, max_motif_edge_weight=1.0,
                                      plot=False,
                                      log=True):
    """
    Create a large sparse graph using a specified base graph model and insert motifs by replacing
    a specified number of nodes with a specified graph model.

    Parameters:
    - base_graph_model (str): The base graph model to use 
                              ('ER' for Erdős-Rényi or 'BA' for Barabási-Albert or 'WS' for Watts-Strogatz).
    - motif_graph_model (str): The motif graph model to use.
    - n (int): Number of nodes in the `base_graph_model` graph.
    - p (float): Probability of an edge between nodes in the ER graph (0 <= p <= 1), 
                 or probability of rewiring each edge in the WS graph.
    - m (int): Number of edges to attach from a new node to existing nodes in the BA model (m >= 1).
    - k (int): Each node is joined with its `k` nearest neighbors in a ring topology in the WS model.
    - min_motif_size (int): Minimum number of nodes in `motif_graph_model`.
    - max_motif_size (int): Maximum number of nodes in the `motif_graph_model`.
    - min_num_motifs (int): Minimum number of motifs to embed.
    - max_num_motifs (int): Maximum number of motifs to embed.
    - motif_overlap (bool): Whether motifs can overlap in nodes (default is False).
    - min_base_edge_weight (float): Minimum edge weight for edges in the base graph.
    - max_base_edge_weight (float): Maximum edge weight for edges in the base graph.
    - min_motif_edge_weight (float): Minimum edge weight for edges in the motif graph.
    - max_motif_edge_weight (float): Maximum edge weight for edges in the motif graph.
    - plot (bool): Whether to plot the graphs (default is False).
    - log (bool): Whether to log variable values for debugging (default is True).

    Returns:
    - np.ndarray or None: The resulting adjacency matrix of the modified graph, or None if invalid parameters.
    """
    if log:  # Check if logging is enabled
        logging.info(f"base_graph_model= {base_graph_model}, motif_graph_model= {motif_graph_model}")
        logging.info(f"n = {n}, p = {p}, m = {m}, min_motif_size = {min_motif_size}, max_motif_size = {max_motif_size}")
        logging.info(f"min_num_motifs = {min_num_motifs}, max_num_motifs = {max_num_motifs}, motif_overlap = {motif_overlap}")
        logging.info(f"min_base_edge_weight = {min_base_edge_weight}, max_base_edge_weight = {max_base_edge_weight}")
        logging.info(f"min_motif_edge_weight = {min_motif_edge_weight}, max_motif_edge_weight = {max_motif_edge_weight}")
        logging.info(f"plot = {plot}")

    # Create the ER graph adjacency matrix
    if base_graph_model == 'ER':
        base_graph = nx.erdos_renyi_graph(n, p)
    elif base_graph_model == 'BA':
        base_graph = nx.barabasi_albert_graph(n, m)
    elif base_graph_model == 'WS':
        base_graph = nx.watts_strogatz_graph(n, k, p)
    base_adjacency = nx.to_numpy_array(base_graph)
    # replace edge weights
    if min_base_edge_weight != max_base_edge_weight:
        assert min_base_edge_weight < max_base_edge_weight
        num_edges = int(base_adjacency.sum()/2)
        # sample `num_edges`
        edge_weights = np.random.uniform(min_base_edge_weight, max_base_edge_weight, num_edges)
        #[1.0, 2.0, 3.0, 1.0, 2.0, ...]
        I, J = np.where(base_adjacency) 
        # get upper triangular matrix
        idx = np.where(J > I)[0]
        I = I[idx] 
        J = J[idx]
        num_edges = len(I) 
        edge_weights = np.random.uniform(0., 1., (num_edges,))
        base_adjacency[I, J] = edge_weights
        base_adjacency[J, I] = edge_weights
        
    # check degree 0 nodes in the original ER graph
    if np.sum(base_adjacency.sum(axis=1) == 0) > 0:
        logging.info(f"Original {base_graph_model} graph contains nodes with degree 0.")

    # plot original ER graph
    if plot:
        plt.figure(figsize=(14, 8))
        plt.subplot(221)
        nx.draw_networkx(nx.Graph(base_adjacency), with_labels=True, node_color='skyblue', node_size=300)
        plt.title(f'Original {base_graph_model} graph')

    # Randomly determine the number of motifs to embed
    num_motifs = random.randint(min_num_motifs, max_num_motifs)

    node_colors = ['skyblue'] * n
    edge_colors = {}  # Initialize edge colors as a dictionary
    # Randomly select 'n' colors from the named_colors list 
    edge_colors_list = random.sample(list(mcolors.CSS4_COLORS.keys()), num_motifs)

    # nodes available to create subgraphs: for the first motif, all nodes are available
    nodes_available = list(np.arange(base_adjacency.shape[0]))

    for motif_i in range(num_motifs):
        # Generate a random motif size within the specified range
        motif_size = random.randint(min_motif_size, max_motif_size)

        # Select nodes to replace
        nodes_to_replace = np.random.choice(nodes_available, size=(motif_size,), replace=False)
        # if motif overlapping not allowed
        if not motif_overlap:    
            nodes_available = [node for node in nodes_available if node not in nodes_to_replace]

        if log:  # Check if logging is enabled
            logging.info(f'motif_{motif_i}: {nodes_to_replace}')

        # change BA nodes' color
        for replaceable_node in nodes_to_replace:
            # if overlapping motifs not allowed, color nodes of each motif differently
            if not motif_overlap:
                node_colors[replaceable_node] = edge_colors_list[motif_i]
            # color all nodes of motifs same, but edge colors are decided motif-wise later
            else:
                node_colors[replaceable_node] = 'lightcoral'

        # Create the BA model adjacency matrix for the motif
        if motif_size >= m:
            if motif_graph_model == 'ER':
                motif_adjacency = nx.to_numpy_array(nx.erdos_renyi_graph(motif_size, p))
            elif motif_graph_model == 'BA':
                motif_adjacency = nx.to_numpy_array(nx.barabasi_albert_graph(motif_size, m))
            elif motif_graph_model == 'WS':
                motif_adjacency = nx.to_numpy_array(nx.watts_strogatz_graph(n, k, p))

            # replace edge weights
            if min_motif_edge_weight != max_motif_edge_weight:
                assert min_motif_edge_weight < max_motif_edge_weight
                num_edges = int(motif_adjacency.sum()/2)
                # sample `num_edges`
                edge_weights = np.random.uniform(min_motif_edge_weight, max_motif_edge_weight, num_edges)
                #[1.0, 2.0, 3.0, 1.0, 2.0, ...]
                I, J = np.where(motif_adjacency) 
                # get upper triangular matrix
                idx = np.where(J > I)[0]
                I = I[idx] 
                J = J[idx]
                num_edges = len(I) 
                edge_weights = np.random.uniform(0., 1., (num_edges,))
                motif_adjacency[I, J] = edge_weights
                motif_adjacency[J, I] = edge_weights

            if np.sum(motif_adjacency.sum(axis=1) == 0) > 0:
                logging.info(f"{motif_graph_model} graph contains nodes with degree 0.")

            if plot:
                # Set edge colors within the subgraph to distinguish from others
                for i, idx in enumerate(nodes_to_replace):
                    for j, jdx in enumerate(nodes_to_replace):
                        if i != j:
                            edge_colors[(idx, jdx)] = edge_colors_list[motif_i]
        else:
            logging.warning(f"{motif_graph_model} model requires motif_size >= m")
            return None

        # Replace selected nodes in ER adjacency matrix with BA model motif
        for i, idx in enumerate(nodes_to_replace):
            base_adjacency[idx, nodes_to_replace] = motif_adjacency[i]

    # Ensure that none of the nodes have degree 0 in the original ER graph
    if np.sum(base_adjacency.sum(axis=1) == 0) > 0:
        logging.warning(f"{base_graph_model} with {motif_graph_model} subgraph contains nodes with degree 0:"
                         "{np.sum(base_adjacency.sum(axis=1) == 0)}")

    # Connect zero-degree nodes to random other nodes
    final_graph = nx.Graph(base_adjacency)
    degree_dict = dict(final_graph.degree())
    zero_degree_nodes = [k for k in degree_dict if degree_dict[k] == 0]
    logging.info(f'Nodes with zero degree in {base_graph_model} with {motif_graph_model} subgraph: {zero_degree_nodes}')
    nonzero_degrees = np.sort([v for k, v in degree_dict.items() if k not in zero_degree_nodes] )
    nonzero_degree_nodes = [k for k, v in degree_dict.items() if v != 0]
    degrees, counts = np.unique(nonzero_degrees, return_counts=True) 
    probs = counts / np.sum(counts)
    for node1 in zero_degree_nodes:
        sample_degree = np.random.choice(degrees, p=probs) 
        sample_connections = np.random.choice(nonzero_degree_nodes, size=(sample_degree,), replace=False)
        for node2 in sample_connections: 
            final_graph.add_edge(node1, node2)

    degree_dict = dict(final_graph.degree())
    zero_degree_nodes = [k for k in degree_dict if degree_dict[k] == 0]
    if len(zero_degree_nodes) == 0:
        logging.info("Nodes with degree 0 connected!")
    # update adjacency matrix
    base_adjacency = nx.to_numpy_array(final_graph)

    # Plot final graph with subgraphs
    if plot:
        plt.subplot(222)
        pos = nx.spring_layout(final_graph)  # Position nodes using a layout algorithm
        edge_color_l = [edge_colors.get((u, v), 'gray') for u, v in final_graph.edges()]
        nx.draw_networkx(final_graph, pos, with_labels=True, node_color=node_colors, node_size=300, edge_color=edge_color_l)
        plt.title(f'{base_graph_model} graph with {num_motifs} {motif_graph_model} subgraphs')

        # Plot degree distribution
        plt.subplot(223)
        for g, l, c in zip([base_graph, final_graph],
                                    [f'{base_graph_model}', f'{base_graph_model} with {motif_graph_model} subgraphs'],
                                    ['skyblue', 'lightcoral']):
            plt.plot(np.sort([j for _, j in g.degree()])[::-1], marker='.', alpha=0.7, label=l, color=c)
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.title('Degree Distribution')
        plt.legend()

        # plot Clustering coefficient plot
        plt.subplot(224)
        for g, l, c in zip([base_graph, final_graph],
                                    [f'{base_graph_model}', f'{base_graph_model} with {motif_graph_model} subgraphs'],
                                    ['skyblue', 'lightcoral']):
            clustering_coefficient = nx.average_clustering(g)
            # Create a histogram of clustering coefficients
            clustering_values = list(nx.clustering(g).values())
            plt.hist(clustering_values, bins=20, alpha=0.5, color=c, label=l)
            # Add a vertical line for the average clustering coefficient
            plt.axvline(x=clustering_coefficient, linestyle='--', color=c, 
                        label=f'avg clustering coeff ({clustering_coefficient:.2f})')
            plt.xlabel('Clustering Coefficient')
            plt.ylabel('Frequency')
        plt.title('Clustering Coefficient')
        plt.legend()

    return base_adjacency
