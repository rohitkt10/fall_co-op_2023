import networkx as nx
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import logging
from .graph_visualizer import GraphVisualizer


class GraphWithMotifs:
    """
    Create graph embedded with motifs.
    """ 
    def __init__(self,
                 base_graph_model='ER',
                 motif_graph_model='BA',
                 n=300, p=0.2, m=3, k=4,
                 min_motif_size=10, max_motif_size=30,
                 min_num_motifs=3, max_num_motifs=5,
                 motif_overlap=False,
                 min_base_edge_weight=1.0, max_base_edge_weight=1.0,
                 min_motif_edge_weight=1.0, max_motif_edge_weight=1.0,
                 node_feat_dim=10,
                 plot=False,
                 log=True):
        """
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
        - node_feat_dim (int): Dimension of the node features (default is 10).
        - plot (bool): Whether to plot the graphs (default is False).
        - log (bool): Whether to log variable values for debugging (default is True).
        """
        self.base_graph_model = base_graph_model
        self.motif_graph_model = motif_graph_model
        self.n = n
        self.p = p
        self.m = m
        self.k = k
        self.min_motif_size = min_motif_size
        self.max_motif_size = max_motif_size
        self.min_num_motifs = min_num_motifs
        self.max_num_motifs = max_num_motifs
        self.motif_overlap = motif_overlap
        self.min_base_edge_weight = min_base_edge_weight
        self.max_base_edge_weight = max_base_edge_weight
        self.min_motif_edge_weight = min_motif_edge_weight
        self.max_motif_edge_weight = max_motif_edge_weight
        self.node_feat_dim = node_feat_dim
        self.node_features = {}  # a dictionary mapping node ID to its one-hot encoded feature
        self.motif_nodes = []  # list of motif nodes
        self.plot = plot
        self.log = log
        self.base_graph = None  # base graph created as per the specified model
        self.final_graph = None # final graph with motifs embedded
        self.graph_visualizer = None # GraphVisualizer object

    def create_graph_with_motif_adjacency(self, motifs_list=None):
        """
        Create a large sparse graph using a specified base graph model and insert motifs by replacing
        a specified number of nodes with a specified graph model.

        Args:
        - motifs_list (list): List of motifs {'adj_matrix': ..., 'node_features': ..., 'class_label': ...}.
        Returns:
        - np.ndarray or None: The resulting adjacency matrix of the modified graph, or None if invalid parameters.
        - list: List of motifs.
        """
        if self.log:  # Check if logging is enabled
            logging.info(f"base_graph_model= {self.base_graph_model}, motif_graph_model= {self.motif_graph_model}")
            logging.info(f"n = {self.n}, p = {self.p}, m = {self.m}, min_motif_size = {self.min_motif_size}, max_motif_size = {self.max_motif_size}")
            logging.info(f"min_num_motifs = {self.min_num_motifs}, max_num_motifs = {self.max_num_motifs}, motif_overlap = {self.motif_overlap}")
            logging.info(f"min_base_edge_weight = {self.min_base_edge_weight}, max_base_edge_weight = {self.max_base_edge_weight}")
            logging.info(f"min_motif_edge_weight = {self.min_motif_edge_weight}, max_motif_edge_weight = {self.max_motif_edge_weight}")
            logging.info(f"plot = {self.plot}")

        # Create the ER graph adjacency matrix
        self.base_graph = self.create_graph_model(self.base_graph_model, self.n)
        base_adjacency = nx.to_numpy_array(self.base_graph)

        # replace edge weights
        base_adjacency = self.replace_edge_weights(base_adjacency, self.min_base_edge_weight, self.max_base_edge_weight)
        # update base_graph
        self.base_graph = nx.Graph(base_adjacency)
            
        # check degree 0 nodes in the original ER graph
        if np.sum(base_adjacency.sum(axis=1) == 0) > 0:
            logging.info(f"Original {self.base_graph_model} graph contains nodes with degree 0.")

        # Randomly determine the number of motifs to embed
        num_motifs = random.randint(self.min_num_motifs, self.max_num_motifs)

        node_colors = ['skyblue'] * self.n
        edge_colors = {}  # Initialize edge colors as a dictionary
        # Randomly select 'n' colors from the named_colors list 
        edge_colors_list = random.sample(list(mcolors.CSS4_COLORS.keys()), num_motifs)

        # nodes available to create subgraphs: for the first motif, all nodes are available
        nodes_available = list(np.arange(base_adjacency.shape[0]))
        # list of motifs
        motifs = []

        for motif_i in range(num_motifs):
            # Generate a random motif size within the specified range
            motif_size = random.randint(self.min_motif_size, self.max_motif_size)

            # if unique motifs required
            if motifs_list is not None:  
                # pick random motif from the list of motifs
                picked_motif = random.choice(motifs_list)
                motif_adjacency = picked_motif['adj_matrix']
                motif_size = motif_adjacency.shape[0]

            # Select nodes to replace
            nodes_to_replace = np.random.choice(nodes_available, size=(motif_size,), replace=False)
            motifs.append(list(nodes_to_replace))
            # if motif overlapping not allowed
            if not self.motif_overlap:    
                nodes_available = [node for node in nodes_available if node not in nodes_to_replace]

            if self.log:  # Check if logging is enabled
                logging.info(f'motif_{motif_i}: {nodes_to_replace}')

            # change motif nodes' color
            for replaceable_node in nodes_to_replace:
                # if overlapping motifs not allowed, color nodes of each motif differently
                if not self.motif_overlap:
                    node_colors[replaceable_node] = edge_colors_list[motif_i]
                # color all nodes of motifs same, but edge colors are decided motif-wise later
                else:
                    node_colors[replaceable_node] = 'lightcoral'

            # Create the motif model adjacency matrix
            if motif_size >= self.m:
                # if unique motifs not required
                if motif_adjacency is None:
                    motif_adjacency = nx.to_numpy_array(self.create_graph_model(self.motif_graph_model, motif_size))

                # replace edge weights
                motif_adjacency = self.replace_edge_weights(motif_adjacency, self.min_motif_edge_weight, self.max_motif_edge_weight)

                if np.sum(motif_adjacency.sum(axis=1) == 0) > 0:
                    logging.info(f"{self.motif_graph_model} graph contains nodes with degree 0.")

                if self.plot:
                    # Set edge colors within the subgraph to distinguish from others
                    for i, idx in enumerate(nodes_to_replace):
                        for j, jdx in enumerate(nodes_to_replace):
                            if i != j:
                                edge_colors[(idx, jdx)] = edge_colors_list[motif_i]
            else:
                logging.warning(f"{self.motif_graph_model} model requires motif_size >= m")
                return None

            # Replace selected nodes in base graph adjacency matrix with motif model
            for i, idx in enumerate(nodes_to_replace):
                base_adjacency[idx, nodes_to_replace] = motif_adjacency[i]
                # update motif node features
                if picked_motif is not None:
                    self.node_features[idx] = picked_motif['node_features'][i]
                else:
                    self.node_features[idx] = self.assign_node_feature()

        # Ensure that none of the nodes have degree 0 in the original base graph
        if np.sum(base_adjacency.sum(axis=1) == 0) > 0:
            logging.warning(f"{self.base_graph_model} with {self.motif_graph_model} subgraph contains nodes with degree 0:"
                            "{np.sum(base_adjacency.sum(axis=1) == 0)}")

        # Connect zero-degree nodes to random other nodes
        self.final_graph = nx.Graph(base_adjacency)
        self.connect_nodes_with_zero_degree()

        degree_dict = dict(self.final_graph.degree())
        zero_degree_nodes = [k for k in degree_dict if degree_dict[k] == 0]
        if len(zero_degree_nodes) == 0:
            logging.info("Nodes with degree 0 connected!")

        # Plot graphs
        if self.plot:
            self.graph_visualizer = GraphVisualizer(
                [self.base_graph, self.final_graph], 
                [f'Base {self.base_graph_model} graph', f'{self.base_graph_model} with {self.motif_graph_model} subgraphs'], 
                ['skyblue', 'lightcoral'])
            _, ax = plt.subplots(3, 2, figsize=(14, 10))
            self.graph_visualizer.plot_network_graph(ax[0, 0], graph_index=0, with_labels=True, node_size=300)
            self.plot_final_graph_with_motifs(edge_colors, node_colors, num_motifs, ax[0, 1])
            self.graph_visualizer.plot_degree_distribution(ax[1, 0])
            self.graph_visualizer.plot_clustering_coefficient_distribution(ax[1, 1])
            self.graph_visualizer.plot_betweenness_centrality_distribution(ax[2, 0])
            self.graph_visualizer.plot_path_length_distribution(ax[2, 1], graph_index=1)
            plt.tight_layout()
            plt.show()

        # Assign a random one-hot encoded feature to each node in the base graph
        self.assign_nonmotif_node_features()

        # list of motif nodes
        self.motif_nodes = [node for motif in motifs for node in motifs]

        # return updated adjacency matrix
        base_adjacency = nx.to_numpy_array(self.final_graph)
        return base_adjacency, motifs, self.node_features


    def create_graph_model(self, graph_model, n):
        """
        Create a graph using the specified model.
        """
        if graph_model == 'ER':
            return nx.erdos_renyi_graph(n, self.p)
        elif graph_model == 'BA':
            return nx.barabasi_albert_graph(n, self.m)
        elif graph_model == 'WS':
            return nx.watts_strogatz_graph(self.n, self.k, self.p)
        else:
            logging.warning(f"Invalid graph model: {graph_model}")
            return None
    
    def replace_edge_weights(self, adjacency_matrix, min_edge_weight, max_edge_weight):
        """
        Replace edge weights in the graph adjacency matrix with random values between the specified range.
        TODO: make edge weights a vector of size `num_edge_features`
        """
        if min_edge_weight != max_edge_weight:
            assert min_edge_weight < max_edge_weight
            num_edges = int(adjacency_matrix.sum()/2)
            # sample `num_edges`
            #[1.0, 2.0, 3.0, 1.0, 2.0, ...]
            I, J = np.where(adjacency_matrix) 
            # get upper triangular matrix
            idx = np.where(J > I)[0]
            I = I[idx] 
            J = J[idx]
            num_edges = len(I) 
            edge_weights = np.random.uniform(min_edge_weight, max_edge_weight, num_edges)
            adjacency_matrix[I, J] = edge_weights
            adjacency_matrix[J, I] = edge_weights

        return adjacency_matrix

    def connect_nodes_with_zero_degree(self):
        """
        Connect nodes with zero degree to random other nodes
        """
        degree_dict = dict(self.final_graph.degree())
        zero_degree_nodes = [k for k in degree_dict if degree_dict[k] == 0]
        logging.info(f'Nodes with zero degree in {self.base_graph_model} with {self.motif_graph_model} subgraph: {zero_degree_nodes}')
        nonzero_degrees = np.sort([v for k, v in degree_dict.items() if k not in zero_degree_nodes] )
        nonzero_degree_nodes = [k for k, v in degree_dict.items() if v != 0]
        degrees, counts = np.unique(nonzero_degrees, return_counts=True) 
        probs = counts / np.sum(counts)
        for node1 in zero_degree_nodes:
            sample_degree = np.random.choice(degrees, p=probs) 
            sample_connections = np.random.choice(nonzero_degree_nodes, size=(sample_degree,), replace=False)
            for node2 in sample_connections: 
                self.final_graph.add_edge(node1, node2)

    def assign_node_feature(self):
        """
        Randomly generate a one-hot encoded feature of dimension `node_feat_dim` for a node.
        """
        one_hot_feature = np.zeros(self.node_feat_dim, dtype=int)
        atom_type = np.random.randint(0, self.node_feat_dim)
        one_hot_feature[atom_type] = 1
        return list(one_hot_feature)

    def assign_nonmotif_node_features(self):
        """
        Assign random one-hot encoded features of dimension `node_feat_dim` to nodes of the base graph.
        """
        for node in self.final_graph.nodes():
            if node not in self.motif_nodes:
                self.node_features[node] = self.assign_node_feature()

    def plot_final_graph_with_motifs(self, edge_colors, node_colors, num_motifs, ax):
        """
        Plot final graph with motifs
        """
        pos = nx.spring_layout(self.final_graph)  # Position nodes using a layout algorithm
        edge_color_l = [edge_colors.get((u, v), 'gray') for u, v in self.final_graph.edges()]
        nx.draw_networkx(self.final_graph, pos, with_labels=True, node_color=node_colors, node_size=300, edge_color=edge_color_l, ax=ax)
        ax.set_title(f'{self.base_graph_model} graph with {num_motifs} {self.motif_graph_model} subgraphs')
