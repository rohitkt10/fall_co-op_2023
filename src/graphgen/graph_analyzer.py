import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from .graph_visualizer import GraphVisualizer

class GraphAnalyzer:
    """
    A class for analyzing and visualizing graphs.
    """
    def __init__(self, edge_indices_file=None, nodes_file_name=None, dataset_name=None, remove_zero_nodes=False, generate_graphs=True):
        """
        Initializes a GraphAnalyzer object with the given edge indices file, nodes file name, and dataset name.

        The shape of the edge index matrix is (2, num edges). 
        Each column is an (i, j) index pair where the adjacency matrix will have 1.
        Other locations in the adj matrix will be 0. 
        The number of nodes is stored in a json file.

        Parameters
        - edge_indices_file: str   - The path to the edge indices file
        - nodes_file_name: str     - The path to the nodes file
        - dataset_name: str        - The name of the dataset
        - remove_zero_nodes: bool  - Whether to remove nodes with degree 0
        - generate_graphs: bool    - Whether to generate the Erdos-Renyi, Barabasi-Albert, Watts-Strogatz, and Powerlaw Cluster models.
        """
        self.edge_indices_file = edge_indices_file
        self.nodes_file_name = nodes_file_name
        self.dataset_name = dataset_name
        self.remove_zero_nodes = remove_zero_nodes
        self.graph = None
        self.load_graph()
        if generate_graphs:
            self.er_model = self.generate_erdos_renyi()
            self.ba_model = self.generate_barabasi_albert()
            self.ws_model = self.generate_watts_strogatz()
            self.pc_model = self.generate_powerlaw_cluster()
            self.graph_visualizer = GraphVisualizer(
                [self.graph, self.er_model, self.ba_model, self.ws_model, self.pc_model], 
                [self.dataset_name, 'Erdos Renyi', 'Barabasi Albert', 'Watts Strogatz', 'Powerlaw Cluster'],
                ['skyblue', 'lightcoral', 'lightgreen', 'pink', 'lavender'])

    def load_graph(self):
        """
        Load the graph from the edge indices file and nodes file.
        """
        if self.nodes_file_name is not None:
            with open(self.nodes_file_name, 'r') as json_file:
                data = json.load(json_file)
                num_nodes = data.get(self.dataset_name, 0)
        else:
            raise Exception("No nodes file provided")
        if self.edge_indices_file is not None:
            edge_indices = np.load(self.edge_indices_file)
            self.graph = nx.Graph()
            # Create a NetworkX graph from the edge indices
            self.graph.add_nodes_from(range(num_nodes))
            for i in range(edge_indices.shape[1]):
                source, target = edge_indices[:, i]
                self.graph.add_edge(source, target)
                             
            # Remove nodes with degree 0
            if self.remove_zero_nodes:
                self.graph.remove_nodes_from(list(nx.isolates(self.graph)))
                num_nodes = len(self.graph.nodes)

            # a sparse matrix in Compressed Sparse Row (CSR) format
            # loaded_graph_adj = nx.adjacency_matrix(loaded_graph)
            # dense numpy array
            graph_adj = nx.to_numpy_array(self.graph)
            assert graph_adj.shape == (num_nodes, num_nodes)
            assert np.array_equal(graph_adj, graph_adj.T)
        else:
            raise Exception("No edge indices file provided")
        
    def get_graph(self):
        """
        Return the loaded graph.
        """
        return self.graph
        
    def _calculate_graph_properties(self, graph, graph_name):
        """
        Calculate the graph properties for the given graph.
        """
        communities = nx.algorithms.community.greedy_modularity_communities(graph)
        properties = {
            "Number of Nodes": len(graph.nodes()),
            "Number of Edges": len(graph.edges()),
            "Average Degree": f'{np.mean(list(dict(graph.degree()).values())):.3f}',
            # measure of the degree to which its neighbors are connected to each other. 
            # The average clustering coefficient provides a measure of the degree to which the graph is clustered or "cliquish".
            "Clustering Coefficient": f'{nx.average_clustering(graph):.3f}',
            # measure of the degree to which a node lies on the shortest path between other nodes in the graph.
            # Nodes with high betweenness centrality are important for maintaining the connectivity of the graph.
            "Betweenness Centrality": f'{np.mean(list(nx.betweenness_centrality(graph).values())):.4f}',
            # Modularity is a measure of the degree to which a graph can be divided into communities or modules. 
            # In genetic datasets, modularity can be used to identify groups of genes that are co-expressed across different samples.
            "Modularity": f'{nx.algorithms.community.modularity(graph, communities):.3f}',
             # Assortativity is a measure of the tendency of nodes in a graph to be connected to other nodes with similar degrees. 
             # In genetic datasets, assortativity can be used to identify genes that are highly connected to other genes with similar expression levels.
            "Assortativity": f'{nx.degree_assortativity_coefficient(graph):.3f}',
            "Jaccard Similarity": f'{self.calculate_jaccard_similarity()[graph_name]:.3f}'
        }
        return properties

    def calculate_graph_properties(self):
        """
        Calculate the graph properties for the loaded graph, Erdos-Renyi, Barabasi-Albert, and Watts-Strogatz models.
        """
        properties = {'Loaded Graph': self._calculate_graph_properties(self.graph, 'Loaded Graph'),
                      'Erdos Renyi': self._calculate_graph_properties(self.er_model, 'Erdos Renyi'),
                      'Barabasi Albert': self._calculate_graph_properties(self.ba_model, 'Barabasi Albert'),
                      'Watts Strogatz': self._calculate_graph_properties(self.ws_model, 'Watts Strogatz'),
                      'Powerlaw Cluster': self._calculate_graph_properties(self.pc_model, 'Powerlaw Cluster'),
                      }
        return properties

    def generate_erdos_renyi(self):
        """
        Generate an Erdos-Renyi random graph with the same number of nodes and edges as the loaded graph.
        """
        num_nodes = len(self.graph.nodes)
        # num_edges = len(self.graph.edges)
        p = len(self.graph.edges) / (num_nodes * (num_nodes - 1) / 2)
        er_model = nx.erdos_renyi_graph(num_nodes, p)
        return er_model

    def generate_barabasi_albert(self, m=5):
        """
        Generate a Barabasi-Albert random graph with the same number of nodes and edges as the loaded graph.
        - m: int - Number of edges to attach from a new node to existing nodes
        """
        num_nodes = len(self.graph.nodes)
        ba_model = nx.barabasi_albert_graph(num_nodes, m)
        return ba_model

    def generate_watts_strogatz(self, k=4, p=0.2):
        """
        Generate a Watts-Strogatz random graph with the same number of nodes and edges as the loaded graph.
        - k: int - Each node is joined with its k nearest neighbors in a ring topology.
        - p: float - The probability of rewiring each edge.
        """
        num_nodes = len(self.graph.nodes)
        ws_model = nx.watts_strogatz_graph(num_nodes, k, p)
        return ws_model
    
    def generate_powerlaw_cluster(self, m=5, p=0.2):
        """
        Generate a powerlaw cluster graph with the same number of nodes and edges as the loaded graph.
        - n: int - Number of nodes
        - m: int - Number of random edges to add for each new node
        - p: float - Probability of adding a triangle after adding a random edge
        """
        num_nodes = len(self.graph.nodes)
        pc_model = nx.powerlaw_cluster_graph(num_nodes, m, p)
        return pc_model

    def visualize_graph(self):
        """
        Visualize the loaded graph, Erdos-Renyi, Barabasi-Albert, and Watts-Strogatz models.
        """
        _, ax = plt.subplots(1, 5, figsize=(12, 3))
        self.graph_visualizer.plot_network_graph(ax[0], graph_index=0, with_labels=False, node_size=10)
        self.graph_visualizer.plot_network_graph(ax[1], graph_index=1, with_labels=False, node_size=10)
        self.graph_visualizer.plot_network_graph(ax[2], graph_index=2, with_labels=False, node_size=10)
        self.graph_visualizer.plot_network_graph(ax[3], graph_index=3, with_labels=False, node_size=10)
        self.graph_visualizer.plot_network_graph(ax[4], graph_index=4, with_labels=False, node_size=10)
        plt.tight_layout()
        plt.show()

    def visualize_distribution(self):
        """
        Visualize the degree, clustering coefficient, and betweenness centrality distributions.
        """
        _, ax = plt.subplots(2, 2, figsize=(12, 8))
        self.graph_visualizer.plot_degree_distribution(ax[0, 0])
        self.graph_visualizer.plot_clustering_coefficient_distribution(ax[0, 1])
        self.graph_visualizer.plot_betweenness_centrality_distribution(ax[1, 0])
        self.graph_visualizer.plot_modularity(ax[1, 1])
        plt.tight_layout()
        plt.show()

    def calculate_jaccard_similarity(self):
        """
        Calculate the Jaccard similarity between the loaded graph and Erdos-Renyi, Barabasi-Albert, and Watts-Strogatz models.
        """
        # Calculate the Jaccard similarity indices for each generated graph
        jaccard_er = nx.jaccard_coefficient(self.graph, self.er_model.edges())
        jaccard_ba = nx.jaccard_coefficient(self.graph, self.ba_model.edges())
        jaccard_ws = nx.jaccard_coefficient(self.graph, self.ws_model.edges())
        jaccard_pc = nx.jaccard_coefficient(self.graph, self.pc_model.edges())

        # Extract the Jaccard similarity coefficients
        jaccard_coefficients = {
            'Loaded Graph': 1.0,
            'Erdos Renyi': sum([coeff for _, _, coeff in jaccard_er]) / len(self.er_model.edges()),
            'Barabasi Albert': sum([coeff for _, _, coeff in jaccard_ba]) / len(self.ba_model.edges()),
            'Watts Strogatz': sum([coeff for _, _, coeff in jaccard_ws]) / len(self.ws_model.edges()),
            'Powerlaw Cluster': sum([coeff for _, _, coeff in jaccard_pc]) / len(self.pc_model.edges())
            }

        return jaccard_coefficients
