import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class GraphAnalyzer:
    """
    A class for analyzing and visualizing graphs.
    """
    def __init__(self, edge_indices_file=None, nodes_file_name=None, dataset_name=None):
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
        """
        self.edge_indices_file = edge_indices_file
        self.nodes_file_name = nodes_file_name
        self.dataset_name = dataset_name
        self.graph = None
        self.load_graph()
        self.er_model = self.generate_erdos_renyi()
        self.ba_model = self.generate_barabasi_albert()
        self.ws_model = self.generate_watts_strogatz()

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

            # a sparse matrix in Compressed Sparse Row (CSR) format
            # loaded_graph_adj = nx.adjacency_matrix(loaded_graph)
            # dense numpy array
            graph_adj = nx.to_numpy_array(self.graph)
            assert graph_adj.shape == (num_nodes, num_nodes)
            assert np.array_equal(graph_adj, graph_adj.T)
        else:
            raise Exception("No edge indices file provided")
        
    def _calculate_graph_properties(self, graph):
        """
        Calculate the graph properties for the given graph.
        """
        properties = {
            "Number of Nodes": len(graph.nodes()),
            "Number of Edges": len(graph.edges()),
            "Average Degree": f'{np.mean(list(dict(graph.degree()).values())):.3f}',
            "Clustering Coefficient": f'{nx.average_clustering(graph):.3f}',
            "Betweenness Centrality": f'{np.mean(list(nx.betweenness_centrality(graph).values())):.4f}',
            "Average Shortest Path Length": f'{nx.average_shortest_path_length(graph):.3f}'
        }
        return properties

    def calculate_graph_properties(self):
        """
        Calculate the graph properties for the loaded graph, Erdos-Renyi, Barabasi-Albert, and Watts-Strogatz models.
        """
        properties = {'Loaded Graph': self._calculate_graph_properties(self.graph),
                      'Erdos-Renyi': self._calculate_graph_properties(self.er_model),
                      'Barabasi-Albert': self._calculate_graph_properties(self.ba_model),
                      'Watts-Strogatz': self._calculate_graph_properties(self.ws_model)
                      }
        return properties

    def generate_erdos_renyi(self):
        """
        Generate an Erdos-Renyi random graph with the same number of nodes and edges as the loaded graph.
        """
        num_nodes = len(self.graph.nodes)
        # num_edges = len(self.graph.edges)
        # er_model = nx.gnm_random_graph(num_nodes, num_edges)
        p = len(self.graph.edges) / (num_nodes * (num_nodes - 1) / 2)
        er_model = nx.erdos_renyi_graph(num_nodes, p)
        return er_model

    def generate_barabasi_albert(self, m=5):
        """
        Generate a Barabasi-Albert random graph with the same number of nodes and edges as the loaded graph.
        """
        num_nodes = len(self.graph.nodes)
        ba_model = nx.barabasi_albert_graph(num_nodes, m)
        return ba_model

    def generate_watts_strogatz(self, k=4, p=0.2):
        """
        Generate a Watts-Strogatz random graph with the same number of nodes and edges as the loaded graph.
        """
        num_nodes = len(self.graph.nodes)
        ws_model = nx.watts_strogatz_graph(num_nodes, k, p)
        return ws_model

    def visualize_graph(self):
        """
        Visualize the loaded graph, Erdos-Renyi, Barabasi-Albert, and Watts-Strogatz models.
        """
        _, ax = plt.subplots(2, 2, figsize=(12, 8))
        pos = nx.spring_layout(self.graph)
        nx.draw_networkx(self.graph, pos, with_labels=False, node_size=10, ax=ax[0, 0])
        ax[0, 0].set_title(self.dataset_name)
        pos = nx.spring_layout(self.er_model)
        nx.draw_networkx(self.er_model, pos, with_labels=False, node_size=10, ax=ax[0, 1])
        ax[0, 1].set_title('Erdo-Renyi')
        pos = nx.spring_layout(self.ba_model)
        nx.draw_networkx(self.ba_model, pos, with_labels=False, node_size=10, ax=ax[1, 0])
        ax[1, 0].set_title('Barabasi-Albert')
        pos = nx.spring_layout(self.ws_model)
        nx.draw_networkx(self.ws_model, pos, with_labels=False, node_size=10, ax=ax[1, 1])
        ax[1, 1].set_title('Watts-Strogatz')

    def visualize_distribution(self):
        """
        Visualize the degree, clustering coefficient, and betweenness centrality distributions.
        """
        _, ax = plt.subplots(2, 2, figsize=(12, 8))
        self.plot_degree_distribution(ax[0, 0])
        self.plot_clustering_coefficient_distribution(ax[0, 1])
        self.plot_betweenness_centrality_distribution(ax[1, 0])
        self.plot_avg_shortest_path_length(ax[1, 1])

    def plot_degree_distribution(self, ax):
        """
        Plot degree distribution
        """
        for g, l, c in zip([self.graph, self.er_model, self.ba_model, self.ws_model],
                            [self.dataset_name, 'Erdos-Renyi', 'Barabasi-Albert', 'Watts-Strogatz'],
                            ['skyblue', 'lightcoral', 'lightgreen', 'pink']):
            ax.plot(np.sort([j for _, j in g.degree()])[::-1], marker='.', alpha=0.7, label=l, color=c)
            ax.set_xlabel('Degree')
            ax.set_ylabel('Frequency')
            ax.set_title('Degree Distribution')
            ax.legend()

    def plot_clustering_coefficient_distribution(self, ax):
        """
        Plot Clustering coefficient plot
        """
        for g, l, c in zip([self.graph, self.er_model, self.ba_model, self.ws_model],
                            [self.dataset_name, 'Erdos-Renyi', 'Barabasi-Albert', 'Watts-Strogatz'],
                            ['skyblue', 'lightcoral', 'lightgreen', 'pink']):
            clustering_coefficient = nx.average_clustering(g)
            # Create a histogram of clustering coefficients
            clustering_values = list(nx.clustering(g).values())
            ax.hist(clustering_values, bins=20, alpha=0.5, color=c, label=l)
            # Add a vertical line for the average clustering coefficient
            ax.axvline(x=clustering_coefficient, linestyle='--', color=c, 
                        label=f'avg clustering coeff ({clustering_coefficient:.2f})')
            ax.set_xlabel('Clustering Coefficient')
            ax.set_ylabel('Frequency')
            ax.set_title('Clustering Coefficient')
            ax.legend()

    def plot_betweenness_centrality_distribution(self, ax):
        """
        Plot betweenness centrality plot 
        """
        for g, l, c in zip([self.graph, self.er_model, self.ba_model, self.ws_model],
                            [self.dataset_name, 'Erdos-Renyi', 'Barabasi-Albert', 'Watts-Strogatz'],
                            ['skyblue', 'lightcoral', 'lightgreen', 'pink']):
            betweenness_centrality = nx.betweenness_centrality(g)
            # Create a histogram of betweenness centrality values
            betweenness_values = list(betweenness_centrality.values())
            ax.hist(betweenness_values, bins=20, alpha=0.5, color=c, label=l)
            # Add a vertical line for the average betweenness centrality
            avg_betweenness_centrality = np.mean(betweenness_values)
            ax.axvline(x=avg_betweenness_centrality, linestyle='--', color=c, 
                        label=f'avg betweenness centrality ({avg_betweenness_centrality:.3f})')
            ax.set_xlabel('Betweenness Centrality')
            ax.set_ylabel('Frequency')
            ax.set_title('Betweenness Centrality Distribution')
            ax.legend()

    def plot_avg_shortest_path_length(self, ax):
        path_lengths = []
        for g in [self.graph, self.er_model, self.ba_model, self.ws_model]:
            path_lengths.append(nx.average_shortest_path_length(g))
       
        ax.bar(['Loaded Graph', 'Erdos-Renyi', 'Barabasi-Albert', 'Watts-Strogatz'], 
               path_lengths, 
               color=['skyblue', 'lightcoral', 'lightgreen', 'pink'])
        ax.set_xlabel('Graph Type')
        ax.set_ylabel('Avg Path length')
        ax.set_title('Avg Path Length')

