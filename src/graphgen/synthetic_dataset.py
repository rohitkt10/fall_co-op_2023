from .graph_with_motifs import GraphWithMotifs

import random
import pickle
import networkx as nx

class GraphWithMotifsDataset:
    def __init__(self, num_graphs=10, min_nodes=80, max_nodes=100, **kwargs):
        """
        Args:
        - num_graphs (int): Number of graphs in the dataset.
        - min_nodes (int): Minimum number of nodes in a graph.
        - max_nodes (int): Maximum number of nodes in a graph.
        - kwargs: Arguments for the GraphWithMotifs class.
        """
        self.num_graphs = num_graphs
        self.kwargs = kwargs
        self.dataset = []
        n = random.randint(min_nodes, max_nodes)
        self.graph_with_motifs = GraphWithMotifs(n=n, **kwargs)
        self.motif_graph_model = kwargs.get('motif_graph_model', 'ER')
        self.min_motif_size = kwargs.get('min_motif_size', 10)
        self.max_motif_size = kwargs.get('max_motif_size', 30)
        self.max_num_motifs = kwargs.get('max_num_motifs', 5)

    def generate_dataset(self):
        """
        Generate the dataset of graphs with motifs.

        Returns:
        - dataset (list): List of tuples containing the graph adjacency matrix, the motifs, and the class label.
        """

        # Generate unique motifs
        classes = 2
        motif_adjacencies = []
        for _ in range(classes):
            # Generate a random motif size within the specified range
            motif_size = random.randint(self.min_motif_size, self.max_motif_size)
            motif_adjacencies.append(nx.to_numpy_array(self.graph_with_motifs.create_graph_model(self.motif_graph_model, motif_size)))

        # Generate samples for each class
        for i in range(self.num_graphs):
            # Create a graph with a positive motif
            label = random.choice(range(classes))
            graph_adj, motifs, node_features = self.graph_with_motifs.create_graph_with_motif_adjacency([motif_adjacencies[label]])
            self.dataset.append((graph_adj, motifs, node_features, label))

    def get_dataset(self):
        return self.dataset
    
    def save_dataset(self, dataset_name):
        """
        Save the dataset to a file.
        """
        with open(f'{dataset_name}.pkl', 'wb') as f:
            pickle.dump(self.dataset, f)


