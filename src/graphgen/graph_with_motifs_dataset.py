from .graph_with_motifs import GraphWithMotifs

import random
import pickle
import numpy as np
import networkx as nx

class GraphWithMotifsDataset:
    def __init__(self, num_graphs=100, min_nodes=80, max_nodes=100, **kwargs):
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
        # TODO: Embed different number of motifs in graph 
        # a) non-overlapping classes b) overlapping (class decided based on probability)
        classes = 2
        motifs_dict = {}
        for i in range(classes):
            # Generate a random motif size within the specified range
            motif_size = random.randint(self.min_motif_size, self.max_motif_size)
            motif_adj = nx.to_numpy_array(self.graph_with_motifs.create_graph_model(self.motif_graph_model, motif_size))
            motif_features = []
            # Get the indices of the non-zero elements in the matrix
            row_indices, col_indices = np.where(motif_adj != 0)
            # Combine the row and column indices to get a list of nodes
            motif_nodes = list(set(row_indices) | set(col_indices))
            for _ in motif_nodes:
                motif_features.append(self.graph_with_motifs.assign_node_feature())
            motifs_dict[f'motif_{i}'] = {'name': f'motif_{i}', 'adj_matrix': motif_adj, 'node_features': motif_features, 'class_label': i}
            

        # Generate samples for each class
        for _ in range(self.num_graphs):
            # Create a graph with a positive motif
            motif_name = random.choice(list(motifs_dict.keys()))
            graph_adj, motifs, node_features = self.graph_with_motifs.create_graph_with_motif_adjacency([motifs_dict[motif_name]])
            self.dataset.append((graph_adj, motifs, node_features, motifs_dict[motif_name]['class_label']))

    def get_dataset(self):
        return self.dataset
    
    def save_dataset(self, dataset_name):
        """
        Save the dataset to a file.
        """
        with open(f'{dataset_name}.pkl', 'wb') as f:
            pickle.dump(self.dataset, f)


