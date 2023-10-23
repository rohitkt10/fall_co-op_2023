import pickle
import torch
from torch_geometric.data import Dataset, Data

class SyntheticGraphDataset(Dataset):
    """
    A PyTorch Geometric dataset of synthetic graphs with motifs.
    """
    def __init__(self, dataset_file):
        super().__init__()

        # Load the dataset from file
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)

        # Convert the dataset to a list of PyTorch Geometric Data objects
        self.data_list = []
        all_labels = set()  # Store all unique labels.
        for graph_adj, motifs, node_features, label in dataset:
            # Convert the graph adjacency matrix to a COO sparse tensor
            edge_index = torch.tensor(graph_adj.nonzero(), dtype=torch.long).t()
            edge_attr = torch.tensor(graph_adj[edge_index[0], edge_index[1]], dtype=torch.float)

            # Convert the node features to a tensor
            x = torch.tensor(list(node_features.values()), dtype=torch.float)

            # Convert the label to a tensor
            y = torch.tensor(label, dtype=torch.long)

            # Create a PyTorch Geometric Data object
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

            self.data_list.append(data)
            all_labels.add(label)  # Add the label to the set of all unique labels.

        # Calculate the number of classes dynamically
        self._num_classes = len(all_labels)

    @property
    def num_classes(self):
        return self._num_classes
    
    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]
