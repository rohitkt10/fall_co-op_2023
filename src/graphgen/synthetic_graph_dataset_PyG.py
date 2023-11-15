import numpy as np
import torch
from torch_geometric.data import Dataset, Data
import pickle

class SyntheticGraphDatasetPyG(Dataset):
    """
    A dataset for synthetic graphs.
    """
    def __init__(self, dataset_file, transform=None, pre_transform=None):
        super().__init__(transform, pre_transform)
        # Load your custom dataset from the saved file.
        with open(dataset_file, 'rb') as f:
            self.data = pickle.load(f)

    def len(self):
        """
        The number of graphs in the dataset. 
        """
        return len(self.data)

    def get(self, idx):
        """
        Get the graph data object at index idx.
        """
        graph_adj, motifs, node_features, label = self.data[idx]

        # Convert your data to PyTorch Geometric format.
        edge_index = torch.tensor(np.array(graph_adj.nonzero()), dtype=torch.long)
        x = torch.tensor(list(node_features.values()), dtype=torch.float)
        y = torch.tensor([label], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y, motifs=motifs)
        return data
