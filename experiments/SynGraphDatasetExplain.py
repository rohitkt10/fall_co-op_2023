# Calculate integrated gradients for both classes for test set
# Calculate correlation (pearson, spearman) between node importance and node degree
# Calculate avg of correlation of all graphs in test set and this to results as new column

import sys
sys.path.append(".")

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch_geometric.nn import GCNConv, GraphConv, ChebConv
from src.models.graphconv_residualnet import GraphConvResidualNet
from captum.attr import IntegratedGradients
from scipy.stats import pearsonr, spearmanr

saved_dir = 'syndata_exp1/'

#load dataset
print("Loading dataset...")
train_dataset = torch.load(f'{saved_dir}/train_dataset.pt')
val_dataset = torch.load(f'{saved_dir}/val_dataset.pt')
test_dataset = torch.load(f'{saved_dir}/test_dataset.pt')
num_features = 10
num_classes = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def model_forward(mask, data):
    """
    Helper function to perform forward pass of model with masked input
    """
    x = data.x
    y_list = []
    for m in mask:
        x_masked = x*m
        y = model(x_masked, data.edge_index, data.batch)[0]
        y_list.append(y)
    return torch.stack(y_list).squeeze()

# Initialize results
results = {"model_name": [],
           "pearson_corr_avg_class_0": [], "pearson_corr_avg_class_1": [], 
           "spearman_corr_avg_class_0": [], "spearman_corr_avg_class_1": []}

model_list = glob.glob(f'{saved_dir}/graphgen_model_*.pt')

# test on all models saved in saved_dir
for i, model_path in enumerate(model_list):
    model_name = model_path.split('/')[-1].split('.')[0]
    n_layer = int(model_name.split('_')[2])
    conv_types = {'GraphConv': GraphConv, 'GCNConv': GCNConv, 'ChebConv': ChebConv}
    conv_type = conv_types[model_name.split('_')[3]]
    dropout = float(model_name.split('_')[4])
    l2 = float(model_name.split('_')[5])
    print(f"Testing #{i+1}/{len(model_list)} model with {n_layer} {conv_type.__name__} convolution, dropout={dropout}, and l2={l2}")

    model = GraphConvResidualNet(dim=32,
                                num_features=num_features,
                                num_classes=num_classes,
                                num_layers=n_layer,
                                conv_type=conv_type,
                                dropout=dropout).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    results["model_name"].append(model_name)

    # For each class
    for class_label in [0, 1]:
        # For each graph in the test set
        pearson_corrs = []
        spearman_corrs = []
        for input_data in test_dataset:
            input_data = input_data.to(device)

            # Check if the class label matches the current iteration
            if input_data.y.item() == class_label:
                # Generate node-level attributions for the corresponding class
                mask = torch.stack([torch.ones_like(input_data.x).requires_grad_(True).to(device)] * 1)
                out = model_forward(mask, input_data)

                ig = IntegratedGradients(model_forward)

                # Calculate IG scores based on the class label
                ig_attr = ig.attribute(mask, target=class_label, additional_forward_args=(input_data,))[0]

                # Calculate node degree
                node_degree = input_data.edge_index[1].bincount(minlength=input_data.num_nodes)

                # Calculate correlation (Pearson and Spearman) between node importance and node degree
                pearson_corr, _ = pearsonr(torch.abs(ig_attr).sum(dim=1).cpu().detach().numpy(), node_degree.cpu().detach().numpy())
                spearman_corr, _ = spearmanr(torch.abs(ig_attr).sum(dim=1).cpu().detach().numpy(), node_degree.cpu().detach().numpy())

                pearson_corrs.append(pearson_corr)
                spearman_corrs.append(spearman_corr)

        # Calculate average correlation across all graphs for the current class
        avg_pearson_corr = np.mean(pearson_corrs)
        avg_spearman_corr = np.mean(spearman_corrs)

        # Store results for the current model and class
        results[f"pearson_corr_avg_class_{class_label}"].append(avg_pearson_corr)
        results[f"spearman_corr_avg_class_{class_label}"].append(avg_spearman_corr)

# Convert results to pandas DataFrame and display as table
results_df = pd.DataFrame(results)
print(results_df)

# Save results to csv
results_df.to_csv(f'{saved_dir}/graphgenexp_results_with_correlation_per_class.csv', index=False)
