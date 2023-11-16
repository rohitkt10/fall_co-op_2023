# Calculate integrated gradients for both classes for test set
# Calculate correlation (pearson, spearman) between node importance and node degree
# Calculate avg of correlation of all graphs in test set and this to results as new column

import sys
sys.path.append("..")

import os
import glob
import csv
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch_geometric.nn import GCNConv, GraphConv, ChebConv
from src.models.graphconv_residualnet import GraphConvResidualNet
from captum.attr import IntegratedGradients
from scipy.stats import pearsonr, spearmanr

saved_dir = '/Users/kumarh/Documents/fall_co-op_2023/experiments/syndata_exp_2023-11-15_19-51-04/'

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

# Initialize the CSV file
training_csv_file = f'{saved_dir}/graphgenexp_results.csv'
model_data = pd.read_csv(training_csv_file)
out_csv_file = f'{saved_dir}/graphgenexp_correlation_results.csv'
csv_exists = os.path.exists(out_csv_file)

# If the CSV file doesn't exist, create it with header
with open(out_csv_file, 'a', newline='') as file:
    writer = csv.writer(file)
    if not csv_exists:
        writer.writerow(list(model_data.columns.values) + 
                        ["pearson_corr_avg_class_0", "pearson_corr_avg_class_1", 
                         "spearman_corr_avg_class_0", "spearman_corr_avg_class_1"])

print(f'Writing results to {out_csv_file}')
def format_dropout(dropout):
    if dropout.is_integer():
        return str(int(dropout))
    else:
        return str(dropout)
    
for index, row in model_data.iterrows():
    print(f"Testing #{index+1}/{len(model_data)}: {row['conv_type']} with layers={row['num_layers']}, dropout={row['dropout']}, and l2={row['l2']}")
    model_path = f'{saved_dir}/graphgen_model_{row["conv_type"]}_{row["num_layers"]}_{format_dropout(row["dropout"])}_{row["l2"]}.pt'
    conv_types = {'GraphConv': GraphConv, 'GCNConv': GCNConv, 'ChebConv': ChebConv}
    model = GraphConvResidualNet(dim=32,
                                num_features=num_features,
                                num_classes=num_classes,
                                num_layers=row["num_layers"],
                                conv_type=conv_types[row["conv_type"]],
                                dropout=row['l2']).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    corr_results = {}
    # For each class
    for class_label in tqdm([0, 1]):
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
        corr_results[f"pearson_corr_avg_class_{class_label}"] = np.mean(pearson_corrs)
        corr_results[f"spearman_corr_avg_class_{class_label}"] = np.mean(spearman_corrs)

    # Append results to CSV
    with open(out_csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(list(row.values) +  
                         [corr_results[f"pearson_corr_avg_class_0"], corr_results[f"pearson_corr_avg_class_1"],
                         corr_results[f"spearman_corr_avg_class_0"], corr_results[f"spearman_corr_avg_class_1"]])
