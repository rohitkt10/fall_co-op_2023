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
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, ChebConv
from src.models.graphconv_residualnet import GraphConvResidualNet
from captum.attr import IntegratedGradients, KernelShap, ShapleyValueSampling
from torch_geometric.explain import Explainer, GNNExplainer
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

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

def kernelSHAP_model_forward(mask, data):
    """
    Helper function to perform forward pass of model with masked input
    """
    x = data.x
    y_list = []
    for m in mask:
        x_masked = x*m
        y = model(x_masked, data.edge_index, data.batch)[0]
        y_list.append(y)
    return torch.stack(y_list)

# Initialize the CSV file
training_csv_file = f'{saved_dir}/graphgenexp_results.csv'
model_data = pd.read_csv(training_csv_file)
out_csv_file = f'{saved_dir}/graphgenexp_correlation_crossentropy_results.csv'
csv_exists = os.path.exists(out_csv_file)

# If the CSV file doesn't exist, create it with header
with open(out_csv_file, 'a', newline='') as file:
    writer = csv.writer(file)
    if not csv_exists:
        writer.writerow(list(model_data.columns.values) + 
                        ["IG_degree_pearson_corr_avg_class_0", "IG_degree_pearson_corr_avg_class_1", 
                         "IG_degree_spearman_corr_avg_class_0", "IG_degree_spearman_corr_avg_class_1",
                         "IG_cross_entropy_avg_class_0", "IG_cross_entropy_avg_class_1",
                         "SHAP_cross_entropy_avg_class_0", "SHAP_cross_entropy_avg_class_1",
                         "GNNExplainer_cross_entropy_avg_class_0", "GNNExplainer_cross_entropy_avg_class_1",
                         "IG_auc_avg_class_0", "IG_auc_avg_class_1",
                         "SHAP_auc_avg_class_0", "SHAP_auc_avg_class_1",
                         "GNNExplainer_auc_avg_class_0", "GNNExplainer_auc_avg_class_1"])

print(f'Writing results to {out_csv_file}')

def calculate_attributions(model, input_data, mask, class_label):
    """
    Helper function to calculate IG, KernelSHAP, and GNNExplainer attributions.
    Returns node attributions for each method.
    """

    # calculate IG attributions
    ig = IntegratedGradients(model_forward)
    ig_attr = ig.attribute(mask, target=class_label, additional_forward_args=(input_data,), n_steps=100)[0]

    # calcualte KernelSHAP attributions
    shapley_value = KernelShap(kernelSHAP_model_forward)
    shap_attr = shapley_value.attribute(mask, target=class_label, additional_forward_args=(input_data,))[0]

    # calculate max absolute attribution across all features for each node
    ig_node_attr = torch.sum(torch.abs(ig_attr), axis=1).detach().cpu().numpy()
    shap_node_attr = torch.sum(torch.abs(shap_attr), axis=1).detach().cpu().numpy()

    # Create an instance of GNNExplainer, passing the model
    gnnExplainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200, lr=1e-3),
        explanation_type='phenomenon',
        node_mask_type='object',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='log_probs',
        ),
    )

    node_index = 0
    gnn_explanation = gnnExplainer(x=input_data.x, 
                            edge_index=input_data.edge_index, 
                            batch=input_data.batch, 
                            target=torch.tensor(class_label, dtype=torch.long).unsqueeze(0),
                            index=None)

    # gnnexplainer_node_attr = gnn_explanation.node_mask.detach().cpu().numpy()
    gnnexplainer_node_attr = torch.sum(torch.abs(gnn_explanation.node_mask), axis=1).detach().cpu().numpy()

    return ig_node_attr, shap_node_attr, gnnexplainer_node_attr
    
for index, row in model_data.iterrows():
    print(f"Testing #{index+1}/{len(model_data)}: {row['conv_type']} with layers={row['num_layers']}, dropout={row['dropout']}, and l2={row['l2']}")
    model_path = f'{saved_dir}/graphgen_model_{row["conv_type"]}_{row["num_layers"]}_{row["dropout"]}_{row["l2"]}.pt'
    conv_types = {'GraphConv': GraphConv, 'GCNConv': GCNConv, 'ChebConv': ChebConv}
    model = GraphConvResidualNet(dim=32,
                                num_features=num_features,
                                num_classes=num_classes,
                                num_layers=row["num_layers"],
                                conv_type=conv_types[row["conv_type"]],
                                dropout=row['dropout']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()

    ig_corr_results = {}
    ig_cross_entropy_results = {}
    shap_cross_entropy_results = {}
    gnnexplainer_cross_entropy_results = {}
    ig_auc_results = {}
    shap_auc_results = {}
    gnnexplainer_auc_results = {}
    # For each class
    for class_label in tqdm([0, 1]):
        # For each graph in the test set
        ig_pearson_corrs = []
        ig_spearman_corrs = []
        ig_cross_entropy_scores = []
        shap_cross_entropy_scores = []
        gnnexplainer_cross_entropy_scores = []
        ig_auc_scores = []
        shap_auc_scores = []
        gnnexplainer_auc_scores = []
        for input_data in test_dataset:
            input_data = input_data.to(device)

            # Check if the class label matches the current iteration
            if input_data.y.item() == class_label:
                # Generate node-level attributions for the corresponding class
                mask = torch.stack([torch.ones_like(input_data.x).requires_grad_(True).to(device)] * 1)
                # calculate attributions
                ig_node_attr, shap_node_attr, gnnexplainer_node_attr = calculate_attributions(model, input_data, mask, class_label)
                # Calculate node degree
                node_degree = input_data.edge_index[1].bincount(minlength=input_data.num_nodes)
                # Calculate correlation (Pearson and Spearman) between node importance and node degree
                ig_pearson_corr, _ = pearsonr(ig_node_attr, node_degree.cpu().detach().numpy())
                ig_spearman_corr, _ = spearmanr(ig_node_attr, node_degree.cpu().detach().numpy())
                ig_pearson_corrs.append(ig_pearson_corr)
                ig_spearman_corrs.append(ig_spearman_corr)

                # calculate weighted cross-entropy on attribution scores
                ground_truth = torch.zeros_like(input_data.x, dtype=torch.float64)
                for motif in input_data.motifs:
                    ground_truth[motif] = 1
                ground_truth = ground_truth.to(device)
                ground_truth = torch.max(torch.abs(ground_truth), axis=1).values
                wt = torch.unique(ground_truth, return_counts=True)[1]
                wt  = 1 - (wt / torch.sum(wt))
                wtts = torch.ones_like(ground_truth)
                # sigmoid of attributions
                ig_sigmoid_scores = torch.sigmoid(torch.tensor(ig_node_attr, dtype=torch.float64))
                shap_sigmoid_scores = torch.sigmoid(torch.tensor(shap_node_attr, dtype=torch.float64))
                gnnexplainer_sigmoid_scores = torch.sigmoid(torch.tensor(gnnexplainer_node_attr, dtype=torch.float64))
                # weighted cross-entropy
                ig_cross_entropy = F.binary_cross_entropy(ig_sigmoid_scores, ground_truth, weight=wtts)
                shap_cross_entropy = F.binary_cross_entropy(shap_sigmoid_scores, ground_truth, weight=wtts)
                gnnexplainer_cross_entropy = F.binary_cross_entropy(gnnexplainer_sigmoid_scores, ground_truth, weight=wtts)

                ig_cross_entropy_scores.append(ig_cross_entropy.item())
                shap_cross_entropy_scores.append(shap_cross_entropy.item())
                gnnexplainer_cross_entropy_scores.append(gnnexplainer_cross_entropy.item())

                # log AUROC
                ig_auc = roc_auc_score(ground_truth.detach().cpu().numpy(), ig_sigmoid_scores.detach().cpu().numpy())
                shap_auc = roc_auc_score(ground_truth.detach().cpu().numpy(), shap_sigmoid_scores.detach().cpu().numpy())
                gnnexplainer_auc = roc_auc_score(ground_truth.detach().cpu().numpy(), gnnexplainer_sigmoid_scores.detach().cpu().numpy())
                ig_auc_scores.append(ig_auc)
                shap_auc_scores.append(shap_auc)
                gnnexplainer_auc_scores.append(gnnexplainer_auc)

        # Calculate average correlation across all graphs for the current class
        ig_corr_results[f"IG_degree_pearson_corr_avg_class_{class_label}"] = np.mean(ig_pearson_corrs)
        ig_corr_results[f"IG_degree_spearman_corr_avg_class_{class_label}"] = np.mean(ig_spearman_corrs)
        # Calculate average cross-entropy across all graphs for the current class
        ig_cross_entropy_results[f"IG_cross_entropy_avg_class_{class_label}"] = np.mean(ig_cross_entropy_scores)
        shap_cross_entropy_results[f"SHAP_cross_entropy_avg_class_{class_label}"] = np.mean(shap_cross_entropy_scores)
        gnnexplainer_cross_entropy_results[f"GNNExplainer_cross_entropy_avg_class_{class_label}"] = np.mean(gnnexplainer_cross_entropy_scores)
        # Calculate average AUROC across all graphs for the current class
        ig_auc_results[f"IG_auc_avg_class_{class_label}"] = np.mean(ig_auc_scores)
        shap_auc_results[f"SHAP_auc_avg_class_{class_label}"] = np.mean(shap_auc_scores)
        gnnexplainer_auc_results[f"GNNExplainer_auc_avg_class_{class_label}"] = np.mean(gnnexplainer_auc_scores)

    # Append results to CSV
    with open(out_csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(list(row.values) +  
                         [ig_corr_results[f"IG_degree_pearson_corr_avg_class_0"], ig_corr_results[f"IG_degree_pearson_corr_avg_class_1"],
                         ig_corr_results[f"IG_degree_spearman_corr_avg_class_0"], ig_corr_results[f"IG_degree_spearman_corr_avg_class_1"],
                         ig_cross_entropy_results[f"IG_cross_entropy_avg_class_0"], ig_cross_entropy_results[f"IG_cross_entropy_avg_class_1"],
                         shap_cross_entropy_results[f"SHAP_cross_entropy_avg_class_0"], shap_cross_entropy_results[f"SHAP_cross_entropy_avg_class_1"],
                         gnnexplainer_cross_entropy_results[f"GNNExplainer_cross_entropy_avg_class_0"], gnnexplainer_cross_entropy_results[f"GNNExplainer_cross_entropy_avg_class_1"],
                         ig_auc_results[f"IG_auc_avg_class_0"], ig_auc_results[f"IG_auc_avg_class_1"],
                         shap_auc_results[f"SHAP_auc_avg_class_0"], shap_auc_results[f"SHAP_auc_avg_class_1"],
                         gnnexplainer_auc_results[f"GNNExplainer_auc_avg_class_0"], gnnexplainer_auc_results[f"GNNExplainer_auc_avg_class_1"]])

