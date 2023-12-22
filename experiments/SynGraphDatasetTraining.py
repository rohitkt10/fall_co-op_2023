import sys
sys.path.append("..")

import os
import csv
import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from src.graphgen.synthetic_graph_dataset_PyG import *
from torch_geometric.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.nn import GCNConv, GraphConv, ChebConv
from src.models.graphconv_residualnet import GraphConvResidualNet


print("Loading dataset...")
# dataset = SyntheticGraphDatasetPyG('/Users/kumarh/Documents/fall_co-op_2023/experiments/graphGenDataset500.pkl')

# Split the dataset into training, validation, and testing subsets
# num_samples = len(dataset)
num_classes = 2
# num_train = int(0.7 * num_samples)
# num_val = int(0.2 * num_samples)
# num_test = num_samples - num_train - num_val

# train_dataset, val_dataset, test_dataset = random_split(dataset, [num_train, num_val, num_test])

# create folder based on timestamp
now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
saved_dir = f'syndata_exp_jacobian_{timestamp}/'
os.mkdir(saved_dir)

# # save dataset
# torch.save(train_dataset, f'{saved_dir}/train_dataset.pt')
# torch.save(val_dataset, f'{saved_dir}/val_dataset.pt')
# torch.save(test_dataset, f'{saved_dir}/test_dataset.pt')

#load dataset
train_dataset = torch.load('/Users/kumarh/Documents/fall_co-op_2023/experiments/syndata_exp_2023-12-07_00-00-16/train_dataset.pt')
val_dataset = torch.load('/Users/kumarh/Documents/fall_co-op_2023/experiments/syndata_exp_2023-12-07_00-00-16/val_dataset.pt')
test_dataset = torch.load('/Users/kumarh/Documents/fall_co-op_2023/experiments/syndata_exp_2023-12-07_00-00-16/test_dataset.pt')

# Create data loaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

print('Dataset splits:')
print(len(train_dataset), len(val_dataset), len(test_dataset))
print(f'Train: {len(train_dataset)}, Valid: {len(val_dataset)}, Test: {len(test_dataset)}')
# Check class-wise distribution of each set
for dname, dset in [('Train', train_dataset), ('Valid', val_dataset), ('Test', test_dataset)]:
    labels = [label[1].item() for _, _, label, _ in dset]
    print(f'{dname}:', Counter(labels))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

# Define hyperparameters to iterate over
num_layers = [2, 4]
conv_types = [GraphConv, GCNConv, ChebConv]
dropouts = [0.1, 0.3, 0.5]
l2_values = [0.00001, 0.001, 0.01]
epochs = 50

# Initialize the CSV file
csv_file = f'{saved_dir}/graphgenexp_results.csv'
csv_exists = os.path.exists(csv_file)

# If the CSV file doesn't exist, create it with header
with open(csv_file, 'a', newline='') as file:
    writer = csv.writer(file)
    if not csv_exists:
        writer.writerow(["conv_type", "num_layers", "dropout", "l2", "best_model_epoch", "train_loss", "train_acc", "val_loss", "val_acc", "test_acc", 'test_acc_cls0', 'test_acc_cls1'])

print(f'Writing overall training results to {csv_file}...')

def jacobian_regularization(logits, node_masks, num_proj=5, coeff=1e-4):
    B, C = logits.shape
    Jreg = 0.
    for z, x in zip(logits, node_masks):
        # Check if x requires gradient and is involved in logits computation
        if not x.requires_grad:
            continue

        V = torch.randn(num_proj, C, device=logits.device)  # Ensure V is on the same device as logits
        Vnorms = V.norm(p=2, dim=1, keepdim=True)
        Vnormed = V / Vnorms
        Jvs = [torch.autograd.grad(z, x, v, retain_graph=True, create_graph=True, allow_unused=True)[0] for v in Vnormed]

        # Check for None in gradients and handle appropriately
        J = [Jv.norm(2)**2 if Jv is not None else torch.tensor(0.0, device=logits.device) for Jv in Jvs]

        J = torch.stack(J).sum()
        jreg = 0.5 * (J / num_proj)
        Jreg += jreg

    return coeff * Jreg

best_model_name = ''
total_models = len(num_layers) * len(conv_types) * len(dropouts) * len(l2_values)
model_count = 0

# Iterate over hyperparameters and train model
for conv_type in conv_types:
    for n_layer in num_layers:
        for dropout in dropouts:
            for l2 in l2_values:
                model_count += 1
                print(f"Training #{model_count}/{total_models}: {conv_type.__name__} with layers={n_layer}, dropout={dropout}, and l2={l2}")

                # Initialize model with current hyperparameters
                model = GraphConvResidualNet(dim=32, 
                                            num_features=10, 
                                            num_classes=2, 
                                            num_layers=n_layer,
                                            conv_type=conv_type, 
                                            dropout=dropout).to(device)

                # Initialize optimizer and criterion
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=l2)
                criterion = torch.nn.NLLLoss()

                # Training loop with validation and model checkpointing
                best_val_accuracy = 0

                # Initialize metrics dictionary for current hyperparameters
                metrics = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

                # Train model for specified number of epochs
                for epoch in tqdm(range(1, epochs + 1)):
                    model.train()
                    train_loss = 0
                    train_correct = 0

                    if epoch == 51:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = 0.5 * param_group['lr']

                    for data in train_loader:
                        data = data.to(device)
                        optimizer.zero_grad()
                        # Jacobian Regularization
                        # set up node masks (indicates all nodes are active, multiplying by 1)
                        node_masks = [torch.ones(len(x)).requires_grad_(True) for x in data.x]
                        # mask the nodes in the sample graphs
                        masked_x = data.x * node_masks[0]  # Assuming one mask per graph in the batch
                        output = model(masked_x, data.edge_index, data.batch)
                        # output = model(data.x, data.edge_index, data.batch)
                        loss = criterion(output, data.y)
                        jreg = jacobian_regularization(output, node_masks, num_proj=5, coeff=1e-3)
                        total_loss = loss + jreg
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()
                        pred = output.argmax(dim=1)
                        train_correct += pred.eq(data.y).sum().item()

                    train_loss /= len(train_loader)
                    train_accuracy = train_correct / len(train_dataset)

                    model.eval()
                    val_loss = 0
                    val_correct = 0
                    for data in val_loader:
                        data = data.to(device)
                        output = model(data.x, data.edge_index, data.batch)
                        loss = criterion(output, data.y)
                        val_loss += loss.item()
                        pred = output.argmax(dim=1)
                        val_correct += pred.eq(data.y).sum().item()

                    val_loss /= len(val_loader)
                    val_accuracy = val_correct / len(val_dataset)

                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy
                        # Save the best model
                        best_model_name = f'{saved_dir}/graphgen_model_{conv_type.__name__}_{n_layer}_{dropout}_{l2}.pt'
                        best_model_epoch = epoch
                        torch.save(model.state_dict(), best_model_name)

                    # Store metrics for current epoch
                    metrics["train_loss"].append(train_loss)
                    metrics["train_acc"].append(train_accuracy)
                    metrics["val_loss"].append(val_loss)
                    metrics["val_acc"].append(val_accuracy)

                # save metrics to csv
                metrics_df = pd.DataFrame(metrics)
                metrics_df.to_csv(f'{saved_dir}/graphgen_metrics_{conv_type.__name__}_{n_layer}_{dropout}_{l2}.csv', index=False)

                # Load the best model for evaluation
                model.load_state_dict(torch.load(best_model_name))

                def test(loader):
                    model.eval()
                    class_correct = defaultdict(int) # default value of int is 0
                    class_total = defaultdict(int)
                    total_correct = 0
                    total = 0

                    for data in loader:
                        data = data.to(device)
                        output = model(data.x, data.edge_index, data.batch)
                        pred = output.max(dim=1)[1]
                        total_correct += pred.eq(data.y).sum().item()
                        total += data.y.size(0)

                        for label, prediction in zip(data.y, pred):
                            class_total[label.item()] += 1
                            if label == prediction:
                                class_correct[label.item()] += 1

                    overall_accuracy = total_correct / len(loader.dataset)
                    class_accuracy = {cls: class_correct[cls] / class_total[cls] for cls in class_correct}

                    return overall_accuracy, class_accuracy

                test_acc, test_class_acc = test(test_loader)

                # Append results to CSV
                with open(csv_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([conv_type.__name__, n_layer, dropout, l2, best_model_epoch, metrics["train_loss"][-1], metrics["train_acc"][-1], metrics["val_loss"][-1], metrics["val_acc"][-1], test_acc, test_class_acc[0], test_class_acc[1]])
