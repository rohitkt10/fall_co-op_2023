import sys
sys.path.append("..")

from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import Counter
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
dataset = SyntheticGraphDatasetPyG('/Users/kumarh/Documents/fall_co-op_2023/experiments/graphGenDataset8000.pkl')

# Split the dataset into training, validation, and testing subsets
num_samples = len(dataset)
num_train = int(0.7 * num_samples)
num_val = int(0.2 * num_samples)
num_test = num_samples - num_train - num_val

train_dataset, val_dataset, test_dataset = random_split(dataset, [num_train, num_val, num_test])

saved_dir = 'syndata_exp1/'

# save dataset
torch.save(train_dataset, f'{saved_dir}/train_dataset.pt')
torch.save(val_dataset, f'{saved_dir}/val_dataset.pt')
torch.save(test_dataset, f'{saved_dir}/test_dataset.pt')

#load dataset
# train_dataset = torch.load(f'{saved_dir}/train_dataset.pt')
# val_dataset = torch.load(f'{saved_dir}/val_dataset.pt')
# test_dataset = torch.load(f'{saved_dir}/test_dataset.pt')

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
dropouts = [0, 0.1, 0.4, 0.5, 0.6, 0.8]
l2_values = [0.00001, 0.0001, 0.001, 0.01, 0.1]
epochs = 50

# Initialize dictionary to store results
results = {"num_layers": [], "conv_type": [], "dropout": [], "l2": [], "best_model_epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "test_acc": []}

best_model_name = ''
total_models = len(num_layers) * len(conv_types) * len(dropouts) * len(l2_values)
model_count = 0

# Iterate over hyperparameters and train model
for n_layer in num_layers:
    for conv_type in conv_types:
        for dropout in dropouts:
            for l2 in l2_values:
                model_count += 1
                print(f"Training #{model_count}/{total_models} model with {n_layer} {conv_type.__name__} convolution, dropout={dropout}, and l2={l2}")

                # Initialize model with current hyperparameters
                model = GraphConvResidualNet(dim=32, 
                                            num_features=dataset.num_features, 
                                            num_classes=dataset.num_classes, 
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
                        output = model(data.x, data.edge_index, data.batch)
                        loss = criterion(output, data.y)
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
                        best_model_name = f'{saved_dir}/graphgen_model_{n_layer}_{conv_type.__name__}_{dropout}_{l2}.pt'
                        best_model_epoch = epoch
                        torch.save(model.state_dict(), best_model_name)

                    # Store metrics for current epoch
                    metrics["train_loss"].append(train_loss)
                    metrics["train_acc"].append(train_accuracy)
                    metrics["val_loss"].append(val_loss)
                    metrics["val_acc"].append(val_accuracy)

                # save metrics to csv
                metrics_df = pd.DataFrame(metrics)
                metrics_df.to_csv(f'{saved_dir}/graphgen_metrics_{n_layer}_{conv_type.__name__}_{dropout}_{l2}.csv', index=False)

                # Store results for current hyperparameters
                results["conv_type"].append(conv_type.__name__)
                results["dropout"].append(dropout)
                results["l2"].append(l2)
                results["best_model_epoch"].append(best_model_epoch)
                results["train_loss"].append(metrics["train_loss"][-1])
                results["train_acc"].append(metrics["train_acc"][-1])
                results["val_loss"].append(metrics["val_loss"][-1])
                results["val_acc"].append(metrics["val_acc"][-1])

                # Load the best model for evaluation
                model.load_state_dict(torch.load(best_model_name))

                # ... (Use the model for evaluation)
                def test(loader):
                    model.eval()
                    correct = 0
                    for data in loader:
                        data = data.to(device)
                        output = model(data.x, data.edge_index, data.batch)
                        pred = output.max(dim=1)[1]
                        correct += pred.eq(data.y).sum().item()
                    return correct / len(loader.dataset)

                test_acc = test(test_loader)
                results["test_acc"].append(test_acc)

                print(f'Results: train_loss={results["train_loss"][-1]}, train_acc={results["train_acc"][-1]}, val_loss={results["val_loss"][-1]}, val_acc={results["val_acc"][-1]}, test_acc={results["test_acc"][-1]}')

# Convert results to pandas DataFrame and display as table
results_df = pd.DataFrame(results)
print(results_df)

# save results to csv
results_df.to_csv(f'{saved_dir}/graphgenexp_results.csv', index=False)
