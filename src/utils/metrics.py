import torch
from sklearn.metrics import roc_auc_score, roc_curve

def accuracy(outputs, labels):
    """
    Calculate accuracy for the given outputs and labels.
    """
    with torch.no_grad():
        predicted = torch.round(torch.sigmoid(outputs))
        correct = predicted == labels
        acc = torch.mean(correct.float())
    return acc.item()

def auroc(outputs, labels):
    """
    Calculate AUROC for the given outputs and labels.
    """
    with torch.no_grad():
        predicted_probs = torch.sigmoid(outputs).cpu().numpy()
        labels_np = labels.cpu().numpy()
        auroc = roc_auc_score(labels_np, predicted_probs)
    return auroc