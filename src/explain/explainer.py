import numpy as np
import torch
from torch import nn 

class Explainer:
    """
    A basic explainer class.
    """
    def __init__(self, model, class_idx=-1, device="cpu"):
        model = model.to(device)
        model.eval()
        self.model = model
        self.class_idx = class_idx 
        self.device = device

    def saliency_map(self, x):
        """
        Arguments
        ---------
        x <torch.tensor> - Input tensors to explain. Expected shape - (batch_size, num_features, ...)
        
        Returns
        -------
        scores <torch.tensor> - The saliency scores for each feature in each sample; same shape as input. 
        """

        x.requires_grad = True
        y = self.model(x)  # shape (batch_size, num_classes)
        y = y[:, self.class_idx]  # shape (batch_size,)
        # Create a tensor with ones to match the shape of y for grad_outputs
        grad_outputs = torch.ones_like(y)
        grad = torch.autograd.grad(y, x, grad_outputs=grad_outputs)[0]  # shape same as x
        scores = x * grad
        return scores
    
    def smoothgrad(self, x, num_samples=50, noise_level=0.1):
        """
        Arguments
        ---------
        x <torch.tensor> - Input tensors to explain. Expected shape - (batch_size, num_features, ...)
        num_samples <int> - Number of samples for estimating the smoothgrad score. 
        
        Returns
        -------
        scores <torch.tensor> - The saliency scores for each feature in each sample; same shape as input. 
        """
        scores_list = []
        for _ in range(num_samples):
            # Generate noise and add it to the input
            noisy_x = x + torch.randn_like(x) * noise_level
            # Calculate saliency map for the noisy input
            saliency = self.saliency_map(noisy_x)
            scores_list.append(saliency)
        # Average the saliency maps
        averaged_scores = torch.stack(scores_list).mean(dim=0)
        return averaged_scores

    def integrated_gradients(self, x):
        """
        Arguments
        ---------
        x <torch.tensor> - Input tensors to explain. Expected shape - (batch_size, num_features, ...)
        baseline <torch.tensor> - Baseline input to start the integration path. If None, uses zeros as the baseline.
        num_steps <int> - Number of steps for approximating the integral.
        
        Returns
        -------
        scores <torch.tensor> - The integrated gradients for each feature in the input x.
        """
        baseline = None
        num_steps = 50
        if baseline is None:
            baseline = torch.zeros_like(x).to(self.device)
        else:
            baseline = baseline.to(self.device)

        # Calculate the step size for the integration path
        alpha = torch.linspace(0, 1, num_steps).view(-1, 1, 1, 1).to(self.device)

        # Create the integrated inputs along the path
        interpolated_inputs = baseline + alpha * (x - baseline)

        # Calculate the saliency maps for each interpolated input
        saliency_maps = self.saliency_map(interpolated_inputs)

        # Integrate the saliency maps along the path (using the trapezoidal rule)
        integrated_gradients = torch.mean(saliency_maps, dim=0) * (x - baseline)

        return integrated_gradients

