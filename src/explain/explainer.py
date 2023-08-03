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
    
    def smoothgrad(self, x, num_samples=50):
        """
        Arguments
        ---------
        x <torch.tensor> - Input tensors to explain. Expected shape - (batch_size, num_features, ...)
        num_samples <int> - Number of samples for estimating the smoothgrad score. 
        
        Returns
        -------
        scores <torch.tensor> - The saliency scores for each feature in each sample; same shape as input. 
        """
        if num_samples == 1:
            return self.saliency_map(x)

        # Initialize an array to store the perturbed samples
        perturbed_samples = torch.zeros((num_samples,) + x.shape).to(self.device)
        
        # Create num_samples perturbations with Gaussian noise
        for i in range(num_samples):
            noise = torch.randn_like(x).to(self.device)  # Gaussian noise with the same shape as x
            perturbed_samples[i] = x + noise
        
        # Calculate the saliency maps for the perturbed samples
        perturbed_saliency_maps = self.saliency_map(perturbed_samples)
        
        # Average the saliency maps over the num_samples
        scores = torch.mean(perturbed_saliency_maps, dim=0)
        return scores
        
        # # Step 1: Create a tensor with random noise (mean=0, std=1) of the same shape as the input
        # noise = torch.randn_like(x)

        # # Step 2: Expand the dimensions of the noise tensor to match the batch size of perturbed inputs
        # noise = noise.unsqueeze(1).repeat(1, num_samples, 1)  # Replace ... with the actual size along the sample dimension
        # # Step 3: Add the noise to the input tensor to generate a batch of perturbed inputs
        # perturbed_inputs = x + noise.expand(-1, num_samples, -1, ...)

        # # Step 4: Pass the batch of perturbed inputs through the model
        # # and collect the output predictions for each sample in the batch
        # model_outputs = self.model(perturbed_inputs)

        # # Step 5: Compute the mean of the saliency scores across the perturbed samples for each input feature
        # saliency_scores = torch.mean(self.saliency_map(perturbed_inputs), dim=0)

        # return saliency_scores

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

