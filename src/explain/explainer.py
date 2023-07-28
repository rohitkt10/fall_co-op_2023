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
        y = self.model(x)  # shape (batch size, num classes)
        y = y[:, class_idx]  # shape (batch size,)
        grad = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(x))[0] # shape same as x 
        scores = x*grad 
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
        
        ## fill out the rest here
        return 