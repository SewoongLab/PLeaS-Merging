"""
Utilities for loading and processing models.
"""

import torch
from pleas.datasets.model_loader import load_model

def load_pretrained_model(model_type, variant, dataset=None):
    """
    Load a pretrained model.
    
    Args:
        model_type (str): Type of model (e.g., 'rn18', 'rn50', 'rn101')
        variant (str): Variant of the model (e.g., 'v1a', 'v2b')
        dataset (str, optional): Dataset the model was trained on
        
    Returns:
        torch.nn.Module: Loaded model
    """
    model = load_model(model_type, variant)
    
    if dataset:
        # Load dataset-specific weights if provided
        model_path = f"/path/to/models/{model_type}_{variant}_{dataset}.pth"
        try:
            model.load_state_dict(torch.load(model_path))
        except FileNotFoundError:
            print(f"Warning: Model weights not found at {model_path}")
    
    return model
