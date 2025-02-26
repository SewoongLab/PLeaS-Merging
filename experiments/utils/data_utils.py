"""
Utilities for loading and processing datasets.
"""

import torch
from torchvision import transforms

def get_train_test_loaders(dataset_name, batch_size=32, shuffle=True, directory_suffix=""):
    """
    Get train and test data loaders for a dataset.
    
    Args:
        dataset_name (str): Name of the dataset
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the data
        directory_suffix (str): Optional suffix for the dataset directory
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Add your implementation here
    # This is a placeholder that you'll need to adapt based on your existing code
    
    # Example implementation:
    from pleas.datasets.loaders import get_dataset_loaders
    return get_dataset_loaders(dataset_name, batch_size, shuffle, directory_suffix)
