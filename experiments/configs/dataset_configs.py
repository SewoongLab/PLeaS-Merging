"""
Configuration for datasets used in the experiments.
"""

DATASETS = {
    # TorchVision datasets
    'cub': {
        'num_classes': 200,
        'image_size': 224,
    },
    'nabird': {
        'num_classes': 555,
        'image_size': 224,
    },
    'oxford_pets': {
        'num_classes': 37,
        'image_size': 224,
    },
    'stanford_dogs': {
        'num_classes': 120,
        'image_size': 224,
    },
    
    # DomainNet domains
    'clipart': {
        'num_classes': 345,
        'image_size': 224,
    },
    'infograph': {
        'num_classes': 345,
        'image_size': 224,
    },
    'painting': {
        'num_classes': 345,
        'image_size': 224,
    },
    'real': {
        'num_classes': 345,
        'image_size': 224,
    },
}
