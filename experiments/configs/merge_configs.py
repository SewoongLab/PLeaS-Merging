"""
Configuration for merging strategies used in the experiments.
"""

MERGE_STRATEGIES = {
    'weight_matching': {
        'description': 'Standard weight matching with no PLeaS optimization',
        'max_iter': 100,
    },
    'activation_matching': {
        'description': 'Activation matching with no PLeaS optimization',
        'num_batches': 100,
    },
    'pleas': {
        'description': 'PLeaS merging with gradient masking',
        'max_steps': 400,
        'lr': 5e-4,
    },
    'mean': {
        'description': 'Simple mean of model weights',
    },
}

# Budget ratios to experiment with
BUDGET_RATIOS = {'rn18': [1.0, 1.24, 1.46, 1.71, 2.0],
                 'rn50': [1.0, 1.2, 1.55, 1.8, 2.0],
                 'rn101': [1.0, 1.2, 1.55, 1.8, 2.0],}

