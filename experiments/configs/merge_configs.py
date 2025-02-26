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
BUDGET_RATIOS = [1.0, 1.24, 1.46, 1.71, 2.0]

# Zipping strategy for progressive merging
def get_zip_ratios(initial_ratios, budget_ratio):
    """
    Create layer-wise ratios based on the budget ratio using a zipping strategy.
    """
    layer_dict = {1.0: 4, 1.24: 3, 1.46: 2, 1.71: 1, 2.0: 0}
    new_ratios = {}
    for k, v in initial_ratios.items():
        if k.startswith('layer'):
            layernum = int(k.split('.')[0].split('layer')[1])
            if layernum <= layer_dict[budget_ratio]:
                new_ratios[k] = 0.
            else:
                new_ratios[k] = 1.
        else:
            new_ratios[k] = 0.
    return new_ratios
