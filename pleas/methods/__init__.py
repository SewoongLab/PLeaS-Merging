"""
Model merging methods for neural network permutation and combination.

This module provides different approaches to merge neural networks:

1. Activation Matching: Find permutations that align activations
2. Weight Matching: Find permutations that directly align weights
3. Partial Matching: Merge models partially based on budget constraints
4. PLeaS Merging: Optimize weights after matching to improve performance
"""

from pleas.methods.activation_matching import (
    activation_matching,
    cross_features_cdist,
    cross_features_inner_product,
)

from pleas.methods.weight_matching import (
    weight_matching,
)

from pleas.methods.partial_matching import (
    partial_merge,
    get_blocks,
    qp_ratios,
    expand_ratios,
    partial_merge_flops,
)

from pleas.methods.pleas_merging import (
    train,
    train_eval_linear_probe,
    eval_perm_model,
    eval_whole_model,
    get_fc_perm,
)