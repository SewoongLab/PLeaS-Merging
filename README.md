# PLeaS: Merging Models with Permutations and Least Squares

This repository contains the implementation for our paper "PLeaS - Merging Models with Permutations and Least Squares", which introduces a novel algorithm for parameter-efficient merging of neural networks.

## Overview

Neural networks are not uniquely identified due to weight-space symmetries. Two networks with the same architecture but different weights might represent the same function if their weights differ only by permutations. This repository provides:

1. **A modular implementation of Git Re-Basin** for finding permutations between neural networks using activation or weight matching
2. **An extension to partial model merging** using permutation matching and budget constraints
3. **Our novel PLeaS algorithm**, which optimizes merged models through least-squares training while respecting permutation structure
4. **Scripts to reproduce our experiments** on various datasets for ResNet models.

## Installation

```bash
# Clone the repository
git clone https://github.com/SewoongLab/PLeaS-Merging.git
cd PLeaS-Merging

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```
Data is at - `s3://pleas-merging-artiacts-sewoonglab`
## Key Components

### 1. Git Re-Basin Implementation

The core permutation-matching algorithms are implemented in:

- `pleas/core/compiler.py` - Analyzes model architectures and creates permutation specifications
- `pleas/methods/activation_matching.py` - Implements activation-based permutation matching
- `pleas/methods/weight_matching.py` - Implements direct weight-based permutation matching

### 2. Partial Merging

- `pleas/methods/partial_matching.py` - Extends Git Re-Basin to partial merging with budget constraints

### 3. PLeaS Algorithm

- `pleas/methods/pleas_merging.py` - Implements our novel PLeaS algorithm that combines permutation matching with least squares optimization

### 4. Experiment Reproduction

- `experiments/domainnet/run_domainnet.py` - Scripts for DomainNet experiments with ResNet-101
- `experiments/torchvision/run_torchvision.py` - Scripts for TorchVision dataset experiments with ResNet-18

## Basic Usage

### Permutation Matching and Merging

```python
import torch
from pleas.core.compiler import get_permutation_spec
from pleas.methods.activation_matching import activation_matching
from pleas.methods.partial_matching import partial_merge

# Load two pretrained models
model1 = torch.load("path/to/model1.pth")
model2 = torch.load("path/to/model2.pth")

# Create permutation specification
spec = get_permutation_spec(model1, ((1, 3, 224, 224),))

# Perform activation matching
dataloader = get_your_dataloader()
perm, costs = activation_matching(
    spec,
    model1,
    model2,
    dataloader,
    num_batches=100,
    output_costs=True,
)

# Partial merging with 50% computation budget
budget_ratios = {key: 0.5 for key in spec.keys()}  # 50% computation budget
merged_model = partial_merge(
    spec, model1, model2, perm, costs, budget_ratios
)
```

### PLeaS Optimization

```python
from pleas.methods.pleas_merging import train as pleas_train

# Optimize the merged model using PLeaS
optimized_model = pleas_train(
    dataloader,
    model1,
    model2,
    merged_model,
    spec,
    perm,
    costs,
    budget_ratios,
    WANDB=True,  # Set to True to log to Weights & Biases
    MAX_STEPS=400,
    wandb_run=wandb_run
)
```

### Evaluation with Linear Probes

```python
from pleas.methods.pleas_merging import train_eval_linear_probe

# Train and evaluate linear probes on top of the merged model
accuracy = train_eval_linear_probe(
    optimized_model,
    train_dataloader,
    test_dataloader,
    num_classes,
    wandb_run,
    dataset_name,
    lr=1e-3,
    epochs=10
)
```

## Running Experiments

### DomainNet Experiments

```bash
python -m experiments.domainnet.run_domainnet \
    --domain1 clipart \
    --domain2 real \
    --model_type rn101 \
    --variant1 v2a \
    --variant2 v2b \
    --merging pleas \
    --budget_ratio 1.5 \
    --wandb \
    --output_dir ./outputs
```

### TorchVision Experiments

```bash
python -m experiments.torchvision.run_torchvision \
    --dataset1 cub \
    --dataset2 oxford_pets \
    --model_type rn18 \
    --variant1 v1a \
    --variant2 v1b \
    --merging pleas \
    --budget_ratio 1.46 \
    --wandb \
    --output_dir ./outputs
```

## Experiment Analysis

We provide utilities for analyzing experiment results:

```python
from experiments.utils.analysis import (
    load_experiment_results,
    plot_accuracy_vs_budget,
    plot_method_comparison
)

# Load results
results = load_experiment_results("./outputs", "torchvision")

# Plot accuracy vs. budget ratio
fig = plot_accuracy_vs_budget(results)
fig.savefig("accuracy_vs_budget.png")

# Compare methods at a specific budget
fig = plot_method_comparison(results, budget_value=1.5)
fig.savefig("method_comparison.png")
```

## Citation

If you use this code in your research, please cite our paper:

```
@article{pleas2025,
  title={PLeaS - Merging Models with Permutations and Least Squares},
  author={},
  journal={},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work builds upon the Git Re-Basin approach introduced by [Ainsworth et al.](https://arxiv.org/abs/2209.04836).