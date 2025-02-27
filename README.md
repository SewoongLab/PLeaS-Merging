# PLeaS: Merging Models with Permutations and Least Squares

This repository contains the implementation for our CVPR'25 paper [PLeaS - Merging Models with Permutations and Least Squares](https://arxiv.org/abs/2407.02447), which introduces a novel algorithm for merging of neural networks.

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

# Create merging ratios
# A ratio of 0.0 indicates complete merging, while 1.0 indicates no merging
budget_ratios = {key: 0. for key in spec.keys()}  
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

## Running Experiments

### Downloading Checkpoints

To download model checkpoints from our S3 bucket (`s3://pleas-merging-artiacts-sewoonglab`):

```bash
# Change to the scripts directory
cd experiments/scripts

# Default directory (./pretrained_models/)
python download_checkpoints.py

# Custom directory
python download_checkpoints.py --output-dir /path/to/models
```

Or use the shell wrapper:

```bash
# Change to the scripts directory
cd experiments/scripts

# Make the shell script executable
chmod +x download_checkpoints.sh

# Default directory (./pretrained_models/)
./download_checkpoints.sh

# Custom directory
./download_checkpoints.sh /path/to/models
```

The script will:
1. Download model checkpoints from `s3://pleas-merging-artiacts-sewoonglab`
2. Store them in the specified directory (defaults to `./pretrained_models/`)
3. Set the `MODEL_CHECKPOINTS_DIR` environment variable for the current session

### Requirements
- AWS CLI must be installed and in your PATH
- Python 3.x

### Environment Variable

To permanently set the `MODEL_CHECKPOINTS_DIR` environment variable:

**Linux/Mac:**
```bash
echo 'export MODEL_CHECKPOINTS_DIR="/absolute/path/to/models"' >> ~/.bashrc
source ~/.bashrc
```

**Windows:**
```bash
setx MODEL_CHECKPOINTS_DIR "C:\path\to\models"
```

This environment variable can be used in your code to reference the models directory.

## Experiment Launcher Scripts

The repository includes scripts to automate the launching of experiments with various configurations. These scripts are located in the `experiments/scripts` directory:

### Different Label Space Experiments

To launch a series of experiments for models trained on different label spaces:

```bash
# Make the script executable
chmod +x ./scripts/launch_different_label_space_experiments.sh


# Run with default settings (rn50 model)
./scripts/launch_different_label_space_experiments.sh

# Run with custom settings
./scripts/launch_different_label_space_experiments.sh \
  --data_dir "/path/to/data" \
  --model_type "rn18" \
  --merging "pleas_activation" \
  --max_steps 800 \
  --seed 100 \
  --output_dir "./custom_outputs" \
  --use_zip_ratios
```

This script will run experiments with the following configurations:
- **Dataset pairs**: 
  - cub and oxford_pets
  - cub and stanford_dogs
  - oxford_pets and nabird
  - nabird and stanford_dogs
  - cub and nabird
  - stanford_dogs and oxford_pets
- **Variant pairs**: (v1a,v1b) and (v1b,v1a)
- **Budget ratios** (for rn50): 1.0, 1.2, 1.55, 1.8, 2.0

### Shared Label Space Experiments

To launch a series of experiments for models trained on shared label spaces:

```bash
# Make the script executable
chmod +x ./scripts/launch_shared_label_space_experiments.sh

# Run with default settings (rn50 model)
./scripts/launch_shared_label_space_experiments.sh

# Run with custom settings
./scripts/launch_shared_label_space_experiments.sh \
  --model_type "rn101" \
  --merging "weight_matching" \
  --use_zip_ratios
```

This script will run experiments with the following configurations:
- **Domain pairs**: 
  - clipart and painting
  - clipart and infograph
  - clipart and real
  - painting and infograph
  - painting and real
  - infograph and real
- **Variant pairs**: (v1a,v1b) and (v1b,v1a)
- **Budget ratios** depend on the model type:
  - rn18: 1.0, 1.24, 1.46, 1.71, 2.0
  - rn50: 1.0, 1.2, 1.55, 1.8, 2.0 (default)
  - rn101: 1.0, 1.1, 1.7, 1.8, 2.0

### Features of Both Scripts

- Logs all experiment commands and outputs to a timestamped log file
- Creates separate output directories for each experiment
- Enables Weights & Biases logging
- Supports customization via command-line options
- Shows experiment progress and counting


## Citation

If you use this code in your research, please cite our paper:

```
@misc{nasery2024pleasmergingmodels,
      title={PLeaS -- Merging Models with Permutations and Least Squares}, 
      author={Anshul Nasery and Jonathan Hayase and Pang Wei Koh and Sewoong Oh},
      year={2024},
      eprint={2407.02447},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.02447}, 
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work builds upon the Git Re-Basin approach introduced by [Ainsworth et al.](https://arxiv.org/abs/2209.04836).