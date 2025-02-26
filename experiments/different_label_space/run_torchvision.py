#!/usr/bin/env python
"""
Script to run PLeaS merging experiments on TorchVision datasets.

This script performs model merging experiments between models trained on
different TorchVision datasets using ResNet-18 models.
"""

import argparse
import os
import torch
import torchvision
import torchmetrics
import wandb
import gc
import numpy as np
from copy import deepcopy, copy

from pleas.core.utils import Axis, count_linear_flops
from pleas.core.compiler import get_permutation_spec
from pleas.methods.activation_matching import activation_matching
from pleas.methods.weight_matching import weight_matching
from pleas.methods.partial_matching import partial_merge, qp_ratios, get_blocks
from pleas.methods.pleas_merging import train, train_eval_linear_probe
from experiments.utils.data_utils import get_train_test_loaders

# Configuration for experiments
MODEL_PATH = os.environ.get('MODEL_PATH', '/path/to/models')


def get_zip_ratios(initial_ratios, budget_ratio):
    """
    Create layer-wise ratios based on the budget ratio using a zipping strategy.
    
    Args:
        initial_ratios (dict): Initial ratios dictionary
        budget_ratio (float): Target budget ratio
        
    Returns:
        dict: Updated ratios dictionary
    """
    layer_dict = {1.0: 4, 1.24: 3, 1.46: 2, 1.71: 1, 2.0: 0}
    new_ratios = {}
    for k, v in initial_ratios.items():
        if k.startswith('layer'):
            layernum = int(k.split('.')[0].split('layer')[1])
            print(k, layernum)
            if layernum <= layer_dict[budget_ratio]:
                new_ratios[k] = 0.
            else:
                new_ratios[k] = 1.
        else:
            new_ratios[k] = 0.
    return new_ratios


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run TorchVision experiments")
    
    parser.add_argument("--dataset1", type=str, required=True, 
                        choices=["cub", "nabird", "oxford_pets", "stanford_dogs"],
                        help="First dataset")
    parser.add_argument("--dataset2", type=str, required=True, 
                        choices=["cub", "nabird", "oxford_pets", "stanford_dogs"],
                        help="Second dataset")
    parser.add_argument("--model_type", type=str, default="rn18",
                        choices=["rn18", "rn50"],
                        help="Model architecture")
    parser.add_argument("--variant1", type=str, default="v1a",
                        help="Variant of first model")
    parser.add_argument("--variant2", type=str, default="v1b",
                        help="Variant of second model")
    
    parser.add_argument("--merging", type=str, default="pleas",
                        choices=["pleas", "weight_matching", "perm_gradmask", 
                                "mean", "mean_eval_only", "perm_eval_only"],
                        help="Merging strategy")
    parser.add_argument("--budget_ratio", type=float, default=1.46,
                        choices=[1.0, 1.24, 1.46, 1.71, 2.0],
                        help="Budget ratio for partial merging")
    
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--max_steps", type=int, default=400,
                        help="Maximum number of training steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to save outputs")
    
    return parser.parse_args()


def main():
    """Main experiment function."""
    args = parse_args()
    torch.manual_seed(args.seed)
    
    # Set up logging
    if args.wandb:
        wandb_run = wandb.init(
            project=f"perm_gd_tv_rn_18_zipsched_seed_{args.seed}", 
            config={
                "budget_ratio": args.budget_ratio, 
                'merge type': f"{args.merging}",
                'd1': args.dataset1, 
                'd2': args.dataset2, 
                'pretrained': args.variant1, 
                'grb_data': "orig-bn-orig-linearprobe"
            }
        )
    else:
        wandb_run = None
    
    # Load datasets
    print(f"Loading datasets: {args.dataset1} and {args.dataset2}...")
    dataset_1_train, dataset_1_test = get_train_test_loaders(
        args.dataset1, 16, True, directory_suffix=str(args.seed))

    dataset_2_train, dataset_2_test = get_train_test_loaders(
        args.dataset2, 16, True, directory_suffix=str(args.seed))
    
    # Get number of classes for each dataset
    try:
        num_classes_1 = len(dataset_1_train.dataset.classes)
    except AttributeError:
        num_classes_1 = np.max(dataset_1_train.dataset.targets) + 1

    try:
        num_classes_2 = len(dataset_2_train.dataset.classes)
    except AttributeError:
        num_classes_2 = np.max(dataset_2_train.dataset.targets) + 1
    
    # Create combined train loader for training the merged model
    bn_train_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset([
            dataset_1_train.dataset, dataset_2_train.dataset
        ]),
        batch_size=32, shuffle=True, num_workers=2
    )
    
    train_loader = bn_train_loader
    
    # Load models
    print("Loading models...")
    model1 = torchvision.models.resnet18().cuda()
    model2 = torchvision.models.resnet18().cuda()
    
    model1.fc = torch.nn.Linear(512, num_classes_1).cuda()
    model2.fc = torch.nn.Linear(512, num_classes_2).cuda()
    
    # Determine the variant to use for the second model (opposite of first)
    non_pt = 'v1b' if args.variant1 == 'v1a' else 'v1a'
    
    # Load model checkpoints
    print("Loading model checkpoints...")
    try:
        model_dict_1 = torch.load(
            f"{MODEL_PATH}/{args.model_type}/{args.dataset1}_{args.variant1}.pth")
        model_dict_2 = torch.load(
            f"{MODEL_PATH}/{args.model_type}/{args.dataset2}_{non_pt}.pth")
        
        model1.load_state_dict(model_dict_1['model'])
        model2.load_state_dict(model_dict_2['model'])
    except:
        print("Error loading model checkpoints. Please check paths.")
        return
    
    # Save the classifiers and replace them with identity
    fc = [model1.fc, model2.fc]
    model1.fc = torch.nn.Identity()
    model2.fc = torch.nn.Identity()
    
    # Create permutation specification
    spec = get_permutation_spec(model1, ((1, 3, 224, 224),), verbose=False)
    
    # Perform permutation matching
    print("Performing permutation matching...")
    if 'weight' in args.merging:
        perm, costs = weight_matching(
            spec=spec, 
            state_as=model1.state_dict(), 
            state_bs=model2.state_dict(), 
            max_iter=100, 
            verbose=False, 
            seed=args.seed, 
            return_costs=True
        )
    else:
        perm, costs = activation_matching(
            spec,
            model1,
            model2,
            train_loader,
            30,  # Use a smaller number for testing, increase for real experiments
            output_costs=True,
        )
    
    # Get budget ratios
    orig_ratios = {k: 0.0 for k in spec.keys()}
    budget_ratios = get_zip_ratios(orig_ratios, args.budget_ratio)
    budget_ratios = {Axis(k, 0): v for k, v in budget_ratios.items()}
    
    # Create merged model
    print("Creating merged model...")
    if 'mean' in args.merging:
        model3 = deepcopy(model1)
    else:
        model3 = partial_merge(spec, model1, model2, perm, costs, budget_ratios)
    model3.cuda()
    
    # First evaluation (before PLeaS optimization)
    print("Initial evaluation...")
    for bidx, batch in enumerate(train_loader):
        model3(batch[0].cuda())
        if bidx > 100:
            break
    
    # Optimize with PLeaS if selected
    if 'eval_only' not in args.merging:
        print("Training with PLeaS...")
        model3 = train(
            train_loader, 
            model1, 
            model2, 
            model3, 
            spec, 
            perm, 
            costs, 
            budget_ratios, 
            args.wandb, 
            args.max_steps, 
            wandb_run, 
            merging=args.merging if 'perm' in args.merging else 'perm_gradmask',
            rn_18=True
        )
    
    # Reset batch norm statistics
    model3.train()
    for bidx, batch in enumerate(train_loader):
        with torch.no_grad():
            model3(batch[0].cuda())
        if bidx > 100:
            break
    
    # Evaluate with linear probes
    print("Evaluating with linear probes...")
    lp_train_loaders = [dataset_1_train, dataset_2_train]
    for didx, test_loader in enumerate([dataset_1_test, dataset_2_test]):
        train_eval_linear_probe(
            model3, 
            lp_train_loaders[didx], 
            test_loader, 
            fc[didx].out_features, 
            wandb_run, 
            [args.dataset1, args.dataset2][didx], 
            lr=1e-3, 
            epochs=50
        )
    
    # Save the merged model
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(
        model3.state_dict(), 
        f"{args.output_dir}/{args.dataset1}_{args.dataset2}_{args.model_type}_{args.merging}_{args.budget_ratio}.pth"
    )
    
    print("Experiment completed!")


if __name__ == "__main__":
    main()