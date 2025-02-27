#!/usr/bin/env python
"""
Script to run PLeaS merging experiments on DomainNet data.

This script performs model merging experiments between different domains
of the DomainNet dataset using ResNet-101 models.
"""

import argparse
import os
import torch
import torchvision
import torchmetrics
import wandb
import numpy as np

from copy import deepcopy, copy
from torchvision import transforms

from pleas.core.utils import Axis, count_linear_flops
from pleas.core.compiler import get_permutation_spec
from pleas.methods.activation_matching import activation_matching
from pleas.methods.weight_matching import weight_matching
from pleas.methods.partial_matching import partial_merge, qp_ratios
from pleas.methods.pleas_merging import train

from experiments.datasets.domainnet import CustomImageFolder
from experiments.configs.merge_configs import BUDGET_RATIOS

# Configuration for experiments
MODEL_PATH = os.environ.get('MODEL_PATH', '/gscratch/sewoong/anasery/rebasin_merging/PLeaS-Merging-Artifacts/models')
DOMAINNET_PATH = os.environ.get('DOMAINNET_PATH', '/scr/domainnet')

def get_zip_ratios(initial_ratios, budget_ratio, base_budget_ratios):
    """
    Create layer-wise ratios based on the budget ratio using a zipping strategy.
    
    Args:
        initial_ratios (dict): Initial ratios dictionary
        budget_ratio (float): Target budget ratio
        
    Returns:
        dict: Updated ratios dictionary
    """
    layer_dict = {base_budget_ratios[0]: 4, base_budget_ratios[1]: 3, base_budget_ratios[2]: 2, base_budget_ratios[3]: 1, base_budget_ratios[4]: 0}
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


def get_gradient_mask(perm_blocks, model_weights):
    """
    Generate gradient masks for PLeaS training.
    
    Args:
        perm_blocks (dict): Permutation blocks
        model_weights (dict): Model weights
        
    Returns:
        list: List of gradient masks
    """
    masks = []
    for layer_name, params in model_weights.items():
        try:
            input_perms = perm_blocks[Axis(f"{layer_name}.weight", 1)]
            bi1, bi2, bi1c, bi2c = input_perms
        except KeyError:
            bi1, bi2, bi1c, bi2c = torch.arange(3), torch.arange(3), [], []
        try:
            output_perms = perm_blocks[Axis(f"{layer_name}.weight", 0)]
            bo1, bo2, bo1c, bo2c = output_perms
        except KeyError:
            bo1, bo2, bo1c, bo2c = torch.arange(1000), torch.arange(1000), [], []

        ni, mi = len(bi1), len(bi1c)
        no, mo = len(bo1), len(bo1c)
        si12, si1, si2 = (
            slice(0, ni),
            slice(ni, ni + mi),
            slice(ni + mi, ni + 2 * mi),
        )
        so12, so1, so2 = (
            slice(0, no),
            slice(no, no + mo),
            slice(no + mo, no + 2 * mo),
        )
        
        for v in params.parameters():
            mask = torch.ones_like(v)

            if len(v.shape) >= 2:
                # These weights take input 1 to output 2 and vice-versa, so we make them non-trainable
                mask[si1,so2] = 0.
                mask[si2,so1] = 0.
            masks.append(mask)
    return masks


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run DomainNet experiments")
    
    parser.add_argument("--domain1", type=str, required=True, 
                        choices=["clipart", "painting", "infograph", "real"],
                        help="First domain")
    parser.add_argument("--domain2", type=str, required=True, 
                        choices=["clipart", "painting", "infograph", "real"],
                        help="Second domain")
    parser.add_argument("--model_type", type=str, default="rn50",
                        choices=["rn18", "rn50", "rn101"],
                        help="Model architecture")
    parser.add_argument("--variant1", type=str, default="v2a",
                        help="Variant of first model")
    parser.add_argument("--variant2", type=str, default="v2b",
                        help="Variant of second model")
    parser.add_argument("--data_dir", type=str, default="/scr/",)
    
    
    parser.add_argument("--merging", type=str, default="pleas_weight",
                        choices=["pleas_weight", "pleas_activation", "weight_matching", 
                                 "perm_activation"],)
    parser.add_argument("--budget_ratio", type=float, default=1.0,
                        help="Budget ratio for partial merging")
    parser.add_argument("--use_zip_ratios", action="store_true",
                        help="Use zipping strategy for ratios")
    
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--max_steps", type=int, default=400,
                        help="Maximum number of training steps")
    parser.add_argument("--seed", type=int, default=0,
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
            project=f"pleas_same_label_space", 
            config={
                "budget_ratio": args.budget_ratio,
                "m1": args.variant1,
                "m2": args.variant2,
                "d1": args.domain1,
                "d2": args.domain2,
                "merging": args.merging,
                'model': args.model_type,
            }
        )
    else:
        wandb_run = None
    
    # Load models
    print("Loading models...")
    if args.model_type == "rn18":
        model1 = torchvision.models.resnet18(pretrained=False).cuda()
        model1.fc = torch.nn.Linear(model1.fc.in_features, 345).cuda()
        model2 = torchvision.models.resnet18(pretrained=False).cuda()
        model2.fc = torch.nn.Linear(model1.fc.in_features, 345).cuda()
    elif args.model_type == "rn50":
        model1 = torchvision.models.resnet50(pretrained=False).cuda()
        model1.fc = torch.nn.Linear(model1.fc.in_features, 345).cuda()
        model2 = torchvision.models.resnet50(pretrained=False).cuda()
        model2.fc = torch.nn.Linear(model1.fc.in_features, 345).cuda()
    elif args.model_type == "rn101":  
        model1 = torchvision.models.resnet101(pretrained=False).cuda()
        model1.fc = torch.nn.Linear(model1.fc.in_features, 345).cuda()
        model2 = torchvision.models.resnet101(pretrained=False).cuda()
        model2.fc = torch.nn.Linear(model1.fc.in_features, 345).cuda()
    
    # Load model weights
    try:
        model1.load_state_dict(torch.load(
            f'{MODEL_PATH}/{args.model_type}/domainnet/{args.domain1}/{args.variant1}/model.pth'))
        model2.load_state_dict(torch.load(
            f'{MODEL_PATH}/{args.model_type}/domainnet/{args.domain2}/{args.variant2}/model.pth'))
    except:
        print("Error loading model weights. Please check paths.")
        return
    
    # Load datasets
    print("Loading datasets...")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Create train and test loaders
    train_loader1 = CustomImageFolder(
        f"{args.data_dir}/domainnet/{args.domain1}/", 
        f"{args.data_dir}/domainnet/train_images.txt", 
        inmemory=False, transform=train_transforms)
    train_loader2 = CustomImageFolder(
        f"{args.data_dir}/domainnet/{args.domain2}/", 
        f"{args.data_dir}/domainnet/{args.domain2}/train_images.txt", 
        inmemory=False, transform=train_transforms)
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset([train_loader1, train_loader2]), 
        batch_size=16, shuffle=True, num_workers=2)
    
    test_loaders = dict([
        (x, torch.utils.data.DataLoader(
            CustomImageFolder(
                f"{args.data_dir}/domainnet/{x}/", 
                f"{args.data_dir}/domainnet/{x}/test_images.txt", 
                split_ratio=1.0, inmemory=False, transform=val_transforms), 
            batch_size=256)) 
        for x in ["clipart", "painting", "infograph", "real"]
    ])
    
    # Create permutation specification
    spec = get_permutation_spec(model1, ((5, 3, 224, 224),))
    
    # Perform permutation matching
    print("Performing permutation matching...")
    if 'weight' in args.merging:
        perm_imnet, costs_imnet = weight_matching(
            spec=spec, 
            state_as=model1.state_dict(), 
            state_bs=model2.state_dict(), 
            max_iter=100, 
            verbose=True, 
            seed=0, 
            return_costs=True
        )
    else:
        perm_imnet, costs_imnet = activation_matching(
            spec,
            model1,
            model2,
            train_loader,
            100,
            output_costs=True,
        )
    
    # Count FLOPs and get terms
    if not args.use_zip_ratios:
        R0 = np.load(
            "/gscratch/sewoong/jhayase/oh/git-re-basin/git-re-basin-fx/archive/rn50-layerwise-0.npy")
        R1 = np.load(
            "/gscratch/sewoong/jhayase/oh/git-re-basin/git-re-basin-fx/archive/rn50-layerwise-1.npy")

        _, terms = count_linear_flops(spec, model1, ((1, 3, 224, 224),))

        
        obj_weights = dict(
            zip(spec, (R1 - 0.4684)*(2 - args.budget_ratio) - (R0 - 0.7743)*(args.budget_ratio - 1)))
        budget_ratios = qp_ratios(spec, terms, args.budget_ratio, obj_weights)
    else:
        orig_ratios = {k: 0.0 for k in spec.keys()}

        budget_ratios = get_zip_ratios(orig_ratios, args.budget_ratio, base_budget_ratios=BUDGET_RATIOS[args.model_type])
    
    # Process permutations based on merging strategy
    new_perm, new_costs = {}, {}
    if 'perm' in args.merging:
        new_perm, new_costs = perm_imnet, costs_imnet
    elif 'reg_mean' in args.merging or 'mean' in args.merging:
        for k, v in perm_imnet.items():
            new_perm[k] = torch.arange(len(v)).cuda()
            new_costs[k] = torch.rand(costs_imnet[k].size(), dtype=costs_imnet[k].dtype).cuda()
    else:
        new_costs = costs_imnet
        new_perm = perm_imnet
    
    # Create merged model
    print("Creating merged model...")
    if 'reg_mean' in args.merging:
        model3 = deepcopy(model1)
    else:                    
        model3 = partial_merge(spec, model1, model2, new_perm, new_costs, budget_ratios)
    model3.cuda()
    
    
    
    if 'pleas' in args.merging:
        # Train with PLeaS
        print("Training with PLeaS...")
        model3 = train(
            train_loader, 
            model1, 
            model2, 
            model3, 
            spec, 
            new_perm, 
            new_costs, 
            budget_ratios, 
            args.wandb, 
            args.max_steps, 
            wandb_run,
            separate_classifier=False,
            num_classes=345,
        )
    
    # Reset batch normalization statistics
    print("Resetting batch normalization statistics...")
    model3.train()
    for m in model3.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()
    
    bnidx = 0
    for batch in train_loader:
        with torch.no_grad():
            x = batch[0].cuda().float()
            _ = model3(x)
        bnidx += 1
        if bnidx > 100: 
            break
    
    # Evaluate the model
    print("Evaluating the model...")
    model3.eval()
    kaccs = {}
    for dset, test_loader in test_loaders.items():
        acc1 = torchmetrics.Accuracy(task="multiclass", num_classes=345).cuda()
        for images, labels in test_loader:
            with torch.no_grad():
                x = images.cuda()
                pred = model3(x)
                acc1(pred, labels.cuda())
        print(f"Dataset: {dset}, Accuracy: {acc1.compute().item():.4f}")
        kaccs[f"{dset}-accuracy"] = acc1.compute().item()
    
    kaccs["budget_ratio"] = args.budget_ratio
    if args.wandb:
        wandb_run.log(kaccs)
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(
        model3.state_dict(), 
        f"{args.output_dir}/{args.domain1}_{args.domain2}_{args.model_type}_{args.merging}_{args.budget_ratio}.pth"
    )
    
    print("Experiment completed!")


if __name__ == "__main__":
    main()