import os
import random
from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader
# import torchvision.models as models

import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import wandb
# import argparse
from tqdm import tqdm
from torchvision import transforms


class CustomImageFolder(ImageFolder):
    def __init__(self, root, text_file, transform=None, target_transform=None, split='train', split_ratio=0.8, inmemory=False):
        random.seed(0)
        super(CustomImageFolder, self).__init__(
            root, transform=transform, target_transform=target_transform)
        # self.valid_extensions = ImageFolder._find_classes(self)[1]

        # Read the text file and filter images
        with open(text_file, 'r') as f:
            all_paths = (
                [f"{root}{'/'.join(x.split(' ')[0].split('/')[1:])}" for x in f.read().splitlines()])

        random.shuffle(all_paths)
        split_index = int(len(all_paths) * split_ratio)

        if split == 'train':
            selected_paths = all_paths[:split_index]
        else:  # 'val'
            selected_paths = all_paths[split_index:]

        self.inmemory = inmemory
        selected_paths_set = set(selected_paths)
        self.samples = [s for s in self.samples if s[0] in selected_paths_set]
        if inmemory:
            self.imgs = [self.loader(x[0]) for x in self.samples]
        else:
            self.imgs = self.samples
            
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        s1, target = self.samples[index]
        # image =
        if self.inmemory:
            sample = self.imgs[index]  # (path)
        else:
            sample = self.loader(s1)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def random_split(self, ratio, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.samples)
        split_idx = int(len(self.samples) * ratio)
        return self.samples[:split_idx], self.samples[split_idx:]
    
    
train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def prepare_train_loaders(config, ds_name):
    return {
        'full': torch.utils.data.DataLoader(
            CustomImageFolder(root=f'/scr/{ds_name}/', split='train', split_ratio=1., text_file=f'/scr/{ds_name}/train_images.txt',transform=train_transforms),
            batch_size=config['batch_size'],
            shuffle=config['shuffle_train'],
            num_workers=config['num_workers']
        )
            
    }

def prepare_test_loaders(config, ds_name):
    
    if config.get('val_fraction', 0) > 0.:
        loaders = {
            'full': torch.utils.data.DataLoader(
                CustomImageFolder(root=f'/scr/{ds_name}/', split='train', split_ratio=1., text_file=f'/scr/{ds_name}/test_images.txt',transform=test_transforms),
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=config['num_workers']
            ),
            'heldout_test': torch.utils.data.DataLoader(
                CustomImageFolder(root=f'/scr/{ds_name}/', split='train', split_ratio=1., text_file=f'/scr/{ds_name}/test_images.txt',transform=test_transforms),
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=config['num_workers']
            ),
            'heldout_val': torch.utils.data.DataLoader(
                CustomImageFolder(root=f'/scr/{ds_name}/', split='val', split_ratio=config['val_fraction'], text_file=f'/scr/{ds_name}/train_images.txt',transform=test_transforms),
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=config['num_workers']
            )                        
        }


    else:
        loaders = {
            'full': torch.utils.data.DataLoader(
                CustomImageFolder(root=f'/scr/{ds_name}/', split='train', split_ratio=1.0, text_file=f'/scr/{ds_name}/test_images.txt',transform=test_transforms),
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=config['num_workers']
            )
        }

    return loaders    