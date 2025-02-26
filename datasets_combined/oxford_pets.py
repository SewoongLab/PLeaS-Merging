import os
import pdb
import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, download_and_extract_archive
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import datetime as dt
import xml.etree.ElementTree as ET
from PIL import Image
import scipy.io
# from utils import create_heldout_split


class OxfordPets(Dataset):
    """`Oxford Pets <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        download (bool, optional): If true, downloads the dataset tar files from the internet and
            puts it in root directory. If the tar files are already downloaded, they are not
            downloaded again.
    """
    folder = 'oxford_pets'
    _RESOURCES = (
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", "5c4f3ee8e5d25df40f4fd59a7f44e54c"),
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", "95a8c909bbe2e81eed6a22bccdf3f68f"),
    )
    _VALID_TARGET_TYPES = ("category", "segmentation")
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 loader=default_loader,
                 download=True,
                 base_set=None):
        
        self.root_og = root
        self.name = 'oxford_pets'
        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.train = train
        self.transform = transform
        self.loader = loader
        if download:
            self._download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")
        
        if base_set is not None:
            self.dataset = base_set
            self.targets = base_set['class_id'].unique()
        else:
            self._load_metadata()

    def __getitem__(self, idx):

        sample = self.dataset.iloc[idx]
        path = os.path.join(self.root, 'images', sample.img_id) + '.jpg'

        target = sample.class_id - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target#, idx
    
    def _load_metadata(self):
        if self.train:
            train_file = os.path.join(self.root, 'annotations', 'trainval.txt')
            self.dataset = pd.read_csv(train_file, sep=' ', names=['img_id', 'class_id', 'species', 'breed_id'])
        else:
            test_file = os.path.join(self.root, 'annotations', 'test.txt')
            self.dataset = pd.read_csv(test_file, sep=' ', names=['img_id', 'class_id', 'species', 'breed_id'])
        
        self.targets = self.dataset['class_id'].unique()
    def _check_exists(self) -> bool:
        for folder in (os.path.join(self.root, 'images'), os.path.join(self.root, 'annotations')):
            if not (os.path.exists(folder) and os.path.isdir(folder)):
                return False
        else:
            return True

    def _download(self) -> None:
        if self._check_exists():
            return

        for url, md5 in self._RESOURCES:
            download_and_extract_archive(url, download_root=str(self.root), md5=md5)

    def __len__(self):
        return len(self.dataset)
    
oxford_pets_train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

oxford_pets_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def oxford_pets_train_loader():
    return OxfordPets('/scr/', train=True, transform=oxford_pets_train_transform)


def oxford_pets_test_loader():
    return OxfordPets('/scr/', train=False, transform=oxford_pets_test_transform)


def prepare_train_loaders(config):
    return {
        'full': torch.utils.data.DataLoader(
            OxfordPets('/scr/', train=True, transform=oxford_pets_train_transform), 
            batch_size=config['batch_size'], 
            shuffle=config['shuffle_train'], 
            num_workers=config['num_workers']
        )
    }

def prepare_test_loaders(config):
    test_set = OxfordPets('/scr/', train=False, transform=oxford_pets_test_transform)
    loaders = {
        'full': torch.utils.data.DataLoader(
            test_set, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=config['num_workers']
        )
    }
    
    if config.get('val_fraction', 0) > 0.:
        # val_set, test_set = train_test_split(test_set.dataset, test_size=config['val_fraction'])
        # val_set = OxfordPets('/srv/share/datasets/', train=False, transform=oxford_pets_test_transform, base_set=val_set)
        # test_set = OxfordPets('/srv/share/datasets/', train=False, transform=oxford_pets_test_transform, base_set=test_set)
        test_set, val_set = create_heldout_split(test_set, config['val_fraction'])
        loaders['heldout_test'] = torch.utils.data.DataLoader(
            test_set,
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=config['num_workers']
        )
        loaders['heldout_val'] = torch.utils.data.DataLoader(
            val_set, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=config['num_workers']
        )
    return loaders
