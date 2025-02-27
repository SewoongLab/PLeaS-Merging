from .cub import CUB2011
from .oxford_pets import OxfordPets
from .nabird import NABird
from .stanford_dogs import StanfordDogs

from torchvision import transforms
from torchvision import datasets as tv_datasets
import torch


def get_train_test_loaders(dataset_name, batch_size, use_augmentations=False, dir='/scr/'):

    if use_augmentations:
        train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    else:
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

    if dataset_name == 'cub':
        train_loader = torch.utils.data.DataLoader(
            CUB2011(root=f'{dir}cub200', train=True, transform=train_transforms),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )        
        test_loader = torch.utils.data.DataLoader(
            CUB2011(root=f'{dir}cub200', train=False, transform=test_transforms),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )        
    elif dataset_name == 'oxford_pets':
        train_loader = torch.utils.data.DataLoader(
            OxfordPets(root=f'{dir}', train=True, transform=train_transforms),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )        
        test_loader = torch.utils.data.DataLoader(
            OxfordPets(root=f'{dir}', train=False, transform=test_transforms),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
    
    elif dataset_name == 'nabird':
        train_loader = torch.utils.data.DataLoader(
            NABird(root=f'{dir}nabirds', train=True, transform=train_transforms),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )        
        test_loader = torch.utils.data.DataLoader(
            NABird(root=f'{dir}nabirds', train=False, transform=test_transforms),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
    elif dataset_name == 'stanford_dogs':
        train_loader = torch.utils.data.DataLoader(
            StanfordDogs(root=f'{dir}stanford_dogs/', train=True, transform=train_transforms),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )        
        test_loader = torch.utils.data.DataLoader(
            StanfordDogs(root=f'{dir}stanford_dogs/', train=False, transform=test_transforms),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
    
    else:
        raise ValueError('Unknown dataset')
    
    return train_loader, test_loader