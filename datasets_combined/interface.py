from .cub import CUB2011
from .dtd import DTD
from .oxford_pets import OxfordPets
from .nabird import NABird
from .stanford_dogs import StanfordDogs

from torchvision import transforms
from torchvision import datasets as tv_datasets
import torch


def get_train_test_loaders(dataset_name, batch_size, use_augmentations=False, directory_suffix=None):
    if directory_suffix is None:
        dir = '/scr/'
    else:
        dir = f'/scr/{directory_suffix}/'
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
    elif dataset_name == 'caltech101':
        train_loader = torch.utils.data.DataLoader(
            Caltech101(root=f'{dir}caltech101', train=True, transform=train_transforms),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )        
        test_loader = torch.utils.data.DataLoader(
            Caltech101(root=f'{dir}caltech101', train=False, transform=test_transforms),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
    elif dataset_name == 'cars':
        train_loader = torch.utils.data.DataLoader(
            PytorchStanfordCars(root=f'{dir}', split='train', transform=train_transforms, download=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )        
        test_loader = torch.utils.data.DataLoader(
            PytorchStanfordCars(root=f'{dir}', split='test', transform=test_transforms, download=True),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
    elif dataset_name == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            Cifar10(root=f'{dir}cifar10', train=True, transform=train_transforms),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )        
        test_loader = torch.utils.data.DataLoader(
            Cifar10(root=f'{dir}cifar10', train=False, transform=test_transforms),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
    elif dataset_name == 'cifar100':
        train_loader = torch.utils.data.DataLoader(
            Cifar100(root=f'{dir}cifar100', train=True, transform=train_transforms),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )        
        test_loader = torch.utils.data.DataLoader(
            Cifar100(root=f'{dir}cifar100', train=False, transform=test_transforms),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
    elif dataset_name == 'dtd':
        train_loader = torch.utils.data.DataLoader(
            tv_datasets.DTD(root=f'{dir}', split='train', transform=train_transforms, download=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )        
        test_loader = torch.utils.data.DataLoader(
            tv_datasets.DTD(root=f'{dir}', split='test', transform=test_transforms, download=True),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
    elif dataset_name == 'eurosat':
        
        train_loader = torch.utils.data.DataLoader(
            EuroSATBase(location='{dir}', transform=train_transforms, test_split='test').train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )        
        test_loader = torch.utils.data.DataLoader(
            EuroSATBase(location='{dir}', transform=test_transforms, test_split='test').test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
    
    elif dataset_name == 'resisc45':
        train_loader = torch.utils.data.DataLoader(
            RESISC45Dataset(root=f'{dir}', split='train', transforms=train_transforms),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )        
        test_loader = torch.utils.data.DataLoader(
            RESISC45Dataset(root=f'{dir}', split='test', transforms=test_transforms),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
    
    elif dataset_name == 'sun397':
        train_loader = torch.utils.data.DataLoader(
            SUN397VD(root=f'{dir}', split='Training_01', transform=train_transforms, download=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )        
        test_loader = torch.utils.data.DataLoader(
            SUN397VD(root=f'{dir}', split='Testing_01', transform=test_transforms, download=True),
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