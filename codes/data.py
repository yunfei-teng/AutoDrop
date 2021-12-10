import os
import random
import numpy
import torch
from torchvision import datasets, transforms

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def get_data_loader(args):
    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
    if args.seed_loader:
        train_kwargs.update({'worker_init_fn': seed_worker})
    test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': False}

    if args.use_cuda:
        cuda_kwargs = {'num_workers': 16 if args.dataset == 'imagenet' else 4,
                       'pin_memory' : True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std =[x / 255.0 for x in [63.0, 62.1, 66.7]])
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std =[x / 255.0 for x in [63.0, 62.1, 66.7]])
        ])

    if args.dataset == 'cifar10':
        dataset1 = datasets.CIFAR10('./data', train=True, download=True,
                        transform=transform_train)
        dataset2 = datasets.CIFAR10('./data', train=False,
                        transform=transform_test)

    if args.dataset == 'cifar100':
        dataset1 = datasets.CIFAR100('./data', train=True, download=True,
                        transform=transform_train)
        dataset2 = datasets.CIFAR100('./data', train=False,
                        transform=transform_test)

    if args.dataset == 'imagenet':
        traindir = os.path.join('imagenet_dataset', 'train')
        valdir = os.path.join('imagenet_dataset', 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
        dataset1 = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))

        dataset2 = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    return train_loader, test_loader