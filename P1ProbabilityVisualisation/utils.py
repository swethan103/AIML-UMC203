# utils.py

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import platform
import numpy as np
import random

# ─────────────────────────────────────────────
# Global reproducibility
# ─────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

NUM_WORKERS = 0 if platform.system() == 'Windows' else 2


# ─────────────────────────────────────────────
# Basic utilities
# ─────────────────────────────────────────────
def flatten(x):
    return x.reshape(-1)   # safer than view


def dequantize(x, n_values=256):
    """
    Adds uniform noise to discrete pixel values.
    """
    return x + torch.rand_like(x) / n_values


# ─────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────
def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
    }, path)
    print(f"  Checkpoint saved → {path}")

def load_checkpoint(path, model, optimizer=None, model_type=None):
    ckpt = torch.load(path, map_location='cpu')

    if model_type == 'glow' and 'model' not in ckpt:
        model.load_state_dict(ckpt)
        return None

    model.load_state_dict(ckpt['model'])

    if optimizer is not None and 'optim' in ckpt:
        optimizer.load_state_dict(ckpt['optim'])

    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
    return ckpt['epoch']

# def load_checkpoint(path, model, optimizer=None):
#     ckpt = torch.load(path, map_location='cpu')
#     model.load_state_dict(ckpt['model'])
#     if optimizer is not None:
#         optimizer.load_state_dict(ckpt['optim'])

#     print(f"  Loaded checkpoint from epoch {ckpt['epoch']}")
#     return ckpt['epoch']


# ─────────────────────────────────────────────
# DataLoader helpers
# ─────────────────────────────────────────────
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_transform():
    """
    Shared transform for all datasets.
    IMPORTANT: No normalization for flow models.
    """
    return transforms.Compose([
        transforms.ToTensor(),        # keeps data in [0,1]
        transforms.Lambda(flatten)
    ])


# ─────────────────────────────────────────────
# Train loader
# ─────────────────────────────────────────────
def get_dataloader(dataset_name, batch_size=200):
    g = torch.Generator()
    g.manual_seed(SEED)

    transform = get_transform()

    if dataset_name == 'mnist':
        ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
        train_ds = Subset(ds, range(0, 50000))

    elif dataset_name == 'cifar10':
        ds = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        train_ds = Subset(ds, range(0, 40000))

    elif dataset_name == 'svhn':
        ds = datasets.SVHN('./data', split='train', download=True, transform=transform)
        train_ds = Subset(ds, range(0, 60000))

    elif dataset_name == 'tfd':
        raise NotImplementedError("TFD requires manual download.")

    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

    return DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )


# ─────────────────────────────────────────────
# Validation loader
# ─────────────────────────────────────────────
def get_dataloader_valid(dataset_name, batch_size=200):
    g = torch.Generator()
    g.manual_seed(SEED)

    transform = get_transform()

    if dataset_name == 'mnist':
        ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
        valid_ds = Subset(ds, range(50000, 60000))

    elif dataset_name == 'cifar10':
        ds = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        valid_ds = Subset(ds, range(40000, 50000))

    elif dataset_name == 'svhn':
        ds = datasets.SVHN('./data', split='train', download=True, transform=transform)
        valid_ds = Subset(ds, range(60000, 73257))  # rest of train set

    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

    return DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )


# ─────────────────────────────────────────────
# Test loader
# ─────────────────────────────────────────────
def get_dataloader_test(dataset_name, batch_size=200):
    g = torch.Generator()
    g.manual_seed(SEED)

    transform = get_transform()

    if dataset_name == 'mnist':
        ds = datasets.MNIST('./data', train=False, download=True, transform=transform)

    elif dataset_name == 'cifar10':
        ds = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

    elif dataset_name == 'svhn':
        ds = datasets.SVHN('./data', split='test', download=True, transform=transform)

    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )

