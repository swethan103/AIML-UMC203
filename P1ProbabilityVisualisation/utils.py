# # utils.py

# import torch
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# import platform
# import numpy as np
# import random

# # At the top of utils.py, outside any function
# NUM_WORKERS = 0 if platform.system() == 'Windows' else 2

# def flatten(x):
#     return x.view(-1)
    
# def dequantize(x, n_values=256):
#     """
#     Adds tiny uniform noise to discrete pixel values.
#     Prevents the model from collapsing probability onto a grid.
#     """
#     return x + torch.rand_like(x) / n_values


# def save_checkpoint(model, optimizer, epoch, path):
#     torch.save({
#         'epoch': epoch,
#         'model': model.state_dict(),
#         'optim': optimizer.state_dict(),
#     }, path)
#     print(f"  Checkpoint saved → {path}")


# def load_checkpoint(path, model, optimizer=None):
#     ckpt = torch.load(path, map_location = 'cpu')
#     model.load_state_dict(ckpt['model'])
#     if optimizer is not None:                        # <- to handle the case of MNIST where i forgot to calculate the validation loss paprallely
#         optimizer.load_state_dict(ckpt['optim'])
#     print(f"  Loaded checkpoint from epoch {ckpt['epoch']}")
#     return ckpt['epoch']

# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)

# def get_dataloader(dataset_name, batch_size=200):
#     g = torch.Generator()
#     g.manual_seed(42)
#     if dataset_name == 'mnist':
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Lambda(flatten)        # 1×28×28 → 784
#         ])
#         ds = datasets.MNIST(
#             './data', train=True, download=True, transform=transform
#         )

#     elif dataset_name == 'cifar10':
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,)*3, (0.5,)*3),      # scale to [-1, 1]
#             transforms.Lambda(flatten)        # 3×32×32 → 3072
#         ])
#         ds = datasets.CIFAR10(
#             './data', train=True, download=True, transform=transform
#         )

#     elif dataset_name == 'svhn':
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,)*3, (0.5,)*3),
#             transforms.Lambda(flatten)        # 3×32×32 → 3072
#         ])
#         # SVHN uses split='train' not train=True
#         ds = datasets.SVHN(
#             './data', split='train', download=True, transform=transform
#         )

#     elif dataset_name == 'tfd':
#         # TFD requires a manual download from:
#         # http://www.cs.toronto.edu/~ranzato/publications/TFD/TFD.zip
#         # Place TFD_48x48.mat in ./data/tfd/ after downloading
#         raise NotImplementedError(
#             "TFD requires manual download. See comment above."
#         )

#     else:
#         raise ValueError(
#             f"Unknown dataset '{dataset_name}'. "
#             f"Choose from: mnist, cifar10, svhn, tfd"
#         )

#     return DataLoader(
#         ds, 
#         batch_size=batch_size, 
#         shuffle=True, 
#         num_workers=NUM_WORKERS, 
#         worker_init_fn = seed_worker, 
#         generator = g
#     )

# def get_dataloader_test(dataset_name, batch_size=200):
#     """Same as get_dataloader but returns the test split."""
#     g = torch.Generator()
#     g.manual_seed(42)
#     if dataset_name == 'mnist':
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Lambda(flatten)
#         ])
#         ds = datasets.MNIST(
#             './data', train=False, download=True, transform=transform
#         )
#     elif dataset_name == 'cifar10':
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,)*3, (0.5,)*3),
#             transforms.Lambda(flatten)
#         ])
#         ds = datasets.CIFAR10(
#             './data', train=False, download=True, transform=transform
#         )
#     elif dataset_name == 'svhn':
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,)*3, (0.5,)*3),
#             transforms.Lambda(flatten)
#         ])
#         ds = datasets.SVHN(
#             './data', split='test', download=True, transform=transform
#         )
#     return DataLoader(
#         ds, 
#         batch_size=batch_size, 
#         shuffle=False, 
#         num_workers=NUM_WORKERS,
#         worker_init_fn=seed_worker, 
#         generator = g
#     )

# def get_dataloader_valid(dataset_name, batch_size=200):
#     g = torch.Generator()
#     g.manual_seed(42)

#     if dataset_name == 'mnist':
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Lambda(flatten)
#         ])
#         ds = datasets.MNIST(
#             './data', train=True, download=True, transform=transform
#         )
#         valid_ds = torch.utils.data.Subset(ds, range(50000, 60000))

#     elif dataset_name == 'cifar10':
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,)*3, (0.5,)*3),
#             transforms.Lambda(flatten)
#         ])
#         ds = datasets.CIFAR10(
#             './data', train=True, download=True, transform=transform
#         )
#         valid_ds = torch.utils.data.Subset(ds, range(40000, 50000))

#     elif dataset_name == 'svhn':
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,)*3, (0.5,)*3),
#             transforms.Lambda(flatten)
#         ])
#         valid_ds = datasets.SVHN(
#             './data', split='extra', download=True, transform=transform
#         )

#     return DataLoader(
#         valid_ds,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=NUM_WORKERS,        
#         generator=g
#     )
# ------------------------------------------------------------------------------------------------------

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



# -----------------------------------------------------------------------------------------------
# import torch
# from torch.utils.data import DataLoader, Subset
# from torchvision import datasets, transforms
# import platform
# import numpy as np
# import random

# # ─────────────────────────────────────────────
# # Global reproducibility
# # ─────────────────────────────────────────────
# SEED = 42
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# random.seed(SEED)

# NUM_WORKERS = 0 if platform.system() == 'Windows' else 2


# # ─────────────────────────────────────────────
# # Basic utilities
# # ─────────────────────────────────────────────
# def flatten(x):
#     return x.reshape(-1)


# def dequantize(x, n_values=256):
#     return x + torch.rand_like(x) / n_values


# # ─────────────────────────────────────────────
# # Checkpointing
# # ─────────────────────────────────────────────
# def save_checkpoint(model, optimizer, epoch, path):
#     torch.save({
#         'epoch': epoch,
#         'model': model.state_dict(),
#         'optim': optimizer.state_dict(),
#     }, path)


# def load_checkpoint(path, model, optimizer=None):
#     ckpt = torch.load(path, map_location='cpu')
#     model.load_state_dict(ckpt['model'])

#     if optimizer is not None:
#         optimizer.load_state_dict(ckpt['optim'])

#     return ckpt['epoch']


# # ─────────────────────────────────────────────
# # Reproducibility helpers
# # ─────────────────────────────────────────────
# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)


# # ─────────────────────────────────────────────
# # Transform factory
# # ─────────────────────────────────────────────
# def get_transform(flatten_input=False, img_size=None):
#     tfms = []

#     if img_size is not None:
#         tfms.append(transforms.Resize((img_size, img_size)))

#     tfms.append(transforms.ToTensor())

#     if flatten_input:
#         tfms.append(transforms.Lambda(flatten))

#     return transforms.Compose(tfms)


# # ─────────────────────────────────────────────
# # Dataset factory
# # ─────────────────────────────────────────────
# def get_dataset(dataset_name, split='train', flatten_input=False):
#     transform = get_transform(flatten_input, img_size = 32)

#     if dataset_name == 'mnist':
#         return datasets.MNIST(
#             './data',
#             train=(split != 'test'),
#             download=True,
#             transform=transform
#         )

#     elif dataset_name == 'cifar10':
#         return datasets.CIFAR10(
#             './data',
#             train=(split != 'test'),
#             download=True,
#             transform=transform
#         )

#     elif dataset_name == 'svhn':
#         if split == 'test':
#             svhn_split = 'test'
#         else:
#             svhn_split = 'train'

#         return datasets.SVHN(
#             './data',
#             split=svhn_split,
#             download=True,
#             transform=transform
#         )

#     else:
#         raise ValueError(f"Unknown dataset '{dataset_name}'")


# # ─────────────────────────────────────────────
# # Unified dataloader
# # ─────────────────────────────────────────────
# def get_dataloader(
#     dataset_name,
#     batch_size=200,
#     split='train',
#     flatten_input=False
# ):
#     g = torch.Generator()
#     g.manual_seed(SEED)

#     ds = get_dataset(dataset_name, split, flatten_input)

#     # Manual train/valid split
#     if split == 'train':
#         if dataset_name == 'mnist':
#             ds = Subset(ds, range(0, 50000))
#         elif dataset_name == 'cifar10':
#             ds = Subset(ds, range(0, 40000))
#         elif dataset_name == 'svhn':
#             ds = Subset(ds, range(0, 60000))

#     elif split == 'valid':
#         if dataset_name == 'mnist':
#             ds = Subset(ds, range(50000, 60000))
#         elif dataset_name == 'cifar10':
#             ds = Subset(ds, range(40000, 50000))
#         elif dataset_name == 'svhn':
#             ds = Subset(ds, range(60000, 73257))

#     elif split == 'test':
#         pass

#     else:
#         raise ValueError("split must be one of: train, valid, test")

#     return DataLoader(
#         ds,
#         batch_size=batch_size,
#         shuffle=(split == 'train'),
#         num_workers=NUM_WORKERS,
#         pin_memory=True,
#         worker_init_fn=seed_worker,
#         generator=g
#     )