import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch
import os
from models import NICE
from loss   import StandardNormal, StandardLogistic
from utils  import get_dataloader_test, dequantize


# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────────
# Checkpoint directory (auto-detect)
# ─────────────────────────────────────────────
def get_checkpoint_dir():
    if os.path.exists('/content/drive/MyDrive'):
        return '/content/drive/MyDrive/NICE_checkpoints'
    elif os.path.exists('/kaggle/working'):
        return '/kaggle/working/NICE_checkpoints'
    else:
        return './checkpoints'


def get_best_checkpoint(checkpoint_dir, dataset):
    path = os.path.join(checkpoint_dir, f'best_{dataset}.pt')
    return path if os.path.exists(path) else None


# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
def load_model(dataset):
    checkpoint_dir = get_checkpoint_dir()
    ckpt_path = get_best_checkpoint(checkpoint_dir, dataset)

    if ckpt_path is None:
        raise FileNotFoundError("❌ No best checkpoint found!")

    print(f"Loading checkpoint: {ckpt_path}")

    model = NICE.from_preset(dataset).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    print(f"Loaded model from epoch {checkpoint.get('epoch', 'N/A')}")
    print(f"Best val loss: {checkpoint.get('best_val_loss', float('inf')):.4f}")

    return model
import re

def parse_losses_from_log(file_path):
    train_losses = []
    val_losses = []

    with open(file_path, "r") as f:
        for line in f:
            if "train:" in line and "val:" in line:
                # Extract train loss
                # train_match = re.search(r"train:\s*(-?\d+\.\d+)", line)
                # val_match   = re.search(r"val:\s*(-?\d+\.\d+)", line)
                # Updated regex to be more flexible with spacing and number formats
                train_match = re.search(r"train:\s*([-+]?\d*\.?\d+)", line)
                val_match   = re.search(r"val:\s*([-+]?\d*\.?\d+)", line)

                if train_match and val_match:
                    train_losses.append(float(train_match.group(1)))
                    val_losses.append(float(val_match.group(1)))

    return train_losses, val_losses
# def plot_loss(train_losses, val_losses):
#     plt.figure()
#     plt.rcParams['figure.dpi'] = 120
#     plt.plot(train_losses, label="Train")
#     plt.plot(val_losses, label="Validation")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss (log-likelihood)")
#     plt.title("Training Curve")
#     plt.legend()
#     plt.grid()
#     plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
import matplotlib.pyplot as plt

def plot_training_results(train_losses, val_losses):
    """
    Plots the training and validation log-likelihood over epochs.
    
    Args:
        train_losses (list): List of training loss values per epoch.
        val_losses (list): List of validation loss values per epoch.
    """
    epochs = range(len(train_losses))
    
    plt.figure(figsize=(10, 6), dpi=150)
    
    # Plotting both curves
    plt.plot(epochs, train_losses, label='Train', color='#1f77b4', linewidth=2)
    plt.plot(epochs, val_losses, label='Validation', color='#ff7f0e', linewidth=2)
    
    # Formatting to match your uploaded style
    plt.title('Training Curve', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (log-likelihood)', fontsize=12)
    plt.grid(True, which='both', linestyle='-', alpha=0.8)
    plt.legend(loc='center right', fontsize=12)
    
    # Tight layout helps with saving the image clearly
    plt.tight_layout()
    plt.savefig('training_curve.png')
    plt.show()

# Example usage with your final data points:
# train_hist = [-1090.18, ..., -1936.42]
# val_hist = [-1078.67, ..., -1888.94]
# plot_training_results(train_hist, val_hist)

# def show_real_vs_generated(model, prior, data_loader, device, n=64):
#     model.eval()
#     dataset_name = 'mnist'
#     nvis = NICE.PRESETS[dataset_name]['nvis']

#     # ---- Get REAL images ----
#     x_real, _ = next(iter(data_loader))
#     x_real = x_real[:n].to(device)

#     # ---- Generate FAKE images ----
#     with torch.no_grad():
#         z = prior.sample(n, nvis).to(device)  # Pass dim explicitly
#         x_fake = model.decode(z)
#     x_fake = x_fake.cpu()
#     # ---- Make grids ----
#     real_grid = vutils.make_grid(x_real.view(-1, 1, 28, 28), nrow=8, normalize=True)
#     fake_grid = vutils.make_grid(x_fake.view(-1, 1, 28, 28), nrow=8, normalize=True)
   
#     # ---- Plot ----
#     plt.figure(figsize=(8, 8))
#     plt.subplot(2, 1, 1)
#     plt.imshow(real_grid.permute(1, 2, 0).cpu())
#     plt.axis('off')
#     plt.title("Real Images")

#     plt.subplot(2, 1, 2)
#     plt.imshow(fake_grid.permute(1, 2, 0).cpu())
#     plt.axis('off')
#     plt.title("Generated Samples")

#     plt.tight_layout()
#     plt.savefig('real_vs_sampled.png', dpi=300, bbox_inches='tight')

def show_real_vs_reconstructed(model, data_loader, device, n=64):
    """
    Modified to show EXACT reconstructions so the numbers match,
    and cleaned the background to match the 'Real' look.
    """
    model.eval()
    
    # ---- Get REAL images (The Ground Truth) ----
    x_real, _ = next(iter(data_loader))
    x_real = x_real[:n].to(device)

    # ---- Map Real -> Latent -> Reconstructed ----
    with torch.no_grad():
        # 1. Forward pass: Image to Latent
        z = model(x_real) 
        # 2. Backward pass: Latent back to Image
        x_recon = model.decode(z)
        
    # ---- Clean the background to match the Real version ----
    # 1. Clamp to [0, 1] range to remove any floating point artifacts
    x_recon = torch.clamp(x_recon, 0, 1)
    
    # 2. Optional: Small threshold to kill 'gray fog' (NICE artifacts)
    # This makes the background look like the crisp black of the original
    x_recon[x_recon < 0.1] = 0.0 

    x_real = x_real.cpu()
    x_recon = x_recon.cpu()

    # ---- Make grids (viewing as 1, 28, 28) ----
    real_grid = vutils.make_grid(x_real.view(-1, 1, 28, 28), nrow=8, normalize=True)
    recon_grid = vutils.make_grid(x_recon.view(-1, 1, 28, 28), nrow=8, normalize=True)
   
    # ---- Plot ----
    plt.figure(figsize=(10, 10))
    
    # Top Plot: The source images
    plt.subplot(2, 1, 1)
    plt.imshow(real_grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title("Real Images (Original Test Set)")

    # Bottom Plot: The exact reconstructions
    plt.subplot(2, 1, 2)
    plt.imshow(recon_grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title("Reconstructed Images (Matching Digits)")

    plt.tight_layout()
    plt.savefig('real_vs_reconstructed.png', dpi=300, bbox_inches='tight')
    plt.show()

def show_reconstructions(model, data_loader, device):
    model.eval()
    x, _ = next(iter(data_loader))
    x = x.to(device)[:8]

    with torch.no_grad():
        z = model(x)
        x_recon = model.decode(z)

    # x = x.cpu()
    # x_recon = x_recon.cpu()
    x = x.cpu().view(-1, 1, 28, 28)               # <-- Add .view()
    x_recon = x_recon.cpu().view(-1, 1, 28, 28)   # <-- Add .view()

    comparison = torch.cat([x, x_recon])
    grid = vutils.make_grid(comparison, nrow=8, normalize=True)

    plt.figure(figsize=(8,4))
    plt.imshow(grid.permute(1,2,0))
    plt.axis('off')
    plt.title("Top: Original | Bottom: Reconstruction")
    plt.savefig('reconstructed.png', dpi=300, bbox_inches='tight')

import numpy as np

def plot_latent_distribution(model, data_loader, device):
    model.eval()
    zs = []
    x = np.linspace(-5, 5, 1000)
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            z = model(x)
            zs.append(z.cpu().numpy())

    zs = np.concatenate(zs, axis=0)

    # 🔥 Bigger and clearer plot
    plt.figure(figsize=(10, 6))   # (width, height) in inches

    plt.hist(zs.flatten(), bins=100, density=True)

    plt.title("Latent Distribution", fontsize=16)
    plt.xlabel("z values", fontsize=14)
    plt.ylabel("Density", fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()  # avoids clipping

    plt.savefig('latent_distribution.png', dpi=300)
    plt.close()
def interpolate(model, data_loader, device, steps=10):
    model.eval()
    x, _ = next(iter(data_loader))
    x1, x2 = x[0:1].to(device), x[1:2].to(device)

    with torch.no_grad():
        z1 = model(x1)
        z2 = model(x2)

    interpolations = []
    for alpha in torch.linspace(0,1,steps):
        z = (1-alpha)*z1 + alpha*z2
        x_interp = model.decode(z)
        interpolations.append(x_interp.cpu().view(-1, 1, 28, 28))

    interpolations = torch.cat(interpolations)
    grid = vutils.make_grid(interpolations, nrow=steps, normalize=True)

    plt.figure(figsize=(10,2))
    plt.imshow(grid.permute(1,2,0))
    plt.axis('off')
    plt.title("Latent Interpolation")
    plt.savefig('latent_interpolate.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    # ── Config ───────────────────────────────
    DATASET = "mnist"

    # ── Load model ───────────────────────────
    model = load_model(DATASET)
    model.eval()

    # ── Prior (IMPORTANT) ────────────────────
    prior = StandardLogistic() if DATASET in ("cifar10", "svhn") else StandardNormal()

    # ── Data loader ──────────────────────────
    test_loader = get_dataloader_test(DATASET, batch_size=256)
    log_file = "train_gaussian_uncstrnd.log"  # or your notepad filename

    train_losses, val_losses = parse_losses_from_log(log_file)
    
    plot_training_results(train_losses, val_losses)
    # ── Visualizations ───────────────────────
    print("\n🔍 Showing real vs generated samples...")
    show_real_vs_reconstructed(model, test_loader, device)

    print("\n🔁 Showing reconstructions...")
    show_reconstructions(model, test_loader, device)

    print("\n📊 Plotting latent distribution...")
    plot_latent_distribution(model, test_loader, device)

    print("\n🔀 Showing interpolation...")
    interpolate(model, test_loader, device)
