import os
import re
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch

from models import NICE
from loss import LearnedDiagonalGaussianPrior
from utils import get_dataloader_test


# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────────
# Checkpoint directory (auto-detect)
# ─────────────────────────────────────────────
def get_checkpoint_dir():
    if os.path.exists('/content/drive/MyDrive'):
        return '/content/drive/MyDrive/NICE_checkpoints_learned_prior_map'
    elif os.path.exists('/kaggle/working'):
        return '/kaggle/working/NICE_checkpoints_learned_prior_map'
    else:
        return './checkpoints_learned_prior_map'


def get_best_checkpoint(checkpoint_dir, dataset):
    path = os.path.join(checkpoint_dir, f'best_{dataset}.pt')
    return path if os.path.exists(path) else None


# ─────────────────────────────────────────────
# Load model + learned prior
# ─────────────────────────────────────────────
def load_model_and_prior(dataset):
    checkpoint_dir = get_checkpoint_dir()
    ckpt_path = get_best_checkpoint(checkpoint_dir, dataset)

    if ckpt_path is None:
        raise FileNotFoundError("❌ No best checkpoint found!")

    print(f"Loading checkpoint: {ckpt_path}")

    model = NICE.from_preset(dataset).to(device)
    prior = LearnedDiagonalGaussianPrior(dim=model.nvis).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    if 'prior_state_dict' in checkpoint:
        prior.load_state_dict(checkpoint['prior_state_dict'])
    else:
        print("⚠️ prior_state_dict not found in checkpoint. Using default initialized prior.")

    print(f"Loaded model from epoch {checkpoint.get('epoch', 'N/A')}")
    print(f"Best val total: {checkpoint.get('best_val_total', float('inf')):.4f}")

    return model, prior


# ─────────────────────────────────────────────
# Parse train_total / val_total from log
# ─────────────────────────────────────────────
def parse_losses_from_log(file_path):
    train_losses = []
    val_losses = []

    if not os.path.exists(file_path):
        print(f"❌ Log file not found: {file_path}")
        return train_losses, val_losses

    with open(file_path, "r") as f:
        for line in f:
            line_lower = line.lower()

            train_match = re.search(
                r"train_total\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
                line_lower
            )
            val_match = re.search(
                r"val_total\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
                line_lower
            )

            if train_match and val_match:
                train_losses.append(float(train_match.group(1)))
                val_losses.append(float(val_match.group(1)))

    return train_losses, val_losses


# ─────────────────────────────────────────────
# Plot training curve
# ─────────────────────────────────────────────
def plot_training_results(train_losses, val_losses):
    if len(train_losses) == 0 or len(val_losses) == 0:
        print("❌ No train_total / val_total values found in the log file.")
        return

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(epochs, train_losses, label='Train Total', linewidth=2)
    plt.plot(epochs, val_losses, label='Validation Total', linewidth=2)

    plt.title('Training Curve', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Total Loss', fontsize=12)
    plt.grid(True, which='both', linestyle='-', alpha=0.8)
    plt.legend(loc='best', fontsize=12)

    plt.tight_layout()
    plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────
# Show real vs reconstructed
# ─────────────────────────────────────────────
def show_real_vs_reconstructed(model, data_loader, device, n=64):
    model.eval()

    x_real, _ = next(iter(data_loader))
    x_real = x_real[:n].to(device)

    with torch.no_grad():
        z = model(x_real)
        x_recon = model.decode(z)

    x_recon = torch.clamp(x_recon, 0, 1)
    x_recon[x_recon < 0.1] = 0.0

    x_real = x_real.cpu()
    x_recon = x_recon.cpu()

    real_grid = vutils.make_grid(x_real.view(-1, 1, 28, 28), nrow=8, normalize=True)
    recon_grid = vutils.make_grid(x_recon.view(-1, 1, 28, 28), nrow=8, normalize=True)

    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)
    plt.imshow(real_grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title("Real Images (Original Test Set)")

    plt.subplot(2, 1, 2)
    plt.imshow(recon_grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title("Reconstructed Images (Matching Digits)")

    plt.tight_layout()
    plt.savefig('real_vs_reconstructed.png', dpi=300, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────
# Show reconstruction strip
# ─────────────────────────────────────────────
def show_reconstructions(model, data_loader, device):
    model.eval()
    x, _ = next(iter(data_loader))
    x = x.to(device)[:8]

    with torch.no_grad():
        z = model(x)
        x_recon = model.decode(z)

    x = x.cpu().view(-1, 1, 28, 28)
    x_recon = x_recon.cpu().view(-1, 1, 28, 28)

    comparison = torch.cat([x, x_recon])
    grid = vutils.make_grid(comparison, nrow=8, normalize=True)

    plt.figure(figsize=(8, 4))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title("Top: Original | Bottom: Reconstruction")
    plt.savefig('reconstructed.png', dpi=300, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────
# Plot latent distribution
# ─────────────────────────────────────────────
def plot_latent_distribution(model, data_loader, device):
    model.eval()
    zs = []

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            z = model(x)
            zs.append(z.cpu().numpy())

    zs = np.concatenate(zs, axis=0)

    plt.figure(figsize=(10, 6))
    plt.hist(zs.flatten(), bins=100, density=True)

    plt.title("Latent Distribution", fontsize=16)
    plt.xlabel("z values", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    plt.savefig('latent_distribution.png', dpi=300)
    plt.show()


# ─────────────────────────────────────────────
# Plot learned prior parameters
# ─────────────────────────────────────────────
def plot_learned_prior(prior):
    mu = prior.mu.detach().cpu().numpy()
    std = torch.exp(torch.clamp(prior.log_std.detach(), min=-5.0, max=5.0)).cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.hist(mu, bins=50, alpha=0.7)
    plt.title("Learned Prior Means")
    plt.xlabel("mu")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("prior_mu_distribution.png", dpi=300)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.hist(std, bins=50, alpha=0.7)
    plt.title("Learned Prior Standard Deviations")
    plt.xlabel("sigma")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("prior_std_distribution.png", dpi=300)
    plt.show()


# ─────────────────────────────────────────────
# Sample from learned prior
# ─────────────────────────────────────────────
def generate_samples(model, prior, device, n=64):
    model.eval()
    prior.eval()

    with torch.no_grad():
        z = prior.sample(n, device=device)
        x_fake = model.decode(z)

    x_fake = torch.clamp(x_fake, 0, 1)
    x_fake[x_fake < 0.1] = 0.0
    x_fake = x_fake.cpu().view(-1, 1, 28, 28)

    fake_grid = vutils.make_grid(x_fake, nrow=8, normalize=True)

    plt.figure(figsize=(8, 8))
    plt.imshow(fake_grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title("Generated Samples from Learned Prior")
    plt.tight_layout()
    plt.savefig('generated_samples.png', dpi=300, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────
# Latent interpolation
# ─────────────────────────────────────────────
def interpolate(model, data_loader, device, steps=10):
    model.eval()
    x, _ = next(iter(data_loader))
    x1, x2 = x[0:1].to(device), x[1:2].to(device)

    with torch.no_grad():
        z1 = model(x1)
        z2 = model(x2)

    interpolations = []
    for alpha in torch.linspace(0, 1, steps):
        z = (1 - alpha) * z1 + alpha * z2
        x_interp = model.decode(z)
        interpolations.append(x_interp.cpu().view(-1, 1, 28, 28))

    interpolations = torch.cat(interpolations)
    grid = vutils.make_grid(interpolations, nrow=steps, normalize=True)

    plt.figure(figsize=(10, 2))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title("Latent Interpolation")
    plt.savefig('latent_interpolate.png', dpi=300, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    DATASET = "mnist"

    model, prior = load_model_and_prior(DATASET)
    model.eval()
    prior.eval()

    test_loader = get_dataloader_test(DATASET, batch_size=256)
    log_file = "outputs.txt"

    print("Reading log from:", os.path.abspath(log_file))
    print("Does log file exist?", os.path.exists(log_file))

    train_losses, val_losses = parse_losses_from_log(log_file)

    print(f"Parsed {len(train_losses)} train_total values")
    print(f"Parsed {len(val_losses)} val_total values")

    plot_training_results(train_losses, val_losses)

    print("\n🔍 Showing real vs reconstructed samples...")
    show_real_vs_reconstructed(model, test_loader, device)

    print("\n🔁 Showing reconstructions...")
    show_reconstructions(model, test_loader, device)

    print("\n📊 Plotting latent distribution...")
    plot_latent_distribution(model, test_loader, device)

    print("\n📈 Plotting learned prior parameters...")
    plot_learned_prior(prior)

    print("\n🎲 Generating samples from learned prior...")
    generate_samples(model, prior, device)

    print("\n🔀 Showing interpolation...")
    interpolate(model, test_loader, device)