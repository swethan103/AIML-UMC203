import os
import argparse
import torch
import torchvision
from models import NICE
from loss import StandardNormal, StandardLogistic


# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# Checkpoint directory
# ─────────────────────────────────────────────
def get_checkpoint_dir():
    if os.path.exists("/content/drive/MyDrive"):
        return "/content/drive/MyDrive/NICE_checkpoints_map_only"
    elif os.path.exists("/kaggle/working"):
        return "/kaggle/working/NICE_checkpoints_map_only"
    else:
        return "./checkpoints_map_only"


def get_final_checkpoint(checkpoint_dir, dataset):
    path = os.path.join(checkpoint_dir, f"final_{dataset}.pt")
    return path if os.path.exists(path) else None


# ─────────────────────────────────────────────
# Output directory
# ─────────────────────────────────────────────
def get_output_dir():
    if os.path.exists("/content/drive/MyDrive"):
        out_dir = "/content/drive/MyDrive/NICE_generated_samples_map_only"
    elif os.path.exists("/kaggle/working"):
        out_dir = "/kaggle/working/NICE_generated_samples_map_only"
    else:
        out_dir = "./generated_samples_map_only"

    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
def load_model(dataset):
    checkpoint_dir = get_checkpoint_dir()
    ckpt_path = get_final_checkpoint(checkpoint_dir, dataset)

    if ckpt_path is None:
        raise FileNotFoundError(
            f"❌ No final checkpoint found at {os.path.join(checkpoint_dir, f'final_{dataset}.pt')}"
        )

    print(f"Loading final checkpoint: {ckpt_path}")

    model = NICE.from_preset(dataset).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded model from epoch {checkpoint.get('epoch', 'N/A')}")
    print(f"Best val total: {checkpoint.get('best_val_total', float('inf')):.4f}")

    return model


# ─────────────────────────────────────────────
# Generate samples
# ─────────────────────────────────────────────
@torch.no_grad()
def generate_samples(model, prior, dataset_name, n_samples=64, temperature=1.0):
    nvis = NICE.PRESETS[dataset_name]["nvis"]

    z = prior.sample(n_samples, nvis, device=device)
    z = z * temperature

    x = model.decode(z)
    x = x.cpu()

    if dataset_name == "mnist":
        x = x.view(n_samples, 1, 28, 28)
    elif dataset_name in ("cifar10", "svhn"):
        x = x.view(n_samples, 3, 32, 32)
    elif dataset_name == "tfd":
        x = x.view(n_samples, 1, 48, 48)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    x = x.clamp(0, 1)
    x[x < 0.10] = 0.0

    return x


# ─────────────────────────────────────────────
# Save sample grid
# ─────────────────────────────────────────────
def save_sample_grid(samples, save_path, nrow=8):
    torchvision.utils.save_image(samples, save_path, nrow=nrow, padding=2)
    print(f"Saved generated samples to: {save_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(args):
    dataset = args.dataset.lower()

    if dataset not in NICE.PRESETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {list(NICE.PRESETS.keys())}")

    model = load_model(dataset)

    prior = StandardLogistic() if dataset in ("cifar10", "svhn") else StandardNormal()

    samples = generate_samples(
        model=model,
        prior=prior,
        dataset_name=dataset,
        n_samples=args.n_samples,
        temperature=args.temperature,
    )

    out_dir = get_output_dir()
    save_name = f"samples_final_{dataset}_{args.n_samples}_temp{args.temperature:.2f}.png"
    save_path = os.path.join(out_dir, save_name)

    nrow = int(args.n_samples ** 0.5)
    if nrow * nrow != args.n_samples:
        nrow = min(8, args.n_samples)

    save_sample_grid(samples, save_path, nrow=nrow)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from final MAP-only NICE model.")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    main(args)