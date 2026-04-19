import os
import re
import math
import torch

from models import NICE
from loss import StandardNormal, StandardLogistic, map_loss
from utils import dequantize, get_dataloader, get_dataloader_valid

torch.set_flush_denormal(True)


def bits_per_dim(nll_value, dim):
    """
    Convert NLL to bits per dimension.
    """
    return nll_value / (dim * math.log(2))


def get_checkpoint_dir():
    if os.path.exists('/content/drive/MyDrive'):
        checkpoint_dir = '/content/drive/MyDrive/NICE_checkpoints_map_only'
    elif os.path.exists('/kaggle/input/datasets/agsmiling/forcontinuingtrainingofnice'):
        checkpoint_dir = '/kaggle/input/datasets/agsmiling/forcontinuingtrainingofnice'
    elif os.path.exists('/kaggle/working'):
        checkpoint_dir = '/kaggle/working/NICE_checkpoints_map_only'
    else:
        checkpoint_dir = './checkpoints_map_only'

    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def get_resume_checkpoint(checkpoint_dir, dataset="mnist", prefer="latest"):
    if not os.path.exists(checkpoint_dir):
        return None

    files = os.listdir(checkpoint_dir)

    if prefer == "best":
        best_path = os.path.join(checkpoint_dir, f"best_{dataset}.pt")
        return best_path if os.path.exists(best_path) else None

    pattern = re.compile(rf"ckpt_{dataset}_(\d+)\.pt")
    checkpoints = []

    for f in files:
        match = pattern.match(f)
        if match:
            epoch = int(match.group(1))
            checkpoints.append((epoch, f))

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0])
    latest_file = checkpoints[-1][1]
    return os.path.join(checkpoint_dir, latest_file)


def save_checkpoint(model, optimizer, epoch, best_val_total, path, config):
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'best_val_total': best_val_total,
        'config': config,
    }, path)


def load_checkpoint(path, model, optimizer=None):
    device = next(model.parameters()).device
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    epoch = checkpoint['epoch']
    best_val_total = checkpoint.get('best_val_total', float('inf'))

    print(f"Resumed from epoch {epoch}")
    return epoch, best_val_total


if __name__ == '__main__':

    # ── Config ─────────────────────────────────────────────
    DATASET         = 'mnist'
    EPOCHS          = 400
    BATCH_SIZE      = 200
    LR              = 1e-4
    BETA1           = 0.9
    BETA2           = 0.999
    EPS             = 1e-4
    CLIP_GRAD       = 5.0

    # MAP hyperparameters
    LAMBDA_MAP      = 1e-4
    PARAM_PRIOR_STD = 1.0

    CHECKPOINT_DIR  = get_checkpoint_dir()
    RESUME_FROM     = get_resume_checkpoint(CHECKPOINT_DIR, DATASET, "latest")

    PRIOR = StandardLogistic() if DATASET in ('cifar10', 'svhn') else StandardNormal()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Dataset : {DATASET}")
    print(f"Device  : {device}")
    print(f"Objective: MAP over model parameters only")
    print(f"lambda_map      = {LAMBDA_MAP}")
    print(f"param_prior_std = {PARAM_PRIOR_STD}")

    # ── Setup ──────────────────────────────────────────────
    model = NICE.from_preset(DATASET).to(device)

    loader = get_dataloader(DATASET, batch_size=BATCH_SIZE)
    valid_loader = get_dataloader_valid(DATASET, batch_size=BATCH_SIZE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        betas=(BETA1, BETA2),
        eps=EPS,
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1 / 1.0005)

    start_epoch = 0
    best_val_total = float('inf')

    if RESUME_FROM:
        start_epoch, best_val_total = load_checkpoint(RESUME_FROM, model, optimizer)
        print(f"Continuing from epoch {start_epoch}")
    else:
        print("No checkpoint found. Training from scratch.")

    nvis = model.nvis

    print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Initial LR : {scheduler.get_last_lr()[0]:.6f}")

    config = {
        "dataset": DATASET,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "lambda_map": LAMBDA_MAP,
        "param_prior_std": PARAM_PRIOR_STD,
        "objective": "map_model_only",
    }

    # ── Training loop ──────────────────────────────────────
    for epoch in range(start_epoch, EPOCHS):

        # TRAIN
        model.train()

        train_total_sum = 0.0
        train_nll_sum = 0.0
        train_map_sum = 0.0

        for x, _ in loader:
            x = x.view(x.size(0), -1)
            x = dequantize(x).to(device)

            loss, stats = map_loss(
                model=model,
                x=x,
                prior=PRIOR,
                lambda_map=LAMBDA_MAP,
                param_prior_std=PARAM_PRIOR_STD,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()

            train_total_sum += stats["total"].item()
            train_nll_sum += stats["nll"].item()
            train_map_sum += stats["map_penalty"].item()

        scheduler.step()

        avg_train_total = train_total_sum / len(loader)
        avg_train_nll = train_nll_sum / len(loader)
        avg_train_map = train_map_sum / len(loader)
        train_bpd = bits_per_dim(avg_train_nll, nvis)

        # VALIDATION
        model.eval()

        val_total_sum = 0.0
        val_nll_sum = 0.0
        val_map_sum = 0.0

        with torch.no_grad():
            for x, _ in valid_loader:
                x = x.view(x.size(0), -1)
                x = dequantize(x).to(device)

                _, stats = map_loss(
                    model=model,
                    x=x,
                    prior=PRIOR,
                    lambda_map=LAMBDA_MAP,
                    param_prior_std=PARAM_PRIOR_STD,
                )

                val_total_sum += stats["total"].item()
                val_nll_sum += stats["nll"].item()
                val_map_sum += stats["map_penalty"].item()

        avg_val_total = val_total_sum / len(valid_loader)
        avg_val_nll = val_nll_sum / len(valid_loader)
        avg_val_map = val_map_sum / len(valid_loader)
        val_bpd = bits_per_dim(avg_val_nll, nvis)

        # Latent stats
        with torch.no_grad():
            x_sample, _ = next(iter(loader))
            x_sample = dequantize(x_sample.view(x_sample.size(0), -1)).to(device)
            z = model(x_sample)
            z_mean = z.mean().item()
            z_std = z.std().item()

        print(
            f"Epoch {epoch+1:4d}/{EPOCHS} | "
            f"train_total: {avg_train_total:.4f} | "
            f"train_nll: {avg_train_nll:.4f} ({train_bpd:.4f} bpd) | "
            f"train_map: {avg_train_map:.6f} | "
            f"val_total: {avg_val_total:.4f} | "
            f"val_nll: {avg_val_nll:.4f} ({val_bpd:.4f} bpd) | "
            f"val_map: {avg_val_map:.6f} | "
            f"z_mean: {z_mean:.3f} | z_std: {z_std:.3f} | "
            f"lr: {scheduler.get_last_lr()[0]:.6f}"
        )

        # SAVE BEST
        if avg_val_total < best_val_total:
            best_val_total = avg_val_total

            if "/kaggle/input/" in CHECKPOINT_DIR:
                save_dir = '/kaggle/working/NICE_checkpoints_map_only'
            else:
                save_dir = CHECKPOINT_DIR

            os.makedirs(save_dir, exist_ok=True)
            best_path = os.path.join(save_dir, f'best_{DATASET}.pt')
            save_checkpoint(model, optimizer, epoch + 1, best_val_total, best_path, config)
            print(f"  New best model at epoch {epoch+1} (val_total: {best_val_total:.4f})")

        # REGULAR CHECKPOINT
        if (epoch + 1) % 5 == 0:
            if "/kaggle/input/" in CHECKPOINT_DIR:
                save_dir = '/kaggle/working/NICE_checkpoints_map_only'
            else:
                save_dir = CHECKPOINT_DIR

            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f'ckpt_{DATASET}_{epoch+1}.pt')
            save_checkpoint(model, optimizer, epoch + 1, best_val_total, path, config)

    # FINAL SAVE
    if "/kaggle/input/" in CHECKPOINT_DIR:
        final_save_dir = '/kaggle/working/NICE_checkpoints_map_only'
    else:
        final_save_dir = CHECKPOINT_DIR

    os.makedirs(final_save_dir, exist_ok=True)
    final_path = os.path.join(final_save_dir, f'final_{DATASET}.pt')
    save_checkpoint(model, optimizer, EPOCHS, best_val_total, final_path, config)
    print(f"Training complete. Final model saved to: {final_path}")