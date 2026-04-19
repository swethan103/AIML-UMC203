# train.py

import os
import torch
import math
from models import NICE
from loss   import StandardNormal, StandardLogistic, nll_loss
from utils  import (dequantize, save_checkpoint, load_checkpoint,
                    get_dataloader, get_dataloader_valid)
torch.set_flush_denormal(True)

def bits_per_dim(loss, dim):
    """
    Convert NLL to bits per dimension.
    """
    return loss / (dim * math.log(2))

import os

def get_checkpoint_dir():
    import os

    if os.path.exists('/content/drive/MyDrive'):
        checkpoint_dir = f'/content/drive/MyDrive/NICE_checkpoints'
    elif os.path.exists('/kaggle/input/datasets/agsmiling/forcontinuingtrainingofnice'):
        checkpoint_dir = '/kaggle/input/datasets/agsmiling/forcontinuingtrainingofnice'
    elif os.path.exists('/kaggle/working'):
        checkpoint_dir = f'/kaggle/working/NICE_checkpoints'
    else:
        checkpoint_dir = './checkpoints'

    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir
import os
import re

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
  
def save_checkpoint(model, optimizer, epoch, best_val_loss, path):
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }, path)

def load_checkpoint(path, model, optimizer=None):
    device = next(model.parameters()).device   # 🔥 key fix

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    epoch = checkpoint['epoch']
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    print(f"Resumed from epoch {epoch}")

    return epoch, best_val_loss
if __name__ == '__main__':

    # ── Config ────────────────────────────────────────────────────────
    DATASET        = 'mnist'
    EPOCHS         = 1000
    BATCH_SIZE     = 200
    CHECKPOINT_DIR = get_checkpoint_dir()
    RESUME_FROM    = get_resume_checkpoint(CHECKPOINT_DIR, DATASET, "latest")
    CLIP_GRAD      = 5.0   # NEW

    PRIOR  = StandardLogistic() if DATASET in ('cifar10', 'svhn') else StandardNormal()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Dataset : {DATASET}")
    print(f"Device  : {device}")

    # ── Setup ─────────────────────────────────────────────────────────
    model        = NICE.from_preset(DATASET).to(device)
    loader       = get_dataloader(DATASET, batch_size=BATCH_SIZE)
    valid_loader = get_dataloader_valid(DATASET, batch_size=BATCH_SIZE)
    # Paper Hyperparameters
    LR = 1e-4  #changed from 1e-3 to 1e-4 to expect smoother training
    BETA1 = 0.9
    BETA2 = 0.999  # in paper 0.01 -> 0.999 because 0.01 lead to unstable epochs
    EPS = 1e-4    # Paper value of 1e-4
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=LR, 
        betas=(BETA1, BETA2), 
        eps=EPS,
    )                       #removed weight decay
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, momentum=0.9)  #<- updated 0.0 to 0.9
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1/1.0005)

    start_epoch   = 0
    best_val_loss = float('inf')

    if RESUME_FROM:
      start_epoch, best_val_loss = load_checkpoint(
          RESUME_FROM, model, optimizer
      )
      print(f"Continuing from epoch {start_epoch}")
    else:
      print("No checkpoint found. Training from scratch.")
    print(f"Using checkpoint: {RESUME_FROM}")
    nvis = model.nvis

    print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Initial LR : {scheduler.get_last_lr()[0]:.6f}")

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(start_epoch, EPOCHS):

        # ── TRAIN ────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0

        for x, _ in loader:
            x = x.view(x.size(0), -1)
            x = dequantize(x).to(device)

            loss = nll_loss(model, x, PRIOR)

            optimizer.zero_grad()
            loss.backward()

            # NEW: gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)

            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_train_loss = epoch_loss / len(loader)
        train_bpd = bits_per_dim(avg_train_loss, nvis)

        # # Momentum warmup
        # if epoch == 5:
        #     for param_group in optimizer.param_groups:
        #         param_group['momentum'] = 0.5
        #     print("  Momentum set to 0.5")              #<- removed this as its not sompatible with pytorch unlike pylearn2

        # ── VALIDATION ────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, _ in valid_loader:
                x = x.view(x.size(0), -1)
                x = dequantize(x).to(device)   # FIXED

                val_loss += nll_loss(model, x, PRIOR).item()

        avg_val_loss = val_loss / len(valid_loader)
        val_bpd = bits_per_dim(avg_val_loss, nvis)

        # ── DEBUG: latent statistics (VERY useful) ────────────────────
        with torch.no_grad():
            x_sample, _ = next(iter(loader))
            x_sample = dequantize(x_sample.view(x_sample.size(0), -1)).to(device)
            z = model(x_sample)

            z_mean = z.mean().item()
            z_std  = z.std().item()

        print(f"Epoch {epoch+1:4d}/{EPOCHS} | "
              f"train: {avg_train_loss:.4f} ({train_bpd:.4f} bpd) | "
              f"val: {avg_val_loss:.4f} ({val_bpd:.4f} bpd) | "
              f"z_mean: {z_mean:.3f} | z_std: {z_std:.3f} | "
              f"lr: {scheduler.get_last_lr()[0]:.6f}")

        # ── SAVE BEST ────────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # Simplified logic: If input is read-only, force save to working
            if "/kaggle/input/" in CHECKPOINT_DIR:
                save_dir = '/kaggle/working/NICE_checkpoints'
            else:
                save_dir = CHECKPOINT_DIR
            
            os.makedirs(save_dir, exist_ok=True)
            best_path = os.path.join(save_dir, f'best_{DATASET}.pt')
            save_checkpoint(model, optimizer, epoch + 1, best_val_loss, best_path)
            print(f"  New best model at epoch {epoch+1} (val: {best_val_loss:.4f})")

        # ── REGULAR CHECKPOINT ───────────────────────────────────────
        if (epoch + 1) % 5 == 0:
            if "/kaggle/input/" in CHECKPOINT_DIR:
                save_dir = '/kaggle/working/NICE_checkpoints'
            else:
                save_dir = CHECKPOINT_DIR
                
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f'ckpt_{DATASET}_{epoch+1}.pt')
            save_checkpoint(model, optimizer, epoch + 1, best_val_loss, path)
    # if CHECKPOINT_DIR == '/kaggle/input/datasets/agsmiling/forcontinuingtrainingofnice':
    #           final_path = os.path.join('/kaggle/working/NICE_checkpoints', f'final_{DATASET}.pt')
    # final_path = os.path.join(CHECKPOINT_DIR, f'final_{DATASET}.pt')
    # save_checkpoint(model, optimizer, EPOCHS, best_val_loss, final_path)
    # Ensure we save to /working/ if we started from /input/
    if "/kaggle/input/" in CHECKPOINT_DIR:
        final_save_dir = '/kaggle/working/NICE_checkpoints'
    else:
        final_save_dir = CHECKPOINT_DIR
        
    os.makedirs(final_save_dir, exist_ok=True)
    final_path = os.path.join(final_save_dir, f'final_{DATASET}.pt')
    
    save_checkpoint(model, optimizer, EPOCHS, best_val_loss, final_path)
    print(f"Training complete. Final model saved to: {final_path}")
