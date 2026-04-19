# import torch
# import os
# import random
# import numpy as np
# import sys

# # ─────────────────────────────────────────────
# # Environment Detection & Path Setup
# # ─────────────────────────────────────────────
# def detect_environment():
#     if 'COLAB_GPU' in os.environ:
#         return 'colab'
#     elif 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
#         return 'kaggle'
#     return 'pc'

# ENV = detect_environment()
# print(f"Running on: {ENV.upper()}")

# # Set base directory for checkpoints and data
# # Kaggle requires output to be in /kaggle/working/
# BASE_DIR = "." 
# def get_checkpoint_dir():
#     import os

#     if os.path.exists('/content/drive/MyDrive'):
#         checkpoint_dir = f'/content/drive/MyDrive/NICE_checkpoints'
#     elif os.path.exists('/kaggle/working'):
#         checkpoint_dir = f'/kaggle/working/NICE_checkpoints'
#     else:
#         checkpoint_dir = './checkpoints'

#     os.makedirs(checkpoint_dir, exist_ok=True)
#     return checkpoint_dir

# # ─────────────────────────────────────────────
# # Config
# # ─────────────────────────────────────────────
# DATASET    = 'mnist'
# # Use os.path.join for cross-platform compatibility (Windows vs Linux)
# CHECKPOINT = os.path.join(get_checkpoint_dir(),'ckpt_mnist_1500.pt')
# N_SAMPLES  = 100
# SEED       = 42

# # ─────────────────────────────────────────────
# # Imports (After path setup if necessary)
# # ─────────────────────────────────────────────
# try:
#     from models import NICE
#     from loss import StandardNormal, StandardLogistic
#     from utils import load_checkpoint
#     from evaluate import compute_test_loglikelihood, generate_samples
# except ImportError:
#     print("Error: Ensure 'models', 'loss', 'utils', and 'evaluate' are in the current directory.")
#     # In Colab/Kaggle, you might need to clone the repo first:
#     # !git clone <repo_url>
#     sys.exit(1)

# # ─────────────────────────────────────────────
# # Reproducibility
# # ─────────────────────────────────────────────
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# random.seed(SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(SEED)

# # ─────────────────────────────────────────────
# # Device & Directory Safety
# # ─────────────────────────────────────────────
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

# # Create directories if they don't exist (Critical for Colab/Kaggle)
# os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
# os.makedirs(os.path.join(BASE_DIR, 'samples'), exist_ok=True)

# # ─────────────────────────────────────────────
# # Model + Prior
# # ─────────────────────────────────────────────
# # Ensure model is on the correct device immediately
# model = NICE.from_preset(DATASET).to(device)

# if DATASET in ('cifar10', 'svhn', 'mnist'):
#     prior = StandardLogistic()
# else:
#     prior = StandardNormal()

# # ─────────────────────────────────────────────
# # Load checkpoint
# # ─────────────────────────────────────────────
# if not os.path.exists(CHECKPOINT):
#     print(f"⚠️ Checkpoint NOT found at {CHECKPOINT}")
#     print("Please upload your checkpoint or check the path.")
# else:
#     # Pass device to load_checkpoint if your util supports it, 
#     # otherwise model.to(device) handled it.
#     load_checkpoint(CHECKPOINT, model, optimizer=None)
#     model.eval()

#     # ─────────────────────────────────────────────
#     # Run evaluation
#     # ─────────────────────────────────────────────
#     print("\nEvaluating model...")
#     # Wrap in try-except in case of memory issues on smaller instances
#     try:
#         avg_ll, bpd = compute_test_loglikelihood(
#             model,
#             prior,
#             DATASET
#         )
        
#         print("\nGenerating samples...")
#         generate_samples(
#             model,
#             prior,
#             DATASET,
#             n_samples=N_SAMPLES
#         )

#         # ─────────────────────────────────────────────
#         # Summary
#         # ─────────────────────────────────────────────
#         print("\n" + "="*30)
#         print("FINAL SUMMARY")
#         print("="*30)
#         print(f"Dataset:        {DATASET}")
#         print(f"Log-likelihood: {avg_ll:.2f} nats")
#         print(f"BPD:            {bpd:.4f}")
#         print(f"Samples saved to: {os.path.join(BASE_DIR, 'samples')}")

#     except RuntimeError as e:
#         if "out of memory" in str(e):
#             print("Error: Out of VRAM. Try reducing batch size in evaluate.py")
#         else:
#             raise e

import torch
import os
import random
import numpy as np
import sys

# ─────────────────────────────────────────────
# Environment Detection & Path Setup
# ─────────────────────────────────────────────
def detect_environment():
    if 'COLAB_GPU' in os.environ:
        return 'colab'
    elif 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        return 'kaggle'
    return 'pc'

ENV = detect_environment()
print(f"Running on: {ENV.upper()}")

# Set base directory for saving outputs (like generated samples)
if ENV == 'kaggle':
    BASE_DIR = '/kaggle/working'
elif ENV == 'colab':
    BASE_DIR = '/content'
else:
    BASE_DIR = '.'

# ─────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────
try:
    from models import NICE
    from loss import StandardNormal, StandardLogistic
    from evaluate import compute_test_loglikelihood, generate_samples
    # Import all your robust checkpoint functions from utils!
    from train import load_checkpoint, get_checkpoint_dir, get_resume_checkpoint
except ImportError as e:
    print(f"Import Error: {e}")
    print("Error: Ensure 'models', 'loss', 'utils', and 'evaluate' are in the current directory.")
    sys.exit(1)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DATASET    = 'mnist'
N_SAMPLES  = 400
SEED       = 42

# Automatically find the right directory and the latest checkpoint
CHECKPOINT_DIR = get_checkpoint_dir()
CHECKPOINT     = get_resume_checkpoint(CHECKPOINT_DIR, dataset=DATASET, prefer="best")

# 💡 NOTE: If you strictly want to run epoch 1500 regardless of later epochs, 
# comment out the line above and uncomment the line below:
# CHECKPOINT = os.path.join(CHECKPOINT_DIR, 'ckpt_mnist_1500.pt')

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ─────────────────────────────────────────────
# Device & Directory Safety
# ─────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create directories if they don't exist
SAMPLES_DIR = os.path.join(BASE_DIR, 'samples')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Model + Prior
# ─────────────────────────────────────────────
model = NICE.from_preset(DATASET).to(device)

if DATASET in ('cifar10', 'svhn', 'mnist'):
    prior = StandardLogistic()
else:
    prior = StandardNormal()

# ─────────────────────────────────────────────
# Load checkpoint & Evaluate
# ─────────────────────────────────────────────
if CHECKPOINT is None or not os.path.exists(CHECKPOINT):
    print(f"\n⚠️ Error: Checkpoint NOT found in {CHECKPOINT_DIR}")
    print("Please check your training run or upload your checkpoints.")
    sys.exit(1)
else:
    print(f"\nLoading checkpoint from: {CHECKPOINT}")
    load_checkpoint(CHECKPOINT, model, optimizer=None)
    model.eval()

    # ─────────────────────────────────────────────
    # Run evaluation
    # ─────────────────────────────────────────────
    print("\nEvaluating model...")
    try:
        avg_ll, bpd = compute_test_loglikelihood(
            model,
            prior,
            DATASET
        )
        
        print("\nGenerating samples...")
        # Note: Depending on how generate_samples is written, 
        # ensure it saves to SAMPLES_DIR internally.
        generate_samples(
            model,
            prior,
            DATASET,
            n_samples=N_SAMPLES
        )

        # ─────────────────────────────────────────────
        # Summary
        # ─────────────────────────────────────────────
        print("\n" + "="*30)
        print("FINAL SUMMARY")
        print("="*30)
        print(f"Dataset:        {DATASET}")
        print(f"Log-likelihood: {avg_ll:.2f} nats")
        print(f"BPD:            {bpd:.4f}")
        print(f"Samples saved to: {SAMPLES_DIR}")

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("Error: Out of VRAM. Try reducing batch size in evaluate.py")
        else:
            raise e
