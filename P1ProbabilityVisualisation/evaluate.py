# # evaluate.py

# import torch
# import torchvision
# import matplotlib.pyplot as plt
# import os
# from models import NICE
# from loss   import StandardNormal, StandardLogistic, nll_loss
# from utils  import get_dataloader, load_checkpoint

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def compute_test_loglikelihood(model, prior, dataset_name, batch_size=200):
#     """
#     Runs the model on the test set and reports average log-likelihood in nats.
#     This is the number the paper reports in Table 1.
#     """
#     model.eval()

#     # Get the test split (not train)
#     from utils import get_dataloader_test
#     loader = get_dataloader_test(dataset_name, batch_size)

#     total_log_prob = 0.0
#     total_samples  = 0

#     with torch.no_grad():
#         for x, _ in loader:
#             x = x.view(x.size(0), -1).to(device)

#             # No dequantization at test time — we evaluate on clean data
#             z      = model.encode(x)
#             log_pz = prior.log_prob(z)
#             ldj    = model.log_det_jacobian()
#             log_px = (log_pz + ldj)          # shape: (batch,)

#             total_log_prob += log_px.sum().item()
#             total_samples  += x.size(0)

#     avg_log_likelihood = total_log_prob / total_samples
#     print(f"Test log-likelihood: {avg_log_likelihood:.2f} nats")
#     print(f"(Paper reports: MNIST=-1454, CIFAR=-5371, SVHN=-5853, TFD=-4483)")
#     return avg_log_likelihood


# def generate_samples(model, prior, dataset_name, n_samples=100):
#     """
#     Generates new images by:
#       1. Sampling z from the prior (random noise)
#       2. Running z backwards through the model (decode)
#       3. Saving as an image grid
#     """
#     model.eval()

#     nvis = NICE.PRESETS[dataset_name]['nvis']

#     with torch.no_grad():
#         # Step 1: sample from prior
#         z = prior.sample(n_samples, nvis).to(device)

#         # Step 2: decode z → x
#         x = model.decode(z)
#     x = x.cpu()

#     # Step 3: reshape flat vectors back to images for display
#     if dataset_name == 'mnist':
#         x = x.view(n_samples, 1, 28, 28)
#         x = x.clamp(0, 1)
#     elif dataset_name in ('cifar10', 'svhn'):
#         x = x.view(n_samples, 3, 32, 32)
#         x = (x * 0.5 + 0.5).clamp(0, 1)  # undo the [-1,1] normalization
#     elif dataset_name == 'tfd':
#         x = x.view(n_samples, 1, 48, 48)
#         x = x.clamp(0, 1)

#     # Save as a grid image
#     os.makedirs('./outputs', exist_ok=True)
#     grid_path = f'./outputs/samples_{dataset_name}.png'
#     torchvision.utils.save_image(x, grid_path, nrow=10, padding=2)
#     print(f"Saved {n_samples} samples → {grid_path}")

#     return x

# evaluate.py

# import torch
# import torchvision
# import os
# from models import NICE
# from loss   import StandardNormal, StandardLogistic, nll_loss
# from utils  import get_dataloader_test, load_checkpoint, dequantize

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# # ─────────────────────────────────────────────
# # Test Log-Likelihood
# # ─────────────────────────────────────────────
# def compute_test_loglikelihood(model, prior, dataset_name, batch_size=200):
#     """
#     Computes average log-likelihood on test set.
#     Reports:
#         - nats
#         - bits per dimension (bpd)
#     """
#     model.eval()

#     loader = get_dataloader_test(dataset_name, batch_size)

#     total_log_prob = 0.0
#     total_samples  = 0

#     nvis = NICE.PRESETS[dataset_name]['nvis']

#     with torch.no_grad():
#         for x, _ in loader:
#             x = x.to(device)

#             # IMPORTANT: dequantization (same as training)
#             x = dequantize(x)

#             # Forward
#             z      = model.encode(x)
#             log_pz = prior.log_prob(z)
#             ldj    = model.log_det_jacobian()
#             log_px = log_pz + ldj

#             total_log_prob += log_px.sum().item()
#             total_samples  += x.size(0)

#     avg_log_likelihood = total_log_prob / total_samples

#     # Convert to bits per dimension
#     bpd = -avg_log_likelihood / (nvis * torch.log(torch.tensor(2.0)).item())

#     print(f"\nTest Results for {dataset_name.upper()}")
#     print(f"Log-likelihood: {avg_log_likelihood:.2f} nats")
#     print(f"Bits per dimension (bpd): {bpd:.4f}")

#     return avg_log_likelihood, bpd


# # ─────────────────────────────────────────────
# # Sample Generation
# # ─────────────────────────────────────────────
# def generate_samples(model, prior, dataset_name, n_samples=100):
#     """
#     Generates new samples from the trained flow model.
#     """
#     model.eval()

#     nvis = NICE.PRESETS[dataset_name]['nvis']

#     with torch.no_grad():
#         # Sample latent
#         z = prior.sample(n_samples, nvis).to(device)

#         # Decode
#         x = model.decode(z)

#     x = x.cpu()

#     # Reshape to images
#     if dataset_name == 'mnist':
#         x = x.view(n_samples, 1, 28, 28)

#     elif dataset_name in ('cifar10', 'svhn'):
#         x = x.view(n_samples, 3, 32, 32)

#     elif dataset_name == 'tfd':
#         x = x.view(n_samples, 1, 48, 48)

#     # Clamp to valid image range
#     x = x.clamp(0, 1)

#     # Save grid
#     os.makedirs('./outputs', exist_ok=True)
#     grid_path = f'./outputs/samples_{dataset_name}.png'

#     torchvision.utils.save_image(
#         x,
#         grid_path,
#         nrow=10,
#         padding=2
#     )

#     print(f"Saved {n_samples} samples → {grid_path}")

#     return x

import os
import math
import torch
import torchvision

from models import NICE
from loss   import StandardNormal, StandardLogistic
from utils  import get_dataloader_test, dequantize
from distribution_analysis import forward_with_intermediates, forward_with_intermediates_nice


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
        return './checkpoints_Gaussian_unconstrained'


def get_best_checkpoint(checkpoint_dir, dataset):
    path = os.path.join(checkpoint_dir, f'best_mnist.pt')
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


# ─────────────────────────────────────────────
# Test Log-Likelihood
# ─────────────────────────────────────────────
def compute_test_loglikelihood(model, prior, dataset_name, batch_size=200):
    model.eval()

    loader = get_dataloader_test(dataset_name, batch_size)
    nvis = NICE.PRESETS[dataset_name]['nvis']

    total_log_prob = 0.0
    total_samples  = 0

    with torch.no_grad():
        for x, _ in loader:
            # 🔥 FIX: reshape + dequantize
            x = x.view(x.size(0), -1)
            x = dequantize(x).to(device)

            # 🔥 safer forward
            z = model(x)
            log_pz = prior.log_prob(z)
            ldj = model.log_det_jacobian()

            log_px = log_pz + ldj

            total_log_prob += log_px.sum().item()
            total_samples  += x.size(0)

    avg_log_likelihood = total_log_prob / total_samples
    bpd = -avg_log_likelihood / (nvis * math.log(2))

    print(f"\n📊 Test Results for {dataset_name.upper()}")
    print(f"Log-likelihood: {avg_log_likelihood:.2f} nats")
    print(f"Bits per dimension (bpd): {bpd:.4f}")

    return avg_log_likelihood, bpd


# ─────────────────────────────────────────────
# Sample Generation
# ─────────────────────────────────────────────
def generate_samples(model, prior, dataset_name, n_samples=100):
    model.eval()

    nvis = NICE.PRESETS[dataset_name]['nvis']

    with torch.no_grad():
        z = (prior.sample(n_samples, nvis)*0.7).to(device)    #added 0.7 to reduce temperature 
        x = model.decode(z)
        
    x = x.cpu()

    # reshape
    if dataset_name == 'mnist':
        x = x.view(n_samples, 1, 28, 28)
    elif dataset_name in ('cifar10', 'svhn'):
        x = x.view(n_samples, 3, 32, 32)
    elif dataset_name == 'tfd':
        x = x.view(n_samples, 1, 48, 48)

    x = x.clamp(0, 1)
    #x = (x - x.min())/(x.max() - x.min())
    x[x < 0.10] = 0

    # output directory (adaptive)
    if os.path.exists('/kaggle/working'):
        out_dir = '/kaggle/working/outputs'
    elif os.path.exists('/content/drive'):
        out_dir = f'/content/drive/MyDrive/NICE_Outputs'
    else:
        out_dir = './outputs'

    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, f'samples_{dataset_name}.png')

    torchvision.utils.save_image(x, path, nrow=10, padding=2)

    print(f"Saved samples → {path}")

    return x
def extract_intermediate_latents(model, dataset_name, batch_size=200, max_batches=None):
    """
    Extracts layer-wise latent representations z^(l) across dataset.

    Returns:
        all_zs: list of tensors
            all_zs[i] → shape [N, dim] for layer i
    """
    model.eval()

    loader = get_dataloader_test(dataset_name, batch_size)

    all_zs = []

    with torch.no_grad():
        for b, (x, _) in enumerate(loader):

            # optional limit (for speed)
            if max_batches is not None and b >= max_batches:
                break

            # preprocess
            x = x.view(x.size(0), -1)
            x = dequantize(x).to(device)

            # 🔥 THIS IS THE KEY CHANGE
            z = model(x)
            zs = forward_with_intermediates(model, x, "nice")
            # initialize container
            if len(all_zs) == 0:
                all_zs = [[] for _ in range(len(zs))]

            # store per-layer outputs
            for i, layer_z in enumerate(zs):
                all_zs[i].append(layer_z.cpu())

    # concatenate across batches
    all_zs = [torch.cat(z_list, dim=0) for z_list in all_zs]

    print("\n✅ Extracted intermediate representations:")
    for i, z in enumerate(all_zs):
        print(f"Layer {i}: {z.shape}")

    return all_zs


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":

    DATASET = 'mnist'

    # Prior
    PRIOR = StandardLogistic() if DATASET in ('cifar10', 'svhn') else StandardNormal()

    # Load model
    model = load_model(DATASET)

    # Evaluate
    compute_test_loglikelihood(model, PRIOR, DATASET)

    # Generate samples
    generate_samples(model, PRIOR, DATASET)
    
    # Intermediate Extraction 
    # all_zs = extract_intermediate_latents(
    #     model,
    #     DATASET,
    #     batch_size=200,
    #     max_batches=10   # keep small first!
    # )

    # save_path = f"intermediate_latents_{DATASET}.pt"
    # torch.save(all_zs, save_path)
    # print(f"Saved intermediate latents → {save_path}")
    # print("Saved intermediate latents → intermediate_latents.pt")

    # import matplotlib.pyplot as plt

    # z_final = all_zs[-1]

    # # sample from logistic prior
    # # logistic_samples = torch.distributions.Logistic(0, 1).sample(z_final.shape)   logistic isnt defined in torch.distributions
    # if isinstance(PRIOR, StandardLogistic):
    #     u = torch.rand_like(z_final)
    #     prior_samples = torch.log(u) - torch.log(1 - u)
    # else:
    #     prior_samples = torch.randn_like(z_final)
    # # plt.hist(z_final.flatten().numpy(), bins=100, density=True, alpha=0.5, label="Model")
    # # plt.hist(logistic_samples.flatten().numpy(), bins=100, density=True, alpha=0.5, label="Logistic Prior")
    # # plt.xlim(-10, 10)
    # # plt.legend()
    # # plt.title("Final latent vs Logistic prior")
    # # plt.savefig("Final_Latent_vs_Logistic_Prior.png")
    # import seaborn as sns

    # sns.kdeplot(z_final.flatten().numpy(), label="Model")
    # sns.kdeplot(prior_samples.flatten().numpy(), label=f"{PRIOR} Prior")

    # plt.xlim(-10, 10)
    # plt.legend()
    # plt.title(f"Final latent vs {PRIOR} prior")
    # plt.savefig(f"Final_Latent_vs_{PRIOR}_Prior.png")
    # plt.show()