# ------------------------------------------------------------------------------------------------------------------------------------------
# Model_type is RealNVP
# ------------------------------------------------------------------------------------------------------------------------------------------

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from models.real_nvp.coupling_layer import CouplingLayer, MaskType
from models import RealNVP
from util import squeeze_2x2

def convert_parametrized_weightnorm_to_legacy(state_dict):
    new_state = {}

    for k, v in state_dict.items():
        if "parametrizations.weight.original0" in k:
            new_k = k.replace(
                "parametrizations.weight.original0",
                "weight_g"
            )
        elif "parametrizations.weight.original1" in k:
            new_k = k.replace(
                "parametrizations.weight.original1",
                "weight_v"
            )
        else:
            new_k = k

        new_state[new_k] = v

    return new_state

def preprocess_for_realnvp(x, model, device):
    """
    Preprocess input for RealNVP, matching the training pipeline exactly.

    Steps:
        1. Pad 28x28 -> 32x32 (matches transforms.Pad(2) in train.py)
        2. Dequantize + logit transform via model's own _pre_process

    Args:
        x (torch.Tensor): Raw MNIST tensor in [0, 1], shape (B, 1, 28, 28)
                          or already padded (B, 1, 32, 32).
        model (RealNVP): Unwrapped RealNVP model.
        device: torch.device.

    Returns:
        y (torch.Tensor): Preprocessed tensor in logit space, shape (B, 1, 32, 32).
    """
    x = x.to(device)

    # Pad if needed (28x28 -> 32x32), matching transforms.Pad(2) in train.py
    if x.shape[-1] == 28:
        x = F.pad(x, (2, 2, 2, 2), mode='constant', value=0)

    # Use model's own _pre_process (dequantize + logit transform)
    unwrapped = model.module if isinstance(model, torch.nn.DataParallel) else model
    with torch.no_grad():
        y, _ = unwrapped._pre_process(x)

    return y
# =========================================================
# 0. LOG PARSING
# =========================================================

def parse_losses_from_log(file_path):
    train_losses = []
    val_losses = []

    with open(file_path, "r") as f:
        for line in f:
            if "train:" in line and "val:" in line:
                train_match = re.search(r"train:\s*([-+]?\d*\.?\d+)", line)
                val_match   = re.search(r"val:\s*([-+]?\d*\.?\d+)", line)

                if train_match and val_match:
                    train_losses.append(float(train_match.group(1)))
                    val_losses.append(float(val_match.group(1)))

    return train_losses, val_losses


# =========================================================
# 1. WASSERSTEIN (SLICED)
# =========================================================

def wasserstein_1d(x, y):
    x_sorted, _ = torch.sort(x)
    y_sorted, _ = torch.sort(y)
    return torch.mean(torch.abs(x_sorted - y_sorted))


def sliced_wasserstein(z, n_projections=100):
    z = z.reshape(z.size(0), -1)
    dim = z.size(1)

    z_ref = torch.randn_like(z)

    swd = 0.0
    for _ in range(n_projections):
        direction = torch.randn(dim, device=z.device)
        direction = F.normalize(direction, dim=0)

        proj_z   = z @ direction
        proj_ref = z_ref @ direction

        swd += wasserstein_1d(proj_z, proj_ref)

    return swd / n_projections


# =========================================================
# 2. MMD
# =========================================================

def gaussian_kernel(x, y, sigma=1.0):
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    return torch.exp(-((x - y) ** 2).sum(2) / (2 * sigma ** 2))


def compute_mmd(x, y, sigma=1.0):
    Kxx = gaussian_kernel(x, x, sigma)
    Kyy = gaussian_kernel(y, y, sigma)
    Kxy = gaussian_kernel(x, y, sigma)

    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()


def mmd_to_gaussian(z, sigma=1.0):
    z = z.reshape(z.size(0), -1)
    z_ref = torch.randn_like(z)
    return compute_mmd(z, z_ref, sigma)

# =========================================================
# 2B. SKEWNESS / KURTOSIS
# =========================================================

def skewness_metric(z):
    """
    Average absolute skewness across latent dimensions.
    Lower is better (Gaussian = 0).

    z: (batch, ...)
    """
    z = z.reshape(z.size(0), -1)

    mean = z.mean(dim=0)
    std = z.std(dim=0, unbiased=True) + 1e-8

    skew = torch.mean(((z - mean) / std) ** 3, dim=0)

    return torch.mean(torch.abs(skew))


def kurtosis_metric(z):
    """
    Average absolute excess from Gaussian kurtosis.
    Lower is better (Gaussian kurtosis = 3).

    z: (batch, ...)
    """
    z = z.reshape(z.size(0), -1)

    mean = z.mean(dim=0)
    std = z.std(dim=0, unbiased=True) + 1e-8

    kurt = torch.mean(((z - mean) / std) ** 4, dim=0)

    return torch.mean(torch.abs(kurt - 3.0))

def covariance_error_metric(z, normalize=True):
    """
    Frobenius norm ||Cov(z) - I||_F

    Lower is better (Gaussian/whitened latent => 0)

    z: (batch, ...)
    """
    z = z.reshape(z.size(0), -1)

    N, D = z.shape

    # Center
    z_centered = z - z.mean(dim=0, keepdim=True)

    # Covariance
    cov = (z_centered.T @ z_centered) / (N - 1)

    I = torch.eye(D, device=z.device)

    cov_err = torch.norm(cov - I, p='fro')

    if normalize:
        cov_err = cov_err / D

    return cov_err
# =========================================================
# 3. REALNVP LATENT EVOLUTION EXTRACTOR
# =========================================================

def realnvp_forward_with_intermediates(model, x):
    """
    Returns cumulative latent representations after each scale block of RealNVP.

    RealNVP splits off part of the latent at each scale (except the last),
    so we collect those split-off chunks and accumulate them across scales,
    mirroring how Glow emits z_new at each block.

    Each entry in `intermediates` is the concatenation of all latents
    emitted so far (flattened), giving a view of how Gaussianity evolves
    as we go deeper through the multi-scale flow.

    Args:
        model (RealNVP): The unwrapped RealNVP model.
        x (torch.Tensor): Pre-processed input in logit space, shape (B, C, H, W).

    Returns:
        intermediates (list of torch.Tensor): One tensor per scale, each of
            shape (B, cumulative_latent_dim).
    """
    model.eval()

    # We hook into _RealNVP's forward manually to capture scale-wise latents.
    # This mirrors the recursive structure in _RealNVP.forward (reverse=False).

    cumulative_z = []
    intermediates = []

    def extract_from_block(block, x, sldj):
        """
        Recursively traverse _RealNVP scales, collecting split-off latents
        exactly as the forward pass does, and recording cumulative state.
        """
        # In-couplings (checkerboard)
        for coupling in block.in_couplings:
            x, sldj = coupling(x, sldj, reverse=False)

        if not block.is_last_block:
            # Squeeze -> channel-wise couplings
            x = squeeze_2x2(x, reverse=False)
            for coupling in block.out_couplings:
                x, sldj = coupling(x, sldj, reverse=False)
            x = squeeze_2x2(x, reverse=True)

            # Re-squeeze -> split: the split-off half is the latent for this scale
            x = squeeze_2x2(x, reverse=False, alt_order=True)
            x, x_split = x.chunk(2, dim=1)

            # Record this scale's latent contribution cumulatively
            cumulative_z.append(x_split.flatten(1))
            intermediates.append(torch.cat(cumulative_z, dim=1).detach().clone())

            # Recurse into the next scale
            x, sldj = extract_from_block(block.next_block, x, sldj)

            x = torch.cat((x, x_split), dim=1)
            x = squeeze_2x2(x, reverse=True, alt_order=True)
        else:
            # Last block: the entire remaining x is the final latent
            cumulative_z.append(x.flatten(1))
            intermediates.append(torch.cat(cumulative_z, dim=1).detach().clone())

        return x, sldj

    with torch.no_grad():
        # RealNVP._pre_process is already applied upstream (caller's responsibility).
        # We access the inner _RealNVP flows directly.
        flows = model.flows if not isinstance(model, torch.nn.DataParallel) \
                else model.module.flows

        # We need sldj to pass through couplings (they update it),
        # but we don't use its final value here — only the latents matter.
        sldj = torch.zeros(x.size(0), device=x.device)

        extract_from_block(flows, x, sldj)

    return intermediates


# def preprocess_for_realnvp(model, x):
#     """
#     Apply RealNVP's own pre-processing (dequantize + logit transform).

#     Args:
#         model (RealNVP or DataParallel-wrapped RealNVP): The model.
#         x (torch.Tensor): Raw input in [0, 1].

#     Returns:
#         y (torch.Tensor): Pre-processed tensor in logit space.
#     """
#     unwrapped = model.module if isinstance(model, torch.nn.DataParallel) else model
#     with torch.no_grad():
#         y, _ = unwrapped._pre_process(x)
#     return y


# =========================================================
# 4. DISTANCE ACROSS REALNVP SCALES
# =========================================================

def compute_distances_across_layers(
    model,
    data_loader,
    device,
    metric="wasserstein",
    n_projections=100,
):
    """
    Compute the average distance from Gaussian for each cumulative latent
    across RealNVP's scale blocks.

    Args:
        model: RealNVP model (wrapped or unwrapped).
        data_loader: DataLoader yielding (x, label) pairs with x in [0, 1].
        device: torch.device.
        metric: "wasserstein" or "mmd".
        n_projections: Number of random projections for sliced Wasserstein.

    Returns:
        list of float: Mean distance per scale.
    """
    model.eval()

    layer_distances = []

    unwrapped = model.module if isinstance(model, torch.nn.DataParallel) else model

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            x = preprocess_for_realnvp(x,unwrapped,DEVICE)
            zs = realnvp_forward_with_intermediates(unwrapped, x)

            for i, z in enumerate(zs):

                if metric == "wasserstein":
                    d = sliced_wasserstein(z, n_projections)
                elif metric == "mmd":
                    d = mmd_to_gaussian(z)
                elif metric == "skewness":
                    d = skewness_metric(z)
                elif metric == "kurtosis":
                    d = kurtosis_metric(z)
                elif metric == "covariance":
                    d = covariance_error_metric(z)
                else:
                    raise ValueError("Metric must be 'wasserstein' or 'mmd'")

                if len(layer_distances) <= i:
                    layer_distances.append([])

                layer_distances[i].append(d.item())

    return [np.mean(layer) for layer in layer_distances]


# =========================================================
# 5. EPOCH-WISE TRACKING
# =========================================================

def track_during_training(
    model,
    data_loader,
    device,
    selected_layers=None,
    metric="wasserstein",
    n_projections=20,
):
    """
    Snapshot the per-scale latent distances for one batch of data.
    Intended to be called once per epoch checkpoint to track training dynamics.

    Args:
        model: RealNVP model.
        data_loader: DataLoader.
        device: torch.device.
        selected_layers: List of scale indices to evaluate. None = all scales.
        metric: "wasserstein" or "mmd".
        n_projections: Random projections for sliced Wasserstein.

    Returns:
        dict mapping scale index -> distance value.
    """
    model.eval()

    unwrapped = model.module if isinstance(model, torch.nn.DataParallel) else model

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            x = preprocess_for_realnvp(x, unwrapped,DEVICE)
            zs = realnvp_forward_with_intermediates(unwrapped, x)
            break

    if selected_layers is None:
        selected_layers = list(range(len(zs)))

    results = {}

    for l in selected_layers:
        if l >= len(zs):
            continue

        z = zs[l]

        if metric == "wasserstein":
            d = sliced_wasserstein(z, n_projections)
        else:
            d = mmd_to_gaussian(z)

        results[l] = d.item()

    return results


# =========================================================
# 6. PLOTTING
# =========================================================

def plot_layer_vs_distance(distances, metric, title="RealNVP Scale vs Distance to Gaussian"):
    plt.figure()

    plt.plot(distances, marker='o')

    plt.xlabel("RealNVP Scale (cumulative latent)")
    plt.ylabel("Distance to Gaussian")
    plt.title(title)

    plt.grid()
    plt.tight_layout()
    plt.savefig(f"layer_vs_distance_{metric}.png")
    plt.show()


def plot_epoch_vs_distance(epoch_results, title="Epoch vs Distance"):
    plt.figure()

    for layer, values in epoch_results.items():
        plt.plot(values, marker='o', label=f"Scale {layer}")

    plt.xlabel("Epoch Checkpoint")
    plt.ylabel("Distance to Gaussian")
    plt.title(title)

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("epoch_vs_distance.png")
    plt.show()


def plot_pca_evolution(model, data_loader, device, layers_to_plot=None):
    """
    PCA scatter plot comparing the cumulative RealNVP latent at each scale
    against a standard Gaussian reference of the same shape.
    """
    model.eval()

    unwrapped = model.module if isinstance(model, torch.nn.DataParallel) else model

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            x = preprocess_for_realnvp(x, unwrapped, DEVICE)
            zs = realnvp_forward_with_intermediates(unwrapped, x)
            break

    if layers_to_plot is None:
        layers_to_plot = list(range(len(zs)))

    for l in layers_to_plot:

        if l >= len(zs):
            print(f"Scale {l} unavailable. Skipping.")
            continue

        z = zs[l].cpu().numpy()

        z_ref = np.random.randn(*z.shape)

        combined = np.concatenate([z, z_ref], axis=0)

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(combined)

        z_pca   = reduced[:len(z)]
        ref_pca = reduced[len(z):]

        plt.figure()

        plt.scatter(z_pca[:, 0], z_pca[:, 1], alpha=0.5, label="RealNVP Latent")
        plt.scatter(ref_pca[:, 0], ref_pca[:, 1], alpha=0.5, label="Gaussian")

        plt.title(f"PCA at RealNVP Scale {l} (cumulative latent)")

        plt.legend()
        plt.grid()
        plt.tight_layout()

        plt.savefig(f"pca_scale_{l}.png")
        plt.show()



# =========================================================
# 7. MAIN
# =========================================================

if __name__ == "__main__":

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 512
    # Update this path to point to your RealNVP checkpoint.
    CHECKPOINT_PATH = "ckpts/best.pth.tar"

    METRICS = ["covariance"]   # "wasserstein" or "mmd" "wasserstein", "mmd", "kurtosis", "skewness"

    # --- Data ---
    import torchvision
    import torchvision.transforms as transforms
    import torch.utils.data as data

    print("Loading test data...")
    # transform_test = transforms.Compose([transforms.ToTensor()])
    transform_test = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor()
    ])
    testset = torchvision.datasets.MNIST('./data', train=False, download=True,
                                         transform=transform_test)
    test_loader = data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=2)

    # --- Model ---
    print("Loading RealNVP model...")
    # net = RealNVP(num_scales=2, in_channels=1, mid_channels=64, num_blocks=8)
    # net = net.to(DEVICE)
    # # Trigger weight norm initialization before loading checkpoint
    # dummy = torch.zeros(1, 1, 32, 32, device=DEVICE)
    # with torch.no_grad():
    #     net(dummy, reverse=False)

    # checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    # state_dict = checkpoint['net']

    # # Strip DataParallel prefix if present
    # # if all(k.startswith("module.") for k in state_dict.keys()):
    # #     state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    # net.load_state_dict(state_dict)
    # net.eval()
    # net = RealNVP(num_scales=2, in_channels=1, mid_channels=64, num_blocks=8)
    # net = net.to(DEVICE)

    # # Wrap in DataParallel exactly as during training
    # if DEVICE.type == 'cuda':
    #     net = torch.nn.DataParallel(net, [0])

    # # Trigger weight norm initialization
    # dummy = torch.zeros(1, 1, 32, 32, device=DEVICE)
    # with torch.no_grad():
    #     net(dummy, reverse=False)

    # # Load checkpoint directly — no need to strip module. prefix
    # # since the DataParallel-wrapped model expects those keys
    # checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    # net.load_state_dict(checkpoint['net'])
    # net.eval()

    net = RealNVP(
        num_scales=2,
        in_channels=1,
        mid_channels=64,
        num_blocks=8
    ).to(DEVICE)

    # Trigger initialization of all weight norm params
    dummy = torch.zeros(1, 1, 32, 32, device=DEVICE)
    with torch.no_grad():
        net(dummy, reverse=False)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state_dict = checkpoint['net']
    state_dict = convert_parametrized_weightnorm_to_legacy(state_dict)
    # Remove DataParallel prefix from saved checkpoint
    state_dict = {
        k.replace("module.", "", 1): v
        for k, v in state_dict.items()
    }
    print(next(iter(net.state_dict().keys())))
    print(next(iter(checkpoint['net'].keys())))

    print("Checkpoint num params:", len(state_dict))

    model = RealNVP(
        num_scales=2,
        in_channels=1,
        mid_channels=64,
        num_blocks=8
    )
    dummy = torch.zeros(1,1,32,32)
    with torch.no_grad():
        model(dummy, reverse=False)

    print("Current model num params:", len(model.state_dict()))

    for k in list(state_dict.keys())[:50]:
        print("CKPT:", k)

    for k in list(model.state_dict().keys())[:50]:
        print("MODEL:", k)
    # --------------------------------------------------------------------------
    net.load_state_dict(state_dict)

    # Wrap AFTER loading if desired
    if DEVICE.type == "cuda":
        net = torch.nn.DataParallel(net)

    net.eval()
    # --- Distance across scales ---
    for metric in METRICS:
        print(f"\nComputing {metric} distances across RealNVP scales...")
        distances = compute_distances_across_layers(
            model=net,
            data_loader=test_loader,
            device=DEVICE,
            metric=metric,
            n_projections=100,
        )

        print("\nRealNVP Latent Gaussianity Across Scales:")
        for i, d in enumerate(distances):
            print(f"  Scale {i} (cumulative): {d:.6f}")

        plot_layer_vs_distance(distances, metric, title = f"{metric}: RealNVP Scale vs Distance to Gaussian")

    # --- PCA evolution ---
    print("\nRunning PCA evolution...")
    plot_pca_evolution(
        model=net,
        data_loader=test_loader,
        device=DEVICE,
        layers_to_plot=list(range(len(distances))),
    )

    print("\nAnalysis complete.")