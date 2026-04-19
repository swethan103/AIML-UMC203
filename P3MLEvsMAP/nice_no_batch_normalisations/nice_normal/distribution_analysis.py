import re
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from models import NICE, Glow
from utils import get_dataloader

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
    """
    z: (batch_size, dim)
    Flattens z to 2D if needed.
    """
    device = z.device
    z = z.reshape(z.size(0), -1)           # flatten to (B, D)
    dim = z.size(1)

    z_ref = torch.randn_like(z)
    swd = 0.0

    for _ in range(n_projections):
        direction = torch.randn(dim, device=device)
        direction = F.normalize(direction, dim=0)

        proj_z   = z   @ direction
        proj_ref = z_ref @ direction

        swd += wasserstein_1d(proj_z, proj_ref)

    return swd / n_projections


# =========================================================
# 2. MMD (Maximum Mean Discrepancy)
# =========================================================

def gaussian_kernel(x, y, sigma=1.0):
    x = x.unsqueeze(1)          # (N, 1, D)
    y = y.unsqueeze(0)          # (1, M, D)
    return torch.exp(-((x - y) ** 2).sum(2) / (2 * sigma ** 2))


def compute_mmd(x, y, sigma=1.0):
    """x, y: (batch_size, dim)"""
    Kxx = gaussian_kernel(x, x, sigma)
    Kyy = gaussian_kernel(y, y, sigma)
    Kxy = gaussian_kernel(x, y, sigma)
    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()


def mmd_to_gaussian(z, sigma=1.0):
    z = z.reshape(z.size(0), -1)           # flatten to (B, D)
    z_ref = torch.randn_like(z)
    return compute_mmd(z, z_ref, sigma)


# =========================================================
# 3. GENERIC HOOK-BASED INTERMEDIATE EXTRACTOR
#    Works for NICE, RealNVP, Glow — any nn.Module composed
#    of sub-layers, regardless of internal naming.
# =========================================================

def get_layer_container(model, model_type: str):
    """
    Returns the iterable of coupling/flow layers for each architecture.
    Tries common attribute names so it works with most implementations.
    """
    model_type = model_type.lower()

    # -- NICE --
    if model_type == "nice":
        for attr in ("layers", "coupling_layers", "flow_layers", "transforms"):
            if hasattr(model, attr):
                layers = list(getattr(model, attr))
                # Append scaling layer if present
                if hasattr(model, "scaling_layer"):
                    layers.append(model.scaling_layer)
                return layers

    # -- RealNVP --
    elif model_type == "realnvp":
        for attr in ("flows", "layers", "coupling_layers", "flow_layers", "transforms"):
            if hasattr(model, attr):
                return list(getattr(model, attr))

    # -- Glow --
    elif model_type == "glow":
        for attr in ("flows", "layers", "blocks", "flow_layers", "transforms"):
            if hasattr(model, attr):
                return list(getattr(model, attr))

    raise AttributeError(
        f"Cannot find layer container for model_type='{model_type}'. "
        f"Available attributes: {[n for n, _ in model.named_children()]}"
    )


def forward_with_intermediates(model, x, model_type: str):
    """
    Runs the model forward and collects the output after every
    top-level flow layer via forward hooks.

    Returns
    -------
    zs : list of Tensors, one per layer (including any scaling/final layer)
    """
    layers = get_layer_container(model, model_type)

    intermediates = []
    hooks = []

    def make_hook():
        def hook(module, input, output):
            # output may be a tuple (z, log_det); grab the tensor part
            if isinstance(output, (tuple, list)):
                intermediates.append(output[0].detach())
            else:
                intermediates.append(output.detach())
        return hook

    for layer in layers:
        hooks.append(layer.register_forward_hook(make_hook()))

    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in hooks:
            h.remove()

    return intermediates


# =========================================================
# 4. COMPUTE DISTANCE ACROSS LAYERS
# =========================================================

def compute_distances_across_layers(
    model,
    data_loader,
    device,
    metric="wasserstein",
    n_projections=100,
    model_type="nice",
):
    model.eval()
    layer_distances = []

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)

            zs = forward_with_intermediates(model, x, model_type)

            for i, z in enumerate(zs):
                if metric == "wasserstein":
                    d = sliced_wasserstein(z, n_projections)
                elif metric == "mmd":
                    d = mmd_to_gaussian(z)
                else:
                    raise ValueError(f"Unknown metric: '{metric}'. Use 'wasserstein' or 'mmd'.")

                if len(layer_distances) <= i:
                    layer_distances.append([])
                layer_distances[i].append(d.item())

    # Average over batches
    layer_distances = [np.mean(layer) for layer in layer_distances]
    return layer_distances


# =========================================================
# 5. TRACK DURING TRAINING (EPOCH-WISE)
# =========================================================

def track_during_training(
    model,
    data_loader,
    device,
    selected_layers=None,
    metric="wasserstein",
    n_projections=20,
    model_type="nice",
):
    """
    Call once per epoch checkpoint to record layer distances.

    Returns
    -------
    dict  {layer_index: distance_value}   (single-batch snapshot)
    """
    model.eval()

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            zs = forward_with_intermediates(model, x, model_type)
            break   # one batch is enough

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
# 6. PLOTTING FUNCTIONS
# =========================================================

def plot_layer_vs_distance(distances_dict, title="Layer vs Distance to Gaussian"):
    """
    distances_dict: {"NICE": [...], "RealNVP": [...], "Glow": [...]}
    """
    plt.figure()
    for name, distances in distances_dict.items():
        plt.plot(distances, marker='o', label=name)

    plt.xlabel("Layer")
    plt.ylabel("Distance to Gaussian")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("layer_vs_distance.png")
    plt.show()


def plot_epoch_vs_distance(epoch_results, title="Epoch vs Distance"):
    """
    epoch_results: {layer_idx: [values_over_epochs]}
    """
    plt.figure()
    for layer, values in epoch_results.items():
        plt.plot(values, marker='o', label=f"Layer {layer}")

    plt.xlabel("Epoch checkpoint")
    plt.ylabel("Distance to Gaussian")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("epoch_vs_distance.png")
    plt.show()


def plot_pca_evolution(model, data_loader, device, model_type="nice", layers_to_plot=None):
    model.eval()

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            zs = forward_with_intermediates(model, x, model_type)
            break

    if layers_to_plot is None:
        layers_to_plot = list(range(len(zs)))

    for l in layers_to_plot:
        if l >= len(zs):
            print(f"Layer {l} not available (model has {len(zs)} layers). Skipping.")
            continue

        z = zs[l].cpu().numpy()
        z = z.reshape(len(z), -1)          # flatten

        z_ref = np.random.randn(*z.shape)

        combined = np.concatenate([z, z_ref], axis=0)

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(combined)

        z_pca   = reduced[:len(z)]
        ref_pca = reduced[len(z):]

        plt.figure()
        plt.scatter(z_pca[:, 0],   z_pca[:, 1],   alpha=0.5, label="Model")
        plt.scatter(ref_pca[:, 0], ref_pca[:, 1], alpha=0.5, label="Gaussian")
        plt.title(f"PCA at Layer {l}")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"pca_layer_{l}.png")
        plt.show()


# =========================================================
# 7. MAIN
# =========================================================

if __name__ == "__main__":
    from models import Glow
    from utils import get_dataloader

    # --------------------------------------------------
    # CONFIG
    # --------------------------------------------------
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATASET    = "mnist"
    BATCH_SIZE = 512
    MODEL_TYPE = "glow"         # "nice" | "realnvp" | "glow"

    CHECKPOINT_PATH = f"checkpoints_GLOW/model_008001.pt"

    METRIC = "wasserstein"      # "wasserstein" | "mmd"

    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------
    print("Loading test data...")
    test_loader = get_dataloader(
    DATASET,
    BATCH_SIZE,
    split='test',
    flatten_input=False
)
    # --------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------
    print(f"Loading {MODEL_TYPE.upper()} model...")
    # if MODEL_TYPE == "nice":
    #     model = NICE.from_preset(DATASET).to(DEVICE)
    # elif MODEL_TYPE == "glow":
    #     if DATASET == "mnist":
    #         model = Glow(
    #             in_channel=1,
    #             n_flow=32,
    #             n_block=3,
    #             affine=True,
    #             conv_lu=True
    #         ).to(DEVICE)
    # checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    # model.load_state_dict(
    #     checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    # )
    # model.eval()
    if MODEL_TYPE == "nice":
        model = NICE.from_preset(DATASET).to(DEVICE)

    elif MODEL_TYPE == "glow":
        if DATASET == "mnist":
            model = Glow(
                in_channel=1,
                n_flow=16,
                n_block=3,
                affine=False,
                conv_lu=True
            ).to(DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    # Handle DataParallel-saved checkpoints automatically
    if all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    model.eval()
    # --------------------------------------------------
    # LAYER-WISE DISTANCES
    # --------------------------------------------------
    print(f"Computing {METRIC} distances across layers...")
    # distances = compute_distances_across_layers(
    #     model=model,
    #     data_loader=test_loader,
    #     device=DEVICE,
    #     metric=METRIC,
    #     n_projections=100,
    #     model_type=MODEL_TYPE,
    # )

    # print("\nLayer-wise distances:")
    # for i, d in enumerate(distances):
    #     print(f"  Layer {i}: {d:.6f}")

    # plot_layer_vs_distance({"NICE": distances})

    # --------------------------------------------------
    # EPOCH-WISE TRACKING  (dummy example — replace with
    # real calls to track_during_training() during training)
    # --------------------------------------------------
    # epoch_results = {
    #     0: [0.95, 0.82, 0.70, 0.60],
    #     2: [0.88, 0.65, 0.42, 0.28],
    #     4: [0.80, 0.50, 0.25, 0.12],
    # }
    # plot_epoch_vs_distance(epoch_results, title="Epoch vs Distance")

    # --------------------------------------------------
    # PCA ANALYSIS
    # --------------------------------------------------
    print("\nRunning PCA evolution...")
    plot_pca_evolution(
        model=model,
        data_loader=test_loader,
        device=DEVICE,
        model_type=MODEL_TYPE,
        layers_to_plot=[0, 2, 4],
    )

    print("\nAnalysis complete.")

    