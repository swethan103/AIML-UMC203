import torch
import numpy as np

from models import NICE
from utils import get_dataloader_test  # your existing loader
from distribution_analysis import (
    compute_distances_across_layers,
    plot_layer_vs_distance
)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET = "mnist"
BATCH_SIZE = 256      #because during analysis, higher batchsize means more statistical estimation quality

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
loader = get_dataloader_test(DATASET, batch_size=BATCH_SIZE)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
def load_model(model_class, checkpoint_path):
    model = model_class.from_preset(DATASET).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    # model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


nice_model = load_model(NICE, "checkpoints_unconstrained/best_mnist.pt")

# Add these when ready:
# realnvp_model = load_model(RealNVP, "checkpoints/realnvp.pth")
# glow_model    = load_model(GLOW, "checkpoints/glow.pth")


# --------------------------------------------------
# ANALYSIS FUNCTION
# --------------------------------------------------
def analyze_model(model, name):
    print(f"\nAnalyzing {name}...")

    distances = compute_distances_across_layers(
        model,
        loader,
        device,
        metric="wasserstein",   # or "mmd"
        n_projections=100
    )

    print(f"{name} distances:", distances)
    return distances


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    results = {}

    # Analyze NICE
    results["NICE"] = analyze_model(nice_model, "NICE")

    # Add later:
    # results["RealNVP"] = analyze_model(realnvp_model, "RealNVP")
    # results["GLOW"]    = analyze_model(glow_model, "GLOW")

    # Plot comparison
    plot_layer_vs_distance(results)


if __name__ == "__main__":
    main()