# import matplotlib.pyplot as plt
# Wass = []
# mmd = []
# skew = []
# kurt = []
# covar = []
# nice_wass = [0.640747, 0.671400, 0.700001, 0.740926, 0.364072]
# nice_mmd = [0.004131, 0.004358, 0.005635, 0.037564, 0.004079]
# nice_skew = [3.441732, 2.975109, 2.528668 ,2.177581,2.177588]
# nice_kurt = [46.156089, 38.643318, 37.091521, 34.885608, 34.885756]
# nice_covar = [0.035099, 0.035051, 0.035231, 0.035515, 0.141966]
# rnvp_wass = [0.096897, 0.090569]
# rnvp_mmd = [ 0.004079, 0.004079]
# rnvp_skew = [0.317838, 0.286184]
# rnvp_kurt = [0.097693, 0.095442]
# rnvp_covar = [0.041806, 0.042379]
# layers = [0, 1, 2, 3, 4]

import matplotlib.pyplot as plt

# -----------------------------
# Data
# -----------------------------
nice_wass = [0.640747, 0.671400, 0.700001, 0.740926, 0.364072]
nice_mmd  = [0.004131, 0.004358, 0.005635, 0.037564, 0.004079]
nice_skew = [3.441732, 2.975109, 2.528668, 2.177581, 2.177588]
nice_kurt = [46.156089, 38.643318, 37.091521, 34.885608, 34.885756]
nice_covar= [0.035099, 0.035051, 0.035231, 0.035515, 0.141966]

rnvp_wass = [0.096897, 0.090569]
rnvp_mmd  = [0.004079, 0.004079]
rnvp_skew = [0.317838, 0.286184]
rnvp_kurt = [0.097693, 0.095442]
rnvp_covar= [0.041806, 0.042379]

nice_layers = [0, 1, 2, 3, 4]
rnvp_layers = [0, 1]

# -----------------------------
# Metric dictionary
# -----------------------------
metrics = {
    "Wasserstein Distance": (nice_wass, rnvp_wass),
    "MMD": (nice_mmd, rnvp_mmd),
    "Skewness": (nice_skew, rnvp_skew),
    "Kurtosis": (nice_kurt, rnvp_kurt),
    "Covariance Error": (nice_covar, rnvp_covar),
}

# -----------------------------
# Plot separate figure for each metric
# -----------------------------
for title, (nice_vals, rnvp_vals) in metrics.items():
    plt.figure(figsize=(6, 4))

    plt.plot(
        nice_layers, nice_vals,
        marker='o',
        linewidth=2,
        label='NICE'
    )

    plt.plot(
        rnvp_layers, rnvp_vals,
        marker='s',
        linewidth=2,
        label='RealNVP'
    )

    plt.title(title)
    plt.xlabel("Layer / Flow Block")
    plt.ylabel(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{title}_vs_Layers_Plot")
    plt.show()
