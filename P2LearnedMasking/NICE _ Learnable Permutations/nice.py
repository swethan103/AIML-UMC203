import random
import numpy as np
import torch
import torch.nn as nn


def _coord_to_index(coord):
    r, c = coord
    return r * 8 + c


def _apply_transform(coords, transform_name):
    def transform(coord):
        r, c = coord

        if transform_name == "identity":
            return (r, c)
        if transform_name == "rot90":
            return (c, 7 - r)
        if transform_name == "rot180":
            return (7 - r, 7 - c)
        if transform_name == "rot270":
            return (7 - c, r)
        if transform_name == "flip_h":
            return (r, 7 - c)
        if transform_name == "flip_v":
            return (7 - r, c)
        if transform_name == "transpose":
            return (c, r)
        if transform_name == "anti_transpose":
            return (7 - c, 7 - r)

        raise ValueError(f"Unknown transform: {transform_name}")

    return [transform(coord) for coord in coords]


def _raster_order():
    return [(r, c) for r in range(8) for c in range(8)]


def _snake_rows_order():
    coords = []
    for r in range(8):
        cols = range(8) if r % 2 == 0 else range(7, -1, -1)
        for c in cols:
            coords.append((r, c))
    return coords


def _snake_cols_order():
    coords = []
    for c in range(8):
        rows = range(8) if c % 2 == 0 else range(7, -1, -1)
        for r in rows:
            coords.append((r, c))
    return coords


def _diag_sum_order():
    coords = [(r, c) for r in range(8) for c in range(8)]
    coords.sort(key=lambda x: (x[0] + x[1], x[0], x[1]))
    return coords


def _spiral_order():
    coords = []
    top, bottom = 0, 7
    left, right = 0, 7

    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            coords.append((top, c))
        top += 1

        for r in range(top, bottom + 1):
            coords.append((r, right))
        right -= 1

        if top <= bottom:
            for c in range(right, left - 1, -1):
                coords.append((bottom, c))
            bottom -= 1

        if left <= right:
            for r in range(bottom, top - 1, -1):
                coords.append((r, left))
            left += 1

    return coords


def _quadrant_order():
    coords = []
    quadrants = [
        (range(0, 4), range(0, 4)),
        (range(0, 4), range(4, 8)),
        (range(4, 8), range(0, 4)),
        (range(4, 8), range(4, 8)),
    ]

    for rows, cols in quadrants:
        for r in rows:
            for c in cols:
                coords.append((r, c))

    return coords


def build_permutation_bank(num_permutations=40):
    base_orders = [
        ("raster", _raster_order()),
        ("snake_rows", _snake_rows_order()),
        ("snake_cols", _snake_cols_order()),
        ("diag_sum", _diag_sum_order()),
        ("spiral", _spiral_order()),
        ("quadrant", _quadrant_order()),
    ]

    transforms = [
        "identity",
        "rot90",
        "rot180",
        "rot270",
        "flip_h",
        "flip_v",
        "transpose",
        "anti_transpose",
    ]

    permutations = []
    permutation_names = []
    seen = set()

    for base_name, base in base_orders:
        for transform_name in transforms:
            transformed = _apply_transform(base, transform_name)
            perm = tuple(_coord_to_index(coord) for coord in transformed)

            if len(set(perm)) != 64:
                raise ValueError("Invalid permutation generated.")

            if perm in seen:
                continue

            seen.add(perm)
            permutations.append(list(perm))
            permutation_names.append(f"{base_name}__{transform_name}")

            if len(permutations) == num_permutations:
                return (
                    torch.tensor(permutations, dtype=torch.long),
                    permutation_names,
                )

    raise ValueError(
        f"Could only generate {len(permutations)} unique structured permutations, "
        f"but {num_permutations} were requested."
    )


def inverse_permutation_bank(perm_bank):
    inv_bank = torch.empty_like(perm_bank)
    base = torch.arange(perm_bank.size(1), dtype=torch.long)

    for i in range(perm_bank.size(0)):
        inv = torch.empty_like(base)
        inv[perm_bank[i]] = base
        inv_bank[i] = inv

    return inv_bank


class AdditiveCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config, perm_bank, inv_perm_bank):
        super(AdditiveCoupling, self).__init__()
        self.in_out_dim = in_out_dim
        self.mid_dim = mid_dim
        self.hidden = hidden
        self.mask_config = mask_config

        self.register_buffer("perm_bank", perm_bank.clone())
        self.register_buffer("inv_perm_bank", inv_perm_bank.clone())

        layers = []
        input_dim = in_out_dim // 2

        layers.append(nn.Linear(input_dim, mid_dim))
        layers.append(nn.ReLU())

        for _ in range(hidden - 2):
            layers.append(nn.Linear(mid_dim, mid_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(mid_dim, in_out_dim // 2))
        self.m = nn.Sequential(*layers)

    def forward(self, x, log_det_J, arm_index, reverse=False):
        perm = self.perm_bank[arm_index]
        inv_perm = self.inv_perm_bank[arm_index]

        x_perm = x.index_select(dim=1, index=perm)

        x_1 = x_perm[:, self.mask_config::2]
        x_2 = x_perm[:, (1 - self.mask_config)::2]

        y_1 = x_1
        if not reverse:
            y_2 = x_2 + self.m(x_1)
        else:
            y_2 = x_2 - self.m(x_1)

        ordered_cat = [y_1, y_2] if self.mask_config == 0 else [y_2, y_1]
        y_perm = torch.stack(ordered_cat, dim=2).reshape(-1, self.in_out_dim)
        y = y_perm.index_select(dim=1, index=inv_perm)

        return y, log_det_J


class Scaling(nn.Module):
    def __init__(self, dim):
        super(Scaling, self).__init__()
        self.scale = nn.Parameter(torch.zeros((1, dim)), requires_grad=True)
        self.eps = 1e-5

    def forward(self, x, reverse=False):
        scale = torch.exp(self.scale) + self.eps
        log_scale = torch.log(scale)

        if not reverse:
            y = x * scale
            log_det_J = log_scale.sum(dim=1)
        else:
            y = x / scale
            log_det_J = -log_scale.sum(dim=1)

        return y, log_det_J


class NICE(nn.Module):
    def __init__(
        self,
        coupling,
        in_out_dim,
        mid_dim,
        hidden,
        device,
        num_permutations=40,
        fixed_mask_config=0,
        data_bins=1.0,
    ):
        super(NICE, self).__init__()
        self.device = device
        self.in_out_dim = in_out_dim
        self.coupling = coupling
        self.mid_dim = mid_dim
        self.hidden = hidden
        self.fixed_mask_config = fixed_mask_config
        self.data_bins = float(data_bins)

        if in_out_dim != 64:
            raise ValueError(
                "This version is for sklearn digits only, so in_out_dim must be 64."
            )

        perm_bank, perm_names = build_permutation_bank(num_permutations=num_permutations)
        inv_perm_bank = inverse_permutation_bank(perm_bank)

        self.register_buffer("global_perm_bank", perm_bank)
        self.register_buffer("global_inv_perm_bank", inv_perm_bank)
        self.permutation_names = perm_names

        self.layers = nn.ModuleList()
        for _ in range(self.coupling):
            self.layers.append(
                AdditiveCoupling(
                    in_out_dim=in_out_dim,
                    mid_dim=mid_dim,
                    hidden=hidden,
                    mask_config=fixed_mask_config,
                    perm_bank=perm_bank,
                    inv_perm_bank=inv_perm_bank,
                )
            )

        self.scaling_layer = Scaling(dim=in_out_dim)

        self.register_buffer("fixed_prior_mu", torch.zeros(1, in_out_dim))
        self.register_buffer("fixed_prior_log_sigma", torch.zeros(1, in_out_dim))

        num_arms = perm_bank.size(0)
        self.register_buffer("bandit_counts", torch.zeros(coupling, num_arms))
        self.register_buffer("bandit_values", torch.zeros(coupling, num_arms))

    def num_arms(self):
        return self.global_perm_bank.size(0)

    def get_permutation_metadata(self):
        metadata = []
        for arm_idx in range(self.num_arms()):
            metadata.append(
                {
                    "arm_index": arm_idx,
                    "name": self.permutation_names[arm_idx],
                    "permutation": self.global_perm_bank[arm_idx].detach().cpu().tolist(),
                    "inverse_permutation": self.global_inv_perm_bank[arm_idx].detach().cpu().tolist(),
                }
            )
        return metadata

    def greedy_arm_indices(self):
        arm_indices = []
        for layer_idx in range(self.coupling):
            arm = int(torch.argmax(self.bandit_values[layer_idx]).item())
            arm_indices.append(arm)
        return arm_indices

    def select_arm_indices(self, epsilon):
        arm_indices = []
        for layer_idx in range(self.coupling):
            if random.random() < epsilon:
                arm = random.randrange(self.num_arms())
            else:
                arm = int(torch.argmax(self.bandit_values[layer_idx]).item())
            arm_indices.append(arm)
        return arm_indices

    def update_bandits(self, arm_indices, reward):
        reward = float(reward)

        with torch.no_grad():
            for layer_idx, arm_idx in enumerate(arm_indices):
                self.bandit_counts[layer_idx, arm_idx] += 1.0
                n = self.bandit_counts[layer_idx, arm_idx]
                q = self.bandit_values[layer_idx, arm_idx]
                self.bandit_values[layer_idx, arm_idx] = q + (reward - q) / n

    def f_inverse(self, z, arm_indices=None):
        if arm_indices is None:
            arm_indices = self.greedy_arm_indices()

        x = z
        x, _ = self.scaling_layer(x, reverse=True)

        for layer, arm_idx in zip(reversed(list(self.layers)), reversed(arm_indices)):
            x, _ = layer(x, 0, arm_index=arm_idx, reverse=True)

        return x

    def f(self, x, arm_indices=None):
        if arm_indices is None:
            arm_indices = self.greedy_arm_indices()

        log_det_J = torch.zeros(x.size(0), device=x.device)

        for layer, arm_idx in zip(self.layers, arm_indices):
            x, log_det_J = layer(x, log_det_J, arm_index=arm_idx, reverse=False)

        x, log_det_scaling_J = self.scaling_layer(x, reverse=False)
        log_det_J = log_det_J + log_det_scaling_J

        return x, log_det_J

    def latent_log_prob(self, z):
        mu = self.fixed_prior_mu
        log_sigma = self.fixed_prior_log_sigma
        sigma = torch.exp(log_sigma)

        log_prob = (
            -0.5 * (((z - mu) / sigma) ** 2)
            - log_sigma
            - 0.5 * np.log(2.0 * np.pi)
        )

        return log_prob.sum(dim=1)

    def log_prob(self, x, arm_indices=None):
        z, log_det_J = self.f(x, arm_indices=arm_indices)

        if self.data_bins > 1.0:
            log_det_J = log_det_J - np.log(self.data_bins) * self.in_out_dim

        log_ll = self.latent_log_prob(z)
        return log_ll + log_det_J

    def sample(self, size, arm_indices=None):
        if arm_indices is None:
            arm_indices = self.greedy_arm_indices()

        mu = self.fixed_prior_mu
        log_sigma = self.fixed_prior_log_sigma
        sigma = torch.exp(log_sigma)

        eps = torch.randn(size, self.in_out_dim, device=self.device)
        z = mu + sigma * eps

        return self.f_inverse(z, arm_indices=arm_indices)

    def forward(self, x, arm_indices=None):
        return self.log_prob(x, arm_indices=arm_indices)