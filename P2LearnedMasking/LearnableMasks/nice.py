import numpy as np
import torch
import torch.nn as nn


def _ensure_tensor_mask(mask, device=None):
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    if not torch.is_tensor(mask):
        mask = torch.tensor(mask, dtype=torch.float32)
    mask = mask.float().view(-1)
    if device is not None:
        mask = mask.to(device)
    return mask


class MaskedAdditiveCoupling(nn.Module):
    """
    Additive coupling layer with an arbitrary binary mask of length in_out_dim.
    Mask entries:
        1 -> dimensions kept unchanged and used as conditioner
        0 -> dimensions transformed
    """
    def __init__(self, in_out_dim, mid_dim, hidden, mask):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.mid_dim = mid_dim
        self.hidden = hidden

        layers = [nn.Linear(in_out_dim, mid_dim), nn.ReLU()]
        for _ in range(hidden - 1):
            layers.append(nn.Linear(mid_dim, mid_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(mid_dim, in_out_dim))
        self.net = nn.Sequential(*layers)

        mask = _ensure_tensor_mask(mask)
        self.register_buffer("mask", mask)

    def set_mask(self, mask):
        mask = _ensure_tensor_mask(mask, device=self.mask.device)
        if mask.numel() != self.in_out_dim:
            raise ValueError(
                f"Mask has {mask.numel()} entries, expected {self.in_out_dim}."
            )
        if torch.any((mask != 0.0) & (mask != 1.0)):
            raise ValueError("Mask must be binary.")
        self.mask.copy_(mask)

    def forward(self, x, log_det_J, reverse=False):
        mask = self.mask
        x_kept = x * mask
        shift = self.net(x_kept) * (1.0 - mask)

        if not reverse:
            y = x_kept + (1.0 - mask) * (x + shift)
        else:
            y = x_kept + (1.0 - mask) * (x - shift)

        return y, log_det_J


class Scaling(nn.Module):
    """
    NICE scaling layer. This is the only source of log-determinant in additive NICE.
    """
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1, dim))
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
    """
    Basic NICE:
    - additive coupling
    - fixed standard Gaussian prior
    - externally settable direct masks
    """
    def __init__(
        self,
        coupling,
        in_out_dim,
        mid_dim,
        hidden,
        device,
        init_mask=None,
    ):
        super().__init__()
        self.device = device
        self.in_out_dim = in_out_dim
        self.coupling = coupling
        self.mid_dim = mid_dim
        self.hidden = hidden

        if init_mask is None:
            init_mask = torch.cat(
                [torch.ones(in_out_dim // 2), torch.zeros(in_out_dim - in_out_dim // 2)]
            )

        init_mask = _ensure_tensor_mask(init_mask)
        if init_mask.numel() != in_out_dim:
            raise ValueError(
                f"Initial mask has {init_mask.numel()} entries, expected {in_out_dim}."
            )

        self.layers = nn.ModuleList()
        current_mask = init_mask.clone()
        for _ in range(coupling):
            layer = MaskedAdditiveCoupling(
                in_out_dim=in_out_dim,
                mid_dim=mid_dim,
                hidden=hidden,
                mask=current_mask,
            )
            self.layers.append(layer)
            current_mask = 1.0 - current_mask

        self.scaling_layer = Scaling(dim=in_out_dim)

    def set_mask(self, base_mask):
        """
        Set the chosen arm's mask into the whole flow:
        even-indexed coupling layers use base_mask
        odd-indexed coupling layers use complement(base_mask)
        """
        base_mask = _ensure_tensor_mask(base_mask, device=self.device)
        if base_mask.numel() != self.in_out_dim:
            raise ValueError(
                f"Mask has {base_mask.numel()} entries, expected {self.in_out_dim}."
            )

        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                layer.set_mask(base_mask)
            else:
                layer.set_mask(1.0 - base_mask)

    def f_inverse(self, z):
        x = z
        x, _ = self.scaling_layer(x, reverse=True)

        for layer in reversed(self.layers):
            x, _ = layer(x, 0, reverse=True)

        return x

    def f(self, x):
        log_det_J = torch.zeros(x.size(0), device=x.device)

        for layer in self.layers:
            x, log_det_J = layer(x, log_det_J, reverse=False)

        x, log_det_scaling_J = self.scaling_layer(x, reverse=False)
        log_det_J = log_det_J + log_det_scaling_J

        return x, log_det_J

    def latent_log_prob(self, z):
        log_prob = -0.5 * (z ** 2) - 0.5 * np.log(2.0 * np.pi)
        return log_prob.sum(dim=1)

    def log_prob(self, x):
        z, log_det_J = self.f(x)
        log_ll = self.latent_log_prob(z)
        return log_ll + log_det_J

    def sample(self, size):
        z = torch.randn(size, self.in_out_dim, device=self.device)
        return self.f_inverse(z)

    def forward(self, x):
        return self.log_prob(x)