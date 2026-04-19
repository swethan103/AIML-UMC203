import torch
import torch.nn as nn
from layers import ShiftCouplingLayer


def make_coupling_net(in_dim, hidden_dim, n_hidden_layers, out_dim):
    """
    Builds the small MLP injected into each ShiftCouplingLayer.
    """
    layers = []
    layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]

    for _ in range(n_hidden_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]

    layers += [nn.Linear(hidden_dim, out_dim)]

    # Initialize last layer to zero → identity transform initially
    nn.init.zeros_(layers[-1].weight)
    nn.init.zeros_(layers[-1].bias)

    return nn.Sequential(*layers)


class DiagonalScalingLayer(nn.Module):
    """
    Learnable per-dimension scaling (NICE 'homothety' layer).
    Only contributor to log-determinant.
    """
    def __init__(self, dim):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        log_s = self.log_scale
        return x * torch.exp(log_s)

    def inverse(self, y):
        log_s = self.log_scale
        return y * torch.exp(-log_s)

    def log_det_jacobian(self):
        log_s = self.log_scale
        return torch.sum(log_s)


class NICE(nn.Module):

    PRESETS = {
        'mnist':   dict(nvis=784,  nhid=1000, n_coupling=4),
        'svhn':    dict(nvis=3072, nhid=2000, n_coupling=4),
        'cifar10': dict(nvis=3072, nhid=2400, n_coupling=8),
        'tfd':     dict(nvis=2304, nhid=5000, n_coupling=8),
    }

    def __init__(self, nvis, nhid, n_coupling):
        super().__init__()

        self.nvis = nvis
        half = nvis // 2
        rest = nvis - half

        self.layers = nn.ModuleList()

        for i in range(n_coupling):
            which_half = 'even' if i % 2 == 0 else 'odd'
            in_dim, out_dim = (half, rest) if which_half == 'even' else (rest, half)

            net = make_coupling_net(
                in_dim=in_dim,
                hidden_dim=nhid,
                n_hidden_layers=5,
                out_dim=out_dim
            )

            self.layers.append(ShiftCouplingLayer(nvis, which_half, net))

        self.scaling_layer = DiagonalScalingLayer(nvis)

    def forward(self, x):
        return self.encode(x)

    def encode(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.scaling_layer(x)

    @torch.no_grad()
    def decode(self, z):
        z = self.scaling_layer.inverse(z)
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z

    def log_det_jacobian(self):
        return self.scaling_layer.log_det_jacobian()

    @classmethod
    def from_preset(cls, dataset_name):
        assert dataset_name in cls.PRESETS, \
            f"Unknown dataset. Choose from: {list(cls.PRESETS.keys())}"
        return cls(**cls.PRESETS[dataset_name])