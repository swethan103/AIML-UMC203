# # models.py

# import torch
# import torch.nn as nn
# from layers import ShiftCouplingLayer


# def make_coupling_net(in_dim, hidden_dim, n_hidden_layers, out_dim):
#     """
#     Builds the small MLP injected into each ShiftCouplingLayer.
#     in_dim  → hidden_dim (ReLU) × n_hidden_layers → out_dim (linear)
#     """
#     layers = []
#     layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
#     for _ in range(n_hidden_layers - 1):
#         layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
#     layers += [nn.Linear(hidden_dim, out_dim)]
#     # Zero-init output so every coupling layer starts as identity
#     nn.init.zeros_(layers[-1].weight)
#     nn.init.zeros_(layers[-1].bias)
#     return nn.Sequential(*layers)


# class DiagonalScalingLayer(nn.Module):
#     """
#     Learnable per-dimension scaling. The only layer that contributes
#     to log|det J|. Called 'Homothety' in the original paper.
#     """
#     def __init__(self, dim):
#         super().__init__()
#         self.log_scale = nn.Parameter(torch.zeros(dim))

#     def forward(self, x):
#         return x * self.log_scale.clamp(-3, 3).exp()  #added clamping as part of handling highly drifting values of the loss

#     def inverse(self, y):
#         return y * (-self.log_scale.clamp(-3,3)).exp()

#     def log_det_jacobian(self):
#         return self.log_scale.clamp(-3,3).sum()


# class NICE(nn.Module):

#     PRESETS = {
#         'mnist':   dict(nvis=784,  nhid=1000, n_coupling=4),
#         'svhn':    dict(nvis=3072, nhid=2000, n_coupling=4),
#         'cifar10': dict(nvis=3072, nhid=2400, n_coupling=8),
#         'tfd':     dict(nvis=2304, nhid=5000, n_coupling=8),
#     }

#     def __init__(self, nvis, nhid, n_coupling):
#         super().__init__()
#         half = nvis // 2
#         rest = nvis - half

#         self.coupling_layers = nn.ModuleList()
#         for i in range(n_coupling):
#             which_half = 'even' if i % 2 == 0 else 'odd'
#             in_dim, out_dim = (half, rest) if which_half == 'even' else (rest, half)

#             if n_coupling == 8:
#                 if i in (2, 3):   n_hidden = 3
#                 elif i in (4, 5): n_hidden = 2
#                 else:             n_hidden = 1
#             else:
#                 n_hidden = 1

#             net = make_coupling_net(in_dim, nhid, n_hidden, out_dim)
#             self.coupling_layers.append(ShiftCouplingLayer(nvis, which_half, net))

#         self.scaling_layer = DiagonalScalingLayer(nvis)

#     def encode(self, x):
#         for layer in self.coupling_layers:
#             x = layer(x)
#         return self.scaling_layer(x)

#     @torch.no_grad()
#     def decode(self, z):
#         z = self.scaling_layer.inverse(z)
#         for layer in reversed(self.coupling_layers):
#             z = layer.inverse(z)
#         return z

#     def log_det_jacobian(self):
#         return self.scaling_layer.log_det_jacobian()

#     @classmethod
#     def from_preset(cls, dataset_name):
#         assert dataset_name in cls.PRESETS, \
#             f"Unknown dataset. Choose from: {list(cls.PRESETS.keys())}"
#         return cls(**cls.PRESETS[dataset_name])
# models.py

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
    # def __init__(self, dim):
    #     super().__init__()
    #     self.log_scale = nn.Parameter(torch.zeros(dim))
        
    # def forward(self, x):                               #better presentation
    #     log_s = torch.tanh(self.log_scale) * 3
    #     return x * torch.exp(log_s)
            
    # def inverse(self, y):
    #     log_s = torch.tanh(self.log_scale) * 3
    #     return y * torch.exp(-log_s)   
    
    # def log_det_jacobian(self):
    #     log_s = torch.tanh(self.log_scale) * 3
    #     return torch.sum(log_s)              #removed clamp as it would then be constrained NICE implementation and lead to collapsing of latent

    def __init__(self, dim):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x):                               #better presentation
        log_s = self.log_scale
        return x * torch.exp(log_s)
            
    def inverse(self, y):
        log_s = self.log_scale
        return y * torch.exp(-log_s)   
    
    def log_det_jacobian(self):
        log_s = self.log_scale
        return torch.sum(log_s)     
    # def forward(self, x):
    #     scale = torch.exp(self.log_scale.clamp(-3, 3))  #added the clamping
    #     return x * scale
        
    # def inverse(self, y):
    #     scale = torch.exp(self.log_scale.clamp(-3, 3))
    #     return y / scale

    # def log_det_jacobian(self):
    #     return torch.sum(self.log_scale.clamp(-3,3))
    


# class PermutationLayer(nn.Module):
#     """
#     Fixed random permutation for better mixing between coupling layers.
#     """

#     def __init__(self, dim):
#         super().__init__()
#         perm = torch.randperm(dim)
#         inv_perm = torch.argsort(perm)

#         self.register_buffer("perm", perm)
#         self.register_buffer("inv_perm", inv_perm)

#     def forward(self, x):
#         return x[:, self.perm]

#     def inverse(self, y):
#         return y[:, self.inv_perm]


class NICE(nn.Module):

    PRESETS = {
        'mnist':   dict(nvis=784,  nhid=1000, n_coupling=4),
        'quickdraw':dict(nvis=784,  nhid=1000, n_coupling=4),
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

        # Optional: define hidden depth schedule
        # if n_coupling == 8:
        #     hidden_schedule = [1, 1, 3, 3, 2, 2, 1, 1]
        # else:
        #     hidden_schedule = [1] * n_coupling
        
        for i in range(n_coupling):
            which_half = 'even' if i % 2 == 0 else 'odd'
            in_dim, out_dim = (half, rest) if which_half == 'even' else (rest, half)

            net = make_coupling_net(
                in_dim,
                nhid,
                5,          #2 -> 5
                out_dim
            )

            self.layers.append(ShiftCouplingLayer(nvis, which_half, net))
            # self.layers.append(PermutationLayer(nvis))  # NEW

        self.scaling_layer = DiagonalScalingLayer(nvis)

    # ─────────────────────────────────────────────
    # Forward (x → z)
    # ─────────────────────────────────────────────
    def forward(self, x):
        return self.encode(x)
    # def forward(self, x, return_intermediate=False):
    #     return self.encode(x, return_intermediate)

    def encode(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.scaling_layer(x)
    # def encode(self, x, return_intermediate=False):
    #     zs = []

    #     # Pass through coupling layers
    #     for layer in self.layers:
    #         x = layer(x)

    #         if return_intermediate:
    #             zs.append(x.detach().cpu())

    #     # Final scaling layer
    #     x = self.scaling_layer(x)

    #     if return_intermediate:
    #         zs.append(x.detach())   # final latent #removed .cpu() for doing analysis of other metrics

    #         return x, zs

    #     return x
    # ─────────────────────────────────────────────
    # Inverse (z → x)
    # ─────────────────────────────────────────────
    @torch.no_grad()
    def decode(self, z):
        z = self.scaling_layer.inverse(z)
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z

    # ─────────────────────────────────────────────
    # Log determinant
    # ─────────────────────────────────────────────
    def log_det_jacobian(self):
        return self.scaling_layer.log_det_jacobian()

    # ─────────────────────────────────────────────
    # Factory
    # ─────────────────────────────────────────────
    @classmethod
    def from_preset(cls, dataset_name):
        assert dataset_name in cls.PRESETS, \
            f"Unknown dataset. Choose from: {list(cls.PRESETS.keys())}"
        return cls(**cls.PRESETS[dataset_name])
