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


# ------------------------------------------------------------------------------------------------------------------------------------------
# GLOW
# ------------------------------------------------------------------------------------------------------------------------------------------

import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi, exp
import numpy as np
from scipy import linalg as la

logabs = lambda x: torch.log(torch.abs(x))


class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = height * width * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet

        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, height, width = input.shape

        out = F.conv2d(input, self.weight)
        logdet = (
            height * width * torch.slogdet(self.weight.squeeze().double())[1].float()
        )

        return out, logdet

    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )


class InvConv2dLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, height, width = input.shape

        weight = self.calc_weight()

        out = F.conv2d(input, weight)
        logdet = height * width * torch.sum(self.w_s)

        return out, logdet

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()

        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512, affine=True):
        super().__init__()

        self.affine = affine

        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = torch.sigmoid(log_s + 2)
            # out_a = s * in_a + t
            out_b = (in_b + t) * s

            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)

        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None

        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = torch.sigmoid(log_s + 2)
            # in_a = (out_a - t) / s
            in_b = out_b / s - t

        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)


class Flow(nn.Module):
    def __init__(self, in_channel, affine=True, conv_lu=True):
        super().__init__()

        self.actnorm = ActNorm(in_channel)

        if conv_lu:
            self.invconv = InvConv2dLU(in_channel)

        else:
            self.invconv = InvConv2d(in_channel)

        self.coupling = AffineCoupling(in_channel, affine=affine)

    def forward(self, input):
        out, logdet = self.actnorm(input)
        out, det1 = self.invconv(out)
        out, det2 = self.coupling(out)

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2

        return out, logdet

    def reverse(self, output):
        input = self.coupling.reverse(output)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)

        return input


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class Block(nn.Module):
    def __init__(self, in_channel, n_flow, split=True, affine=True, conv_lu=True):
        super().__init__()

        squeeze_dim = in_channel * 4

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu))

        self.split = split

        if split:
            self.prior = ZeroConv2d(in_channel * 2, in_channel * 4)

        else:
            self.prior = ZeroConv2d(in_channel * 4, in_channel * 8)

    def forward(self, input):
        b_size, n_channel, height, width = input.shape
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)

        logdet = 0

        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det

        if self.split:
            out, z_new = out.chunk(2, 1)
            mean, log_sd = self.prior(out).chunk(2, 1)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)

        else:
            zero = torch.zeros_like(out)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out

        return out, logdet, log_p, z_new

    def reverse(self, output, eps=None, reconstruct=False):
        input = output

        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 1)

            else:
                input = eps

        else:
            if self.split:
                mean, log_sd = self.prior(input).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = torch.cat([output, z], 1)

            else:
                zero = torch.zeros_like(input)
                # zero = F.pad(zero, [1, 1, 1, 1], value=1)
                mean, log_sd = self.prior(zero).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = z

        for flow in self.flows[::-1]:
            input = flow.reverse(input)

        b_size, n_channel, height, width = input.shape

        unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_channel // 4, height * 2, width * 2
        )

        return unsqueezed


class Glow(nn.Module):
    def __init__(
        self, in_channel, n_flow, n_block, affine=True, conv_lu=True
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu))
            n_channel *= 2
        self.blocks.append(Block(n_channel, n_flow, split=False, affine=affine))

    def forward(self, input):
        log_p_sum = 0
        logdet = 0
        out = input
        z_outs = []

        for block in self.blocks:
            out, det, log_p, z_new = block(out)
            z_outs.append(z_new)
            logdet = logdet + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        return log_p_sum, logdet, z_outs

    def reverse(self, z_list, reconstruct=False):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                input = block.reverse(z_list[-1], z_list[-1], reconstruct=reconstruct)

            else:
                input = block.reverse(input, z_list[-(i + 1)], reconstruct=reconstruct)

        return input