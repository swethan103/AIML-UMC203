import torch
import torch.nn as nn
import numpy as np


class AdditiveCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        super(AdditiveCoupling, self).__init__()
        self.in_out_dim = in_out_dim
        self.mid_dim = mid_dim
        self.hidden = hidden
        self.mask_config = mask_config

        layers = []
        input_dim = in_out_dim // 2

        layers.append(nn.Linear(input_dim, mid_dim))
        layers.append(nn.ReLU())

        for _ in range(hidden - 2):
            layers.append(nn.Linear(mid_dim, mid_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(mid_dim, in_out_dim // 2))
        self.m = nn.Sequential(*layers)

    def forward(self, x, log_det_J, reverse=False):
        x_1 = x[:, self.mask_config::2]
        x_2 = x[:, (1 - self.mask_config)::2]

        y_1 = x_1
        if not reverse:
            y_2 = x_2 + self.m(x_1)
        else:
            y_2 = x_2 - self.m(x_1)

        ordered_cat = [y_1, y_2] if self.mask_config == 0 else [y_2, y_1]
        y = torch.stack(ordered_cat, dim=2).view(-1, self.in_out_dim)

        return y, log_det_J


class AffineCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        super(AffineCoupling, self).__init__()
        self.in_out_dim = in_out_dim
        self.mid_dim = mid_dim
        self.hidden = hidden
        self.mask_config = mask_config

        def make_net():
            layers = []
            input_dim = in_out_dim // 2

            layers.append(nn.Linear(input_dim, mid_dim))
            layers.append(nn.ReLU())

            for _ in range(hidden - 2):
                layers.append(nn.Linear(mid_dim, mid_dim))
                layers.append(nn.ReLU())

            layers.append(nn.Linear(mid_dim, in_out_dim // 2))
            return nn.Sequential(*layers)

        self.scale_network = make_net()
        self.shift_network = make_net()

    def forward(self, x, log_det_J, reverse=False):
        x_1 = x[:, self.mask_config::2]
        x_2 = x[:, (1 - self.mask_config)::2]

        y_1 = x_1

        log_scale = torch.tanh(self.scale_network(x_1))
        scale = torch.exp(log_scale)
        shift = self.shift_network(x_1)

        if not reverse:
            y_2 = x_2 * scale + shift
            log_det_J = log_det_J + log_scale.sum(dim=1)
        else:
            y_2 = (x_2 - shift) / scale
            log_det_J = log_det_J - log_scale.sum(dim=1)

        ordered_cat = [y_1, y_2] if self.mask_config == 0 else [y_2, y_1]
        y = torch.stack(ordered_cat, dim=2).view(-1, self.in_out_dim)

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
        prior,
        coupling,
        coupling_type,
        in_out_dim,
        mid_dim,
        hidden,
        device,
    ):
        super(NICE, self).__init__()
        self.device = device
        self.in_out_dim = in_out_dim
        self.coupling = coupling
        self.coupling_type = coupling_type
        self.mid_dim = mid_dim
        self.hidden = hidden

        if prior not in ["gaussian", "learned_gaussian"]:
            raise ValueError("Supported priors: 'gaussian', 'learned_gaussian'")

        self.prior_type = prior

        if coupling_type == "additive":
            Layer = AdditiveCoupling
        elif coupling_type == "affine":
            Layer = AffineCoupling
        else:
            raise ValueError("No such coupling type")

        self.layers = nn.ModuleList()
        for i in range(self.coupling):
            self.layers.append(
                Layer(in_out_dim, mid_dim, hidden, mask_config=i % 2)
            )

        self.scaling_layer = Scaling(dim=in_out_dim)

        if self.prior_type == "learned_gaussian":
            self.prior_mu = nn.Parameter(torch.zeros(1, in_out_dim))
            self.prior_log_sigma = nn.Parameter(torch.zeros(1, in_out_dim))
        else:
            self.register_buffer("fixed_prior_mu", torch.zeros(1, in_out_dim))
            self.register_buffer("fixed_prior_log_sigma", torch.zeros(1, in_out_dim))

    def f_inverse(self, z):
        x = z
        x, _ = self.scaling_layer(x, reverse=True)

        for layer in reversed(list(self.layers)):
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
        if self.prior_type == "learned_gaussian":
            mu = self.prior_mu
            log_sigma = self.prior_log_sigma.clamp(min=-5.0, max=5.0)
        else:
            mu = self.fixed_prior_mu
            log_sigma = self.fixed_prior_log_sigma

        sigma = torch.exp(log_sigma)

        log_prob = (
            -0.5 * (((z - mu) / sigma) ** 2)
            - log_sigma
            - 0.5 * np.log(2.0 * np.pi)
        )

        return log_prob.sum(dim=1)

    def log_prob(self, x):
        z, log_det_J = self.f(x)

        log_det_J = log_det_J - np.log(256.0) * self.in_out_dim
        log_ll = self.latent_log_prob(z)

        return log_ll + log_det_J

    def map_penalty(
        self,
        lambda_model=1e-5,
        lambda_mu=1e-4,
        lambda_log_sigma=1e-4,
    ):
        penalty = torch.tensor(0.0, device=self.device)

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if name in ["prior_mu", "prior_log_sigma"]:
                continue

            penalty = penalty + lambda_model * torch.sum(param ** 2)

        if self.prior_type == "learned_gaussian":
            penalty = penalty + lambda_mu * torch.sum(self.prior_mu ** 2)
            penalty = penalty + lambda_log_sigma * torch.sum(self.prior_log_sigma ** 2)

        return penalty

    def sample(self, size):
        if self.prior_type == "learned_gaussian":
            mu = self.prior_mu
            log_sigma = self.prior_log_sigma.clamp(min=-5.0, max=5.0)
        else:
            mu = self.fixed_prior_mu
            log_sigma = self.fixed_prior_log_sigma

        sigma = torch.exp(log_sigma)
        eps = torch.randn(size, self.in_out_dim, device=self.device)
        z = mu + sigma * eps

        return self.f_inverse(z)

    def forward(self, x):
        return self.log_prob(x)