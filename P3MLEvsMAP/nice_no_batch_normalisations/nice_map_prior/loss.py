import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardNormal:
    def log_prob(self, z):
        dim = z.size(1)
        return -0.5 * z.pow(2).sum(dim=1) - 0.5 * dim * math.log(2 * math.pi)

    def sample(self, n_samples, dim, device=None):
        return torch.randn(n_samples, dim, device=device)


class StandardLogistic:
    def log_prob(self, z):
        return -(F.softplus(z) + F.softplus(-z)).sum(dim=1)

    def sample(self, n_samples, dim, device=None):
        u = torch.rand(n_samples, dim, device=device).clamp(1e-6, 1 - 1e-6)
        return torch.log(u) - torch.log(1 - u)


class LearnedDiagonalGaussianPrior(nn.Module):
    """
    Learnable latent prior:
        p(z) = N(mu, diag(sigma^2))

    mu      : learnable mean vector
    log_std : learnable log standard deviation vector
    """
    def __init__(self, dim, init_mean=0.0, init_log_std=0.0):
        super().__init__()
        self.dim = dim
        self.mu = nn.Parameter(torch.full((dim,), float(init_mean)))
        self.log_std = nn.Parameter(torch.full((dim,), float(init_log_std)))

    def log_prob(self, z):
        """
        z: (B, D)
        returns: (B,)
        """
        log_std = torch.clamp(self.log_std, min=-5.0, max=5.0)
        var = torch.exp(2.0 * log_std)

        centered = z - self.mu
        log_prob_per_dim = (
            -0.5 * (centered ** 2) / var
            - log_std
            - 0.5 * math.log(2.0 * math.pi)
        )
        return log_prob_per_dim.sum(dim=1)

    def sample(self, n_samples, dim=None, device=None):
        if dim is not None and dim != self.dim:
            raise ValueError(f"Requested dim={dim}, but learned prior has dim={self.dim}")

        if device is None:
            device = self.mu.device

        eps = torch.randn(n_samples, self.dim, device=device)
        log_std = torch.clamp(self.log_std, min=-5.0, max=5.0)
        return self.mu.unsqueeze(0) + torch.exp(log_std).unsqueeze(0) * eps


def _mean_squared_parameter_penalty(parameters):
    """
    Returns (1 / N) * sum_i theta_i^2 over all trainable parameters.
    """
    total_sq = None
    total_numel = 0

    for p in parameters:
        if p.requires_grad:
            sq = p.pow(2).sum()
            total_sq = sq if total_sq is None else total_sq + sq
            total_numel += p.numel()

    if total_sq is None or total_numel == 0:
        raise ValueError("No trainable parameters found for MAP penalty.")

    return total_sq / total_numel


def nll_loss(model, x, prior):
    z = model(x)
    log_pz = prior.log_prob(z)
    log_det = model.log_det_jacobian()
    return -(log_pz + log_det).mean()


def learned_prior_map_loss(
    model,
    x,
    prior,
    lambda_model_map=1e-4,
    model_param_prior_std=1.0,
    lambda_prior_map=1e-4,
    prior_param_prior_std=1.0,
):
    """
    Joint objective:
        total = NLL
              + lambda_model_map * model_MAP_penalty
              + lambda_prior_map * prior_MAP_penalty
    """
    z = model(x)
    log_pz = prior.log_prob(z)
    log_det = model.log_det_jacobian()

    nll = -(log_pz + log_det).mean()

    model_mean_sq = _mean_squared_parameter_penalty(model.parameters())
    prior_mean_sq = _mean_squared_parameter_penalty(prior.parameters())

    model_map_penalty = model_mean_sq / (2.0 * (model_param_prior_std ** 2))
    prior_map_penalty = prior_mean_sq / (2.0 * (prior_param_prior_std ** 2))

    total_loss = (
        nll
        + lambda_model_map * model_map_penalty
        + lambda_prior_map * prior_map_penalty
    )

    stats = {
        "nll": nll.detach(),
        "model_map_penalty": model_map_penalty.detach(),
        "prior_map_penalty": prior_map_penalty.detach(),
        "total": total_loss.detach(),
    }

    return total_loss, stats