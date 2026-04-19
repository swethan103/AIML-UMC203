import math
import torch
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
    z = model(x)                              # (B, D)
    log_pz = prior.log_prob(z)                # (B,)
    log_det = model.log_det_jacobian()        # scalar
    return -(log_pz + log_det).mean()


def map_loss(model, x, prior, lambda_map=1e-4, param_prior_std=1.0):
    """
    MAP objective:
        total_loss = NLL + lambda_map * [ 1/(2 sigma^2) * mean(theta^2) ]

    Here sigma = param_prior_std is the std of the Gaussian prior over parameters.
    """
    z = model(x)
    log_pz = prior.log_prob(z)
    log_det = model.log_det_jacobian()

    nll = -(log_pz + log_det).mean()

    mean_sq = _mean_squared_parameter_penalty(model.parameters())
    map_penalty = mean_sq / (2.0 * (param_prior_std ** 2))

    total_loss = nll + lambda_map * map_penalty

    stats = {
        "nll": nll.detach(),
        "map_penalty": map_penalty.detach(),
        "total": total_loss.detach(),
    }

    return total_loss, stats