import torch
import torch.nn.functional as F
import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class StandardNormal:
    def log_prob(self, z):
        dim = z.size(1)
        return -0.5 * z.pow(2).sum(dim=1) - 0.5 * dim * math.log(2 * math.pi)

    def sample(self, n_samples, dim,device = None):
        return torch.randn(n_samples, dim, device = device)


class StandardLogistic:
    def log_prob(self, z):
        return -(F.softplus(z) + F.softplus(-z)).sum(dim=1)

    def sample(self, n_samples, dim, device = None):
        u = torch.rand(n_samples, dim, device=device).clamp(1e-6, 1 - 1e-6)
        return torch.log(u) - torch.log(1 - u)

def nll_loss(model, x, prior):
    z = model(x)                              # (B, D)

    log_pz = prior.log_prob(z)     # (B,)
    log_det = model.log_det_jacobian()         # scalar

    return -(log_pz + log_det).mean()
