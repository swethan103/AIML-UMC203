# # loss.py

# import torch
# import torch.nn.functional as F


# class StandardNormal:
#     """
#     Gaussian prior — good for MNIST and TFD.
#     log p(z) = -0.5 * sum(z^2)
#     """
#     def log_prob(self, z):
#         return -0.5 * z.pow(2).sum(dim=1)
    
#     def sample(self, n_samples, dim):
#         return torch.randn(n_samples, dim)


# class StandardLogistic:
#     """
#     Logistic prior — better for CIFAR-10 and SVHN.
#     log p(z) = -sum( softplus(z) + softplus(-z) )
#     F.softplus is numerically stable; torch.log(1 + exp(z)) is not.
#     """
#     def log_prob(self, z):
#         return -(F.softplus(z) + F.softplus(-z)).sum(dim=1)
    
#     def sample(self, n_samples, dim):
#         # Logistic samples via inverse CDF trick:
#         # if u ~ Uniform(0,1), then log(u/(1-u)) ~ Logistic
#         u = torch.rand(n_samples, dim).clamp(1e-6, 1 - 1e-6)
#         return torch.log(u) - torch.log(1 - u)

# def nll_loss(model, x, prior, base_reg=0.01):
#     z      = model.encode(x)
#     log_pz = prior.log_prob(z)
#     ldj    = model.log_det_jacobian()

#     log_scale     = model.scaling_layer.log_scale
#     current_scale = log_scale.abs().mean().item()
#     adaptive_reg  = base_reg * current_scale
#     scale_penalty = log_scale.pow(2).mean()

#     return -(log_pz + ldj).mean() + adaptive_reg * scale_penalty
# # ```

# # Think of it as two layers of protection:
# # ```

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

# def nll_loss(model, x, prior):
#     z = model(x) # [batch_size, 784]
    
#     log_pz = prior.log_prob(z) # [batch_size]
    
#     # Ensure log_det is broadcasted to the batch size
#     # If model.log_det_jacobian() returns a scalar:
#     log_det = model.log_det_jacobian() 
    
#     # The loss for the batch
#     return -(log_pz + log_det).mean()
def nll_loss(model, x, prior):
    z = model(x)                              # (B, D)

    log_pz = prior.log_prob(z)     # (B,)
    log_det = model.log_det_jacobian()         # scalar

    return -(log_pz + log_det).mean()
# def nll_loss(model, x, prior):
#     z = model(x)

#     log_pz = prior.log_prob(z)
#     log_det = model.log_det_jacobian().expand_as(log_pz) # more safer

#     return -(log_pz + log_det).mean()
# # Clamp      = hard wall    → scales physically cannot exceed exp(3)=20
# # Adaptive   = soft spring  → gets stiffer as scales drift away from 0
