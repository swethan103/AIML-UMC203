"""NICE model
"""

import torch
import torch.nn as nn
from torch.distributions.transforms import Transform,SigmoidTransform,AffineTransform
from torch.distributions import Uniform, TransformedDistribution
import numpy as np
"""Additive coupling layer.
"""
class AdditiveCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize an additive coupling layer.

        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(AdditiveCoupling, self).__init__()
        #TODO fill in
        # class members
        self.in_out_dim = in_out_dim
        self.mid_dim = mid_dim
        self.hidden = hidden
        self.mask_config = mask_config

        layers = nn.ModuleList()
        layer_size = in_out_dim // 2
        for _ in range(1, hidden):
            layers.append(nn.Linear(layer_size, mid_dim))
            layers.append(nn.ReLU())
            layer_size = mid_dim
        # last hidden layer
        layers.append(nn.Linear(mid_dim, in_out_dim // 2))
        # m function
        self.m = nn.Sequential(*layers)

    def forward(self, x, log_det_J, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            log_det_J: log determinant of the Jacobian
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and updated log-determinant of Jacobian.
        """
        #TODO fill in

        # Transform odd units if mask is 1
        x_1, x_2 = x[:, self.mask_config::2], x[:, (1-self.mask_config)::2]

        y_1 = x_1
        if not reverse:
            y_2 = x_2 + self.m(x_1)
        else:
            y_2 = x_2 - self.m(x_1)

       #output = torch.empty_like(x)
        
       # output[:, (1 - self.mask_config)::2] = y_1
        #output[:, self.mask_config::2] = y_2
        
        ordered_cat = [y_1, y_2] if self.mask_config == 0 else [y_2, y_1]
        y = torch.stack(ordered_cat, dim=2).view(-1, self.in_out_dim)

        return y, log_det_J

class AffineCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize an affine coupling layer.

        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(AffineCoupling, self).__init__()
        #TODO fill in
        # class members
        self.in_out_dim = in_out_dim
        self.mid_dim = mid_dim
        self.hidden = hidden
        self.mask_config = mask_config

        scale_layers = nn.ModuleList()
        layer_size = in_out_dim // 2
        for _ in range(1, hidden):
            scale_layers.append(nn.Linear(layer_size, mid_dim))
            scale_layers.append(nn.ReLU())
            layer_size = mid_dim
        # last hidden layer
        scale_layers.append(nn.Linear(mid_dim, in_out_dim // 2))
        scale_layers.append(nn.Tanh())
        # m function
        self.scale_network = nn.Sequential(*scale_layers)

        shift_layers = nn.ModuleList()
        layer_size = in_out_dim // 2
        for _ in range(1, hidden):
            shift_layers.append(nn.Linear(layer_size, mid_dim))
            shift_layers.append(nn.ReLU())
            layer_size = mid_dim
        # last hidden layer
        shift_layers.append(nn.Linear(mid_dim, in_out_dim // 2))
        shift_layers.append(nn.Tanh())
        # m function
        self.shift_network = nn.Sequential(*shift_layers)


    def forward(self, x, log_det_J, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            log_det_J: log determinant of the Jacobian
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and updated log-determinant of Jacobian.
        """
        #TODO fill in

        # Transform odd units if mask is 1
        x_1, x_2 = x[:, self.mask_config::2], x[:, (1-self.mask_config)::2]

        y_1 = x_1
        
        scale = torch.exp(self.scale_network(x_1))
        shift = self.shift_network(x_1)

        if not reverse:
            y_2 = x_2 * scale + shift
        else:
            y_2 = (x_2 - shift) / scale

       # output = torch.empty_like(x)

       # output[:, (1 - self.mask_config)::2] = y_1
       # output[:, self.mask_config::2] = y_2
        ordered_cat = [y_1, y_2] if self.mask_config == 0 else [y_2, y_1]
        y = torch.stack(ordered_cat, dim=2).view(-1, self.in_out_dim)

        log_det_J = log_det_J + scale.log().sum(dim=1)

        return y, log_det_J


"""Log-scaling layer.
"""
class Scaling(nn.Module):
    def __init__(self, dim):
        """Initialize a (log-)scaling layer.

        Args:
            dim: input/output dimensions.
        """
        super(Scaling, self).__init__()
        self.scale = nn.Parameter(
            torch.zeros((1, dim)), requires_grad=True)
        self.eps = 1e-5

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log-determinant of Jacobian.
        """
        scale = torch.exp(self.scale)+ self.eps
        #TODO fill in

        if not reverse:
            y = x * scale
        else:
            y = x / scale

        log_det_J = scale.log().sum(dim=1)

        return y, log_det_J

"""Standard logistic distribution.
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Move parameters to the target device
base_distribution = Uniform(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
affine_transform = AffineTransform(loc=torch.tensor(0.0, device=device), scale=torch.tensor(1.0, device=device))

# Create the transformed distribution
logistic = TransformedDistribution(base_distribution, [SigmoidTransform().inv, affine_transform])


"""NICE main model.
"""
class NICE(nn.Module):
    def __init__(self, prior, coupling, coupling_type,
        in_out_dim, mid_dim, hidden,device):
        """Initialize a NICE.

        Args:
            coupling_type: 'additive' or 'adaptive'
            coupling: number of coupling layers.
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            device: run on cpu or gpu
        """
        super(NICE, self).__init__()
        self.device = device
        if prior == 'gaussian':
            self.prior = torch.distributions.Normal(
                torch.tensor(0.).to(device), torch.tensor(1.).to(device))
        elif prior == 'logistic':
            self.prior = logistic
        else:
            raise ValueError('Prior not implemented.')
        self.in_out_dim = in_out_dim
        self.coupling = coupling
        self.coupling_type = coupling_type
        self.mid_dim = mid_dim
        self.hidden = hidden

        a = AdditiveCoupling
        #TODO fill in

        # Define the layer type we will use
        if coupling_type == 'additive':
            Layer = AdditiveCoupling
        elif coupling_type == 'adaptive':
            Layer = AffineCoupling
        else:
            raise ValueError("No such coupling type")

        # Define the list of layers of our model
        self.layers = nn.ModuleList()
        mask_config = 0
        for i in range(self.coupling):
            self.layers.append(Layer(in_out_dim, mid_dim, hidden, mask_config=i%2))
        self.scaling_layer = Scaling(dim=in_out_dim)


    def f_inverse(self, z):
        """Transformation g: Z -> X (inverse of f).

        Args:
            z: tensor in latent space Z.
        Returns:
            transformed tensor in data space X.
        """
        #TODO fill in
        x = z
        x, _ = self.scaling_layer(x, reverse=True)
        for layer in reversed(list(self.layers)):
            x, _ = layer(x, 0, reverse=True)

        return x


    def f(self, x):
        """Transformation f: X -> Z (inverse of g).

        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z and log determinant Jacobian
        """
        #TODO fill in
        log_det_J = 0.

        for layer in self.layers:
            x, log_det_J = layer(x, log_det_J)
        
        x, log_det_scaling_J = self.scaling_layer(x)

        return x, log_det_J + log_det_scaling_J

    def log_prob(self, x):
        """Computes data log-likelihood.

        (See Section 3.3 in the NICE paper.)

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        z, log_det_J = self.f(x)
        log_det_J -= np.log(256)*self.in_out_dim #log det for rescaling from [0.256] (after dequantization) to [0,1]
        log_ll = torch.sum(self.prior.log_prob(z), dim=1)
        return log_ll + log_det_J

    def sample(self, size):
        """Generates samples.

        Args:
            size: number of samples to generate.
        Returns:
            samples from the data space X.
        """
        z = self.prior.sample((size, self.in_out_dim)).to(self.device)
        #TODO
        return self.f_inverse(z)

    def forward(self, x):
        """Forward pass.

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        return self.log_prob(x)

