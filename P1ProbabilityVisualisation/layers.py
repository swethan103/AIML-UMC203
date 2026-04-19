import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# Column selectors
# ─────────────────────────────────────────────
def pick_even_columns(xs):
    return xs[:, 0::2]

def pick_odd_columns(xs):
    return xs[:, 1::2]


# ─────────────────────────────────────────────
# Fast reweaving (vectorized)
# ─────────────────────────────────────────────
def reweave_columns(first, second, which_was_first):
    B = first.shape[0]
    d1 = first.shape[1]
    d2 = second.shape[1]

    out = torch.zeros(B, d1 + d2, device=first.device, dtype=first.dtype)

    if which_was_first == 'even':
        out[:, 0::2] = first
        out[:, 1::2] = second
    else:
        out[:, 1::2] = first
        out[:, 0::2] = second

    return out


# ─────────────────────────────────────────────
# Base coupling layer
# ─────────────────────────────────────────────
class HalfAndHalfLayer(nn.Module):
    def __init__(self, dim, which_half, small_net):
        super().__init__()

        assert which_half in ['even', 'odd'], \
            "which_half must be 'even' or 'odd'"

        self.dim = dim
        self.which_half = which_half
        self.small_net = small_net   # cleaner

        if which_half == 'even':
            self.frozen_half   = pick_even_columns
            self.changing_half = pick_odd_columns
        else:
            self.frozen_half   = pick_odd_columns
            self.changing_half = pick_even_columns

    def forward(self, x, return_intermediate=False):
        frozen   = self.frozen_half(x)
        changing = self.changing_half(x)

        net_out = self.small_net(frozen)

        #  Safety check
        if net_out.shape != changing.shape:
            raise ValueError(
                f"Shape mismatch: net_out {net_out.shape} vs changing {changing.shape}"
            )

        transformed = self.transform_second_half(changing, net_out)
        out = reweave_columns(frozen, transformed, self.which_half)

        if return_intermediate:
            return out, net_out

        return out

    def inverse(self, y):
        frozen   = self.frozen_half(y)
        changing = self.changing_half(y)

        net_out = self.small_net(frozen)

        untransformed = self.untransform_second_half(changing, net_out)
        return reweave_columns(frozen, untransformed, self.which_half)

    def transform_second_half(self, changing, net_output):
        raise NotImplementedError

    def untransform_second_half(self, changing, net_output):
        raise NotImplementedError


# ─────────────────────────────────────────────
# NICE additive coupling
# ─────────────────────────────────────────────
class ShiftCouplingLayer(HalfAndHalfLayer):
    """
    NICE coupling:
    Forward:  x2 + m(x1)
    Inverse:  x2 - m(x1)
    """

    def transform_second_half(self, changing, net_output):
        return changing + net_output

    def untransform_second_half(self, changing, net_output):
        return changing - net_output


# class ScaleCouplingLayer(HalfAndHalfLayer):
#     """Forward: changing * net(frozen). Inverse: changing / net(frozen)."""
#     def transform_second_half(self, changing, net_output):
#         return torch.mul(changing, net_output)

#     def untransform_second_half(self, changing, net_output):
#         return torch.mul(changing, torch.reciprocal(net_output))


# class ShiftAndScaleCouplingLayer(HalfAndHalfLayer):
#     """Forward: changing * scale + shift. Inverse: not yet implemented."""
#     def transform_second_half(self, changing, net_output):
#         scale = self.frozen_half(net_output)
#         shift = self.changing_half(net_output)
#         return torch.mul(changing, scale) + shift

#     def untransform_second_half(self, changing, net_output):
#         raise NotImplementedError(
#             "ShiftAndScaleCouplingLayer inverse not yet implemented."
#         )
