import torch
from torch import Tensor, nn


class MaxFeatureMap(nn.Module):
    """Apply pairwise maximum across two channel groups."""

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 2:
            raise ValueError(f"expected input with at least 2 dimensions, got {x.ndim}")

        channels = x.shape[1]
        if channels % 2 != 0:
            raise ValueError(f"expected an even number of channels, got {channels}")

        first_half, second_half = x.chunk(2, dim=1)
        return torch.maximum(first_half, second_half)
