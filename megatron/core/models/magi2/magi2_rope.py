# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Three-axis partial rotary embedding used by MAGI-2 Preview."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class Magi2FourierRoPE(nn.Module):
    """Build the official element-wise Fourier embedding from 3D coordinates."""

    def __init__(self, head_dim: int, temperature: float = 10_000.0) -> None:
        super().__init__()
        if head_dim < 8 or head_dim % 8:
            raise ValueError("MAGI-2 attention head_dim must be divisible by 8")
        self.head_dim = head_dim
        num_bands = head_dim // 8
        exponents = torch.arange(num_bands, dtype=torch.float32) / num_bands
        self.register_buffer("bands", 1.0 / (temperature**exponents), persistent=True)

    @property
    def rotary_dim(self) -> int:
        """Number of head channels rotated by the three-axis embedding."""
        return 6 * self.bands.numel()

    def forward(self, coordinates: Tensor) -> Tensor:
        """Return ``[tokens, rotary_dim]`` sine values followed by cosine values."""
        if coordinates.ndim != 2 or coordinates.shape[1] != 9:
            raise ValueError("coordinates must have shape [tokens, 9]")

        coordinates = coordinates.float()
        coordinates_xyz = coordinates[:, :3]
        sizes = coordinates[:, 3:6]
        references = coordinates[:, 6:9]
        scales = (references - 1.0) / (sizes - 1.0)
        scales = torch.where((references == 1.0) & (sizes == 1.0), torch.ones_like(scales), scales)
        if not bool(torch.isfinite(scales).all().item()):
            raise ValueError("coordinate scaling produced a non-finite value")

        centers = (sizes - 1.0) / 2.0
        centers[:, 0] = 0.0
        projection = (coordinates_xyz - centers).unsqueeze(-1) * scales.unsqueeze(-1) * self.bands
        return torch.cat((projection.sin(), projection.cos()), dim=1).flatten(1)


def apply_magi2_rotary_pos_emb(value: Tensor, rope: Tensor) -> Tensor:
    """Apply official non-interleaved partial RoPE to ``[tokens, ..., heads, dim]``."""
    if value.ndim < 3:
        raise ValueError("value must have shape [tokens, ..., heads, head_dim]")
    if rope.ndim != 2 or rope.shape[0] != value.shape[0]:
        raise ValueError("rope must have shape [tokens, rotary_dim]")

    sin, cos = rope.tensor_split(2, dim=-1)
    sin = torch.cat((sin, sin), dim=-1)
    cos = torch.cat((cos, cos), dim=-1)
    rotary_dim = cos.shape[-1]
    if rotary_dim > value.shape[-1]:
        raise ValueError("rotary embedding is wider than the attention head")

    broadcast_shape = (value.shape[0],) + (1,) * (value.ndim - 2) + (rotary_dim,)
    sin = sin.view(broadcast_shape)
    cos = cos.view(broadcast_shape)
    rotary_value = value[..., :rotary_dim]
    first, second = rotary_value.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    embedded = rotary_value * cos + rotated * sin
    return torch.cat((embedded, value[..., rotary_dim:]), dim=-1)
