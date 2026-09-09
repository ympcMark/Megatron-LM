# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Modality-aware tensor operations used by native MCore MAGI-2 layers."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from megatron.core.transformer.module import mark_keep_in_fp32


class Magi2ModalityDispatcher:
    """Stable token permutation and group metadata for modality-specific weights.

    The dispatcher is created once per model invocation from the original token
    order. Transformer layers receive hidden states already permuted into
    contiguous video, audio, and text groups.
    """

    def __init__(self, modality_mapping: Tensor, num_modalities: int = 3) -> None:
        if modality_mapping.ndim != 1:
            raise ValueError("modality_mapping must be one-dimensional")
        if modality_mapping.numel() == 0:
            raise ValueError("modality_mapping must contain at least one token")
        if (
            int(modality_mapping.min().item()) < 0
            or int(modality_mapping.max().item()) >= num_modalities
        ):
            raise ValueError("modality_mapping contains an unsupported modality")

        self.num_modalities = num_modalities
        self.modality_mapping = modality_mapping
        self.permute_mapping = torch.argsort(modality_mapping, stable=True)
        self.inverse_permute_mapping = torch.argsort(self.permute_mapping)
        permuted_mapping = modality_mapping.index_select(0, self.permute_mapping)
        self.group_sizes = torch.bincount(permuted_mapping, minlength=num_modalities).to(
            torch.int32
        )
        self._group_sizes_cpu = tuple(int(size) for size in self.group_sizes.cpu().tolist())

    def permute(self, value: Tensor) -> Tensor:
        """Move equal-modality tokens into contiguous groups."""
        return value.index_select(0, self.permute_mapping)

    def inverse_permute(self, value: Tensor) -> Tensor:
        """Restore original packed-sequence token order."""
        return value.index_select(0, self.inverse_permute_mapping)

    def split(self, value: Tensor) -> tuple[Tensor, ...]:
        """Split an already-permuted tensor into modality groups."""
        if value.shape[0] != self.modality_mapping.numel():
            raise ValueError("value token dimension does not match modality_mapping")
        return torch.split(value, self._group_sizes_cpu, dim=0)


class Magi2ModalityLinear(nn.Module):
    """Linear projection with one checkpoint-compatible weight set per modality."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        num_modalities: int = 1,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        init_method: Callable[[Tensor], None] | None = None,
        perform_initialization: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_modalities = num_modalities
        self.weight = nn.Parameter(
            torch.empty(num_modalities * out_features, in_features, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(num_modalities * out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        if perform_initialization:
            weight = self.weight.view(num_modalities, out_features, in_features)
            for modality_weight in weight.unbind(0):
                if init_method is None:
                    nn.init.xavier_uniform_(modality_weight)
                else:
                    init_method(modality_weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, value: Tensor, dispatcher: Magi2ModalityDispatcher | None = None) -> Tensor:
        """Apply the matching projection to each contiguous modality group."""
        if value.shape[-1] != self.in_features:
            raise ValueError(f"expected last dimension {self.in_features}")
        weight = self.weight.view(self.num_modalities, self.out_features, self.in_features)
        bias = (
            self.bias.view(self.num_modalities, self.out_features)
            if self.bias is not None
            else None
        )
        if self.num_modalities == 1:
            return F.linear(value, weight[0], None if bias is None else bias[0])
        if dispatcher is None:
            raise ValueError("dispatcher is required for modality-specific linear layers")

        outputs = [
            F.linear(group, weight[index], None if bias is None else bias[index])
            for index, group in enumerate(dispatcher.split(value))
        ]
        return torch.cat(outputs, dim=0)


class Magi2MultiModalityRMSNorm(nn.Module):
    """Official zero-centered RMSNorm with optional modality parameters."""

    def __init__(
        self,
        hidden_size: int,
        *,
        num_modalities: int = 1,
        num_patterns: int = 1,
        eps: float = 1e-6,
        out_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or num_modalities <= 0 or num_patterns <= 0:
            raise ValueError("RMSNorm dimensions must be positive")
        self.hidden_size = hidden_size
        self.num_modalities = num_modalities
        self.num_patterns = num_patterns
        self.eps = eps
        self.out_dtype = out_dtype
        self.weight = nn.Parameter(
            torch.zeros(num_modalities * num_patterns * hidden_size, dtype=torch.float32)
        )
        mark_keep_in_fp32(self.weight)

    def _normalize(self, value: Tensor, weight: Tensor) -> Tensor:
        """Normalize in FP32, apply zero-centered weights, then select output dtype."""
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * (weight + 1.0)).to(self.out_dtype or value.dtype)

    def forward(self, value: Tensor, dispatcher: Magi2ModalityDispatcher | None = None) -> Tensor:
        """Normalize over the final dimension and select weights by modality."""
        if value.shape[-1] != self.hidden_size:
            raise ValueError(f"expected last dimension {self.hidden_size}")
        weight = self.weight.view(self.num_modalities, self.num_patterns, self.hidden_size)
        if self.num_modalities == 1:
            return self._normalize(value, weight[0])
        if dispatcher is None:
            raise ValueError("dispatcher is required for multi-modality RMSNorm")

        outputs = [
            self._normalize(group, weight[index])
            for index, group in enumerate(dispatcher.split(value))
        ]
        return torch.cat(outputs, dim=0)
