# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""MAGI-2 manifold-constrained hyper-connection operators."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from megatron.core.models.magi2.magi2_config import Magi2Config
from megatron.core.transformer.module import mark_keep_in_fp32


@dataclass(frozen=True)
class Magi2MHCState:
    """Unactivated post-branch and residual mappings computed before a branch."""

    raw_post: Tensor
    raw_residual: Tensor


def magi2_sinkhorn(logits: Tensor, iterations: int, eps: float) -> Tensor:
    """Apply the official column-then-row Sinkhorn-Knopp normalization."""
    matrix = torch.exp(logits - logits.amax(dim=(-2, -1), keepdim=True))
    for _ in range(iterations):
        matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + eps)
        matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + eps)
    return matrix


class Magi2MHCBranch(nn.Module):
    """One attention or MLP branch of official four-stream MAGI-2 mHC."""

    def __init__(self, config: Magi2Config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_streams = config.magi2_mhc_num_streams
        self.sinkhorn_iterations = config.magi2_mhc_sinkhorn_iterations
        self.sinkhorn_eps = config.magi2_mhc_sinkhorn_eps
        self.matmul_scale = 1.0 / math.sqrt(self.num_streams * self.hidden_size)
        mapping_width = 2 * self.num_streams + self.num_streams**2

        self.alpha_pre = nn.Parameter(
            torch.full((1,), config.magi2_mhc_alpha_init, dtype=torch.float32)
        )
        self.alpha_post = nn.Parameter(
            torch.full((1,), config.magi2_mhc_alpha_init, dtype=torch.float32)
        )
        self.alpha_res = nn.Parameter(
            torch.full((1,), config.magi2_mhc_alpha_init, dtype=torch.float32)
        )
        self.bias_pre = nn.Parameter(torch.zeros(self.num_streams, dtype=torch.float32))
        self.bias_post = nn.Parameter(torch.zeros(self.num_streams, dtype=torch.float32))
        self.bias_res = nn.Parameter(
            torch.zeros(self.num_streams, self.num_streams, dtype=torch.float32)
        )
        self.phi_fused = nn.Parameter(
            torch.empty(self.num_streams * self.hidden_size, mapping_width, dtype=torch.float32)
        )
        for parameter in self.parameters():
            mark_keep_in_fp32(parameter)
        if config.perform_initialization:
            nn.init.xavier_uniform_(self.phi_fused)

    def prepare(
        self, hidden_states: Tensor, normalized_states: Tensor
    ) -> tuple[Tensor, Magi2MHCState]:
        """Compute dynamic mappings and aggregate residual streams to one stream."""
        expected_width = self.num_streams * self.hidden_size
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != expected_width:
            raise ValueError(f"hidden_states must have shape [sequence, batch, {expected_width}]")
        if normalized_states.shape != hidden_states.shape:
            raise ValueError("normalized_states must match hidden_states")

        projected = torch.matmul(normalized_states.float(), self.phi_fused)
        raw_pre, raw_post, raw_residual = torch.split(
            projected, [self.num_streams, self.num_streams, self.num_streams**2], dim=-1
        )
        h_pre = torch.sigmoid(self.alpha_pre * self.matmul_scale * raw_pre + self.bias_pre).to(
            hidden_states.dtype
        )
        streams = hidden_states.view(
            hidden_states.shape[0], hidden_states.shape[1], self.num_streams, self.hidden_size
        )
        branch_input = torch.einsum("sbn,sbnc->sbc", h_pre, streams)
        state = Magi2MHCState(
            raw_post=raw_post,
            raw_residual=raw_residual.view(
                hidden_states.shape[0], hidden_states.shape[1], self.num_streams, self.num_streams
            ),
        )
        return branch_input, state

    def compute_post_residual(
        self, state: Magi2MHCState, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        """Activate the branch expansion and doubly stochastic residual mappings."""
        h_post = (
            2.0
            * torch.sigmoid(self.alpha_post * self.matmul_scale * state.raw_post + self.bias_post)
        ).to(dtype)
        residual_logits = self.alpha_res * self.matmul_scale * state.raw_residual + self.bias_res
        h_residual = magi2_sinkhorn(
            residual_logits.float(), self.sinkhorn_iterations, self.sinkhorn_eps
        ).to(dtype)
        return h_post, h_residual

    def merge(self, residual: Tensor, output: Tensor, state: Magi2MHCState) -> Tensor:
        """Mix residual streams and inject a single-stream branch output."""
        if output.shape != (*residual.shape[:2], self.hidden_size):
            raise ValueError("branch output must have shape [sequence, batch, hidden_size]")
        h_post, h_residual = self.compute_post_residual(state, residual.dtype)
        residual_streams = residual.view(
            residual.shape[0], residual.shape[1], self.num_streams, self.hidden_size
        )
        mixed_residual = torch.einsum("sbij,sbjc->sbic", h_residual, residual_streams)
        expanded_output = torch.einsum("sbn,sbc->sbnc", h_post, output)
        return (mixed_residual + expanded_output).reshape_as(residual)
