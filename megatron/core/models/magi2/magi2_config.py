# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Configuration for the native Megatron Core MAGI-2 model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class Magi2Config(TransformerConfig):
    """Megatron Core configuration for the public MAGI-2 Preview architecture.

    The inherited TransformerConfig fields configure MCore execution. Fields
    prefixed with ``magi2_`` describe topology that is specific to MAGI-2.
    Reduced values are supported for unit tests, but all topology relationships
    are validated in the same way as the released model.
    """

    num_layers: int = 40
    hidden_size: int = 3072
    num_attention_heads: int = 24
    num_query_groups: int | None = 24
    kv_channels: int | None = 128
    ffn_hidden_size: int | None = 8192
    num_moe_experts: int | None = 3072
    moe_ffn_hidden_size: int | None = 1280
    moe_router_topk: int = 6
    moe_router_score_function: str = "sigmoid"
    moe_router_dtype: Literal["fp32", "fp64"] | None = "fp32"
    moe_router_enable_expert_bias: bool = True
    moe_router_load_balancing_type: str = "none"
    moe_token_dispatcher_type: str = "alltoall"
    moe_grouped_gemm: bool = True
    params_dtype: torch.dtype = torch.bfloat16
    pipeline_dtype: torch.dtype | None = torch.bfloat16
    add_bias_linear: bool = False
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    softmax_type: Literal["vanilla", "off-by-one", "learnable"] = "learnable"
    attention_output_gate: bool = True
    normalization: str = "RMSNorm"
    layernorm_epsilon: float = 1e-6
    hetereogenous_dist_checkpoint: bool = True
    # MCore's generic mHC expands H -> nH at block entry. MAGI-2 adapters
    # already produce nH, so MAGI-2 layers own their mHC implementation.
    enable_mhc_connections: bool = False

    magi2_video_in_channels: int = 48
    magi2_audio_in_channels: int = 64
    magi2_text_in_channels: int = 5120
    magi2_intermediate_factor: int = 4
    magi2_mm_layers: tuple[int, ...] = (0, 1, 38, 39)
    magi2_moe_layers: tuple[int, ...] = field(default_factory=lambda: tuple(range(2, 38)))
    magi2_moe_num_heads: int = 12
    magi2_moe_num_experts_per_head: int = 256
    magi2_moe_top_k: int = 6
    magi2_moe_expert_intermediate_size: int = 1280
    magi2_shared_expert_intermediate_size: int = 1280
    magi2_modality_expert_intermediate_size: int = 1280
    magi2_route_scale: float = 4.9
    magi2_route_norm: bool = True
    magi2_route_norm_eps: float = 1e-12
    magi2_sink_token_num: int = 1
    magi2_attention_softcap: float = -1.0
    magi2_mhc_num_streams: int = 4
    magi2_mhc_alpha_init: float = 0.01
    magi2_mhc_sinkhorn_iterations: int = 20
    magi2_mhc_sinkhorn_eps: float = 1e-12

    def __post_init__(self) -> None:
        """Validate MCore execution fields and MAGI-2 topology."""
        super().__post_init__()

        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.kv_channels != self.hidden_size // self.num_attention_heads:
            raise ValueError("kv_channels must equal the MAGI-2 attention head dimension")
        if self.num_query_groups != self.num_attention_heads:
            raise ValueError("MAGI-2 Preview uses one KV head per query head")
        if self.hidden_size % self.magi2_moe_num_heads:
            raise ValueError("hidden_size must be divisible by magi2_moe_num_heads")
        if not 0 < self.magi2_moe_top_k <= self.magi2_moe_num_experts_per_head:
            raise ValueError("magi2_moe_top_k must be in [1, magi2_moe_num_experts_per_head]")
        if self.moe_router_topk != self.magi2_moe_top_k:
            raise ValueError("MCore and MAGI-2 router top-k values must match")
        if self.num_moe_experts != self.magi2_flattened_num_experts:
            raise ValueError("num_moe_experts must equal heads * experts_per_head for MAGI-2")
        if self.moe_ffn_hidden_size != self.magi2_moe_expert_intermediate_size:
            raise ValueError("MCore and MAGI-2 expert intermediate sizes must match")
        if self.moe_router_score_function != "sigmoid":
            raise ValueError("MAGI-2 Preview requires sigmoid expert routing")
        if self.moe_router_dtype != "fp32":
            raise ValueError("MAGI-2 Preview requires FP32 expert routing")
        if self.moe_router_load_balancing_type != "none":
            raise ValueError("MAGI-2 Preview uses bias-balanced routing without an auxiliary loss")
        if not self.moe_router_enable_expert_bias:
            raise ValueError("MAGI-2 Preview requires expert-selection bias")
        if self.moe_token_dispatcher_type != "alltoall":
            raise ValueError("MAGI-2 distributed experts require the all-to-all dispatcher")
        if not self.moe_grouped_gemm:
            raise ValueError("MAGI-2 distributed experts require grouped GEMM")
        if not self.magi2_route_norm:
            raise ValueError("MAGI-2 Preview requires normalized top-k route weights")
        if self.num_moe_experts % self.expert_model_parallel_size:
            raise ValueError("flattened MAGI-2 experts must divide the expert parallel size")
        if self.magi2_mhc_num_streams <= 0:
            raise ValueError("magi2_mhc_num_streams must be positive")
        if self.magi2_sink_token_num != 1:
            raise ValueError("MAGI-2 Preview uses exactly one learnable attention sink per head")
        if self.softmax_type != "learnable":
            raise ValueError("MAGI-2 Preview requires learnable sink softmax")
        if not self.attention_output_gate:
            raise ValueError("MAGI-2 Preview requires scalar attention output gating")
        if self.magi2_attention_softcap != -1.0:
            raise ValueError("MAGI-2 Preview requires attention softcap=-1.0")
        if self.magi2_route_scale <= 0 or self.magi2_route_norm_eps <= 0:
            raise ValueError("MAGI-2 routing scale and epsilon must be positive")
        if self.magi2_mhc_sinkhorn_iterations <= 0 or self.magi2_mhc_sinkhorn_eps <= 0:
            raise ValueError("MAGI-2 mHC Sinkhorn settings must be positive")
        if self.enable_mhc_connections:
            raise ValueError(
                "MCore generic mHC must remain disabled because MAGI-2 adapters already output nH"
            )
        if not self.hetereogenous_dist_checkpoint:
            raise ValueError("MAGI-2 requires non-homogeneous distributed-checkpoint layer keys")

        mm_layers = set(self.magi2_mm_layers)
        moe_layers = set(self.magi2_moe_layers)
        all_layers = set(range(self.num_layers))
        if mm_layers & moe_layers:
            raise ValueError("magi2_mm_layers and magi2_moe_layers must be disjoint")
        if mm_layers | moe_layers != all_layers:
            raise ValueError("every MAGI-2 layer must be assigned to the dense or MoE stack")
        if self.tensor_model_parallel_size != 1:
            raise ValueError("MAGI-2 tensor parallelism is not implemented yet")
        if self.pipeline_model_parallel_size != 1:
            raise ValueError("MAGI-2 pipeline parallelism is not implemented yet")
        if self.context_parallel_size != 1:
            raise ValueError("MAGI-2 context parallelism is not implemented yet")
        if self.sequence_parallel:
            raise ValueError("MAGI-2 sequence parallelism requires tensor parallelism support")

    @property
    def magi2_adapter_width(self) -> int:
        """Width of the flattened MAGI-2 residual streams."""
        return self.hidden_size * self.magi2_mhc_num_streams

    @property
    def magi2_attention_head_dim(self) -> int:
        """Dimension of one attention head."""
        return self.hidden_size // self.num_attention_heads

    @property
    def magi2_moe_head_dim(self) -> int:
        """Dimension routed by one multi-head-MoE head."""
        return self.hidden_size // self.magi2_moe_num_heads

    @property
    def magi2_flattened_num_experts(self) -> int:
        """Number of globally addressable ``(head, expert)`` pairs."""
        return self.magi2_moe_num_heads * self.magi2_moe_num_experts_per_head

    @property
    def magi2_dense_intermediate_size(self) -> int:
        """Aligned QuickGEGLU7 intermediate width used by dense layers."""
        return max(128, int(self.hidden_size * self.magi2_intermediate_factor * 2 / 3) // 128 * 128)

    def magi2_parameter_count_breakdown(self) -> dict[str, int]:
        """Return the logical global parameter count implied by the config."""
        hidden = self.hidden_size
        adapter = self.magi2_adapter_width
        expert_hidden = self.magi2_moe_head_dim
        mapping_width = 2 * self.magi2_mhc_num_streams + self.magi2_mhc_num_streams**2
        mhc_branch = (
            adapter * mapping_width
            + 3
            + 2 * self.magi2_mhc_num_streams
            + self.magi2_mhc_num_streams**2
        )
        routed_experts = (
            len(self.magi2_moe_layers)
            * self.magi2_flattened_num_experts
            * 3
            * expert_hidden
            * self.magi2_moe_expert_intermediate_size
        )
        adapters = (
            adapter
            * (
                self.magi2_video_in_channels
                + self.magi2_audio_in_channels
                + self.magi2_text_in_channels
                + 3
            )
            + 2 * adapter
            + adapter * (self.magi2_video_in_channels + self.magi2_audio_in_channels)
        )

        attention_and_mhc = 0
        other_mlp_and_router = 0
        moe_layers = set(self.magi2_moe_layers)
        mm_layers = set(self.magi2_mm_layers)
        for layer in range(self.num_layers):
            num_modalities = 3 if layer in mm_layers else 1
            attention_and_mhc += (
                num_modalities * hidden
                + 2 * num_modalities * self.magi2_attention_head_dim
                + num_modalities * self.num_attention_heads * hidden
                + num_modalities * 4 * hidden * hidden
                + self.magi2_sink_token_num * self.num_attention_heads
                + adapter * num_modalities
                + 2 * mhc_branch
            )
            if layer in moe_layers:
                other_mlp_and_router += (
                    3 * hidden
                    + 2 * hidden * hidden
                    + self.magi2_flattened_num_experts * expert_hidden
                    + 3 * hidden * self.magi2_shared_expert_intermediate_size
                    + 9 * hidden * self.magi2_modality_expert_intermediate_size
                )
            else:
                other_mlp_and_router += num_modalities * (
                    hidden + 3 * hidden * self.magi2_dense_intermediate_size
                )
        return {
            "adapters": adapters,
            "attention_and_mhc": attention_and_mhc,
            "routed_experts": routed_experts,
            "other_mlp_and_router": other_mlp_and_router,
        }

    @property
    def magi2_parameter_count(self) -> int:
        """Logical global MAGI-2 parameter count, including every expert once."""
        return sum(self.magi2_parameter_count_breakdown().values())
