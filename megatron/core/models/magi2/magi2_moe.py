# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Distributed multi-head MoE operators for native MCore MAGI-2."""

from __future__ import annotations

from dataclasses import fields
from functools import partial
from typing import Literal, TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.fusions.fused_bias_geglu import quick_gelu
from megatron.core.models.magi2.magi2_config import Magi2Config
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import mark_keep_in_fp32
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig

Magi2RouterScoreFunction: TypeAlias = Literal["sigmoid", "softmax"]


def multi_head_topk_routing(
    hidden_states: Tensor,
    gate: Tensor,
    expert_bias: Tensor | None,
    *,
    num_heads: int,
    num_experts_per_head: int,
    top_k: int,
    score_func: Magi2RouterScoreFunction,
    route_norm: bool,
    route_scale: float,
    route_norm_eps: float,
) -> tuple[Tensor, Tensor]:
    """Build an MCore routing map while keeping expert selection head-local.

    Head tokens must be flattened in token-major order. The returned expert IDs
    use the public checkpoint layout ``head * experts_per_head + expert``.
    """
    if hidden_states.ndim < 2:
        raise ValueError("hidden_states must have at least two dimensions")
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if num_experts_per_head <= 0:
        raise ValueError("num_experts_per_head must be positive")
    if not 0 < top_k <= num_experts_per_head:
        raise ValueError("top_k must be in [1, num_experts_per_head]")
    if score_func not in ("sigmoid", "softmax"):
        raise ValueError(f"unknown score function: {score_func}")
    if route_scale <= 0.0:
        raise ValueError("route_scale must be positive")
    if route_norm_eps <= 0.0:
        raise ValueError("route_norm_eps must be positive")

    head_dim = hidden_states.shape[-1]
    flattened_hidden_states = hidden_states.reshape(-1, head_dim)
    if flattened_hidden_states.shape[0] % num_heads:
        raise ValueError("the flattened token count must be divisible by num_heads")

    flattened_num_experts = num_heads * num_experts_per_head
    if gate.shape != (flattened_num_experts, head_dim):
        raise ValueError(
            "gate shape must be " f"({flattened_num_experts}, {head_dim}), got {tuple(gate.shape)}"
        )
    if expert_bias is not None and expert_bias.shape != (flattened_num_experts,):
        raise ValueError(
            f"expert_bias shape must be ({flattened_num_experts},), "
            f"got {tuple(expert_bias.shape)}"
        )

    num_tokens = flattened_hidden_states.shape[0] // num_heads
    input_by_head = flattened_hidden_states.reshape(num_tokens, num_heads, head_dim)
    gate_by_head = gate.reshape(num_heads, num_experts_per_head, head_dim)
    router_logits = torch.einsum("thd,hed->hte", input_by_head.float(), gate_by_head.float())
    if score_func == "sigmoid":
        router_scores = torch.sigmoid(router_logits)
    else:
        router_scores = F.softmax(router_logits, dim=-1)

    selection_scores = router_scores
    if expert_bias is not None:
        selection_scores = selection_scores + expert_bias.float().reshape(
            num_heads, 1, num_experts_per_head
        )
    local_indices = torch.topk(selection_scores, top_k, dim=-1).indices
    topk_probs = router_scores.gather(-1, local_indices)
    if route_norm:
        topk_probs = F.normalize(topk_probs, p=1, dim=-1, eps=route_norm_eps)
    topk_probs = topk_probs * route_scale

    head_offsets = (
        torch.arange(num_heads, device=hidden_states.device).reshape(num_heads, 1, 1)
        * num_experts_per_head
    )
    global_indices = (local_indices + head_offsets).permute(1, 0, 2).reshape(-1, top_k)
    topk_probs = topk_probs.permute(1, 0, 2).reshape(-1, top_k)

    probs = torch.zeros(
        flattened_hidden_states.shape[0],
        flattened_num_experts,
        dtype=topk_probs.dtype,
        device=hidden_states.device,
    )
    routing_map = torch.zeros(
        flattened_hidden_states.shape[0],
        flattened_num_experts,
        dtype=torch.bool,
        device=hidden_states.device,
    )
    probs.scatter_(1, global_indices, topk_probs)
    routing_map.scatter_(1, global_indices, True)
    return probs, routing_map


class Magi2MultiHeadTopKRouter(TopKRouter):
    """MCore router that restricts every head token to its own expert bank."""

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection: ProcessGroupCollection | None = None,
        is_mtp_layer: bool = False,
        *,
        num_heads: int,
        num_experts_per_head: int,
        route_norm: bool,
        route_scale: float,
        route_norm_eps: float,
    ) -> None:
        if config.num_moe_experts != num_heads * num_experts_per_head:
            raise ValueError("config.num_moe_experts must equal num_heads * num_experts_per_head")
        super().__init__(config=config, pg_collection=pg_collection, is_mtp_layer=is_mtp_layer)
        self.num_heads = num_heads
        self.num_experts_per_head = num_experts_per_head
        self.route_norm = route_norm
        self.route_scale = route_scale
        self.route_norm_eps = route_norm_eps
        self.weight.data = self.weight.data.float()
        mark_keep_in_fp32(self.weight)
        if self.expert_bias is None:
            raise ValueError("MAGI-2 routing requires expert bias")
        self.register_buffer("expert_bias_ema", torch.zeros_like(self.expert_bias))

    @property
    def gate(self) -> Tensor:
        """Expose the router weight under the public MAGI-2 checkpoint name."""
        return self.weight

    def _maintain_float32_expert_bias(self) -> None:
        """Keep the training and inference router biases in FP32."""
        super()._maintain_float32_expert_bias()
        if self.expert_bias_ema.dtype != torch.float32:
            self.expert_bias_ema.data = self.expert_bias_ema.data.float()

    def forward(self, input: Tensor, padding_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Route flattened head tokens without cross-head expert choices."""
        self._maintain_float32_expert_bias()
        probs, routing_map = multi_head_topk_routing(
            input,
            self.weight,
            self.expert_bias,
            num_heads=self.num_heads,
            num_experts_per_head=self.num_experts_per_head,
            top_k=self.topk,
            score_func=self.score_function,
            route_norm=self.route_norm,
            route_scale=self.route_scale,
            route_norm_eps=self.route_norm_eps,
        )
        flattened_padding_mask = None
        if padding_mask is not None:
            flattened_padding_mask = padding_mask.reshape(-1)
        self._apply_expert_bias(routing_map, padding_mask=flattened_padding_mask)
        return probs, routing_map

    @torch.no_grad()
    def update_expert_bias_ema(self, decay: float) -> None:
        """Update the inference router-bias EMA after a global training step."""
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        self.expert_bias_ema.mul_(decay).add_(self.expert_bias, alpha=1.0 - decay)


def build_magi2_expert_config(config: Magi2Config) -> TransformerConfig:
    """Derive the head-sized MCore configuration used by routed experts."""
    values = {
        config_field.name: getattr(config, config_field.name)
        for config_field in fields(TransformerConfig)
        if config_field.init
    }
    values.update(
        hidden_size=config.magi2_moe_head_dim,
        ffn_hidden_size=config.magi2_moe_expert_intermediate_size,
        num_attention_heads=1,
        num_query_groups=1,
        kv_channels=config.magi2_moe_head_dim,
        num_moe_experts=config.magi2_flattened_num_experts,
        moe_ffn_hidden_size=config.magi2_moe_expert_intermediate_size,
        moe_layer_freq=1,
        moe_router_load_balancing_type="none",
        moe_router_topk=config.magi2_moe_top_k,
        moe_router_pre_softmax=False,
        moe_router_topk_scaling_factor=config.magi2_route_scale,
        moe_router_score_function=config.moe_router_score_function,
        moe_router_dtype="fp32",
        moe_router_enable_expert_bias=True,
        moe_aux_loss_coeff=0.0,
        moe_grouped_gemm=True,
        moe_shared_expert_intermediate_size=None,
        moe_shared_expert_overlap=False,
        moe_expert_capacity_factor=None,
        moe_pad_expert_input_to_capacity=False,
        moe_apply_probs_on_input=False,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=quick_gelu,
        activation_func_clamp_value=7.0,
        glu_linear_offset=1.0,
        bias_activation_fusion=True,
        use_te_activation_func=False,
        use_transformer_engine_op_fuser=False,
        enable_mhc_connections=False,
    )
    return TransformerConfig(**values)


class Magi2DistributedMultiHeadMoE(MoELayer):
    """MAGI-2 multi-head MoE backed by MCore dispatch and grouped experts."""

    def __init__(
        self,
        config: Magi2Config,
        layer_number: int,
        pg_collection: ProcessGroupCollection,
        is_mtp_layer: bool = False,
        name: str | None = None,
    ) -> None:
        expert_config = build_magi2_expert_config(config)
        router = partial(
            Magi2MultiHeadTopKRouter,
            num_heads=config.magi2_moe_num_heads,
            num_experts_per_head=config.magi2_moe_num_experts_per_head,
            route_norm=config.magi2_route_norm,
            route_scale=config.magi2_route_scale,
            route_norm_eps=config.magi2_route_norm_eps,
        )
        experts = TESpecProvider().grouped_mlp_modules(True)
        super().__init__(
            config=expert_config,
            submodules=MoESubmodules(router=router, experts=experts),
            layer_number=layer_number,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
            name=name,
        )
        self.magi2_config = config
        self.magi2_hidden_size = config.hidden_size
        self.num_heads = config.magi2_moe_num_heads
        self.head_dim = config.magi2_moe_head_dim

    def forward(
        self, hidden_states: Tensor, padding_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Split hidden states into head tokens, dispatch, and restore shape."""
        if hidden_states.ndim not in (2, 3):
            raise ValueError(
                "hidden_states must have shape [tokens, hidden] or " "[sequence, batch, hidden]"
            )
        if hidden_states.shape[-1] != self.magi2_hidden_size:
            raise ValueError(f"hidden_states last dimension must equal {self.magi2_hidden_size}")

        squeeze_batch = hidden_states.ndim == 2
        if squeeze_batch:
            hidden_states = hidden_states.unsqueeze(1)
        sequence_length, batch_size, _ = hidden_states.shape
        head_tokens = hidden_states.reshape(-1, 1, self.head_dim)

        flattened_padding_mask = None
        if padding_mask is not None:
            if padding_mask.shape != (batch_size, sequence_length):
                raise ValueError("padding_mask must have shape [batch, sequence]")
            flattened_padding_mask = (
                padding_mask.T.reshape(-1, 1).expand(-1, self.num_heads).reshape(-1)
            )

        output, output_bias = super().forward(head_tokens, padding_mask=flattened_padding_mask)
        output = output.reshape(sequence_length, batch_size, self.magi2_hidden_size)
        if squeeze_batch:
            output = output.squeeze(1)
        return output, output_bias
