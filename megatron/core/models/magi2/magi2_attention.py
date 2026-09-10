# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Modality-aware packed self-attention for native MCore MAGI-2."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from megatron.core.models.magi2.magi2_config import Magi2Config
from megatron.core.models.magi2.magi2_modalities import (
    Magi2ModalityLinear,
    Magi2MultiModalityRMSNorm,
)
from megatron.core.models.magi2.magi2_rope import apply_magi2_rotary_pos_emb
from megatron.core.models.magi2.magi2_runtime_context import Magi2RuntimeContext
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.spec_utils import ModuleSpec, build_module


@dataclass(frozen=True)
class Magi2AttentionSubmodules:
    """Injectable MCore attention kernel used after MAGI-2 QKV preparation."""

    core_attention: ModuleSpec | type


class Magi2TorchDotProductAttention(MegatronModule):
    """Differentiable packed attention reference with a learnable sink per head.

    This self-contained implementation is the unit-test oracle. Production
    correctness specs use :class:`Magi2DotProductAttention` to share MCore's
    generic attention implementation.
    """

    def __init__(
        self,
        config: Magi2Config,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float | None = None,
        softmax_scale: float | None = None,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
    ) -> None:
        super().__init__(config=config)
        del layer_number, cp_comm_type
        if attn_mask_type is not AttnMaskType.no_mask or attention_type != "self":
            raise ValueError("MAGI-2 uses bidirectional self-attention without an explicit mask")
        if (
            attention_dropout if attention_dropout is not None else config.attention_dropout
        ) != 0.0:
            raise ValueError("the MAGI-2 Preview attention dropout must be zero")
        if config.tensor_model_parallel_size != 1:
            raise ValueError("the PyTorch MAGI-2 attention core only supports TP=1")

        self.pg_collection = pg_collection
        self.num_heads = config.num_attention_heads
        self.head_dim = config.magi2_attention_head_dim
        self.softmax_scale = softmax_scale or self.head_dim**-0.5
        self.softmax_offset = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        mark_keep_in_fp32(self.softmax_offset)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        *,
        attn_mask_type: AttnMaskType,
        attention_bias: Tensor | None = None,
        packed_seq_params: Magi2RuntimeContext | None = None,
    ) -> Tensor:
        """Apply independent full attention inside every packed sequence."""
        if attention_mask is not None or attention_bias is not None:
            raise ValueError("MAGI-2 packed attention does not consume an external mask or bias")
        if attn_mask_type is not AttnMaskType.no_mask:
            raise ValueError("MAGI-2 attention must use AttnMaskType.no_mask")
        if packed_seq_params is None or packed_seq_params.cu_seqlens_q is None:
            raise ValueError("MAGI-2 attention requires packed sequence boundaries")
        if query.shape != key.shape or query.shape != value.shape or query.ndim not in (3, 4):
            raise ValueError("query, key, and value must have matching THD or TBHD shapes")

        thd_input = query.ndim == 3
        if thd_input:
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)

        boundaries = tuple(int(item) for item in packed_seq_params.cu_seqlens_q.cpu().tolist())
        outputs = []
        sink = self.softmax_offset.view(1, self.num_heads, 1, 1)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            q_segment = query[start:end].permute(1, 2, 0, 3).float()
            k_segment = key[start:end].permute(1, 2, 0, 3).float()
            v_segment = value[start:end].permute(1, 2, 0, 3).float()
            logits = torch.matmul(q_segment, k_segment.transpose(-1, -2)) * self.softmax_scale
            max_logits = torch.maximum(logits.amax(dim=-1, keepdim=True), sink)
            exp_logits = torch.exp(logits - max_logits)
            denominator = exp_logits.sum(dim=-1, keepdim=True) + torch.exp(sink - max_logits)
            attended = torch.matmul(exp_logits / denominator, v_segment)
            outputs.append(attended.permute(2, 0, 1, 3).to(query.dtype))

        output = torch.cat(outputs, dim=0)
        if thd_input:
            return output.squeeze(1)
        return output.reshape(output.shape[0], output.shape[1], -1)


class Magi2DotProductAttention(DotProductAttention):
    """Packed-sequence adapter around MCore's native dot-product attention.

    MCore's reference attention already implements MAGI-2's learnable sink
    softmax, mixed-precision score calculation, dropout, and checkpoint
    sharding. MAGI-2 only adds segmentation at the packed-sequence boundaries
    because the generic implementation accepts one dense sequence at a time.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.attn_mask_type is not AttnMaskType.no_mask or self.attention_type != "self":
            raise ValueError("MAGI-2 uses bidirectional self-attention without an explicit mask")
        if self.softmax_offset is None:
            raise ValueError("MAGI-2 requires a learnable attention sink")
        self.softmax_offset.data = self.softmax_offset.data.float()
        mark_keep_in_fp32(self.softmax_offset)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        *,
        attn_mask_type: AttnMaskType | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params: Magi2RuntimeContext | None = None,
    ) -> Tensor:
        """Run native MCore attention independently for each packed sequence."""
        if attention_mask is not None or attention_bias is not None:
            raise ValueError("MAGI-2 packed attention does not consume an external mask or bias")
        if attn_mask_type not in (None, AttnMaskType.no_mask):
            raise ValueError("MAGI-2 attention must use AttnMaskType.no_mask")
        if packed_seq_params is None or packed_seq_params.cu_seqlens_q is None:
            raise ValueError("MAGI-2 attention requires packed sequence boundaries")
        if query.shape != key.shape or query.shape != value.shape or query.ndim not in (3, 4):
            raise ValueError("query, key, and value must have matching THD or TBHD shapes")

        thd_input = query.ndim == 3
        if thd_input:
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)

        boundaries = tuple(int(item) for item in packed_seq_params.cu_seqlens_q.cpu().tolist())
        outputs = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            output = super().forward(
                query[start:end],
                key[start:end],
                value[start:end],
                None,
                attn_mask_type=AttnMaskType.no_mask,
                attention_bias=None,
                packed_seq_params=None,
            )
            outputs.append(
                output.view(
                    end - start,
                    query.shape[1],
                    self.num_attention_heads_per_partition,
                    self.hidden_size_per_attention_head,
                )
            )

        output = torch.cat(outputs, dim=0)
        if thd_input:
            return output.squeeze(1)
        return output.reshape(output.shape[0], output.shape[1], -1)


class Magi2Attention(MegatronModule):
    """Official MAGI-2 attention projections around an MCore attention kernel."""

    def __init__(
        self,
        config: Magi2Config,
        submodules: Magi2AttentionSubmodules,
        layer_number: int,
        num_modalities: int,
        pg_collection: ProcessGroupCollection,
    ) -> None:
        super().__init__(config=config)
        self.config = config
        self.layer_number = layer_number
        self.num_modalities = num_modalities
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.magi2_attention_head_dim

        self.pre_norm = Magi2MultiModalityRMSNorm(
            self.hidden_size, num_modalities=num_modalities, eps=config.layernorm_epsilon
        )
        self.q_norm = Magi2MultiModalityRMSNorm(
            self.head_dim,
            num_modalities=num_modalities,
            eps=config.layernorm_epsilon,
            out_dtype=torch.float32,
        )
        self.k_norm = Magi2MultiModalityRMSNorm(
            self.head_dim,
            num_modalities=num_modalities,
            eps=config.layernorm_epsilon,
            out_dtype=torch.float32,
        )
        linear_kwargs = {
            "num_modalities": num_modalities,
            "dtype": config.params_dtype,
            "init_method": config.init_method,
            "perform_initialization": config.perform_initialization,
        }
        self.linear_g = Magi2ModalityLinear(
            self.hidden_size, self.num_heads, bias=False, **linear_kwargs
        )
        self.linear_qkv = Magi2ModalityLinear(
            self.hidden_size, 3 * self.hidden_size, bias=False, **linear_kwargs
        )
        self.linear_proj = Magi2ModalityLinear(
            self.hidden_size, self.hidden_size, bias=False, **linear_kwargs
        )
        self.core_attention = build_module(
            submodules.core_attention,
            config=config,
            layer_number=layer_number,
            attn_mask_type=AttnMaskType.no_mask,
            attention_type="self",
            attention_dropout=config.attention_dropout,
            softmax_scale=config.softmax_scale,
            cp_comm_type=None,
            pg_collection=pg_collection,
        )

    @property
    def sinks(self) -> Tensor:
        """Return the official ``[1, heads]`` view of the MCore sink parameter."""
        softmax_offset = getattr(self.core_attention, "softmax_offset", None)
        if softmax_offset is None:
            raise RuntimeError("the selected core attention does not expose learnable sinks")
        return softmax_offset.unsqueeze(0)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
        runtime_context: Magi2RuntimeContext,
    ) -> Tensor:
        """Run grouped projections, packed attention, output gate, and projection."""
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden_states must have shape [sequence, batch, hidden_size]")
        if hidden_states.shape[1] != 1:
            raise ValueError("MAGI-2 currently expects packed hidden states with batch size one")
        if runtime_context.rope is None:
            raise ValueError("runtime_context.rope must be prepared by Magi2Model")

        dispatcher = runtime_context.get_modality_dispatcher()
        normalized = self.pre_norm(hidden_states, dispatcher)
        gate = self.linear_g(normalized, dispatcher).view(
            hidden_states.shape[0], hidden_states.shape[1], self.num_heads, 1
        )
        qkv = self.linear_qkv(normalized, dispatcher)
        query, key, value = torch.split(qkv, self.hidden_size, dim=-1)
        query = query.view(*hidden_states.shape[:2], self.num_heads, self.head_dim)
        key = key.view(*hidden_states.shape[:2], self.num_heads, self.head_dim)
        value = value.view(*hidden_states.shape[:2], self.num_heads, self.head_dim)
        query = self.q_norm(query, dispatcher)
        key = self.k_norm(key, dispatcher)

        query = dispatcher.inverse_permute(query)
        key = dispatcher.inverse_permute(key)
        value = dispatcher.inverse_permute(value)
        query = apply_magi2_rotary_pos_emb(query, runtime_context.rope).to(self.config.params_dtype)
        key = apply_magi2_rotary_pos_emb(key, runtime_context.rope).to(self.config.params_dtype)
        value = value.to(self.config.params_dtype)
        output = self.core_attention(
            query.squeeze(1),
            key.squeeze(1),
            value.squeeze(1),
            attention_mask,
            attn_mask_type=AttnMaskType.no_mask,
            attention_bias=None,
            packed_seq_params=runtime_context,
        )
        output = output.reshape(output.shape[0], 1, -1)
        output = output.view(*hidden_states.shape[:2], self.num_heads, self.head_dim)
        output = dispatcher.permute(output) * torch.sigmoid(gate)
        output = output.reshape(*hidden_states.shape[:2], self.hidden_size)
        return self.linear_proj(output.to(self.linear_proj.weight.dtype), dispatcher)
