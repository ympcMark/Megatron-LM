# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Pure-PyTorch training oracle following the public MAGI-2 module layout."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from megatron.core.models.magi2 import Magi2Config, Magi2Modality


class ReferenceDispatcher:
    """Minimal modality dispatcher matching the public implementation."""

    def __init__(self, modality_mapping: Tensor) -> None:
        self.modality_mapping = modality_mapping
        self.permute_mapping = torch.argsort(modality_mapping, stable=True)
        self.inverse_mapping = torch.argsort(self.permute_mapping)
        sorted_mapping = modality_mapping.index_select(0, self.permute_mapping)
        self.group_sizes = tuple(
            int(size) for size in torch.bincount(sorted_mapping, minlength=3).cpu().tolist()
        )

    def permute(self, value: Tensor) -> Tensor:
        return value.index_select(0, self.permute_mapping)

    def inverse_permute(self, value: Tensor) -> Tensor:
        return value.index_select(0, self.inverse_mapping)

    def split(self, value: Tensor) -> tuple[Tensor, ...]:
        return torch.split(value, self.group_sizes, dim=0)


class ReferenceRMSNorm(nn.Module):
    """Public zero-centered RMSNorm with flattened modality weights."""

    def __init__(
        self,
        hidden_size: int,
        *,
        num_modalities: int = 1,
        eps: float = 1e-6,
        out_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_modalities = num_modalities
        self.eps = eps
        self.out_dtype = out_dtype
        self.weight = nn.Parameter(torch.zeros(num_modalities * hidden_size, dtype=torch.float32))

    def _normalize(self, value: Tensor, weight: Tensor) -> Tensor:
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * (weight + 1.0)).to(self.out_dtype or value.dtype)

    def forward(self, value: Tensor, dispatcher: ReferenceDispatcher | None = None) -> Tensor:
        weight = self.weight.view(self.num_modalities, self.hidden_size)
        if self.num_modalities == 1:
            return self._normalize(value, weight[0])
        if dispatcher is None:
            raise ValueError("multi-modality reference norm requires a dispatcher")
        return torch.cat(
            [
                self._normalize(group, weight[index])
                for index, group in enumerate(dispatcher.split(value))
            ]
        )


class ReferenceGroupedLinear(nn.Module):
    """Public grouped-linear parameter layout with a PyTorch forward."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        num_modalities: int = 1,
        bias: bool = False,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_modalities = num_modalities
        self.weight = nn.Parameter(
            torch.empty(num_modalities * out_features, in_features, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_modalities * out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        for weight in self.weight.view(num_modalities, out_features, in_features).unbind(0):
            nn.init.xavier_uniform_(weight)

    def forward(self, value: Tensor, dispatcher: ReferenceDispatcher | None = None) -> Tensor:
        weights = self.weight.view(self.num_modalities, self.out_features, self.in_features)
        biases = (
            self.bias.view(self.num_modalities, self.out_features)
            if self.bias is not None
            else None
        )
        if self.num_modalities == 1:
            return F.linear(value, weights[0], None if biases is None else biases[0])
        if dispatcher is None:
            raise ValueError("multi-modality reference linear requires a dispatcher")
        return torch.cat(
            [
                F.linear(group, weights[index], None if biases is None else biases[index])
                for index, group in enumerate(dispatcher.split(value))
            ]
        )


def reference_quick_geglu(value: Tensor) -> Tensor:
    """Public interleaved QuickGEGLU7 equation."""
    output_dtype = value.dtype
    value = value.float()
    gate = value[..., ::2].clamp(max=7.0)
    linear = value[..., 1::2].clamp(min=-7.0, max=7.0)
    return (gate * torch.sigmoid(1.702 * gate) * (linear + 1.0)).to(output_dtype)


class ReferenceRoPE(nn.Module):
    """Public three-axis Fourier embedding."""

    def __init__(self, head_dim: int) -> None:
        super().__init__()
        bands = 10_000.0 ** (-torch.arange(head_dim // 8, dtype=torch.float32) / (head_dim // 8))
        self.register_buffer("bands", bands)

    def forward(self, coordinates: Tensor) -> Tensor:
        xyz = coordinates[:, :3].float()
        sizes = coordinates[:, 3:6].float()
        references = coordinates[:, 6:9].float()
        scales = (references - 1.0) / (sizes - 1.0)
        scales = torch.where((references == 1.0) & (sizes == 1.0), torch.ones_like(scales), scales)
        centers = (sizes - 1.0) / 2.0
        centers[:, 0] = 0.0
        projection = (xyz - centers).unsqueeze(-1) * scales.unsqueeze(-1) * self.bands
        return torch.cat((projection.sin(), projection.cos()), dim=1).flatten(1)


def _apply_rope(value: Tensor, rope: Tensor) -> Tensor:
    sin, cos = rope.tensor_split(2, dim=-1)
    sin = torch.cat((sin, sin), dim=-1).unsqueeze(1)
    cos = torch.cat((cos, cos), dim=-1).unsqueeze(1)
    rotary_width = cos.shape[-1]
    first, second = value[..., :rotary_width].chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    embedded = value[..., :rotary_width] * cos + rotated * sin
    return torch.cat((embedded, value[..., rotary_width:]), dim=-1)


class ReferenceAttention(nn.Module):
    """Differentiable public attention equations and checkpoint names."""

    def __init__(self, config: Magi2Config, num_modalities: int) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        head_dim = config.magi2_attention_head_dim
        num_heads = config.num_attention_heads
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.pre_norm = ReferenceRMSNorm(hidden_size, num_modalities=num_modalities)
        self.q_norm = ReferenceRMSNorm(
            head_dim, num_modalities=num_modalities, out_dtype=torch.float32
        )
        self.k_norm = ReferenceRMSNorm(
            head_dim, num_modalities=num_modalities, out_dtype=torch.float32
        )
        self.linear_g = ReferenceGroupedLinear(
            hidden_size, num_heads, num_modalities=num_modalities, dtype=config.params_dtype
        )
        self.linear_qkv = ReferenceGroupedLinear(
            hidden_size, 3 * hidden_size, num_modalities=num_modalities, dtype=config.params_dtype
        )
        self.linear_proj = ReferenceGroupedLinear(
            hidden_size, hidden_size, num_modalities=num_modalities, dtype=config.params_dtype
        )
        self.sinks = nn.Parameter(torch.zeros(1, num_heads, dtype=torch.float32))

    def forward(
        self,
        hidden_states: Tensor,
        rope: Tensor,
        cu_seqlens: Tensor,
        dispatcher: ReferenceDispatcher,
    ) -> Tensor:
        normalized = self.pre_norm(hidden_states, dispatcher)
        gate = self.linear_g(normalized, dispatcher).view(-1, self.num_heads, 1)
        qkv = self.linear_qkv(normalized, dispatcher)
        query, key, value = torch.split(qkv, self.hidden_size, dim=-1)
        query = self.q_norm(query.view(-1, self.num_heads, self.head_dim), dispatcher)
        key = self.k_norm(key.view(-1, self.num_heads, self.head_dim), dispatcher)
        value = value.view(-1, self.num_heads, self.head_dim)
        query = _apply_rope(dispatcher.inverse_permute(query), rope)
        key = _apply_rope(dispatcher.inverse_permute(key), rope)
        value = dispatcher.inverse_permute(value)

        output = torch.empty_like(value)
        boundaries = tuple(int(item) for item in cu_seqlens.cpu().tolist())
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            q_segment = query[start:end].transpose(0, 1).float()
            k_segment = key[start:end].transpose(0, 1).float()
            v_segment = value[start:end].transpose(0, 1).float()
            logits = torch.matmul(q_segment, k_segment.transpose(-1, -2)) * (self.head_dim**-0.5)
            sink = self.sinks.T.unsqueeze(1).expand(-1, end - start, -1)
            probabilities = torch.softmax(torch.cat((logits, sink), dim=-1), dim=-1)
            attended = torch.matmul(probabilities[..., : end - start], v_segment)
            output[start:end] = attended.transpose(0, 1).to(output.dtype)

        output = dispatcher.permute(output) * torch.sigmoid(gate)
        return self.linear_proj(output.reshape(-1, self.hidden_size), dispatcher)


class ReferenceDenseMLP(nn.Module):
    """Public dense MLP equations and checkpoint names."""

    def __init__(self, config: Magi2Config, num_modalities: int) -> None:
        super().__init__()
        intermediate = config.magi2_dense_intermediate_size
        self.pre_norm = ReferenceRMSNorm(config.hidden_size, num_modalities=num_modalities)
        self.up_gate_proj = ReferenceGroupedLinear(
            config.hidden_size,
            2 * intermediate,
            num_modalities=num_modalities,
            dtype=config.params_dtype,
        )
        self.down_proj = ReferenceGroupedLinear(
            intermediate,
            config.hidden_size,
            num_modalities=num_modalities,
            dtype=config.params_dtype,
        )

    def forward(self, hidden_states: Tensor, dispatcher: ReferenceDispatcher) -> Tensor:
        normalized = self.pre_norm(hidden_states, dispatcher)
        return self.down_proj(
            reference_quick_geglu(self.up_gate_proj(normalized, dispatcher)), dispatcher
        )


class ReferenceRouterBias(nn.Module):
    """Public router bias buffers."""

    def __init__(self, num_experts: int) -> None:
        super().__init__()
        self.register_buffer("expert_bias", torch.zeros(num_experts, dtype=torch.float32))
        self.register_buffer("expert_bias_ema", torch.zeros(num_experts, dtype=torch.float32))


class ReferenceCoreMultiHeadMoE(nn.Module):
    """Pure-PyTorch routed experts with official parameter names and shapes."""

    def __init__(self, config: Magi2Config) -> None:
        super().__init__()
        num_experts = config.magi2_flattened_num_experts
        head_dim = config.magi2_moe_head_dim
        intermediate = config.magi2_moe_expert_intermediate_size
        self.num_heads = config.magi2_moe_num_heads
        self.num_experts_per_head = config.magi2_moe_num_experts_per_head
        self.top_k = config.magi2_moe_top_k
        self.route_scale = config.magi2_route_scale
        self.route_eps = config.magi2_route_norm_eps
        self.gate = nn.Parameter(torch.empty(num_experts, head_dim, dtype=torch.float32))
        self.W_gate = nn.Parameter(
            torch.empty(num_experts, head_dim, intermediate, dtype=config.params_dtype)
        )
        self.W_up = nn.Parameter(
            torch.empty(num_experts, head_dim, intermediate, dtype=config.params_dtype)
        )
        self.W_down = nn.Parameter(
            torch.empty(num_experts, intermediate, head_dim, dtype=config.params_dtype)
        )
        self.router = ReferenceRouterBias(num_experts)
        nn.init.normal_(self.gate, std=0.02)
        for parameter in (self.W_gate, self.W_up, self.W_down):
            nn.init.normal_(parameter, std=0.02)

    def forward(self, hidden_states: Tensor) -> Tensor:
        tokens = hidden_states.shape[0]
        head_dim = hidden_states.shape[-1] // self.num_heads
        inputs = hidden_states.view(tokens, self.num_heads, head_dim)
        gate = self.gate.view(self.num_heads, self.num_experts_per_head, head_dim)
        scores = torch.sigmoid(torch.einsum("thd,hed->hte", inputs.float(), gate))
        bias = self.router.expert_bias.view(self.num_heads, self.num_experts_per_head)
        indices = torch.topk(scores + bias[:, None, :], self.top_k, dim=-1).indices
        probabilities = (
            F.normalize(scores.gather(-1, indices), p=1, dim=-1, eps=self.route_eps)
            * self.route_scale
        )

        outputs: list[Tensor] = []
        for token_index in range(tokens):
            head_outputs: list[Tensor] = []
            for head_index in range(self.num_heads):
                head_output = torch.zeros_like(inputs[token_index, head_index])
                for route_index in range(self.top_k):
                    expert_index = int(indices[head_index, token_index, route_index])
                    global_index = head_index * self.num_experts_per_head + expert_index
                    gate_value = inputs[token_index, head_index] @ self.W_gate[global_index]
                    up_value = inputs[token_index, head_index] @ self.W_up[global_index]
                    activated = gate_value.float().clamp(max=7.0)
                    activated = activated * torch.sigmoid(1.702 * activated)
                    activated = activated * (up_value.float().clamp(-7.0, 7.0) + 1.0)
                    expert_output = activated.to(up_value.dtype) @ self.W_down[global_index]
                    head_output = (
                        head_output
                        + probabilities[head_index, token_index, route_index].to(head_output.dtype)
                        * expert_output
                    )
                head_outputs.append(head_output)
            outputs.append(torch.cat(head_outputs))
        return torch.stack(outputs)


class ReferenceMultiHeadMoELayer(nn.Module):
    """Official routed, global-shared, and modality-shared MoE composition."""

    def __init__(self, config: Magi2Config) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.pre_norm = ReferenceRMSNorm(hidden_size, num_modalities=3)
        self.split_linear = ReferenceGroupedLinear(
            hidden_size, hidden_size, dtype=config.params_dtype
        )
        self.merge_linear = ReferenceGroupedLinear(
            hidden_size, hidden_size, dtype=config.params_dtype
        )
        self.moe_mlp = ReferenceCoreMultiHeadMoE(config)
        self.shared_expert_fc1 = ReferenceGroupedLinear(
            hidden_size, 2 * config.magi2_shared_expert_intermediate_size, dtype=config.params_dtype
        )
        self.shared_expert_fc2 = ReferenceGroupedLinear(
            config.magi2_shared_expert_intermediate_size, hidden_size, dtype=config.params_dtype
        )
        self.modality_specific_shared_expert_fc1 = ReferenceGroupedLinear(
            hidden_size,
            2 * config.magi2_modality_expert_intermediate_size,
            num_modalities=3,
            dtype=config.params_dtype,
        )
        self.modality_specific_shared_expert_fc2 = ReferenceGroupedLinear(
            config.magi2_modality_expert_intermediate_size,
            hidden_size,
            num_modalities=3,
            dtype=config.params_dtype,
        )

    def forward(self, hidden_states: Tensor, dispatcher: ReferenceDispatcher) -> Tensor:
        normalized = self.pre_norm(hidden_states, dispatcher)
        routed = self.merge_linear(self.moe_mlp(self.split_linear(normalized)))
        shared = self.shared_expert_fc2(reference_quick_geglu(self.shared_expert_fc1(normalized)))
        modality = self.modality_specific_shared_expert_fc2(
            reference_quick_geglu(self.modality_specific_shared_expert_fc1(normalized, dispatcher)),
            dispatcher,
        )
        return routed + shared + modality


def _sinkhorn(logits: Tensor, iterations: int, eps: float) -> Tensor:
    matrix = torch.exp(logits - logits.amax(dim=(-2, -1), keepdim=True))
    for _ in range(iterations):
        matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + eps)
        matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + eps)
    return matrix


class ReferenceTransformerLayer(nn.Module):
    """Official flat mHC parameter layout around attention and MLP."""

    def __init__(self, config: Magi2Config, layer_index: int) -> None:
        super().__init__()
        self.config = config
        self.num_streams = config.magi2_mhc_num_streams
        self.scale = 1.0 / math.sqrt(self.num_streams * config.hidden_size)
        num_modalities = 3 if layer_index in config.magi2_mm_layers else 1
        self.attention = ReferenceAttention(config, num_modalities)
        if layer_index in config.magi2_moe_layers:
            self.mlp: nn.Module = ReferenceMultiHeadMoELayer(config)
        else:
            self.mlp = ReferenceDenseMLP(config, num_modalities)
        self.mhc_norm = ReferenceRMSNorm(
            config.magi2_adapter_width, num_modalities=num_modalities, out_dtype=torch.float32
        )
        width = 2 * self.num_streams + self.num_streams**2
        for branch in ("attn", "mlp"):
            setattr(
                self,
                f"mhc_alpha_pre_{branch}",
                nn.Parameter(torch.full((1,), config.magi2_mhc_alpha_init)),
            )
            setattr(
                self,
                f"mhc_alpha_post_{branch}",
                nn.Parameter(torch.full((1,), config.magi2_mhc_alpha_init)),
            )
            setattr(
                self,
                f"mhc_alpha_res_{branch}",
                nn.Parameter(torch.full((1,), config.magi2_mhc_alpha_init)),
            )
            setattr(self, f"mhc_bias_pre_{branch}", nn.Parameter(torch.zeros(self.num_streams)))
            setattr(self, f"mhc_bias_post_{branch}", nn.Parameter(torch.zeros(self.num_streams)))
            setattr(
                self,
                f"mhc_bias_res_{branch}",
                nn.Parameter(torch.zeros(self.num_streams, self.num_streams)),
            )
            phi = nn.Parameter(torch.empty(config.magi2_adapter_width, width))
            nn.init.xavier_uniform_(phi)
            setattr(self, f"mhc_phi_fused_{branch}", phi)

    def _prepare(
        self, hidden_states: Tensor, normalized: Tensor, branch: str
    ) -> tuple[Tensor, Tensor, Tensor]:
        projected = normalized.float() @ getattr(self, f"mhc_phi_fused_{branch}")
        raw_pre, raw_post, raw_residual = torch.split(
            projected, [self.num_streams, self.num_streams, self.num_streams**2], dim=-1
        )
        pre = torch.sigmoid(
            getattr(self, f"mhc_alpha_pre_{branch}") * self.scale * raw_pre
            + getattr(self, f"mhc_bias_pre_{branch}")
        ).to(hidden_states.dtype)
        post = (
            2.0
            * torch.sigmoid(
                getattr(self, f"mhc_alpha_post_{branch}") * self.scale * raw_post
                + getattr(self, f"mhc_bias_post_{branch}")
            )
        ).to(hidden_states.dtype)
        residual_logits = getattr(self, f"mhc_alpha_res_{branch}") * self.scale * raw_residual.view(
            -1, self.num_streams, self.num_streams
        ) + getattr(self, f"mhc_bias_res_{branch}")
        residual_map = _sinkhorn(
            residual_logits.float(),
            self.config.magi2_mhc_sinkhorn_iterations,
            self.config.magi2_mhc_sinkhorn_eps,
        ).to(hidden_states.dtype)
        streams = hidden_states.view(-1, self.num_streams, self.config.hidden_size)
        branch_input = torch.einsum("tn,tnc->tc", pre, streams)
        return branch_input, post, residual_map

    def _merge(self, residual: Tensor, output: Tensor, post: Tensor, mapping: Tensor) -> Tensor:
        streams = residual.view(-1, self.num_streams, self.config.hidden_size)
        mixed = torch.einsum("tij,tjc->tic", mapping, streams)
        expanded = torch.einsum("tn,tc->tnc", post, output)
        return (mixed + expanded).reshape_as(residual)

    def forward(
        self,
        hidden_states: Tensor,
        rope: Tensor,
        cu_seqlens: Tensor,
        dispatcher: ReferenceDispatcher,
    ) -> Tensor:
        branch_input, post, mapping = self._prepare(
            hidden_states, self.mhc_norm(hidden_states, dispatcher), "attn"
        )
        hidden_states = self._merge(
            hidden_states, self.attention(branch_input, rope, cu_seqlens, dispatcher), post, mapping
        )
        branch_input, post, mapping = self._prepare(
            hidden_states, self.mhc_norm(hidden_states, dispatcher), "mlp"
        )
        return self._merge(hidden_states, self.mlp(branch_input, dispatcher), post, mapping)


class ReferencePreAdapter(nn.Module):
    """Official input adapter names and equations."""

    def __init__(self, config: Magi2Config) -> None:
        super().__init__()
        width = config.magi2_adapter_width
        self.config = config
        self.video_embedder = nn.Linear(config.magi2_video_in_channels, width, dtype=torch.float32)
        self.audio_embedder = nn.Linear(config.magi2_audio_in_channels, width, dtype=torch.float32)
        self.text_embedder = nn.Linear(config.magi2_text_in_channels, width, dtype=torch.float32)
        self.rope = ReferenceRoPE(config.magi2_attention_head_dim)

    def forward(
        self, inputs: Tensor, coordinates: Tensor, mapping: Tensor
    ) -> tuple[Tensor, Tensor]:
        output = torch.zeros(inputs.shape[0], self.config.magi2_adapter_width, device=inputs.device)
        for modality, channels, projection in (
            (Magi2Modality.VIDEO, self.config.magi2_video_in_channels, self.video_embedder),
            (Magi2Modality.AUDIO, self.config.magi2_audio_in_channels, self.audio_embedder),
            (Magi2Modality.TEXT, self.config.magi2_text_in_channels, self.text_embedder),
            (Magi2Modality.TIME, self.config.magi2_text_in_channels, self.text_embedder),
        ):
            indices = torch.nonzero(mapping == modality, as_tuple=False).flatten()
            if indices.numel():
                output.index_copy_(
                    0, indices, projection(inputs.index_select(0, indices)[:, :channels].float())
                )
        return output, self.rope(coordinates)


class ReferencePostAdapter(nn.Module):
    """Official output adapter names and equations."""

    def __init__(self, config: Magi2Config) -> None:
        super().__init__()
        width = config.magi2_adapter_width
        self.config = config
        self.final_norm_video = ReferenceRMSNorm(width)
        self.final_norm_audio = ReferenceRMSNorm(width)
        self.final_linear_video = nn.Linear(
            width, config.magi2_video_in_channels, bias=False, dtype=torch.float32
        )
        self.final_linear_audio = nn.Linear(
            width, config.magi2_audio_in_channels, bias=False, dtype=torch.float32
        )

    def forward(self, hidden_states: Tensor, mapping: Tensor) -> Tensor:
        output = torch.zeros(
            hidden_states.shape[0],
            max(self.config.magi2_video_in_channels, self.config.magi2_audio_in_channels),
            device=hidden_states.device,
        )
        for modality, channels, norm, projection in (
            (
                Magi2Modality.VIDEO,
                self.config.magi2_video_in_channels,
                self.final_norm_video,
                self.final_linear_video,
            ),
            (
                Magi2Modality.AUDIO,
                self.config.magi2_audio_in_channels,
                self.final_norm_audio,
                self.final_linear_audio,
            ),
        ):
            indices = torch.nonzero(mapping == modality, as_tuple=False).flatten()
            if indices.numel():
                projected = projection(norm(hidden_states.index_select(0, indices)).float())
                output[:, :channels].index_copy_(0, indices, projected)
        return output


class ReferenceTransformerBlock(nn.Module):
    """Official block namespace."""

    def __init__(self, config: Magi2Config) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [ReferenceTransformerLayer(config, index) for index in range(config.num_layers)]
        )


class ReferenceMagi2Model(nn.Module):
    """End-to-end differentiable oracle with official state-dict keys."""

    def __init__(self, config: Magi2Config) -> None:
        super().__init__()
        self.config = config
        self.pre_adapter = ReferencePreAdapter(config)
        self.block = ReferenceTransformerBlock(config)
        self.post_adapter = ReferencePostAdapter(config)

    def forward(
        self, inputs: Tensor, coordinates: Tensor, modality_mapping: Tensor, cu_seqlens: Tensor
    ) -> Tensor:
        hidden_states, rope = self.pre_adapter(inputs, coordinates, modality_mapping)
        model_mapping = modality_mapping.clone()
        model_mapping[model_mapping == Magi2Modality.TIME] = Magi2Modality.TEXT
        dispatcher = ReferenceDispatcher(model_mapping)
        hidden_states = dispatcher.permute(hidden_states).to(self.config.params_dtype)
        for layer in self.block.layers:
            hidden_states = layer(hidden_states, rope, cu_seqlens, dispatcher)
        return self.post_adapter(dispatcher.inverse_permute(hidden_states), modality_mapping)


def reference_flow_loss(output: Tensor, target: Tensor, loss_mask: Tensor) -> Tensor:
    """Flow-matching MSE used for the fixed-batch training comparison."""
    squared_error = (output.float() - target.float()).square()
    return (squared_error * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)


__all__ = ["ReferenceMagi2Model", "reference_flow_loss"]
