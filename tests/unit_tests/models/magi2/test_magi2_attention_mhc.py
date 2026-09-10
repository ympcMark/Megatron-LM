# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Numerical and integration tests for native MCore MAGI-2 Attention and mHC."""

from __future__ import annotations

import hashlib

import pytest
import torch

from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.models.magi2 import (
    Magi2Attention,
    Magi2AttentionSubmodules,
    Magi2Config,
    Magi2DotProductAttention,
    Magi2FourierRoPE,
    Magi2MHCBranch,
    Magi2Modality,
    Magi2Model,
    Magi2RuntimeContext,
    Magi2TorchDotProductAttention,
    Magi2TransformerLayer,
    Magi2TransformerLayerSpecs,
    get_magi2_transformer_layer_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from tests.unit_tests.test_utilities import Utils

pytestmark = [pytest.mark.unit]


def _reduced_config(**overrides) -> Magi2Config:
    values = {
        "num_layers": 1,
        "hidden_size": 16,
        "num_attention_heads": 2,
        "num_query_groups": 2,
        "kv_channels": 8,
        "ffn_hidden_size": 128,
        "num_moe_experts": 6,
        "moe_ffn_hidden_size": 8,
        "moe_router_topk": 2,
        "params_dtype": torch.float32,
        "pipeline_dtype": torch.float32,
        "use_cpu_initialization": True,
        "magi2_video_in_channels": 3,
        "magi2_audio_in_channels": 4,
        "magi2_text_in_channels": 5,
        "magi2_intermediate_factor": 2,
        "magi2_mm_layers": (0,),
        "magi2_moe_layers": (),
        "magi2_moe_num_heads": 2,
        "magi2_moe_num_experts_per_head": 3,
        "magi2_moe_top_k": 2,
        "magi2_moe_expert_intermediate_size": 8,
        "magi2_shared_expert_intermediate_size": 8,
        "magi2_modality_expert_intermediate_size": 8,
        "magi2_mhc_num_streams": 2,
    }
    values.update(overrides)
    return Magi2Config(**values)


def _runtime_context(
    config: Magi2Config, device: torch.device | None = None
) -> Magi2RuntimeContext:
    device = device or torch.device("cpu")
    coordinates = torch.tensor(
        [
            [0, 0, 0, 2, 2, 2, 2, 2, 2],
            [0, 0, 1, 2, 2, 2, 2, 2, 2],
            [0, 1, 0, 2, 2, 2, 2, 2, 2],
            [0, 1, 1, 2, 2, 2, 2, 2, 2],
            [1, 0, 0, 2, 2, 2, 2, 2, 2],
            [1, 0, 1, 2, 2, 2, 2, 2, 2],
        ],
        dtype=torch.float32,
        device=device,
    )
    mapping = torch.tensor(
        [
            Magi2Modality.VIDEO,
            Magi2Modality.AUDIO,
            Magi2Modality.TEXT,
            Magi2Modality.TIME,
            Magi2Modality.VIDEO,
            Magi2Modality.AUDIO,
        ],
        device=device,
    )
    context = Magi2RuntimeContext.from_tensors(
        coordinates, mapping, torch.tensor([0, 3, 6], dtype=torch.int32, device=device)
    )
    context.rope = Magi2FourierRoPE(config.magi2_attention_head_dim).to(device)(coordinates)
    return context


def _fill_deterministically(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            offset = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 10_000
            indices = torch.arange(parameter.numel(), dtype=torch.float32)
            values = 0.02 * torch.sin(indices * 0.173 + offset * 0.001)
            parameter.copy_(values.reshape(parameter.shape).to(parameter.dtype))


def _reference_norm(
    value: torch.Tensor, weight: torch.Tensor, sorted_modalities: torch.Tensor, eps: float
) -> torch.Tensor:
    normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + eps)
    selected_weight = weight.index_select(0, sorted_modalities)
    while selected_weight.ndim < normalized.ndim:
        selected_weight = selected_weight.unsqueeze(1)
    return normalized * (selected_weight + 1.0)


def _reference_linear(
    value: torch.Tensor, weight: torch.Tensor, sorted_modalities: torch.Tensor
) -> torch.Tensor:
    selected_weight = weight.index_select(0, sorted_modalities)
    return torch.einsum("tbi,toi->tbo", value, selected_weight)


def _reference_rope(value: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    sin, cos = rope.tensor_split(2, -1)
    sin = torch.cat((sin, sin), -1).view(value.shape[0], 1, 1, -1)
    cos = torch.cat((cos, cos), -1).view(value.shape[0], 1, 1, -1)
    rotary_dim = cos.shape[-1]
    first, second = value[..., :rotary_dim].chunk(2, -1)
    rotated = torch.cat((-second, first), -1)
    return torch.cat((value[..., :rotary_dim] * cos + rotated * sin, value[..., rotary_dim:]), -1)


def _reference_attention(
    module: Magi2Attention, hidden_states: torch.Tensor, context: Magi2RuntimeContext
) -> torch.Tensor:
    dispatcher = context.get_modality_dispatcher()
    sorted_modalities = context.model_modality_mapping.index_select(0, dispatcher.permute_mapping)
    modalities = module.num_modalities
    hidden = module.hidden_size
    heads = module.num_heads
    head_dim = module.head_dim
    pre_weight = module.pre_norm.weight.view(modalities, hidden)
    normalized = _reference_norm(
        hidden_states, pre_weight, sorted_modalities, module.config.layernorm_epsilon
    )
    gate_weight = module.linear_g.weight.view(modalities, heads, hidden)
    qkv_weight = module.linear_qkv.weight.view(modalities, 3 * hidden, hidden)
    gate = _reference_linear(normalized, gate_weight, sorted_modalities).view(6, 1, heads, 1)
    qkv = _reference_linear(normalized, qkv_weight, sorted_modalities)
    query, key, value = torch.split(qkv, hidden, dim=-1)
    query = query.view(6, 1, heads, head_dim)
    key = key.view(6, 1, heads, head_dim)
    value = value.view(6, 1, heads, head_dim)
    q_weight = module.q_norm.weight.view(modalities, head_dim)
    k_weight = module.k_norm.weight.view(modalities, head_dim)
    query = _reference_norm(query, q_weight, sorted_modalities, module.config.layernorm_epsilon)
    key = _reference_norm(key, k_weight, sorted_modalities, module.config.layernorm_epsilon)
    query = _reference_rope(dispatcher.inverse_permute(query), context.rope)
    key = _reference_rope(dispatcher.inverse_permute(key), context.rope)
    value = dispatcher.inverse_permute(value)

    outputs = []
    for start, end in ((0, 3), (3, 6)):
        q_segment = query[start:end].permute(1, 2, 0, 3)
        k_segment = key[start:end].permute(1, 2, 0, 3)
        v_segment = value[start:end].permute(1, 2, 0, 3)
        logits = torch.matmul(q_segment, k_segment.transpose(-1, -2)) * head_dim**-0.5
        sinks = module.sinks.T.view(1, heads, 1, 1).expand(1, heads, end - start, 1)
        weights = torch.softmax(torch.cat((logits, sinks), dim=-1), dim=-1)[..., : end - start]
        outputs.append(torch.matmul(weights, v_segment).permute(2, 0, 1, 3))
    output = dispatcher.permute(torch.cat(outputs, dim=0)) * torch.sigmoid(gate)
    output = output.reshape(6, 1, hidden)
    proj_weight = module.linear_proj.weight.view(modalities, hidden, hidden)
    return _reference_linear(output, proj_weight, sorted_modalities)


class _ZeroMagi2MLP(MegatronModule):
    """Test-only feed-forward stub used to exercise real Attention+mHC assembly."""

    def __init__(
        self,
        config: Magi2Config,
        layer_number: int,
        num_modalities: int,
        pg_collection: ProcessGroupCollection,
    ) -> None:
        super().__init__(config=config)
        self.layer_number = layer_number
        self.num_modalities = num_modalities
        self.pg_collection = pg_collection

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        return torch.zeros_like(hidden_states)


class TestMagi2AttentionMHC:
    """Compare Stage 3 operators with independent official-formula references."""

    def setup_method(self) -> None:
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self) -> None:
        Utils.destroy_model_parallel()

    def test_fourier_rope_matches_public_partial_width(self) -> None:
        config = _reduced_config()
        context = _runtime_context(config)
        rope = context.rope

        assert rope is not None
        assert rope.shape == (6, 6)
        assert torch.isfinite(rope).all()

    def test_mhc_matches_official_formula_and_backward(self) -> None:
        config = _reduced_config(hidden_size=8, num_attention_heads=1, num_query_groups=1)
        branch = Magi2MHCBranch(config)
        _fill_deterministically(branch)
        hidden = torch.randn(4, 1, 16, requires_grad=True)
        normalized = torch.randn(4, 1, 16, requires_grad=True)

        actual_input, state = branch.prepare(hidden, normalized)
        actual = branch.merge(hidden, actual_input, state)

        projected = normalized.float() @ branch.phi_fused
        raw_pre, raw_post, raw_residual = torch.split(projected, [2, 2, 4], dim=-1)
        h_pre = torch.sigmoid(branch.alpha_pre * branch.matmul_scale * raw_pre + branch.bias_pre)
        streams = hidden.view(4, 1, 2, 8)
        reference_input = torch.einsum("sbn,sbnc->sbc", h_pre, streams)
        h_post = 2 * torch.sigmoid(
            branch.alpha_post * branch.matmul_scale * raw_post + branch.bias_post
        )
        logits = (
            branch.alpha_res * branch.matmul_scale * raw_residual.view(4, 1, 2, 2) + branch.bias_res
        )
        residual_mapping = torch.exp(logits - logits.amax(dim=(-2, -1), keepdim=True))
        for _ in range(config.magi2_mhc_sinkhorn_iterations):
            residual_mapping = residual_mapping / (
                residual_mapping.sum(dim=-2, keepdim=True) + config.magi2_mhc_sinkhorn_eps
            )
            residual_mapping = residual_mapping / (
                residual_mapping.sum(dim=-1, keepdim=True) + config.magi2_mhc_sinkhorn_eps
            )
        reference = (
            torch.einsum("sbij,sbjc->sbic", residual_mapping, streams)
            + torch.einsum("sbn,sbc->sbnc", h_post, reference_input)
        ).reshape_as(hidden)

        torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            residual_mapping.sum(dim=-1), torch.ones(4, 1, 2), rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            residual_mapping.sum(dim=-2), torch.ones(4, 1, 2), rtol=1e-5, atol=1e-5
        )
        probe = torch.randn_like(actual)
        actual_grad = torch.autograd.grad(
            actual, (hidden, branch.phi_fused), probe, retain_graph=True
        )
        reference_grad = torch.autograd.grad(reference, (hidden, branch.phi_fused), probe)
        for actual_value, reference_value in zip(actual_grad, reference_grad):
            torch.testing.assert_close(actual_value, reference_value, rtol=1e-5, atol=1e-6)

    def test_attention_matches_official_formula_forward_and_backward(self) -> None:
        config = _reduced_config()
        context = _runtime_context(config)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(["tp", "cp"])
        attention = Magi2Attention(
            config,
            Magi2AttentionSubmodules(core_attention=Magi2TorchDotProductAttention),
            layer_number=1,
            num_modalities=3,
            pg_collection=pg_collection,
        )
        _fill_deterministically(attention)
        original_hidden = torch.randn(6, 1, 16)
        hidden = context.get_modality_dispatcher().permute(original_hidden).requires_grad_(True)

        actual = attention(hidden, None, context)
        reference = _reference_attention(attention, hidden, context)

        torch.testing.assert_close(actual, reference, rtol=2e-6, atol=2e-6)
        probe = torch.randn_like(actual)
        parameters = (hidden, attention.linear_qkv.weight, attention.core_attention.softmax_offset)
        actual_grad = torch.autograd.grad(actual, parameters, probe, retain_graph=True)
        reference_grad = torch.autograd.grad(reference, parameters, probe)
        for actual_value, reference_value in zip(actual_grad, reference_grad):
            torch.testing.assert_close(actual_value, reference_value, rtol=2e-5, atol=2e-6)

    def test_packed_sequences_do_not_attend_across_boundaries(self) -> None:
        config = _reduced_config()
        context = _runtime_context(config)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(["tp", "cp"])
        attention = Magi2Attention(
            config,
            Magi2AttentionSubmodules(core_attention=Magi2TorchDotProductAttention),
            layer_number=1,
            num_modalities=3,
            pg_collection=pg_collection,
        )
        original = torch.randn(6, 1, 16)
        changed = original.clone()
        changed[3:] += 100.0
        dispatcher = context.get_modality_dispatcher()

        first = dispatcher.inverse_permute(attention(dispatcher.permute(original), None, context))
        second = dispatcher.inverse_permute(attention(dispatcher.permute(changed), None, context))

        torch.testing.assert_close(first[:3], second[:3], rtol=0.0, atol=0.0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Transformer Engine GPU")
    def test_transformer_engine_attention_core_forward_backward(self) -> None:
        device = torch.device("cuda")
        config = _reduced_config(
            params_dtype=torch.bfloat16, pipeline_dtype=torch.bfloat16, use_cpu_initialization=False
        )
        context = _runtime_context(config, device)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(["tp", "cp"])
        reference = Magi2Attention(
            config,
            Magi2AttentionSubmodules(core_attention=Magi2TorchDotProductAttention),
            layer_number=1,
            num_modalities=3,
            pg_collection=pg_collection,
        ).to(device)
        production = Magi2Attention(
            config,
            Magi2AttentionSubmodules(core_attention=TEDotProductAttention),
            layer_number=1,
            num_modalities=3,
            pg_collection=pg_collection,
        ).to(device)
        _fill_deterministically(reference)
        projection_names = ("pre_norm", "q_norm", "k_norm", "linear_g", "linear_qkv", "linear_proj")
        for name in projection_names:
            getattr(production, name).load_state_dict(getattr(reference, name).state_dict())
        production.core_attention.softmax_offset.data.copy_(
            reference.core_attention.softmax_offset.data
        )
        original_hidden = torch.randn(6, 1, 16, device=device, dtype=torch.bfloat16)
        hidden = context.get_modality_dispatcher().permute(original_hidden).requires_grad_(True)

        expected = reference(hidden, None, context)
        actual = production(hidden, None, context)

        torch.testing.assert_close(actual, expected, rtol=5e-2, atol=2e-2)
        actual.float().square().mean().backward()
        assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
        assert production.core_attention.softmax_offset.grad is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
    def test_mcore_dot_product_attention_adapter_forward_backward(self) -> None:
        device = torch.device("cuda")
        tensor_parallel.model_parallel_cuda_manual_seed(1234, force_reset_rng=True)
        config = _reduced_config(
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            bf16=True,
            use_cpu_initialization=False,
            attention_softmax_in_fp32=True,
            masked_softmax_fusion=False,
        )
        context = _runtime_context(config, device)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(["tp", "cp"])
        reference = Magi2Attention(
            config,
            Magi2AttentionSubmodules(core_attention=Magi2TorchDotProductAttention),
            layer_number=1,
            num_modalities=3,
            pg_collection=pg_collection,
        ).to(device)
        production = Magi2Attention(
            config,
            Magi2AttentionSubmodules(core_attention=Magi2DotProductAttention),
            layer_number=1,
            num_modalities=3,
            pg_collection=pg_collection,
        ).to(device)
        _fill_deterministically(reference)
        projection_names = ("pre_norm", "q_norm", "k_norm", "linear_g", "linear_qkv", "linear_proj")
        for name in projection_names:
            getattr(production, name).load_state_dict(getattr(reference, name).state_dict())
        production.core_attention.softmax_offset.data.copy_(
            reference.core_attention.softmax_offset.data
        )
        original_hidden = torch.randn(6, 1, 16, device=device, dtype=torch.bfloat16)
        hidden = context.get_modality_dispatcher().permute(original_hidden).requires_grad_(True)

        expected = reference(hidden, None, context)
        actual = production(hidden, None, context)

        torch.testing.assert_close(actual, expected, rtol=5e-2, atol=2e-2)
        actual.float().square().mean().backward()
        assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
        assert production.core_attention.softmax_offset.grad is not None
        assert torch.isfinite(production.core_attention.softmax_offset.grad).all()

    def test_real_attention_mhc_layer_runs_inside_transformer_block(self) -> None:
        config = _reduced_config()
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(["tp", "pp", "cp"])
        layer_spec = get_magi2_transformer_layer_spec(
            Magi2TorchDotProductAttention, ModuleSpec(module=_ZeroMagi2MLP)
        )
        model = Magi2Model(
            config, Magi2TransformerLayerSpecs(dense=layer_spec, moe=layer_spec), pg_collection
        )
        context = _runtime_context(config)
        inputs = torch.randn(6, 5, requires_grad=True)

        output = model(
            inputs,
            context.coordinates,
            context.modality_mapping,
            context.cu_seqlens_q,
            runtime_context=context,
        )
        output.square().mean().backward()

        assert isinstance(model.decoder.layers[0], Magi2TransformerLayer)
        assert sum(parameter.numel() for parameter in model.decoder.layers[0].parameters()) == (
            config.magi2_parameter_count_breakdown()["attention_and_mhc"]
        )
        assert output.shape == (6, 4)
        assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
        assert model.decoder.layers[0].attention_mhc.phi_fused.grad is not None
        assert model.decoder.layers[0].self_attention.linear_qkv.weight.grad is not None
