# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Numerical tests for native MCore MAGI-2 dense and routed MLPs."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from megatron.core.models.magi2 import (
    Magi2Config,
    Magi2DenseMLP,
    Magi2DistributedMultiHeadMoE,
    Magi2Modality,
    Magi2ModalityDispatcher,
    Magi2Model,
    Magi2MoEMLP,
    Magi2RuntimeContext,
    Magi2TorchDotProductAttention,
    build_magi2_expert_config,
    get_magi2_layer_specs,
    magi2_quick_geglu,
    multi_head_topk_routing,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.test_utilities import Utils


def _config(*, moe: bool = False, gpu: bool = False, **overrides) -> Magi2Config:
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
        "params_dtype": torch.bfloat16 if gpu else torch.float32,
        "pipeline_dtype": torch.bfloat16 if gpu else torch.float32,
        "use_cpu_initialization": not gpu,
        "magi2_video_in_channels": 3,
        "magi2_audio_in_channels": 4,
        "magi2_text_in_channels": 5,
        "magi2_intermediate_factor": 2,
        "magi2_mm_layers": () if moe else (0,),
        "magi2_moe_layers": (0,) if moe else (),
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


def _runtime_context(device: torch.device) -> Magi2RuntimeContext:
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
    cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32, device=device)
    return Magi2RuntimeContext.from_tensors(coordinates, mapping, cu_seqlens)


def _reference_norm(
    value: torch.Tensor, weight: torch.Tensor, sorted_modalities: torch.Tensor, eps: float
) -> torch.Tensor:
    normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + eps)
    return normalized * (weight.index_select(0, sorted_modalities) + 1.0)


def _reference_linear(
    value: torch.Tensor, weight: torch.Tensor, sorted_modalities: torch.Tensor
) -> torch.Tensor:
    selected_weight = weight.index_select(0, sorted_modalities)
    return torch.einsum("ti,toi->to", value, selected_weight)


def _reference_quick_geglu(value: torch.Tensor) -> torch.Tensor:
    output_dtype = value.dtype
    value = value.float()
    gate = value[..., ::2].clamp(max=7.0)
    linear = value[..., 1::2].clamp(min=-7.0, max=7.0)
    return (gate * torch.sigmoid(1.702 * gate) * (linear + 1.0)).to(output_dtype)


class TestMagi2DenseMLP:
    """Compare the native dense block with independent public equations."""

    def setup_method(self) -> None:
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self) -> None:
        Utils.destroy_model_parallel()

    def test_quick_geglu_matches_interleaved_public_formula(self) -> None:
        value = torch.tensor([[-8.0, -9.0, 2.0, 3.0, 9.0, 8.0]], dtype=torch.bfloat16)

        actual = magi2_quick_geglu(value)
        expected = _reference_quick_geglu(value)

        assert actual.dtype == value.dtype
        torch.testing.assert_close(actual, expected)

    def test_dense_forward_backward_matches_independent_formula(self) -> None:
        torch.manual_seed(123)
        config = _config()
        dispatcher = Magi2ModalityDispatcher(torch.tensor([2, 0, 1, 0, 2, 1]))
        process_groups = ProcessGroupCollection.use_mpu_process_groups(["tp", "cp"])
        module = Magi2DenseMLP(
            config, layer_number=1, num_modalities=3, pg_collection=process_groups
        )
        original = torch.randn(6, config.hidden_size)
        hidden_states = dispatcher.permute(original).requires_grad_(True)
        sorted_modalities = dispatcher.modality_mapping.index_select(0, dispatcher.permute_mapping)

        actual = module(hidden_states, dispatcher)

        pre_weight = module.pre_norm.weight.view(3, config.hidden_size)
        normalized = _reference_norm(
            hidden_states, pre_weight, sorted_modalities, config.layernorm_epsilon
        )
        up_weight = module.up_gate_proj.weight.view(
            3, 2 * config.magi2_dense_intermediate_size, config.hidden_size
        )
        up = _reference_linear(normalized, up_weight, sorted_modalities)
        activated = _reference_quick_geglu(up)
        down_weight = module.down_proj.weight.view(
            3, config.hidden_size, config.magi2_dense_intermediate_size
        )
        expected = _reference_linear(activated, down_weight, sorted_modalities)

        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        probe = torch.randn_like(actual)
        parameters = (hidden_states, module.up_gate_proj.weight, module.down_proj.weight)
        actual_grad = torch.autograd.grad(actual, parameters, probe, retain_graph=True)
        expected_grad = torch.autograd.grad(expected, parameters, probe)
        for actual_value, expected_value in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual_value, expected_value, rtol=1e-5, atol=1e-6)

        expected_parameter_count = 3 * (
            config.hidden_size + 3 * config.hidden_size * config.magi2_dense_intermediate_size
        )
        assert sum(parameter.numel() for parameter in module.parameters()) == (
            expected_parameter_count
        )


def _reference_route(
    hidden_states: torch.Tensor,
    gate: torch.Tensor,
    expert_bias: torch.Tensor,
    *,
    num_heads: int,
    num_experts_per_head: int,
    top_k: int,
    route_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    flattened = hidden_states.reshape(-1, hidden_states.shape[-1])
    probabilities = torch.zeros(
        flattened.shape[0],
        num_heads * num_experts_per_head,
        dtype=torch.float32,
        device=flattened.device,
    )
    routing_map = torch.zeros_like(probabilities, dtype=torch.bool)
    for token_head, value in enumerate(flattened):
        head = token_head % num_heads
        first_expert = head * num_experts_per_head
        last_expert = first_expert + num_experts_per_head
        scores = torch.sigmoid(value.float() @ gate[first_expert:last_expert].float().T)
        local_indices = torch.topk(scores + expert_bias[first_expert:last_expert], top_k).indices
        selected = F.normalize(scores[local_indices], p=1, dim=-1, eps=1e-12)
        global_indices = local_indices + first_expert
        probabilities[token_head, global_indices] = selected * route_scale
        routing_map[token_head, global_indices] = True
    return probabilities, routing_map


class TestMagi2Router:
    """Verify constrained routing and the derived expert configuration."""

    def test_route_forward_backward_matches_independent_formula(self) -> None:
        torch.manual_seed(456)
        hidden_states = torch.randn(8, 1, 4, requires_grad=True)
        gate = torch.randn(10, 4, requires_grad=True)
        expert_bias = torch.linspace(-0.2, 0.2, 10)
        actual_probs, actual_map = multi_head_topk_routing(
            hidden_states,
            gate,
            expert_bias,
            num_heads=2,
            num_experts_per_head=5,
            top_k=2,
            score_func="sigmoid",
            route_norm=True,
            route_scale=4.9,
            route_norm_eps=1e-12,
        )
        expected_probs, expected_map = _reference_route(
            hidden_states,
            gate,
            expert_bias,
            num_heads=2,
            num_experts_per_head=5,
            top_k=2,
            route_scale=4.9,
        )

        torch.testing.assert_close(actual_probs, expected_probs)
        torch.testing.assert_close(actual_map, expected_map)
        selected_experts = actual_map.int().topk(2, dim=-1).indices
        expected_heads = torch.arange(8).unsqueeze(1) % 2
        assert torch.equal(selected_experts // 5, expected_heads.expand(-1, 2))

        probe = torch.randn_like(actual_probs)
        actual_grad = torch.autograd.grad(
            actual_probs, (hidden_states, gate), probe, retain_graph=True
        )
        expected_grad = torch.autograd.grad(expected_probs, (hidden_states, gate), probe)
        for actual_value, expected_value in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual_value, expected_value, rtol=1e-5, atol=1e-6)

    def test_derived_expert_config_uses_head_width_and_grouped_gemm(self) -> None:
        config = _config(moe=True)
        expert_config = build_magi2_expert_config(config)

        assert expert_config.hidden_size == config.magi2_moe_head_dim
        assert expert_config.num_moe_experts == 6
        assert expert_config.moe_ffn_hidden_size == 8
        assert expert_config.moe_router_topk == 2
        assert expert_config.moe_router_dtype == "fp32"
        assert expert_config.moe_grouped_gemm
        assert expert_config.moe_token_dispatcher_type == "alltoall"
        assert expert_config.activation_func is not None
        assert expert_config.activation_func_clamp_value == 7.0
        assert expert_config.glu_linear_offset == 1.0


def _expert_weights(
    module: Magi2DistributedMultiHeadMoE, linear_name: str
) -> tuple[torch.Tensor, ...]:
    linear = getattr(module.experts, linear_name)
    if linear.single_grouped_weight:
        return tuple(linear.weight.unbind(0))
    return tuple(getattr(linear, f"weight{index}") for index in range(module.num_local_experts))


def _reference_experts(
    module: Magi2DistributedMultiHeadMoE, hidden_states: torch.Tensor
) -> torch.Tensor:
    head_tokens = hidden_states.reshape(-1, module.head_dim)
    probabilities, routing_map = _reference_route(
        head_tokens,
        module.router.weight,
        module.router.expert_bias,
        num_heads=module.num_heads,
        num_experts_per_head=module.magi2_config.magi2_moe_num_experts_per_head,
        top_k=module.magi2_config.magi2_moe_top_k,
        route_scale=module.magi2_config.magi2_route_scale,
    )
    fc1_weights = _expert_weights(module, "linear_fc1")
    fc2_weights = _expert_weights(module, "linear_fc2")
    outputs = []
    for token_index, value in enumerate(head_tokens):
        output = torch.zeros_like(value)
        selected_experts = routing_map[token_index].nonzero().flatten()
        for expert_index in selected_experts:
            expert_id = int(expert_index)
            projected = F.linear(value, fc1_weights[expert_id])
            gate, linear = projected.chunk(2, dim=-1)
            gate = gate.clamp(max=7.0)
            linear = linear.clamp(min=-7.0, max=7.0)
            activated = gate * torch.sigmoid(1.702 * gate) * (linear + 1.0)
            expert_output = F.linear(activated, fc2_weights[expert_id])
            output = output + probabilities[token_index, expert_id].to(output.dtype) * expert_output
        outputs.append(output)
    return torch.stack(outputs).reshape_as(hidden_states)


class TestMagi2ProductionMoE:
    """Exercise the MCore all-to-all and Transformer Engine grouped path."""

    def setup_method(self) -> None:
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=1)

    def teardown_method(self) -> None:
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires grouped GEMM")
    def test_grouped_moe_matches_pytorch_experts_forward_backward(self) -> None:
        torch.manual_seed(789)
        config = _config(moe=True, gpu=True)
        process_groups = ProcessGroupCollection.use_mpu_process_groups()
        module = Magi2DistributedMultiHeadMoE(
            config, layer_number=1, pg_collection=process_groups
        ).cuda()
        hidden_states = torch.randn(
            4, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

        output, output_bias = module(hidden_states)
        expected = _reference_experts(module, hidden_states)

        expected_parameters = (
            config.magi2_flattened_num_experts * config.magi2_moe_head_dim
            + config.magi2_flattened_num_experts
            * 3
            * config.magi2_moe_head_dim
            * config.magi2_moe_expert_intermediate_size
        )
        assert output.shape == hidden_states.shape
        assert output_bias is None
        torch.testing.assert_close(output, expected, rtol=5e-2, atol=2e-2)
        assert sum(parameter.numel() for parameter in module.parameters()) == (expected_parameters)
        routing_map = multi_head_topk_routing(
            hidden_states.reshape(-1, module.head_dim),
            module.router.weight,
            module.router.expert_bias,
            num_heads=module.num_heads,
            num_experts_per_head=config.magi2_moe_num_experts_per_head,
            top_k=config.magi2_moe_top_k,
            score_func="sigmoid",
            route_norm=True,
            route_scale=config.magi2_route_scale,
            route_norm_eps=config.magi2_route_norm_eps,
        )[1]
        selected_expert = int(routing_map.any(dim=0).nonzero()[0])
        parameters = (
            hidden_states,
            module.router.weight,
            _expert_weights(module, "linear_fc1")[selected_expert],
            _expert_weights(module, "linear_fc2")[selected_expert],
        )
        probe = torch.randn_like(output)
        actual_gradients = torch.autograd.grad(output, parameters, probe, retain_graph=True)
        expected_gradients = torch.autograd.grad(expected, parameters, probe)
        for actual, reference in zip(actual_gradients, expected_gradients):
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, reference, rtol=7e-2, atol=3e-2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires grouped GEMM")
    def test_native_specs_run_dense_and_moe_layers_in_transformer_block(self) -> None:
        config = _config(
            moe=True, gpu=True, num_layers=2, magi2_mm_layers=(0,), magi2_moe_layers=(1,)
        )
        process_groups = ProcessGroupCollection.use_mpu_process_groups()
        model = Magi2Model(
            config, get_magi2_layer_specs(Magi2TorchDotProductAttention), process_groups
        ).cuda()
        context = _runtime_context(torch.device("cuda"))
        inputs = torch.randn(
            6, config.magi2_text_in_channels, device="cuda", dtype=torch.float32, requires_grad=True
        )

        output = model(
            inputs,
            context.coordinates,
            context.modality_mapping,
            context.cu_seqlens_q,
            runtime_context=context,
        )
        output.float().square().mean().backward()

        assert isinstance(model.decoder.layers[0].mlp, Magi2DenseMLP)
        assert isinstance(model.decoder.layers[1].mlp, Magi2MoEMLP)
        assert sum(parameter.numel() for parameter in model.parameters()) == (
            config.magi2_parameter_count
        )
        assert output.shape == (6, config.magi2_audio_in_channels)
        assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
        assert model.decoder.layers[1].mlp.routed.router.weight.grad is not None
