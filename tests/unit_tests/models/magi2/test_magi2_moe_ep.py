# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Expert-parallel correctness tests for native MCore MAGI-2 MoE."""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core.models.magi2 import (
    Magi2Config,
    Magi2DistributedMultiHeadMoE,
    multi_head_topk_routing,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.test_utilities import Utils


def _config(ep_size: int) -> Magi2Config:
    return Magi2Config(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        num_query_groups=2,
        kv_channels=8,
        ffn_hidden_size=128,
        num_moe_experts=8,
        moe_ffn_hidden_size=8,
        moe_router_topk=2,
        expert_model_parallel_size=ep_size,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        use_cpu_initialization=False,
        magi2_video_in_channels=3,
        magi2_audio_in_channels=4,
        magi2_text_in_channels=5,
        magi2_intermediate_factor=2,
        magi2_mm_layers=(),
        magi2_moe_layers=(0,),
        magi2_moe_num_heads=2,
        magi2_moe_num_experts_per_head=4,
        magi2_moe_top_k=2,
        magi2_moe_expert_intermediate_size=8,
        magi2_shared_expert_intermediate_size=8,
        magi2_modality_expert_intermediate_size=8,
        magi2_mhc_num_streams=2,
    )


def _expert_weights(
    module: Magi2DistributedMultiHeadMoE, linear_name: str
) -> tuple[torch.Tensor, ...]:
    linear = getattr(module.experts, linear_name)
    return tuple(getattr(linear, f"weight{index}") for index in range(module.num_local_experts))


def _fill_parameters(module: Magi2DistributedMultiHeadMoE) -> None:
    with torch.no_grad():
        router_values = torch.arange(
            module.router.weight.numel(), device=module.router.weight.device, dtype=torch.float32
        )
        module.router.weight.copy_(
            (0.04 * torch.sin(router_values * 0.17)).reshape_as(module.router.weight)
        )
        module.router.expert_bias.copy_(
            torch.linspace(
                -0.1,
                0.1,
                module.magi2_config.magi2_flattened_num_experts,
                device=module.router.expert_bias.device,
            )
        )
        for local_index, global_index in enumerate(module.local_expert_indices):
            weight_groups = (
                _expert_weights(module, "linear_fc1"),
                _expert_weights(module, "linear_fc2"),
            )
            for kind, weights in enumerate(weight_groups):
                weight = weights[local_index]
                values = torch.arange(weight.numel(), device=weight.device, dtype=torch.float32)
                weight.copy_(
                    (0.03 * torch.sin(values * 0.11 + global_index * 0.37 + kind * 0.19))
                    .reshape_as(weight)
                    .to(weight.dtype)
                )


def _gather_expert_weights(
    module: Magi2DistributedMultiHeadMoE, linear_name: str, ep_group: dist.ProcessGroup
) -> torch.Tensor:
    local_weights = torch.stack(_expert_weights(module, linear_name))
    gathered = [torch.empty_like(local_weights) for _ in range(dist.get_world_size(ep_group))]
    dist.all_gather(gathered, local_weights, group=ep_group)
    return torch.cat(gathered, dim=0)


def _reference_forward(
    module: Magi2DistributedMultiHeadMoE,
    hidden_states: torch.Tensor,
    fc1_weights: torch.Tensor,
    fc2_weights: torch.Tensor,
) -> torch.Tensor:
    head_tokens = hidden_states.reshape(-1, module.head_dim)
    probabilities, routing_map = multi_head_topk_routing(
        head_tokens,
        module.router.weight,
        module.router.expert_bias,
        num_heads=module.num_heads,
        num_experts_per_head=module.magi2_config.magi2_moe_num_experts_per_head,
        top_k=module.magi2_config.magi2_moe_top_k,
        score_func="sigmoid",
        route_norm=True,
        route_scale=module.magi2_config.magi2_route_scale,
        route_norm_eps=module.magi2_config.magi2_route_norm_eps,
    )
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


class TestMagi2MoEEP:
    """Compare EP4 MCore dispatch with a gathered PyTorch expert reference."""

    def setup_method(self) -> None:
        self.ep_size = int(os.environ.get("WORLD_SIZE", "1"))
        if self.ep_size != 4:
            pytest.skip("this test requires exactly four distributed ranks")
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=self.ep_size)

    def teardown_method(self) -> None:
        Utils.destroy_model_parallel()

    def test_ep4_forward_backward_matches_gathered_reference(self) -> None:
        config = _config(self.ep_size)
        process_groups = ProcessGroupCollection.use_mpu_process_groups()
        module = Magi2DistributedMultiHeadMoE(
            config, layer_number=1, pg_collection=process_groups
        ).cuda()
        assert module.num_local_experts == 2
        assert module.local_expert_indices == [
            2 * dist.get_rank(process_groups.ep),
            2 * dist.get_rank(process_groups.ep) + 1,
        ]
        _fill_parameters(module)
        fc1_weights = _gather_expert_weights(module, "linear_fc1", process_groups.ep)
        fc2_weights = _gather_expert_weights(module, "linear_fc2", process_groups.ep)
        torch.manual_seed(2468)
        hidden_states = torch.randn(
            5, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

        output, output_bias = module(hidden_states)
        expected = _reference_forward(module, hidden_states, fc1_weights, fc2_weights)

        assert output_bias is None
        torch.testing.assert_close(output, expected, rtol=5e-2, atol=2e-2)
        gathered_outputs = [torch.empty_like(output) for _ in range(self.ep_size)]
        dist.all_gather(gathered_outputs, output, group=process_groups.ep)
        for rank_output in gathered_outputs:
            torch.testing.assert_close(rank_output, output, rtol=0.0, atol=0.0)

        probe = torch.randn_like(output)
        expert_parameters = _expert_weights(module, "linear_fc1") + _expert_weights(
            module, "linear_fc2"
        )
        shared_parameters = (hidden_states, module.router.weight)
        actual_gradients = torch.autograd.grad(output, shared_parameters + expert_parameters, probe)
        expected_gradients = torch.autograd.grad(expected, shared_parameters, probe)
        for actual, reference in zip(actual_gradients[:2], expected_gradients):
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, reference, rtol=7e-2, atol=3e-2)
        assert all(torch.isfinite(gradient).all() for gradient in actual_gradients[2:])
