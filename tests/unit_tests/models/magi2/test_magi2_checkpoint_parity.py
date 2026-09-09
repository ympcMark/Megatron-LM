# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Official-key conversion, training parity, and checkpoint tests for MAGI-2."""

from __future__ import annotations

import os
import shutil
import tempfile

import pytest
import torch

from megatron.core.dist_checkpointing import load, save
from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core.models.magi2 import (
    Magi2Config,
    Magi2Modality,
    Magi2Model,
    Magi2TorchDotProductAttention,
    convert_magi2_official_state_dict,
    get_magi2_layer_specs,
    load_magi2_official_safetensors,
    load_magi2_official_state_dict,
    magi2_official_to_mcore_key,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.models.magi2.magi2_reference import ReferenceMagi2Model, reference_flow_loss
from tests.unit_tests.test_utilities import Utils


def _config() -> Magi2Config:
    return Magi2Config(
        num_layers=2,
        hidden_size=16,
        num_attention_heads=2,
        num_query_groups=2,
        kv_channels=8,
        ffn_hidden_size=128,
        num_moe_experts=6,
        moe_ffn_hidden_size=8,
        moe_router_topk=2,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        magi2_video_in_channels=3,
        magi2_audio_in_channels=4,
        magi2_text_in_channels=5,
        magi2_intermediate_factor=2,
        magi2_mm_layers=(0,),
        magi2_moe_layers=(1,),
        magi2_moe_num_heads=2,
        magi2_moe_num_experts_per_head=3,
        magi2_moe_top_k=2,
        magi2_moe_expert_intermediate_size=8,
        magi2_shared_expert_intermediate_size=8,
        magi2_modality_expert_intermediate_size=8,
        magi2_mhc_num_streams=2,
    )


def _batch(device: torch.device) -> tuple[torch.Tensor, ...]:
    inputs = torch.linspace(-0.8, 0.9, 30, device=device).view(6, 5)
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
    target = torch.linspace(-0.3, 0.4, 24, device=device).view(6, 4)
    loss_mask = torch.zeros_like(target)
    loss_mask[mapping == Magi2Modality.VIDEO, :3] = 1.0
    loss_mask[mapping == Magi2Modality.AUDIO, :4] = 1.0
    return inputs, coordinates, mapping, cu_seqlens, target, loss_mask


def _models(config: Magi2Config) -> tuple[ReferenceMagi2Model, Magi2Model]:
    torch.manual_seed(1234)
    reference = ReferenceMagi2Model(config).cuda()
    with torch.no_grad():
        bias = reference.block.layers[1].mlp.moe_mlp.router.expert_bias
        bias.copy_(torch.tensor([1.0, 0.5, -1.0, 1.0, 0.5, -1.0], device=bias.device))
        reference.block.layers[1].mlp.moe_mlp.router.expert_bias_ema.copy_(bias)
    groups = ProcessGroupCollection.use_mpu_process_groups()
    model = Magi2Model(config, get_magi2_layer_specs(Magi2TorchDotProductAttention), groups).cuda()
    report = load_magi2_official_state_dict(model, reference.state_dict())
    assert report.is_strictly_complete
    return reference, model


def _loss(
    model: torch.nn.Module, batch: tuple[torch.Tensor, ...], *, requires_grad: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs, coordinates, mapping, cu_seqlens, target, loss_mask = batch
    model_inputs = inputs.detach().clone().requires_grad_(requires_grad)
    output = model(model_inputs, coordinates, mapping, cu_seqlens)
    return output, reference_flow_loss(output, target, loss_mask), model_inputs


def test_official_key_mapping_contract() -> None:
    assert (
        magi2_official_to_mcore_key("block.layers.7.attention.linear_qkv.weight")
        == "decoder.layers.7.self_attention.linear_qkv.weight"
    )
    assert (
        magi2_official_to_mcore_key("block.layers.7.mhc_phi_fused_mlp")
        == "decoder.layers.7.mlp_mhc.phi_fused"
    )
    assert (
        magi2_official_to_mcore_key("block.layers.7.mlp.shared_expert_fc1.weight")
        == "decoder.layers.7.mlp.shared_fc1.weight"
    )
    assert magi2_official_to_mcore_key("block.layers.7.mlp.moe_mlp.W_gate") is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires TE Grouped GEMM")
class TestMagi2OfficialParity:
    """Compare a complete dense-plus-MoE model against the public equations."""

    def setup_method(self) -> None:
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=1)

    def teardown_method(self) -> None:
        Utils.destroy_model_parallel()

    def test_forward_loss_backward_and_adamw_step(self) -> None:
        config = _config()
        reference, model = _models(config)
        batch = _batch(torch.device("cuda"))
        reference_output, reference_loss, reference_input = _loss(
            reference, batch, requires_grad=True
        )
        model_output, model_loss, model_input = _loss(model, batch, requires_grad=True)

        print(
            "MAGI2_STAGE5_FORWARD "
            f"reference_loss={reference_loss.item():.9f} "
            f"mcore_loss={model_loss.item():.9f} "
            f"loss_abs_diff={abs(model_loss.item() - reference_loss.item()):.9f} "
            f"output_max_abs_diff={(model_output - reference_output).abs().max().item():.9f}"
        )
        torch.testing.assert_close(model_output, reference_output, rtol=7e-2, atol=3e-2)
        torch.testing.assert_close(model_loss, reference_loss, rtol=2e-2, atol=2e-3)

        reference_optimizer = torch.optim.AdamW(
            reference.parameters(), lr=2e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01
        )
        model_optimizer = torch.optim.AdamW(
            model.parameters(), lr=2e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01
        )
        reference_loss.backward()
        model_loss.backward()

        print(
            "MAGI2_STAGE5_BACKWARD "
            f"input_grad_max_abs_diff={(model_input.grad - reference_input.grad).abs().max().item():.9f}"
        )
        torch.testing.assert_close(model_input.grad, reference_input.grad, rtol=1e-1, atol=4e-2)
        reference_parameters = dict(reference.named_parameters())
        model_parameters = dict(model.named_parameters())
        direct_gradient_pairs = (
            ("pre_adapter.video_embedder.weight", "pre_adapter.video_embedder.weight"),
            (
                "block.layers.0.attention.linear_qkv.weight",
                "decoder.layers.0.self_attention.linear_qkv.weight",
            ),
            ("block.layers.0.mhc_phi_fused_attn", "decoder.layers.0.attention_mhc.phi_fused"),
            ("block.layers.0.mlp.up_gate_proj.weight", "decoder.layers.0.mlp.up_gate_proj.weight"),
            ("block.layers.1.mlp.moe_mlp.gate", "decoder.layers.1.mlp.routed.router.weight"),
            (
                "block.layers.1.mlp.shared_expert_fc1.weight",
                "decoder.layers.1.mlp.shared_fc1.weight",
            ),
            ("post_adapter.final_linear_audio.weight", "post_adapter.final_linear_audio.weight"),
        )
        for reference_key, model_key in direct_gradient_pairs:
            reference_gradient = reference_parameters[reference_key].grad
            model_gradient = model_parameters[model_key].grad
            assert reference_gradient is not None and model_gradient is not None
            torch.testing.assert_close(model_gradient, reference_gradient, rtol=2e-1, atol=5e-2)

        reference_expert = reference_parameters["block.layers.1.mlp.moe_mlp.W_gate"].grad[0].T
        model_expert = model_parameters[
            "decoder.layers.1.mlp.routed.experts.linear_fc1.weight0"
        ].grad[: config.magi2_moe_expert_intermediate_size]
        torch.testing.assert_close(model_expert, reference_expert, rtol=2e-1, atol=5e-2)

        reference_optimizer.step()
        model_optimizer.step()
        reference_optimizer.zero_grad(set_to_none=True)
        model_optimizer.zero_grad(set_to_none=True)
        reference_output, reference_loss, _ = _loss(reference, batch, requires_grad=False)
        model_output, model_loss, _ = _loss(model, batch, requires_grad=False)
        print(
            "MAGI2_STAGE5_ADAMW "
            f"reference_loss={reference_loss.item():.9f} "
            f"mcore_loss={model_loss.item():.9f} "
            f"loss_abs_diff={abs(model_loss.item() - reference_loss.item()):.9f}"
        )
        torch.testing.assert_close(model_output, reference_output, rtol=8e-2, atol=4e-2)
        torch.testing.assert_close(model_loss, reference_loss, rtol=3e-2, atol=3e-3)

    def test_strict_audit_rejects_missing_and_unknown_tensors(self) -> None:
        config = _config()
        reference, model = _models(config)
        missing_source = dict(reference.state_dict())
        missing_source.pop("block.layers.0.attention.linear_qkv.weight")
        with pytest.raises(RuntimeError, match="missing target keys"):
            convert_magi2_official_state_dict(missing_source, model)

        unknown_source = dict(reference.state_dict())
        unknown_source["block.layers.0.unknown.weight"] = torch.ones(1, device="cuda")
        with pytest.raises(RuntimeError, match="unexpected source keys"):
            convert_magi2_official_state_dict(unknown_source, model)

    def test_single_file_safetensors_streaming_load(self) -> None:
        from safetensors.torch import save_file

        config = _config()
        reference, _ = _models(config)
        checkpoint_dir = tempfile.mkdtemp(prefix="magi2_stage5_safe_", dir="/tmp")
        try:
            save_file(
                {
                    key: value.detach().cpu().contiguous()
                    for key, value in reference.state_dict().items()
                },
                os.path.join(checkpoint_dir, "model.safetensors"),
            )
            groups = ProcessGroupCollection.use_mpu_process_groups()
            loaded_model = Magi2Model(
                config, get_magi2_layer_specs(Magi2TorchDotProductAttention), groups
            ).cuda()
            report = load_magi2_official_safetensors(loaded_model, checkpoint_dir)
            assert report.is_strictly_complete
            print(
                "MAGI2_STAGE5_SAFETENSORS "
                f"source_keys={len(report.consumed_source_keys)} "
                f"target_keys={len(report.populated_target_keys)} "
                f"preserved_runtime_keys={len(report.preserved_target_keys)}"
            )

            batch = _batch(torch.device("cuda"))
            inputs, coordinates, mapping, cu_seqlens, _, _ = batch
            expected = reference(inputs, coordinates, mapping, cu_seqlens)
            actual = loaded_model(inputs, coordinates, mapping, cu_seqlens)
            torch.testing.assert_close(actual, expected, rtol=7e-2, atol=3e-2)
        finally:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)

    def test_dcp_model_and_adamw_resume_round_trip(self) -> None:
        config = _config()
        _, model = _models(config)
        batch = _batch(torch.device("cuda"))
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=2e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01
        )
        _, loss_before_save, _ = _loss(model, batch, requires_grad=False)
        loss_before_save.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        checkpoint_dir = tempfile.mkdtemp(prefix="magi2_stage5_", dir="/tmp")
        model_dir = os.path.join(checkpoint_dir, "model")
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        try:
            os.makedirs(model_dir)
            save(model.sharded_state_dict(), model_dir)
            torch.save(optimizer.state_dict(), optimizer_path)

            groups = ProcessGroupCollection.use_mpu_process_groups()
            resumed = Magi2Model(
                config, get_magi2_layer_specs(Magi2TorchDotProductAttention), groups
            ).cuda()
            loaded, missing, unexpected = load(
                resumed.sharded_state_dict(), model_dir, strict=StrictHandling.RETURN_ALL
            )
            assert not missing
            assert not unexpected
            resumed.load_state_dict(loaded, strict=True)
            resumed_optimizer = torch.optim.AdamW(
                resumed.parameters(), lr=2e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01
            )
            resumed_optimizer.load_state_dict(torch.load(optimizer_path, weights_only=True))

            _, uninterrupted_loss, _ = _loss(model, batch, requires_grad=False)
            _, resumed_loss, _ = _loss(resumed, batch, requires_grad=False)
            torch.testing.assert_close(resumed_loss, uninterrupted_loss, rtol=0.0, atol=0.0)
            uninterrupted_loss.backward()
            resumed_loss.backward()
            optimizer.step()
            resumed_optimizer.step()
            print(
                "MAGI2_STAGE5_RESUME "
                f"loss={resumed_loss.item():.9f} model_and_optimizer_next_step_equal=true"
            )
            for original, restored in zip(model.parameters(), resumed.parameters()):
                torch.testing.assert_close(restored, original, rtol=0.0, atol=0.0)
        finally:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
