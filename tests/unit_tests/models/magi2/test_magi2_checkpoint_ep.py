# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""EP4 official-weight mapping and distributed-checkpoint tests for MAGI-2."""

from __future__ import annotations

import os
import shutil
import tempfile

import pytest
import torch
import torch.distributed as dist

from megatron.core.dist_checkpointing import load, save
from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core.models.magi2 import (
    Magi2Config,
    Magi2Model,
    Magi2TorchDotProductAttention,
    get_magi2_layer_specs,
    load_magi2_official_state_dict,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.models.magi2.magi2_reference import ReferenceMagi2Model
from tests.unit_tests.models.magi2.test_magi2_checkpoint_parity import _batch
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


class TestMagi2CheckpointEP:
    """Validate public global experts and MCore DCP local shards under EP4."""

    def setup_method(self) -> None:
        self.ep_size = int(os.environ.get("WORLD_SIZE", "1"))
        if self.ep_size != 4:
            pytest.skip("this test requires exactly four distributed ranks")
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=self.ep_size)

    def teardown_method(self) -> None:
        Utils.destroy_model_parallel()

    def test_official_experts_forward_and_dcp_round_trip(self) -> None:
        config = _config(self.ep_size)
        torch.manual_seed(4321)
        reference = ReferenceMagi2Model(config).cuda()
        groups = ProcessGroupCollection.use_mpu_process_groups()
        model = Magi2Model(
            config, get_magi2_layer_specs(Magi2TorchDotProductAttention), groups
        ).cuda()
        report = load_magi2_official_state_dict(model, reference.state_dict())
        assert report.is_strictly_complete

        routed = model.decoder.layers[0].mlp.routed
        ep_rank = dist.get_rank(groups.ep)
        expected_indices = [2 * ep_rank, 2 * ep_rank + 1]
        assert routed.local_expert_indices == expected_indices
        official_moe = reference.block.layers[0].mlp.moe_mlp
        for local_index, global_index in enumerate(expected_indices):
            expected_fc1 = torch.cat(
                (official_moe.W_gate[global_index].T, official_moe.W_up[global_index].T)
            )
            expected_fc2 = official_moe.W_down[global_index].T
            torch.testing.assert_close(
                getattr(routed.experts.linear_fc1, f"weight{local_index}"), expected_fc1
            )
            torch.testing.assert_close(
                getattr(routed.experts.linear_fc2, f"weight{local_index}"), expected_fc2
            )

        batch = _batch(torch.device("cuda"))
        inputs, coordinates, mapping, cu_seqlens, _, _ = batch
        reference_output = reference(inputs, coordinates, mapping, cu_seqlens)
        output = model(inputs, coordinates, mapping, cu_seqlens)
        if dist.get_rank() == 0:
            print(
                "MAGI2_STAGE5_EP4_FORWARD "
                f"output_max_abs_diff={(output - reference_output).abs().max().item():.9f}"
            )
        torch.testing.assert_close(output, reference_output, rtol=7e-2, atol=3e-2)

        path_holder: list[str | None] = [None]
        if dist.get_rank() == 0:
            path_holder[0] = tempfile.mkdtemp(prefix="magi2_stage5_ep4_", dir="/tmp")
        dist.broadcast_object_list(path_holder, src=0)
        checkpoint_dir = path_holder[0]
        assert checkpoint_dir is not None
        try:
            save(model.sharded_state_dict(), checkpoint_dir)
            resumed = Magi2Model(
                config, get_magi2_layer_specs(Magi2TorchDotProductAttention), groups
            ).cuda()
            loaded, missing, unexpected = load(
                resumed.sharded_state_dict(), checkpoint_dir, strict=StrictHandling.RETURN_ALL
            )
            assert not missing
            assert not unexpected
            resumed.load_state_dict(loaded, strict=True)
            resumed_output = resumed(inputs, coordinates, mapping, cu_seqlens)
            torch.testing.assert_close(resumed_output, output, rtol=0.0, atol=0.0)
            if dist.get_rank() == 0:
                print("MAGI2_STAGE5_EP4_DCP resumed_output_equal=true")
        finally:
            dist.barrier()
            if dist.get_rank() == 0:
                shutil.rmtree(checkpoint_dir, ignore_errors=True)
