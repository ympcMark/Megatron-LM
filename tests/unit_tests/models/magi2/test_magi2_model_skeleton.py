# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for the native MCore MAGI-2 model skeleton."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import megatron.core.models.magi2 as magi2_package
from megatron.core.models.magi2 import (
    Magi2Config,
    Magi2MLPType,
    Magi2Modality,
    Magi2Model,
    Magi2PostAdapter,
    Magi2PreAdapter,
    Magi2RuntimeContext,
    Magi2TransformerLayerSpecs,
    get_magi2_layer_plan,
    get_magi2_transformer_block_submodules,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import BaseTransformerLayer
from tests.unit_tests.test_utilities import Utils


class _IdentityMagi2Layer(MegatronModule, BaseTransformerLayer):
    """Shape-preserving test layer used before MAGI-2 kernels are introduced."""

    def __init__(
        self,
        config: Magi2Config,
        layer_number: int = 1,
        pg_collection: ProcessGroupCollection | None = None,
        vp_stage: int | None = None,
    ) -> None:
        super().__init__(config=config)
        self.layer_number = layer_number
        self.pg_collection = pg_collection
        self.vp_stage = vp_stage

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del attention_mask, kwargs
        return hidden_states, context


def _reduced_config(**overrides) -> Magi2Config:
    values = {
        "num_layers": 2,
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
        "magi2_moe_layers": (1,),
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


def _test_layer_specs() -> Magi2TransformerLayerSpecs:
    return Magi2TransformerLayerSpecs(
        dense=ModuleSpec(module=_IdentityMagi2Layer), moe=ModuleSpec(module=_IdentityMagi2Layer)
    )


class TestMagi2ModelSkeleton:
    """Exercise config, allocation, adapters, context, and model assembly."""

    def setup_method(self) -> None:
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self) -> None:
        Utils.destroy_model_parallel()

    def test_public_config_matches_frozen_contract(self) -> None:
        config = Magi2Config()

        assert config.num_layers == 40
        assert config.magi2_adapter_width == 12_288
        assert config.magi2_flattened_num_experts == 3_072
        assert config.magi2_dense_intermediate_size == 8_192
        assert config.magi2_parameter_count == 113_934_732_336

    def test_generic_mhc_and_incomplete_layer_allocation_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="generic mHC"):
            _reduced_config(enable_mhc_connections=True)
        with pytest.raises(ValueError, match="every MAGI-2 layer"):
            _reduced_config(magi2_moe_layers=())

    def test_layer_plan_and_transformer_block_specs_are_owned_by_mcore(self) -> None:
        config = _reduced_config()
        layer_specs = _test_layer_specs()
        plan = get_magi2_layer_plan(config)
        block_submodules = get_magi2_transformer_block_submodules(config, layer_specs)

        assert [layer.mlp_type for layer in plan] == [Magi2MLPType.DENSE, Magi2MLPType.MOE]
        assert plan[0].attention_num_modalities == 3
        assert plan[1].attention_num_modalities == 1
        assert plan[1].mlp_num_modalities == 3
        assert block_submodules.layer_specs == [layer_specs.dense, layer_specs.moe]

    def test_adapters_preserve_token_order_and_support_backward(self) -> None:
        config = _reduced_config()
        pre_adapter = Magi2PreAdapter(config)
        post_adapter = Magi2PostAdapter(config)
        inputs = torch.arange(20, dtype=torch.float32).reshape(4, 5).requires_grad_(True)
        mapping = torch.tensor(
            [Magi2Modality.VIDEO, Magi2Modality.AUDIO, Magi2Modality.TEXT, Magi2Modality.TIME]
        )

        hidden_states = pre_adapter(inputs, mapping)
        output = post_adapter(hidden_states, mapping)
        output.square().mean().backward()

        assert hidden_states.shape == (4, 32)
        assert output.shape == (4, 4)
        assert torch.count_nonzero(output[2:]) == 0
        assert inputs.grad is not None
        assert pre_adapter.video_embedder.weight.grad is not None
        assert post_adapter.final_linear_audio.weight.grad is not None

    def test_runtime_context_uses_existing_packed_sequence_interface(self) -> None:
        coordinates = torch.zeros(4, 9)
        mapping = torch.tensor(
            [Magi2Modality.VIDEO, Magi2Modality.TIME, Magi2Modality.TEXT, Magi2Modality.AUDIO]
        )
        cu_seqlens = torch.tensor([0, 2, 4], dtype=torch.int32)

        context = Magi2RuntimeContext.from_tensors(coordinates, mapping, cu_seqlens)

        assert context.qkv_format == "thd"
        assert context.max_seqlen_q == 2
        assert context.total_tokens == 4
        assert context.model_modality_mapping.tolist() == [0, 2, 2, 1]

    def test_model_assembles_adapters_and_mcore_transformer_block(self) -> None:
        config = _reduced_config()
        process_groups = ProcessGroupCollection.use_mpu_process_groups(["tp", "pp"])
        model = Magi2Model(config, _test_layer_specs(), process_groups)
        inputs = torch.arange(30, dtype=torch.float32).reshape(6, 5).requires_grad_(True)
        coordinates = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1, 1]], dtype=torch.float32).repeat(6, 1)
        mapping = torch.tensor(
            [
                Magi2Modality.VIDEO,
                Magi2Modality.AUDIO,
                Magi2Modality.TEXT,
                Magi2Modality.TIME,
                Magi2Modality.VIDEO,
                Magi2Modality.AUDIO,
            ]
        )
        cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32)

        output = model(inputs, coordinates, mapping, cu_seqlens)
        output.square().mean().backward()

        assert len(model.decoder.layers) == 2
        assert output.shape == (6, 4)
        assert torch.count_nonzero(output[2:4]) == 0
        assert inputs.grad is not None
        # A successful post-adapter projection also proves TransformerBlock did
        # not incorrectly expand the already-2H residual stream a second time.
        assert model.pre_adapter is not None
        assert model.post_adapter is not None

    def test_mcore_package_has_no_bridge_dependency(self) -> None:
        package_path = Path(magi2_package.__file__).parent
        imported_modules = []
        for source_path in package_path.glob("*.py"):
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_modules.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported_modules.append(node.module)

        assert not [name for name in imported_modules if name.startswith("megatron.bridge")]
