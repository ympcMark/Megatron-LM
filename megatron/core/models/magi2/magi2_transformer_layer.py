# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Thin MAGI-2 Transformer layer hosted by MCore TransformerBlock."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from megatron.core.models.magi2.magi2_config import Magi2Config
from megatron.core.models.magi2.magi2_mhc import Magi2MHCBranch
from megatron.core.models.magi2.magi2_modalities import Magi2MultiModalityRMSNorm
from megatron.core.models.magi2.magi2_runtime_context import Magi2RuntimeContext
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_layer import BaseTransformerLayer


@dataclass(frozen=True)
class Magi2TransformerLayerSubmodules:
    """Attention and feed-forward modules injected into one MAGI-2 layer."""

    self_attention: ModuleSpec
    mlp: ModuleSpec


class Magi2TransformerLayer(MegatronModule, BaseTransformerLayer):
    """Apply independent mHC mappings around MAGI-2 Attention and MLP branches."""

    supports_mhc_connections = True

    def __init__(
        self,
        config: Magi2Config,
        submodules: Magi2TransformerLayerSubmodules,
        layer_number: int = 1,
        pg_collection: ProcessGroupCollection | None = None,
        vp_stage: int | None = None,
    ) -> None:
        super().__init__(config=config)
        if pg_collection is None:
            raise ValueError("Magi2TransformerLayer requires an explicit ProcessGroupCollection")
        self.config = config
        self.layer_number = layer_number
        self.vp_stage = vp_stage
        self.pg_collection = pg_collection
        self.layer_index = layer_number - 1
        self.num_modalities = 3 if self.layer_index in config.magi2_mm_layers else 1
        self.is_moe_layer = self.layer_index in config.magi2_moe_layers

        self.mhc_norm = Magi2MultiModalityRMSNorm(
            config.magi2_adapter_width,
            num_modalities=self.num_modalities,
            eps=config.layernorm_epsilon,
            out_dtype=torch.float32,
        )
        self.attention_mhc = Magi2MHCBranch(config)
        self.mlp_mhc = Magi2MHCBranch(config)
        self.self_attention = build_module(
            submodules.self_attention,
            config=config,
            layer_number=layer_number,
            num_modalities=self.num_modalities,
            pg_collection=pg_collection,
        )
        self.mlp = build_module(
            submodules.mlp,
            config=config,
            layer_number=layer_number,
            num_modalities=(3 if self.is_moe_layer else self.num_modalities),
            pg_collection=pg_collection,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        context: Tensor | None = None,
        packed_seq_params: Magi2RuntimeContext | None = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor | None]:
        """Run the official Attention→MLP sequence while preserving ``[T, B, nH]``."""
        del kwargs
        if not isinstance(packed_seq_params, Magi2RuntimeContext):
            raise TypeError("Magi2TransformerLayer requires Magi2RuntimeContext")
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.config.magi2_adapter_width:
            raise ValueError("hidden_states must have shape [T, B, magi2_adapter_width]")

        dispatcher = packed_seq_params.get_modality_dispatcher()
        attention_input, attention_state = self.attention_mhc.prepare(
            hidden_states, self.mhc_norm(hidden_states, dispatcher)
        )
        attention_output = self.self_attention(attention_input, attention_mask, packed_seq_params)
        hidden_states = self.attention_mhc.merge(hidden_states, attention_output, attention_state)

        mlp_input, mlp_state = self.mlp_mhc.prepare(
            hidden_states, self.mhc_norm(hidden_states, dispatcher)
        )
        mlp_output = self.mlp(mlp_input, dispatcher=dispatcher)
        if isinstance(mlp_output, tuple):
            mlp_output, mlp_bias = mlp_output
            if mlp_bias is not None:
                raise ValueError("MAGI-2 MLPs must be bias-free")
        hidden_states = self.mlp_mhc.merge(hidden_states, mlp_output, mlp_state)
        return hidden_states, context
