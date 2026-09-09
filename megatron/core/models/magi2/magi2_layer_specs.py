# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Layer allocation helpers for the native MCore MAGI-2 model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from megatron.core.models.magi2.magi2_attention import Magi2Attention, Magi2AttentionSubmodules
from megatron.core.models.magi2.magi2_config import Magi2Config
from megatron.core.models.magi2.magi2_mlp import Magi2DenseMLP, Magi2MoEMLP
from megatron.core.models.magi2.magi2_transformer_layer import (
    Magi2TransformerLayer,
    Magi2TransformerLayerSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules


class Magi2MLPType(str, Enum):
    """Feed-forward family used by a MAGI-2 transformer layer."""

    DENSE = "dense"
    MOE = "moe"


@dataclass(frozen=True)
class Magi2LayerPlan:
    """Architecture properties for one MAGI-2 transformer layer."""

    layer_index: int
    mlp_type: Magi2MLPType
    attention_num_modalities: int
    mhc_num_modalities: int
    mlp_num_modalities: int


@dataclass(frozen=True)
class Magi2TransformerLayerSpecs:
    """Injectable implementations for the two MAGI-2 layer families."""

    dense: ModuleSpec
    moe: ModuleSpec


def get_magi2_transformer_layer_spec(
    core_attention: ModuleSpec | type, mlp: ModuleSpec
) -> ModuleSpec:
    """Build one MAGI-2 layer spec from an MCore attention core and MLP."""
    return ModuleSpec(
        module=Magi2TransformerLayer,
        submodules=Magi2TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=Magi2Attention,
                submodules=Magi2AttentionSubmodules(core_attention=core_attention),
            ),
            mlp=mlp,
        ),
    )


def get_magi2_layer_specs(core_attention: ModuleSpec | type) -> Magi2TransformerLayerSpecs:
    """Build the native dense and MoE MAGI-2 Transformer layer specs."""
    return Magi2TransformerLayerSpecs(
        dense=get_magi2_transformer_layer_spec(core_attention, ModuleSpec(module=Magi2DenseMLP)),
        moe=get_magi2_transformer_layer_spec(core_attention, ModuleSpec(module=Magi2MoEMLP)),
    )


def get_magi2_layer_plan(config: Magi2Config) -> tuple[Magi2LayerPlan, ...]:
    """Return the ordered dense/MoE allocation owned by MAGI-2 MCore code."""
    mm_layers = set(config.magi2_mm_layers)
    moe_layers = set(config.magi2_moe_layers)
    return tuple(
        Magi2LayerPlan(
            layer_index=layer_index,
            mlp_type=(Magi2MLPType.MOE if layer_index in moe_layers else Magi2MLPType.DENSE),
            attention_num_modalities=3 if layer_index in mm_layers else 1,
            mhc_num_modalities=3 if layer_index in mm_layers else 1,
            # Routed layers always contain video/audio/text shared experts.
            mlp_num_modalities=(
                3 if layer_index in moe_layers else (3 if layer_index in mm_layers else 1)
            ),
        )
        for layer_index in range(config.num_layers)
    )


def get_magi2_transformer_block_submodules(
    config: Magi2Config, layer_specs: Magi2TransformerLayerSpecs
) -> TransformerBlockSubmodules:
    """Build the heterogeneous 40-layer TransformerBlock specification."""
    ordered_specs = [
        layer_specs.moe if plan.mlp_type is Magi2MLPType.MOE else layer_specs.dense
        for plan in get_magi2_layer_plan(config)
    ]
    return TransformerBlockSubmodules(layer_specs=ordered_specs)
