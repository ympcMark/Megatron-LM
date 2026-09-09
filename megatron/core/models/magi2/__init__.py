# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Native Megatron Core support for MAGI-2 Preview."""

from megatron.core.models.magi2.magi2_adapters import (
    Magi2PostAdapter,
    Magi2PreAdapter,
    Magi2RMSNorm,
)
from megatron.core.models.magi2.magi2_attention import (
    Magi2Attention,
    Magi2AttentionSubmodules,
    Magi2TorchDotProductAttention,
)
from megatron.core.models.magi2.magi2_checkpoint import (
    Magi2CheckpointConversionReport,
    convert_magi2_official_state_dict,
    load_magi2_official_safetensors,
    load_magi2_official_state_dict,
    magi2_official_to_mcore_key,
)
from megatron.core.models.magi2.magi2_config import Magi2Config
from megatron.core.models.magi2.magi2_layer_specs import (
    Magi2LayerPlan,
    Magi2MLPType,
    Magi2TransformerLayerSpecs,
    get_magi2_layer_plan,
    get_magi2_layer_specs,
    get_magi2_transformer_block_submodules,
    get_magi2_transformer_layer_spec,
)
from megatron.core.models.magi2.magi2_mhc import Magi2MHCBranch, Magi2MHCState, magi2_sinkhorn
from megatron.core.models.magi2.magi2_mlp import Magi2DenseMLP, Magi2MoEMLP, magi2_quick_geglu
from megatron.core.models.magi2.magi2_modalities import (
    Magi2ModalityDispatcher,
    Magi2ModalityLinear,
    Magi2MultiModalityRMSNorm,
)
from megatron.core.models.magi2.magi2_model import Magi2Model
from megatron.core.models.magi2.magi2_moe import (
    Magi2DistributedMultiHeadMoE,
    Magi2MultiHeadTopKRouter,
    Magi2RouterScoreFunction,
    build_magi2_expert_config,
    multi_head_topk_routing,
)
from megatron.core.models.magi2.magi2_rope import Magi2FourierRoPE, apply_magi2_rotary_pos_emb
from megatron.core.models.magi2.magi2_runtime_context import Magi2Modality, Magi2RuntimeContext
from megatron.core.models.magi2.magi2_transformer_layer import (
    Magi2TransformerLayer,
    Magi2TransformerLayerSubmodules,
)

__all__ = [
    "Magi2Config",
    "Magi2CheckpointConversionReport",
    "Magi2DenseMLP",
    "Magi2DistributedMultiHeadMoE",
    "Magi2Attention",
    "Magi2AttentionSubmodules",
    "Magi2FourierRoPE",
    "Magi2LayerPlan",
    "Magi2MHCBranch",
    "Magi2MHCState",
    "Magi2MLPType",
    "Magi2Model",
    "Magi2Modality",
    "Magi2ModalityDispatcher",
    "Magi2ModalityLinear",
    "Magi2MultiModalityRMSNorm",
    "Magi2MoEMLP",
    "Magi2MultiHeadTopKRouter",
    "Magi2PostAdapter",
    "Magi2PreAdapter",
    "Magi2RMSNorm",
    "Magi2RuntimeContext",
    "Magi2RouterScoreFunction",
    "Magi2TorchDotProductAttention",
    "Magi2TransformerLayer",
    "Magi2TransformerLayerSubmodules",
    "Magi2TransformerLayerSpecs",
    "apply_magi2_rotary_pos_emb",
    "build_magi2_expert_config",
    "convert_magi2_official_state_dict",
    "get_magi2_layer_specs",
    "get_magi2_layer_plan",
    "get_magi2_transformer_block_submodules",
    "get_magi2_transformer_layer_spec",
    "magi2_sinkhorn",
    "magi2_official_to_mcore_key",
    "magi2_quick_geglu",
    "load_magi2_official_state_dict",
    "load_magi2_official_safetensors",
    "multi_head_topk_routing",
]
