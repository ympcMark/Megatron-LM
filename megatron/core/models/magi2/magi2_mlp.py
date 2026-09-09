# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Dense and routed feed-forward layers for native MCore MAGI-2."""

from __future__ import annotations

import torch
from torch import Tensor

from megatron.core.models.magi2.magi2_config import Magi2Config
from megatron.core.models.magi2.magi2_modalities import (
    Magi2ModalityDispatcher,
    Magi2ModalityLinear,
    Magi2MultiModalityRMSNorm,
)
from megatron.core.models.magi2.magi2_moe import Magi2DistributedMultiHeadMoE
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule


def magi2_quick_geglu(value: Tensor) -> Tensor:
    """Apply the public interleaved QuickGEGLU7 activation in FP32."""
    output_dtype = value.dtype
    value = value.float()
    gate = value[..., ::2].clamp(max=7.0)
    linear = value[..., 1::2].clamp(min=-7.0, max=7.0)
    return (gate * torch.sigmoid(1.702 * gate) * (linear + 1.0)).to(output_dtype)


class Magi2DenseMLP(MegatronModule):
    """Modality-aware dense MLP used by the four multimodal layers."""

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
        intermediate_size = config.magi2_dense_intermediate_size
        linear_kwargs = {
            "dtype": config.params_dtype,
            "init_method": config.init_method,
            "perform_initialization": config.perform_initialization,
        }
        self.pre_norm = Magi2MultiModalityRMSNorm(
            config.hidden_size, num_modalities=num_modalities, eps=config.layernorm_epsilon
        )
        self.up_gate_proj = Magi2ModalityLinear(
            config.hidden_size,
            2 * intermediate_size,
            num_modalities=num_modalities,
            **linear_kwargs,
        )
        self.down_proj = Magi2ModalityLinear(
            intermediate_size, config.hidden_size, num_modalities=num_modalities, **linear_kwargs
        )

    def forward(self, hidden_states: Tensor, dispatcher: Magi2ModalityDispatcher) -> Tensor:
        """Apply modality norm, QuickGEGLU7, and the down projection."""
        hidden_states = self.pre_norm(hidden_states, dispatcher)
        hidden_states = magi2_quick_geglu(self.up_gate_proj(hidden_states, dispatcher))
        return self.down_proj(hidden_states, dispatcher)


class Magi2MoEMLP(MegatronModule):
    """Routed head experts plus global and modality-specific shared experts."""

    def __init__(
        self,
        config: Magi2Config,
        layer_number: int,
        num_modalities: int,
        pg_collection: ProcessGroupCollection,
    ) -> None:
        super().__init__(config=config)
        if num_modalities != 3:
            raise ValueError("MAGI-2 MoE layers require three modality-shared experts")
        self.layer_number = layer_number
        self.num_modalities = num_modalities
        self.pg_collection = pg_collection
        hidden_size = config.hidden_size
        linear_kwargs = {
            "dtype": config.params_dtype,
            "init_method": config.init_method,
            "perform_initialization": config.perform_initialization,
        }

        self.pre_norm = Magi2MultiModalityRMSNorm(
            hidden_size, num_modalities=num_modalities, eps=config.layernorm_epsilon
        )
        self.split_linear = Magi2ModalityLinear(hidden_size, hidden_size, **linear_kwargs)
        self.merge_linear = Magi2ModalityLinear(hidden_size, hidden_size, **linear_kwargs)
        self.routed = Magi2DistributedMultiHeadMoE(
            config, layer_number=layer_number, pg_collection=pg_collection
        )
        self.shared_fc1 = Magi2ModalityLinear(
            hidden_size, 2 * config.magi2_shared_expert_intermediate_size, **linear_kwargs
        )
        self.shared_fc2 = Magi2ModalityLinear(
            config.magi2_shared_expert_intermediate_size, hidden_size, **linear_kwargs
        )
        self.modality_fc1 = Magi2ModalityLinear(
            hidden_size,
            2 * config.magi2_modality_expert_intermediate_size,
            num_modalities=num_modalities,
            **linear_kwargs,
        )
        self.modality_fc2 = Magi2ModalityLinear(
            config.magi2_modality_expert_intermediate_size,
            hidden_size,
            num_modalities=num_modalities,
            **linear_kwargs,
        )

    def forward(self, hidden_states: Tensor, dispatcher: Magi2ModalityDispatcher) -> Tensor:
        """Sum routed, global-shared, and modality-shared expert outputs."""
        normed = self.pre_norm(hidden_states, dispatcher)
        routed, routed_bias = self.routed(self.split_linear(normed))
        if routed_bias is not None:
            raise RuntimeError("MAGI-2 routed experts must be bias-free")
        routed = self.merge_linear(routed)
        shared = self.shared_fc2(magi2_quick_geglu(self.shared_fc1(normed)))
        modality = self.modality_fc2(
            magi2_quick_geglu(self.modality_fc1(normed, dispatcher)), dispatcher
        )
        return routed + shared + modality
