# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Native Megatron Core model shell for MAGI-2 Preview."""

from __future__ import annotations

from torch import Tensor

from megatron.core.models.magi2.magi2_adapters import Magi2PostAdapter, Magi2PreAdapter
from megatron.core.models.magi2.magi2_config import Magi2Config
from megatron.core.models.magi2.magi2_layer_specs import (
    Magi2TransformerLayerSpecs,
    get_magi2_transformer_block_submodules,
)
from megatron.core.models.magi2.magi2_runtime_context import Magi2RuntimeContext
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_block import TransformerBlock


class Magi2Model(MegatronModule):
    """MAGI-2 model owned and assembled by Megatron Core.

    The model shell owns the modality adapters and the MCore TransformerBlock.
    Attention, mHC, and dense/MoE implementations are selected through the two
    injected layer specs; the ordered 40-layer allocation remains owned here in
    MCore rather than in a Bridge provider.
    """

    def __init__(
        self,
        config: Magi2Config,
        layer_specs: Magi2TransformerLayerSpecs,
        pg_collection: ProcessGroupCollection,
        *,
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: int | None = None,
    ) -> None:
        super().__init__(config=config)
        if config.enable_mhc_connections:
            raise ValueError("MAGI-2 must not use TransformerBlock's automatic mHC expansion")
        self.config = config
        self.pre_process = pre_process
        self.post_process = post_process
        self.pg_collection = pg_collection

        self.pre_adapter = Magi2PreAdapter(config) if pre_process else None
        self.decoder = TransformerBlock(
            config=config,
            spec=get_magi2_transformer_block_submodules(config, layer_specs),
            post_layer_norm=False,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )
        self.post_adapter = Magi2PostAdapter(config) if post_process else None

    def set_input_tensor(self, input_tensor: Tensor | list[Tensor]) -> None:
        """Set the tensor received from the previous pipeline stage."""
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        if len(input_tensor) != 1:
            raise ValueError("Magi2Model expects exactly one pipeline input tensor")
        self.decoder.set_input_tensor(input_tensor[0])

    def forward(
        self,
        inputs: Tensor | None,
        coordinates: Tensor,
        modality_mapping: Tensor,
        cu_seqlens: Tensor,
        attention_mask: Tensor | None = None,
        *,
        cp_split_sizes: Tensor | None = None,
        time_token_sequence: Tensor | None = None,
        runtime_context: Magi2RuntimeContext | None = None,
    ) -> Tensor:
        """Run a packed multimodal sequence through the MAGI-2 model shell."""
        token_count = int(modality_mapping.numel())
        if runtime_context is None:
            runtime_context = Magi2RuntimeContext.from_tensors(
                coordinates,
                modality_mapping,
                cu_seqlens,
                cp_split_sizes=cp_split_sizes,
                time_token_sequence=time_token_sequence,
            )
        else:
            runtime_context.validate(token_count)

        dispatcher = runtime_context.get_modality_dispatcher()

        if self.pre_process:
            if inputs is None or self.pre_adapter is None:
                raise ValueError("inputs are required on the first MAGI-2 pipeline stage")
            hidden_states = self.pre_adapter(inputs, modality_mapping)
            time_tokens = runtime_context.time_token_sequence
            if time_tokens is not None and time_tokens.shape[-1] > 0:
                if time_tokens.ndim != 2 or time_tokens.shape[0] != token_count:
                    raise ValueError("time_token_sequence must have shape [tokens, channels]")
                if time_tokens.shape[-1] > hidden_states.shape[-1]:
                    raise ValueError("time_token_sequence is wider than the adapter output")
                hidden_states[:, : time_tokens.shape[-1]] = time_tokens.to(hidden_states.dtype)
            runtime_context.rope = self.pre_adapter.build_rope(coordinates)
            hidden_states = dispatcher.permute(hidden_states)
            hidden_states = hidden_states.to(self.config.params_dtype).unsqueeze(1)
        else:
            hidden_states = None

        hidden_states = self.decoder(
            hidden_states, attention_mask, packed_seq_params=runtime_context
        )

        if not self.post_process:
            return hidden_states
        if self.post_adapter is None:
            raise RuntimeError("post_adapter is missing on the final MAGI-2 pipeline stage")
        if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
            raise ValueError("MAGI-2 currently expects packed hidden states with batch size one")
        hidden_states = dispatcher.inverse_permute(hidden_states.squeeze(1))
        return self.post_adapter(hidden_states, modality_mapping)
