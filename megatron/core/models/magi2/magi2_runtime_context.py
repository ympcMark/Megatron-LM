# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Per-forward runtime metadata for packed MAGI-2 sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import torch
from torch import Tensor

from megatron.core.models.magi2.magi2_modalities import Magi2ModalityDispatcher
from megatron.core.packed_seq_params import PackedSeqParams


class Magi2Modality(IntEnum):
    """Token modality identifiers used by MAGI-2 Preview."""

    VIDEO = 0
    AUDIO = 1
    TEXT = 2
    TIME = 3


@dataclass
class Magi2RuntimeContext(PackedSeqParams):
    """Packed sequence metadata consumed by MAGI-2 transformer layers.

    Extending :class:`PackedSeqParams` lets MCore's TransformerBlock pass this
    object through its existing attention and MLP call path. Runtime data stays
    attached to the forward invocation rather than mutable model state, which
    keeps recomputation and multiple in-flight microbatches safe.
    """

    coordinates: Tensor | None = None
    modality_mapping: Tensor | None = None
    cp_split_sizes: Tensor | None = None
    cp_global_indices: Tensor | None = None
    time_token_sequence: Tensor | None = None
    rope: Tensor | None = None
    _modality_dispatcher: Magi2ModalityDispatcher | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Initialize base packed-sequence fields and validate MAGI-2 metadata."""
        super().__post_init__()
        if self.coordinates is not None and self.modality_mapping is not None:
            self.validate(self.modality_mapping.numel())

    @classmethod
    def from_tensors(
        cls,
        coordinates: Tensor,
        modality_mapping: Tensor,
        cu_seqlens: Tensor,
        *,
        cu_seqlens_padded: Tensor | None = None,
        cp_split_sizes: Tensor | None = None,
        cp_global_indices: Tensor | None = None,
        time_token_sequence: Tensor | None = None,
    ) -> "Magi2RuntimeContext":
        """Construct runtime metadata for a packed self-attention batch."""
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must be one-dimensional with at least two entries")
        sequence_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        if torch.any(sequence_lengths < 0):
            raise ValueError("cu_seqlens must be monotonically non-decreasing")
        max_sequence_length = int(sequence_lengths.max().item())
        if cu_seqlens_padded is not None:
            if cu_seqlens_padded.shape != cu_seqlens.shape:
                raise ValueError("cu_seqlens_padded must match cu_seqlens")
            padded_lengths = cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
            if torch.any(padded_lengths < sequence_lengths):
                raise ValueError("padded sequence lengths must cover logical sequence lengths")
            total_tokens = int(cu_seqlens_padded[-1].item())
        else:
            total_tokens = int(cu_seqlens[-1].item())
        context = cls(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
            max_seqlen_q=max_sequence_length,
            max_seqlen_kv=max_sequence_length,
            total_tokens=total_tokens,
            coordinates=coordinates,
            modality_mapping=modality_mapping,
            cp_split_sizes=cp_split_sizes,
            cp_global_indices=cp_global_indices,
            time_token_sequence=time_token_sequence,
        )
        context.validate(modality_mapping.numel())
        return context

    def validate(self, token_count: int) -> None:
        """Validate shapes, token counts, and modality identifiers."""
        if self.coordinates is None or self.coordinates.shape != (token_count, 9):
            raise ValueError("coordinates must have shape [tokens, 9]")
        if self.modality_mapping is None or self.modality_mapping.shape != (token_count,):
            raise ValueError("modality_mapping must have shape [tokens]")
        if self.modality_mapping.numel() == 0:
            raise ValueError("MAGI-2 requires at least one token")
        if self.modality_mapping.min().item() < Magi2Modality.VIDEO:
            raise ValueError("modality_mapping contains a negative modality")
        if self.modality_mapping.max().item() > Magi2Modality.TIME:
            raise ValueError("modality_mapping contains an unsupported modality")
        if self.cu_seqlens_q is None or self.cu_seqlens_q.ndim != 1:
            raise ValueError("cu_seqlens_q must be one-dimensional")
        if self.cu_seqlens_q[0].item() != 0:
            raise ValueError("cu_seqlens must start at zero")
        physical_cu_seqlens = (
            self.cu_seqlens_q_padded if self.cu_seqlens_q_padded is not None else self.cu_seqlens_q
        )
        if physical_cu_seqlens.ndim != 1 or physical_cu_seqlens.shape != self.cu_seqlens_q.shape:
            raise ValueError("padded sequence boundaries must match logical sequence boundaries")
        global_token_count = int(physical_cu_seqlens[-1].item())
        if self.cp_split_sizes is None:
            if global_token_count != token_count:
                raise ValueError("cu_seqlens must end at the packed token count")
        else:
            if self.cp_split_sizes.ndim != 1 or torch.any(self.cp_split_sizes <= 0):
                raise ValueError("cp_split_sizes must contain one positive size per CP rank")
            if int(self.cp_split_sizes.sum().item()) != global_token_count:
                raise ValueError("cp_split_sizes must sum to the padded packed token count")
            if not torch.any(self.cp_split_sizes == token_count):
                raise ValueError("local packed token count is absent from cp_split_sizes")
            if self.cp_global_indices is None or self.cp_global_indices.shape != (token_count,):
                raise ValueError("cp_global_indices must contain one index per local token")
            if (
                self.cp_global_indices.min().item() < 0
                or self.cp_global_indices.max().item() >= global_token_count
            ):
                raise ValueError("cp_global_indices contains an out-of-range global token index")

    @property
    def model_modality_mapping(self) -> Tensor:
        """Return the mapping used by modality-specific transformer weights."""
        if self.modality_mapping is None:
            raise RuntimeError("modality_mapping has not been initialized")
        mapping = self.modality_mapping.clone()
        mapping[mapping == Magi2Modality.TIME] = Magi2Modality.TEXT
        return mapping

    def get_modality_dispatcher(self) -> Magi2ModalityDispatcher:
        """Return the stable video/audio/text dispatcher for this invocation."""
        if self._modality_dispatcher is None:
            self._modality_dispatcher = Magi2ModalityDispatcher(self.model_modality_mapping)
        return self._modality_dispatcher

    def indices(self, modality: Magi2Modality) -> Tensor:
        """Return packed token indices for one original input modality."""
        if self.modality_mapping is None:
            raise RuntimeError("modality_mapping has not been initialized")
        return torch.nonzero(self.modality_mapping == modality, as_tuple=False).flatten()
