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
        cp_split_sizes: Tensor | None = None,
        time_token_sequence: Tensor | None = None,
    ) -> "Magi2RuntimeContext":
        """Construct runtime metadata for a packed self-attention batch."""
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must be one-dimensional with at least two entries")
        sequence_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        if torch.any(sequence_lengths < 0):
            raise ValueError("cu_seqlens must be monotonically non-decreasing")
        max_sequence_length = int(sequence_lengths.max().item())
        context = cls(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_sequence_length,
            max_seqlen_kv=max_sequence_length,
            total_tokens=int(modality_mapping.numel()),
            coordinates=coordinates,
            modality_mapping=modality_mapping,
            cp_split_sizes=cp_split_sizes,
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
        if self.cu_seqlens_q[-1].item() != token_count:
            raise ValueError("cu_seqlens must end at the packed token count")

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
