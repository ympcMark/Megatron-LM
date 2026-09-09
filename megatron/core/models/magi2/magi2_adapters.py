# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Input and output adapters for the native MCore MAGI-2 model."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from megatron.core.models.magi2.magi2_config import Magi2Config
from megatron.core.models.magi2.magi2_modalities import Magi2MultiModalityRMSNorm
from megatron.core.models.magi2.magi2_rope import Magi2FourierRoPE
from megatron.core.models.magi2.magi2_runtime_context import Magi2Modality

Magi2RMSNorm = Magi2MultiModalityRMSNorm


class Magi2PreAdapter(nn.Module):
    """Project packed video, audio, text, and time tokens into residual streams."""

    def __init__(self, config: Magi2Config) -> None:
        super().__init__()
        self.config = config
        width = config.magi2_adapter_width
        self.video_embedder = nn.Linear(
            config.magi2_video_in_channels, width, bias=True, dtype=torch.float32
        )
        self.audio_embedder = nn.Linear(
            config.magi2_audio_in_channels, width, bias=True, dtype=torch.float32
        )
        self.text_embedder = nn.Linear(
            config.magi2_text_in_channels, width, bias=True, dtype=torch.float32
        )
        self.rope = Magi2FourierRoPE(config.magi2_attention_head_dim)

    def forward(self, inputs: Tensor, modality_mapping: Tensor) -> Tensor:
        """Apply the matching modality projection while preserving token order."""
        if inputs.ndim != 2:
            raise ValueError("inputs must have shape [tokens, input_channels]")
        if modality_mapping.shape != (inputs.shape[0],):
            raise ValueError("modality_mapping must have shape [tokens]")
        if inputs.shape[1] < max(
            self.config.magi2_video_in_channels,
            self.config.magi2_audio_in_channels,
            self.config.magi2_text_in_channels,
        ):
            raise ValueError("inputs do not contain all configured modality channels")

        output = torch.zeros(
            inputs.shape[0],
            self.config.magi2_adapter_width,
            device=inputs.device,
            dtype=torch.float32,
        )
        for modality, channels, projection in (
            (Magi2Modality.VIDEO, self.config.magi2_video_in_channels, self.video_embedder),
            (Magi2Modality.AUDIO, self.config.magi2_audio_in_channels, self.audio_embedder),
            (Magi2Modality.TEXT, self.config.magi2_text_in_channels, self.text_embedder),
            (Magi2Modality.TIME, self.config.magi2_text_in_channels, self.text_embedder),
        ):
            indices = torch.nonzero(modality_mapping == modality, as_tuple=False).flatten()
            if indices.numel():
                selected = inputs.index_select(0, indices)[:, :channels].float()
                output.index_copy_(0, indices, projection(selected))
        return output

    def build_rope(self, coordinates: Tensor) -> Tensor:
        """Build the shared per-forward partial rotary embedding."""
        return self.rope(coordinates)


class Magi2PostAdapter(nn.Module):
    """Project residual streams to packed video and audio velocity predictions."""

    def __init__(self, config: Magi2Config) -> None:
        super().__init__()
        self.config = config
        width = config.magi2_adapter_width
        self.final_norm_video = Magi2RMSNorm(width, eps=config.layernorm_epsilon)
        self.final_norm_audio = Magi2RMSNorm(width, eps=config.layernorm_epsilon)
        self.final_linear_video = nn.Linear(
            width, config.magi2_video_in_channels, bias=False, dtype=torch.float32
        )
        self.final_linear_audio = nn.Linear(
            width, config.magi2_audio_in_channels, bias=False, dtype=torch.float32
        )

    def forward(self, hidden_states: Tensor, modality_mapping: Tensor) -> Tensor:
        """Project video/audio tokens and leave text/time output rows at zero."""
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must have shape [tokens, n * hidden_size]")
        if hidden_states.shape[-1] != self.config.magi2_adapter_width:
            raise ValueError("hidden_states width does not match MAGI-2 residual streams")
        if modality_mapping.shape != (hidden_states.shape[0],):
            raise ValueError("modality_mapping must have shape [tokens]")

        output_width = max(self.config.magi2_video_in_channels, self.config.magi2_audio_in_channels)
        output = torch.zeros(
            hidden_states.shape[0], output_width, device=hidden_states.device, dtype=torch.float32
        )
        for modality, norm, projection, channels in (
            (
                Magi2Modality.VIDEO,
                self.final_norm_video,
                self.final_linear_video,
                self.config.magi2_video_in_channels,
            ),
            (
                Magi2Modality.AUDIO,
                self.final_norm_audio,
                self.final_linear_audio,
                self.config.magi2_audio_in_channels,
            ),
        ):
            indices = torch.nonzero(modality_mapping == modality, as_tuple=False).flatten()
            if indices.numel():
                selected = hidden_states.index_select(0, indices)
                projected = projection(norm(selected).float())
                output[:, :channels].index_copy_(0, indices, projected)
        return output
