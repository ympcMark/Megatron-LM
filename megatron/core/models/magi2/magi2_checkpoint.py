# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Official MAGI-2 Preview checkpoint interop for the native MCore model."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn

_LAYER_KEY = re.compile(r"^block\.layers\.(\d+)\.(.+)$")
_EXPERT_WEIGHT_NAMES = ("W_gate", "W_up", "W_down")
_EXPERT_READ_CHUNK_SIZE = 16
_MHC_NAMES = {
    "mhc_alpha_pre_attn": "attention_mhc.alpha_pre",
    "mhc_alpha_post_attn": "attention_mhc.alpha_post",
    "mhc_alpha_res_attn": "attention_mhc.alpha_res",
    "mhc_bias_pre_attn": "attention_mhc.bias_pre",
    "mhc_bias_post_attn": "attention_mhc.bias_post",
    "mhc_bias_res_attn": "attention_mhc.bias_res",
    "mhc_phi_fused_attn": "attention_mhc.phi_fused",
    "mhc_alpha_pre_mlp": "mlp_mhc.alpha_pre",
    "mhc_alpha_post_mlp": "mlp_mhc.alpha_post",
    "mhc_alpha_res_mlp": "mlp_mhc.alpha_res",
    "mhc_bias_pre_mlp": "mlp_mhc.bias_pre",
    "mhc_bias_post_mlp": "mlp_mhc.bias_post",
    "mhc_bias_res_mlp": "mlp_mhc.bias_res",
    "mhc_phi_fused_mlp": "mlp_mhc.phi_fused",
}
_MOE_NAMES = {
    "pre_norm.weight": "pre_norm.weight",
    "split_linear.weight": "split_linear.weight",
    "merge_linear.weight": "merge_linear.weight",
    "shared_expert_fc1.weight": "shared_fc1.weight",
    "shared_expert_fc2.weight": "shared_fc2.weight",
    "modality_specific_shared_expert_fc1.weight": "modality_fc1.weight",
    "modality_specific_shared_expert_fc2.weight": "modality_fc2.weight",
    "moe_mlp.gate": "routed.router.weight",
    "moe_mlp.router.expert_bias": "routed.router.expert_bias",
    "moe_mlp.router.expert_bias_ema": "routed.router.expert_bias_ema",
}


@dataclass(frozen=True)
class Magi2CheckpointConversionReport:
    """Audit information returned by an official-to-MCore conversion."""

    consumed_source_keys: tuple[str, ...]
    populated_target_keys: tuple[str, ...]
    preserved_target_keys: tuple[str, ...]
    missing_target_keys: tuple[str, ...]
    unexpected_source_keys: tuple[str, ...]

    @property
    def is_strictly_complete(self) -> bool:
        """Whether every public tensor and every MCore model tensor was accounted for."""
        return not self.missing_target_keys and not self.unexpected_source_keys


def _without_wrapper_prefix(key: str) -> str:
    if key.startswith("module."):
        return key[len("module.") :]
    return key


def _is_official_expert_weight(key: str) -> bool:
    return any(key.endswith(f".mlp.moe_mlp.{name}") for name in _EXPERT_WEIGHT_NAMES)


def magi2_official_to_mcore_key(source_key: str) -> str | None:
    """Map one non-expert official state key to its native MCore key.

    The three routed-expert matrices require a tensor transform and therefore
    return ``None`` here; :func:`convert_magi2_official_state_dict` handles them
    as one group.
    """
    source_key = _without_wrapper_prefix(source_key)
    if source_key.startswith(("pre_adapter.", "post_adapter.")):
        return source_key

    match = _LAYER_KEY.match(source_key)
    if match is None:
        return None
    layer_index, suffix = match.groups()
    target_prefix = f"decoder.layers.{layer_index}."

    if suffix == "mhc_norm.weight":
        return target_prefix + suffix
    if suffix in _MHC_NAMES:
        return target_prefix + _MHC_NAMES[suffix]
    if suffix.startswith("attention."):
        attention_suffix = suffix[len("attention.") :]
        if attention_suffix == "sinks":
            attention_suffix = "core_attention.softmax_offset"
        return target_prefix + "self_attention." + attention_suffix
    if not suffix.startswith("mlp."):
        return None

    mlp_suffix = suffix[len("mlp.") :]
    if mlp_suffix in _MOE_NAMES:
        return target_prefix + "mlp." + _MOE_NAMES[mlp_suffix]
    if mlp_suffix.startswith("moe_mlp."):
        return None
    return target_prefix + "mlp." + mlp_suffix


def _target_value(source: Tensor, target: Tensor, source_key: str, target_key: str) -> Tensor:
    value = source
    if source_key.endswith(".attention.sinks"):
        if source.ndim != 2 or source.shape[0] != 1:
            raise ValueError(f"{source_key} must have shape [1, num_attention_heads]")
        value = source.squeeze(0)
    if tuple(value.shape) != tuple(target.shape):
        raise ValueError(
            f"shape mismatch for {source_key} -> {target_key}: "
            f"{tuple(value.shape)} != {tuple(target.shape)}"
        )
    if value.is_floating_point() and target.is_floating_point() and value.dtype != target.dtype:
        value = value.to(dtype=target.dtype)
    return value


def _local_expert_indices(model: nn.Module) -> dict[int, Sequence[int]]:
    decoder = getattr(model, "decoder", None)
    layers = getattr(decoder, "layers", None)
    if layers is None:
        raise TypeError("model must expose decoder.layers like Magi2Model")
    result: dict[int, Sequence[int]] = {}
    for layer_index, layer in enumerate(layers):
        routed = getattr(getattr(layer, "mlp", None), "routed", None)
        if routed is not None:
            result[layer_index] = tuple(int(index) for index in routed.local_expert_indices)
    return result


def _conversion_report(
    source_keys: set[str], target_keys: set[str], model: nn.Module
) -> Magi2CheckpointConversionReport:
    consumed: set[str] = set()
    populated: set[str] = set()
    for source_key in source_keys:
        if _is_official_expert_weight(source_key):
            continue
        target_key = magi2_official_to_mcore_key(source_key)
        if target_key is not None and target_key in target_keys:
            consumed.add(source_key)
            populated.add(target_key)

    for layer_index, global_expert_indices in _local_expert_indices(model).items():
        source_prefix = f"block.layers.{layer_index}.mlp.moe_mlp."
        expert_sources = {source_prefix + name for name in _EXPERT_WEIGHT_NAMES}
        if expert_sources <= source_keys:
            consumed.update(expert_sources)
            target_prefix = f"decoder.layers.{layer_index}.mlp.routed.experts."
            for local_index in range(len(global_expert_indices)):
                populated.add(target_prefix + f"linear_fc1.weight{local_index}")
                populated.add(target_prefix + f"linear_fc2.weight{local_index}")

    preserved = {
        key for key in target_keys if key.endswith("._extra_state") or "._extra_state" in key
    }
    return Magi2CheckpointConversionReport(
        consumed_source_keys=tuple(sorted(consumed)),
        populated_target_keys=tuple(sorted(populated)),
        preserved_target_keys=tuple(sorted(preserved)),
        missing_target_keys=tuple(sorted(target_keys - populated - preserved)),
        unexpected_source_keys=tuple(sorted(source_keys - consumed)),
    )


def convert_magi2_official_state_dict(
    official_state_dict: Mapping[str, Tensor], model: nn.Module, *, strict: bool = True
) -> tuple[dict[str, Any], Magi2CheckpointConversionReport]:
    """Convert official public tensors to a loadable native-MCore state dict.

    Official routed matrices use ``[global_expert, input, output]``. MCore TE
    Grouped GEMM owns one local ``[output, input]`` parameter per expert. Gate
    and up matrices are transposed and concatenated into FC1; down matrices are
    transposed into FC2. Global expert IDs remain head-major.

    Args:
        official_state_dict: State dict emitted by the public SandAI model.
        model: Target native :class:`Magi2Model`, already constructed with its
            desired expert-parallel process groups.
        strict: Raise when a public key or a non-runtime MCore key is unaccounted.

    Returns:
        A loadable MCore state dict and a complete conversion audit report.
    """
    normalized_source: dict[str, Tensor] = {}
    for original_key, value in official_state_dict.items():
        key = _without_wrapper_prefix(original_key)
        if key in normalized_source:
            raise ValueError(f"duplicate official key after prefix normalization: {key}")
        normalized_source[key] = value

    target_state = model.state_dict()
    converted: dict[str, Any] = {}
    consumed: set[str] = set()
    populated: set[str] = set()

    for source_key, source_value in normalized_source.items():
        if _is_official_expert_weight(source_key):
            continue
        target_key = magi2_official_to_mcore_key(source_key)
        if target_key is None or target_key not in target_state:
            continue
        converted[target_key] = _target_value(
            source_value, target_state[target_key], source_key, target_key
        )
        consumed.add(source_key)
        populated.add(target_key)

    for layer_index, global_expert_indices in _local_expert_indices(model).items():
        source_prefix = f"block.layers.{layer_index}.mlp.moe_mlp."
        expert_source_keys = {name: source_prefix + name for name in _EXPERT_WEIGHT_NAMES}
        present = {name: key in normalized_source for name, key in expert_source_keys.items()}
        if not any(present.values()):
            continue
        if not all(present.values()):
            absent = sorted(name for name, is_present in present.items() if not is_present)
            raise KeyError(f"layer {layer_index} is missing routed expert tensors: {absent}")

        gate = normalized_source[expert_source_keys["W_gate"]]
        up = normalized_source[expert_source_keys["W_up"]]
        down = normalized_source[expert_source_keys["W_down"]]
        if gate.ndim != 3 or up.shape != gate.shape:
            raise ValueError(
                "official W_gate and W_up must have matching [experts, input, output] shapes"
            )
        if down.ndim != 3 or down.shape[0] != gate.shape[0]:
            raise ValueError("official W_down must have shape [experts, output, input]")

        target_prefix = f"decoder.layers.{layer_index}.mlp.routed.experts."
        for local_index, global_index in enumerate(global_expert_indices):
            if global_index >= gate.shape[0]:
                raise ValueError(
                    f"global expert {global_index} is outside official layer {layer_index} weights"
                )
            fc1_key = target_prefix + f"linear_fc1.weight{local_index}"
            fc2_key = target_prefix + f"linear_fc2.weight{local_index}"
            if fc1_key not in target_state or fc2_key not in target_state:
                raise KeyError(
                    f"target model is missing local expert {local_index} in layer {layer_index}"
                )
            fc1 = torch.cat(
                (gate[global_index].transpose(0, 1), up[global_index].transpose(0, 1)), dim=0
            )
            fc2 = down[global_index].transpose(0, 1)
            converted[fc1_key] = _target_value(
                fc1, target_state[fc1_key], expert_source_keys["W_gate"], fc1_key
            )
            converted[fc2_key] = _target_value(
                fc2, target_state[fc2_key], expert_source_keys["W_down"], fc2_key
            )
            populated.update((fc1_key, fc2_key))
        consumed.update(expert_source_keys.values())

    preserved = {
        key for key in target_state if key.endswith("._extra_state") or "._extra_state" in key
    }
    for key in preserved:
        converted[key] = target_state[key]

    missing = set(target_state) - populated - preserved
    unexpected = set(normalized_source) - consumed
    report = Magi2CheckpointConversionReport(
        consumed_source_keys=tuple(sorted(consumed)),
        populated_target_keys=tuple(sorted(populated)),
        preserved_target_keys=tuple(sorted(preserved)),
        missing_target_keys=tuple(sorted(missing)),
        unexpected_source_keys=tuple(sorted(unexpected)),
    )
    if strict and not report.is_strictly_complete:
        raise RuntimeError(
            "incomplete MAGI-2 checkpoint conversion: "
            f"missing target keys={report.missing_target_keys}, "
            f"unexpected source keys={report.unexpected_source_keys}"
        )
    return converted, report


def _apply_runtime_router_bias(
    model: nn.Module, router_bias_source: Literal["ema", "main"]
) -> None:
    """Select the runtime router bias paired with the public checkpoint weights."""
    if router_bias_source not in ("ema", "main"):
        raise ValueError("router_bias_source must be 'ema' or 'main'")
    if router_bias_source == "main":
        return
    buffers = dict(model.named_buffers())
    with torch.no_grad():
        for key, ema_bias in buffers.items():
            if not key.endswith(".routed.router.expert_bias_ema"):
                continue
            main_key = key.removesuffix("_ema")
            if main_key not in buffers:
                raise KeyError(f"missing runtime router bias paired with {key}")
            buffers[main_key].copy_(ema_bias)


def load_magi2_official_state_dict(
    model: nn.Module,
    official_state_dict: Mapping[str, Tensor],
    *,
    strict: bool = True,
    router_bias_source: Literal["ema", "main"] = "ema",
) -> Magi2CheckpointConversionReport:
    """Convert and strictly load public MAGI-2 tensors into a native MCore model."""
    converted, report = convert_magi2_official_state_dict(official_state_dict, model, strict=strict)
    incompatible = model.load_state_dict(converted, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "converted MAGI-2 state dict failed strict load: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    _apply_runtime_router_bias(model, router_bias_source)
    return report


def _safetensors_weight_map(checkpoint_dir: str | Path) -> dict[str, Path]:
    """Read a HuggingFace-style safetensors index without loading tensor data."""
    checkpoint_path = Path(checkpoint_dir)
    index_path = checkpoint_path / "model.safetensors.index.json"
    single_path = checkpoint_path / "model.safetensors"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        return {
            key: checkpoint_path / relative_path
            for key, relative_path in index["weight_map"].items()
        }
    if not single_path.is_file():
        raise FileNotFoundError(
            f"no model.safetensors.index.json or model.safetensors under {checkpoint_path}"
        )
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ImportError("loading official MAGI-2 weights requires safetensors") from error
    with safe_open(single_path, framework="pt", device="cpu") as handle:
        return {key: single_path for key in handle.keys()}


def load_magi2_official_safetensors(
    model: nn.Module,
    checkpoint_dir: str | Path,
    *,
    strict: bool = True,
    router_bias_source: Literal["ema", "main"] = "ema",
) -> Magi2CheckpointConversionReport:
    """Stream an official safetensors directory into a native MCore model.

    Routed expert tensors are sliced by the target rank's global expert IDs,
    so peak host memory is bounded by individual non-expert tensors rather than
    the full public checkpoint.
    """
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ImportError("loading official MAGI-2 weights requires safetensors") from error

    raw_weight_map = _safetensors_weight_map(checkpoint_dir)
    weight_map: dict[str, tuple[str, Path]] = {}
    for raw_key, path in raw_weight_map.items():
        key = _without_wrapper_prefix(raw_key)
        if key in weight_map:
            raise ValueError(f"duplicate official key after prefix normalization: {key}")
        weight_map[key] = (raw_key, path)

    target_state = model.state_dict(keep_vars=True)
    report = _conversion_report(set(weight_map), set(target_state), model)
    if strict and not report.is_strictly_complete:
        raise RuntimeError(
            "incomplete MAGI-2 safetensors conversion: "
            f"missing target keys={report.missing_target_keys}, "
            f"unexpected source keys={report.unexpected_source_keys}"
        )

    with ExitStack() as stack:
        handles = {
            path: stack.enter_context(safe_open(path, framework="pt", device="cpu"))
            for path in set(raw_weight_map.values())
        }

        def read_tensor(source_key: str) -> Tensor:
            raw_key, path = weight_map[source_key]
            return handles[path].get_tensor(raw_key)

        def read_experts(source_key: str, global_indices: Sequence[int]) -> Tensor:
            raw_key, path = weight_map[source_key]
            view = handles[path].get_slice(raw_key)
            start = global_indices[0]
            if tuple(global_indices) == tuple(range(start, start + len(global_indices))):
                return view[start : start + len(global_indices)]
            return torch.stack([view[index] for index in global_indices])

        with torch.no_grad():
            for source_key in report.consumed_source_keys:
                if _is_official_expert_weight(source_key):
                    continue
                target_key = magi2_official_to_mcore_key(source_key)
                if target_key is None:
                    raise AssertionError(f"missing direct mapping for {source_key}")
                value = _target_value(
                    read_tensor(source_key), target_state[target_key], source_key, target_key
                )
                target_state[target_key].copy_(value)

            for layer_index, global_expert_indices in _local_expert_indices(model).items():
                source_prefix = f"block.layers.{layer_index}.mlp.moe_mlp."
                gate_key = source_prefix + "W_gate"
                up_key = source_prefix + "W_up"
                down_key = source_prefix + "W_down"
                if gate_key not in weight_map:
                    continue
                target_prefix = f"decoder.layers.{layer_index}.mlp.routed.experts."
                for local_start in range(0, len(global_expert_indices), _EXPERT_READ_CHUNK_SIZE):
                    global_chunk = global_expert_indices[
                        local_start : local_start + _EXPERT_READ_CHUNK_SIZE
                    ]
                    gate_chunk = read_experts(gate_key, global_chunk)
                    up_chunk = read_experts(up_key, global_chunk)
                    down_chunk = read_experts(down_key, global_chunk)
                    for chunk_index, (gate, up, down) in enumerate(
                        zip(gate_chunk, up_chunk, down_chunk)
                    ):
                        local_index = local_start + chunk_index
                        fc1_key = target_prefix + f"linear_fc1.weight{local_index}"
                        fc2_key = target_prefix + f"linear_fc2.weight{local_index}"
                        fc1 = torch.cat((gate.T, up.T))
                        fc2 = down.T
                        target_state[fc1_key].copy_(
                            _target_value(fc1, target_state[fc1_key], gate_key, fc1_key)
                        )
                        target_state[fc2_key].copy_(
                            _target_value(fc2, target_state[fc2_key], down_key, fc2_key)
                        )
    _apply_runtime_router_bias(model, router_bias_source)
    return report


__all__ = [
    "Magi2CheckpointConversionReport",
    "convert_magi2_official_state_dict",
    "load_magi2_official_state_dict",
    "load_magi2_official_safetensors",
    "magi2_official_to_mcore_key",
]
