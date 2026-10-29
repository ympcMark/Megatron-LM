# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared helpers for Mixture-of-Transformers branch execution."""

from typing import Optional, Tuple

from torch import Tensor, nn


def get_mot_local_seqlens(packed_seq_params, config) -> Tuple[int, int]:
    """Return the UND/GEN lengths held by one sequence-parallel rank."""
    tp_size = config.tensor_model_parallel_size if config.sequence_parallel else 1
    und_seqlen = packed_seq_params.padded_und_seqlen
    gen_seqlen = packed_seq_params.padded_gen_seqlen
    if und_seqlen % tp_size or gen_seqlen % tp_size:
        raise ValueError(
            "BAGEL MoT branch lengths must be divisible by the tensor-parallel size "
            "when sequence parallelism is enabled"
        )
    return und_seqlen // tp_size, gen_seqlen // tp_size


def attach_zero_grad_dependency(output: Tensor, *modules: Optional[nn.Module]) -> Tensor:
    """Keep skipped branch parameters in autograd without executing the branch.

    Empty MoT branches must not enter modules that may reject zero-token inputs or
    launch unnecessary collectives.  Distributed wrappers still need every trainable
    parameter to participate in backward, so attach a zero-sized view of each skipped
    parameter to the branch output.  This produces explicit zero (or empty-shard)
    gradients without reading the parameter values.
    """
    zero = None
    seen_parameters = set()
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            parameter_id = id(parameter)
            if not parameter.requires_grad or parameter_id in seen_parameters:
                continue
            seen_parameters.add(parameter_id)
            dependency = (
                parameter.reshape(-1)[:0].sum().to(device=output.device, dtype=output.dtype)
            )
            zero = dependency if zero is None else zero + dependency

    return output if zero is None else output + zero
