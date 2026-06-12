# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M329: Megatron 9026b86d8 — Initialization fixes: allowing simple case like
#        pytest pass, also making apex optional
# Source: megatron/initialize.py (NVIDIA/Megatron-LM commit 9026b86d8)
# Author: Boris Fomitchev <bfomitchev@nvidia.com>  Date: 2020-07-22
#
# Mapping: megatron/initialize.py → deepspeed/compile/megatron_initialize.py
#          (project convention: megatron top-level init → deepspeed/compile/)
#
# Changes ported:
#   1. initialize_megatron(): add early-return guard via
#      mpu.model_parallel_is_initialized() — prevents double-init when
#      pytest (or any caller) invokes initialize_megatron with the same
#      args twice in the same process.
#
# 20% adaptation: uses deepspeed.comm / mpu_initialize instead of
# megatron.mpu directly; keep guard semantics identical; adds print marker.
# ---------------------------------------------------------------------------

print('[M329]')

import deepspeed.comm as dist
from deepspeed.compile.mpu_initialize import (
    get_model_parallel_rank,
    get_model_parallel_world_size,
)


def model_parallel_is_initialized():
    """Check whether model and data parallel groups are initialized.

    Megatron 9026b86d8 initialize.py — WAR to allow simple cases like
    pytest calling initialize_megatron with the same args twice.
    Returns True once the model-parallel comm group has been set up.
    """
    if not dist.is_initialized():
        return False
    try:
        # A world_size > 0 and a valid rank both indicate that the
        # model-parallel group has been initialised at least once.
        world_size = get_model_parallel_world_size()
        rank = get_model_parallel_rank()
        return world_size is not None and rank is not None
    except Exception:
        return False


def initialize_megatron(extra_args_provider=None,
                        args_defaults=None,
                        ignore_unknown_args=False,
                        allow_no_cuda=False):
    """Set global variables, initialize distributed, and set random seeds.

    Megatron 9026b86d8 initialize.py — idempotent init: if the
    model-parallel group is already set up (e.g. a second pytest call)
    we return immediately without re-running expensive distributed setup.

    `allow_no_cuda` should not be set unless using megatron/deepspeed for
    CPU-only data processing. In general this arg should not be set unless
    you know what you are doing.
    """
    if args_defaults is None:
        args_defaults = {}

    if not allow_no_cuda:
        import torch
        # Make sure cuda is available.
        assert torch.cuda.is_available(), 'DeepSpeed/Megatron requires CUDA.'

    # This is temporary WAR to make simple case like pytest calling with
    # same args twice.  Need to implement clean factory init.
    if model_parallel_is_initialized():
        return

    # Pytorch distributed — delegate to deepspeed.comm initialisation.
    if not dist.is_initialized():
        dist.init_distributed()

    print('[M329] initialize_megatron: distributed initialised, '
          f'world_size={get_model_parallel_world_size()}, '
          f'rank={get_model_parallel_rank()}')
