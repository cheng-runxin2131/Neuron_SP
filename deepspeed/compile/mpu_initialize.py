# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M54: Megatron 57c2060fe — Model parallel merger
# Source: megatron/mpu/initialize.py (NVIDIA/Megatron-LM commit 57c2060fe)
# Author: Mohammad Shoeybi <mshoeybi@nvidia.com>  Date: 2020-02-10
#
# Mapping: mpu/* → deepspeed/compile/  (project convention)
#
# Changes ported:
#   1. Add module-level _MPU_WORLD_SIZE / _MPU_RANK override globals.
#   2. Add set_model_parallel_world_size() setter.
#   3. Modify get_model_parallel_world_size() to honour override first.
#   4. Add set_model_parallel_rank() setter.
#   5. Modify get_model_parallel_rank() to honour override first.
#
# These overrides allow checkpoint-merge tooling to pretend the process
# group has world_size=1 / rank=0 without rebuilding NCCL groups — the
# primary use case in merge_mp_partitions.py.
#
# 20% adaptation: uses deepspeed.comm instead of torch.distributed directly;
# falls back to 1/0 when not initialised (test-safe); adds print markers.
# ---------------------------------------------------------------------------
# M345: Megatron 5c04ceb31 — Implementing lazy parallel initialization
# Source: megatron/mpu/__init__.py + megatron/mpu/initialize.py
#         (NVIDIA/Megatron-LM commit 5c04ceb31)
# Author: Boris Fomitchev <bfomitchev@nvidia.com>  Date: 2020-08-05
#
# Mapping: megatron/mpu/initialize.py → deepspeed/compile/mpu_initialize.py
#
# Changes ported:
#   1. megatron/mpu/__init__.py: export set_model_parallel_rank and
#      set_model_parallel_world_size (already present from M54 in DS).
#   2. megatron/mpu/initialize.py: remove set_model_parallel_group() and
#      set_data_parallel_group() helpers — these were not present in the DS
#      mapping (no _MODEL_PARALLEL_GROUP / _DATA_PARALLEL_GROUP globals here),
#      so no deletion needed; the export additions are the meaningful change.
#
# 20% adaptation: set_model_parallel_rank / set_model_parallel_world_size
# already exported; this entry records the upstream mpu/__init__.py change.
# ---------------------------------------------------------------------------

import deepspeed.comm as dist

print('[M345]')

# These values enable us to change the mpu sizes on the fly.
_MPU_WORLD_SIZE = None
_MPU_RANK = None


def set_model_parallel_world_size(world_size):
    """Set the model parallel size.

    Megatron 57c2060fe mpu/initialize.py — allows callers (e.g. checkpoint
    merge tools) to override the distributed world size without touching
    process groups.
    """
    global _MPU_WORLD_SIZE
    _MPU_WORLD_SIZE = world_size
    print(f'[M54-MPU] set_model_parallel_world_size({world_size})')


def get_model_parallel_world_size():
    """Return world size for the model parallel group.

    Megatron 57c2060fe mpu/initialize.py — returns override if set,
    otherwise queries the distributed group.
    """
    global _MPU_WORLD_SIZE
    if _MPU_WORLD_SIZE is not None:
        return _MPU_WORLD_SIZE
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def set_model_parallel_rank(rank):
    """Set model parallel rank.

    Megatron 57c2060fe mpu/initialize.py — allows callers to override
    the distributed rank without touching process groups.
    """
    global _MPU_RANK
    _MPU_RANK = rank
    print(f'[M54-MPU] set_model_parallel_rank({rank})')


def get_model_parallel_rank():
    """Return my rank for the model parallel group.

    Megatron 57c2060fe mpu/initialize.py — returns override if set,
    otherwise queries the distributed group.
    """
    global _MPU_RANK
    if _MPU_RANK is not None:
        return _MPU_RANK
    if dist.is_initialized():
        return dist.get_rank()
    return 0
