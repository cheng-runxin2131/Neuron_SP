# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M573: Megatron 816fb8902 — fixed a minor bug
# Source: megatron/training.py (NVIDIA/Megatron-LM commit 816fb8902)
# Author: Mostofa Patwary <mostofa.patwary@gmail.com>  Date: 2021-02-17
#
# Mapping: megatron/training.py → deepspeed/compile/megatron_training.py
#          (project convention: megatron training helpers → deepspeed/compile/)
#
# Changes ported:
#   1. setup_model_and_optimizer(): fix assertion guard — change
#      `if len(model) == 1` to `if len(model) > 1` so that multi-model
#      pipeline-parallel configurations correctly require local DDP,
#      rather than accidentally enforcing it only for single-model cases.
#
# 20% adaptation: uses deepspeed.comm / mpu_initialize instead of
# megatron.mpu directly; DDP_impl check mirrors DeepSpeed engine
# conventions; adds print marker.
# ---------------------------------------------------------------------------
print('[M573]')

import deepspeed.comm as dist
from deepspeed.compile.mpu_initialize import get_pipeline_model_parallel_world_size


def _check_ddp_impl_constraints(model, args):
    """Validate DDP implementation constraints for the given model list.

    Megatron 816fb8902 training.py — local DDP is required whenever more
    than one micro-batch is used, when the model list contains multiple
    pipeline stages (len > 1), or when pipeline model parallelism spans
    more than one rank.  The original bug had `len(model) == 1` which only
    enforced local DDP for single-model configs, the opposite of correct.
    """
    ddp_impl = getattr(args, 'DDP_impl', 'local')

    num_microbatches = getattr(args, 'gradient_accumulation_steps', 1)
    if num_microbatches > 1:
        assert ddp_impl == 'local', \
            f'Local DDP required with {num_microbatches} micro-batches; got DDP_impl={ddp_impl!r}'

    # Bug fix (816fb8902): was `== 1`, must be `> 1` — pipeline parallelism
    # splits the model across multiple stages, each element of `model` is one
    # stage; all stages must use local DDP for correctness.
    if len(model) > 1:
        assert ddp_impl == 'local', \
            f'Local DDP required for pipeline-parallel model (len={len(model)}); got {ddp_impl!r}'

    pp_world_size = get_pipeline_model_parallel_world_size()
    if pp_world_size is not None and pp_world_size > 1:
        assert ddp_impl == 'local', \
            f'Local DDP required for pipeline_model_parallel_world_size={pp_world_size}; got {ddp_impl!r}'
