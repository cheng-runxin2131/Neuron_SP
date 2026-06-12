# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M512: Megatron 78066ab08 — Fixing merge_mp_partitions
# Source: megatron/checkpointing.py (NVIDIA/Megatron-LM commit 78066ab08)
# Author: Jared Casper <jcasper@nvidia.com>  Date: 2021-01-21
#
# Mapping: megatron/checkpointing.py → deepspeed/compile/megatron_checkpointing.py
#          (project convention: megatron top-level → deepspeed/compile/)
#
# Changes ported from checkpointing.py:
#
#   1. set_checkpoint_version(value):
#        BEFORE: assert _CHECKPOINT_VERSION is None, "checkpoint version already set"
#        AFTER:  if _CHECKPOINT_VERSION is not None:
#                    assert _CHECKPOINT_VERSION == value, \
#                        "checkpoint versions do not match"
#      Allows multiple callers to set the same version without error (needed
#      by merge_mp_partitions which loads N partition checkpoints sequentially).
#
#   2. save_checkpoint():
#        if torch.distributed.get_rank() == 0:  →  print_rank_0(...)
#        if mpu.get_data_parallel_rank() == 0:  →  if not dist.is_initialized()
#                                                      or mpu.get_data_parallel_rank() == 0:
#        torch.distributed.barrier()            →  if dist.is_initialized():
#                                                      dist.barrier()
#        (Allows save_checkpoint to be called without distributed init, as
#        required by the merge_mp_partitions main() script.)
#
#   3. load_checkpoint():
#        Same dist.is_initialized() guards for barrier() calls.
#        Converts remaining torch.distributed.get_rank() == 0 prints to
#        print_rank_0().
#
# 20% adaptation: standalone module with simplified state; uses Python
# logging-compatible print_rank_0; does not depend on full Megatron global
# state machinery (get_args / mpu).  Adds print('[M512]') marker.
# ---------------------------------------------------------------------------

print('[M512]')

import torch

_CHECKPOINT_VERSION = None


def print_rank_0(message):
    """Print only on rank 0 (or when distributed is not initialised)."""
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(message, flush=True)


def set_checkpoint_version(value):
    """Set global checkpoint version; allow re-setting to the same value.

    Megatron 78066ab08 checkpointing.py set_checkpoint_version():
      BEFORE: assert _CHECKPOINT_VERSION is None, 'checkpoint version already set'
      AFTER:  if _CHECKPOINT_VERSION is not None:
                  assert _CHECKPOINT_VERSION == value, \
                      'checkpoint versions do not match'

    The relaxed check enables merge_mp_partitions to call load_checkpoint()
    for each rank partition without hitting the 'already set' assertion on
    the second and subsequent partitions.
    """
    global _CHECKPOINT_VERSION
    if _CHECKPOINT_VERSION is not None:
        assert _CHECKPOINT_VERSION == value, \
            f'checkpoint versions do not match: {_CHECKPOINT_VERSION} vs {value}'
    _CHECKPOINT_VERSION = value
    print(f'[M512] set_checkpoint_version: version={value}')


def get_checkpoint_version():
    """Return the global checkpoint version."""
    return _CHECKPOINT_VERSION


def _barrier_if_initialized():
    """Call torch.distributed.barrier() only when distributed is initialized.

    Megatron 78066ab08: guards every barrier() call with is_initialized() so
    that utilities like merge_mp_partitions can run without a process group.
    """
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def save_checkpoint_safe(iteration, save_path, state_dict_fn,
                          get_checkpoint_name_fn,
                          ensure_directory_exists_fn):
    """Save a checkpoint guarded for non-distributed (single-process) use.

    Megatron 78066ab08 checkpointing.py save_checkpoint() key changes:
      • print_rank_0() instead of if get_rank()==0: print(...)
      • if not dist.is_initialized() or mpu.get_data_parallel_rank() == 0:
        (replaces bare  if mpu.get_data_parallel_rank() == 0:)
      • barrier guards: if dist.is_initialized(): dist.barrier()

    Args:
        iteration: training iteration number.
        save_path: directory to save checkpoint.
        state_dict_fn: callable() → dict to save.
        get_checkpoint_name_fn: callable(save_path, iteration) → filename.
        ensure_directory_exists_fn: callable(filename).
    """
    print_rank_0(f'saving checkpoint at iteration {iteration:7d} to {save_path}')

    # Only data-parallel rank 0 saves (or if not distributed at all).
    save_this_rank = True
    if torch.distributed.is_initialized():
        try:
            from .mpu_initialize import get_data_parallel_rank
            save_this_rank = (get_data_parallel_rank() == 0)
        except (ImportError, AttributeError):
            save_this_rank = (torch.distributed.get_rank() == 0)

    if save_this_rank:
        checkpoint_name = get_checkpoint_name_fn(save_path, iteration)
        ensure_directory_exists_fn(checkpoint_name)
        state_dict = state_dict_fn()
        state_dict['iteration'] = iteration
        torch.save(state_dict, checkpoint_name)

    # Wait so everyone is done (necessary).
    _barrier_if_initialized()

    print_rank_0(f'  successfully saved checkpoint at iteration {iteration:7d} '
                 f'to {save_path}')

    # Update latest iteration tracker.
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        tracker_filename = save_path.rstrip('/') + '/latest_checkpointed_iteration.txt'
        with open(tracker_filename, 'w') as f:
            f.write(str(iteration))

    # Wait so everyone is done (not strictly necessary but matches Megatron).
    _barrier_if_initialized()


def load_checkpoint_safe(load_path, checkpoint_name,
                          model_load_fn,
                          set_checkpoint_version_fn=None):
    """Load a checkpoint guarded for non-distributed (single-process) use.

    Megatron 78066ab08 checkpointing.py load_checkpoint() key changes:
      • print_rank_0() instead of if get_rank()==0: print(...)
      • if dist.is_initialized(): dist.barrier()  (replaces unconditional barrier)

    Args:
        load_path: directory from which the checkpoint was loaded.
        checkpoint_name: full path to the checkpoint file.
        model_load_fn: callable(state_dict) — loads model state.
        set_checkpoint_version_fn: optional callable(version) — registers
            checkpoint version; defaults to set_checkpoint_version().
    """
    if set_checkpoint_version_fn is None:
        set_checkpoint_version_fn = set_checkpoint_version

    print_rank_0(f' loading checkpoint from {load_path}')

    try:
        state_dict = torch.load(checkpoint_name, map_location='cpu')
    except Exception as e:
        print(f'[M512] load_checkpoint_safe: failed to load {checkpoint_name}: {e}')
        raise

    # Set checkpoint version.
    if 'checkpoint_version' in state_dict:
        set_checkpoint_version_fn(state_dict['checkpoint_version'])

    # Load model.
    model_load_fn(state_dict)

    # Some utilities want to load without distributed being initialized.
    _barrier_if_initialized()

    print_rank_0(f'  successfully loaded checkpoint from {load_path}')

    return state_dict.get('iteration', 0)
