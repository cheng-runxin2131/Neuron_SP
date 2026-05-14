import logging

import torch
from torch.fx import GraphModule
from .passes.sp_compile import apply_autosp
from .custom_ops.sp_dp_registry import extract_mesh_size
from .custom_ops.sp_compat import _check_autosp_compatibility
from .custom_ops import all_to_all as _force_register_a2a
from .passes.long_context_checkpointing import register_long_context_checkpointing

logger = logging.getLogger(__name__)


def _resolve_head_counts(param_dict):
    n_heads = param_dict.get('n_heads', param_dict.get('num_attention_heads', 0))
    n_kv_heads = param_dict.get('num_key_value_heads',
                                 param_dict.get('n_kv_heads', n_heads))
    min_heads = min(n_heads, n_kv_heads) if n_kv_heads > 0 else n_heads
    return n_heads, n_kv_heads, min_heads


def _auto_downgrade_sp_size(sp_size, min_heads, world_size):
    for cand in range(sp_size - 1, 0, -1):
        if min_heads % cand == 0 and world_size % cand == 0:
            return cand, world_size // cand
    return 1, world_size


def _parse_desloc_config(param_dict):
    cfg = param_dict.get('desloc', {})
    return {
        'enabled': cfg.get('enabled', False),
        'Kx': cfg.get('Kx', 1),
    }


def _parse_hetero_config(param_dict):
    cfg = param_dict.get('hetero_mesh', {})
    return {
        'strategy': cfg.get('strategy', 'contiguous'),
    }


def _parse_histogram_config(param_dict):
    cfg = param_dict.get('sp_histogram', {})
    return {
        'enabled': cfg.get('enabled', False),
        'num_bins': cfg.get('num_bins', 256),
    }


def init_autosp(config):
    _check_autosp_compatibility()

    from .custom_ops.sp_dp_registry import cleanup_sp_groups, is_setup, pending_handle_count
    if is_setup():
        pending = pending_handle_count()
        if pending > 0:
            logger.warning(
                f"[AutoSP] Reinitializing with {pending} pending A2A handles. "
                f"Fencing before cleanup to prevent NCCL errors.")
        cleanup_sp_groups()

    sp_size, dp_size = extract_mesh_size(config._param_dict)
    register_long_context_checkpointing()

    import deepspeed.comm as dist
    n_heads, n_kv_heads, min_heads = _resolve_head_counts(config._param_dict)

    if min_heads > 0 and sp_size > 1 and min_heads % sp_size != 0:
        old_sp = sp_size
        sp_size, dp_size = _auto_downgrade_sp_size(sp_size, min_heads, dist.get_world_size())
        logger.warning(
            f"[AutoSP] n_heads={n_heads}, n_kv_heads={n_kv_heads} "
            f"(min={min_heads}) not divisible by sp_size={old_sp}. "
            f"Reduced to sp_size={sp_size}, dp_size={dp_size}.")
        if sp_size <= 1:
            raise RuntimeError(
                f"[AutoSP] Cannot find valid sp_size for n_kv_heads={n_kv_heads}. "
                f"All candidates down to 2 fail divisibility against "
                f"min(n_heads,n_kv_heads)={min_heads} and world_size={dist.get_world_size()}. "
                f"Set sequence_parallel_size=1 explicitly to disable SP.")

    config._param_dict['_effective_sp_size'] = sp_size
    config._param_dict['_effective_dp_size'] = dp_size

    desloc_cfg = _parse_desloc_config(config._param_dict)
    hetero_cfg = _parse_hetero_config(config._param_dict)
    histogram_cfg = _parse_histogram_config(config._param_dict)

    if hetero_cfg['strategy'] != 'contiguous':
        from .custom_ops.hetero_mesh import populate_hetero_registry
        populate_hetero_registry(sp_size, dp_size, strategy=hetero_cfg['strategy'])
        from .custom_ops.sp_dp_registry import mark_heterogeneous
        mark_heterogeneous(True)

    if histogram_cfg['enabled']:
        from .custom_ops.sp_histogram import get_histogram_kernel
        get_histogram_kernel(num_bins=histogram_cfg['num_bins'])

    logger.info(
        f"[AutoSP] sp={sp_size} dp={dp_size} desloc={desloc_cfg['enabled']} "
        f"Kx={desloc_cfg['Kx']} mesh_strategy={hetero_cfg['strategy']} "
        f"histogram={histogram_cfg['enabled']}")

    def backend_fn(gm: GraphModule, real_inputs):
        apply_autosp(gm, real_inputs, debug=False, sp_size=sp_size, dp_size=dp_size)
        return torch._inductor.compile(gm, real_inputs)

    return backend_fn
