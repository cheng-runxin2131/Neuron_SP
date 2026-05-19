import inspect
import logging
import threading
import textwrap
import torch._functorch.partitioners as _partitioners

logger = logging.getLogger(__name__)

_CUSTOM_SHOULD_BAN = (
    'def should_ban_recomputation(node):\n'
    '    if node.op != "call_function":\n'
    '        return None\n'
    '    if node.target == operator.getitem:\n'
    '        return None\n'
    '    if node.meta.get("recompute", None) == CheckpointPolicy.MUST_SAVE:\n'
    '        return "autosp: MUST_SAVE policy"\n'
    '    if config.recompute_views and op_types.is_view(node):\n'
    '        return None\n'
    '    if node.target in [aten.lift_fresh_copy.default, aten.lift_fresh.default]:\n'
    '        return None\n'
    '\n'
    '    must_save_set = [\n'
    '        aten.convolution,\n'
    '        aten.convolution_backward,\n'
    '        aten._scaled_dot_product_flash_attention,\n'
    '        aten._scaled_dot_product_efficient_attention,\n'
    '        aten._flash_attention_forward,\n'
    '        aten._efficient_attention_forward,\n'
    '        aten.upsample_bilinear2d,\n'
    '        aten.native_dropout,\n'
    '        aten.rand_like,\n'
    '        aten.randn_like,\n'
    '    ]\n'
    '\n'
    '    if get_aten_target(node) in must_save_set:\n'
    '        return "autosp: attention/stochastic op"\n'
    '\n'
    '    if hasattr(node.target, \'__module__\') and \'autosp\' in str(node.target):\n'
    '        return "autosp: collective op"\n'
    '\n'
    '    def heuristic(node):\n'
    '        if "val" in node.meta:\n'
    '            if isinstance(node.meta["val"], torch.Tensor) and node.meta["val"].dim() >= 2:\n'
    '                return node.meta["val"].shape[1] >= 4096\n'
    '        return False\n'
    '\n'
    '    if min_cut_options.ban_if_not_in_allowlist:\n'
    '        if not op_types.is_recomputable(node):\n'
    '            return None\n'
    '\n'
    '    if min_cut_options.ban_if_materialized_backward and is_materialized_backwards(node):\n'
    '        if heuristic(node):\n'
    '            return None\n'
    '        return "autosp: materialized backward (small tensor)"\n'
    '\n'
    '    if node.dist_from_bw < 1000 and node.dist_from_bw > config.max_dist_from_bw:\n'
    '        return None\n'
    '\n'
    '    if min_cut_options.ban_if_reduction:\n'
    '        input_tensors_size = sum(\n'
    '            _size_of(i) for i in node.args if isinstance(i, fx.Node)\n'
    '        )\n'
    '        output_size = _size_of(node)\n'
    '        if output_size * 4 < input_tensors_size:\n'
    '            return "autosp: reduction op"\n'
    '    return None\n'
)

_NEEDLE = '    def should_ban_recomputation('

_ORIGINAL_SOLVE_MIN_CUT = _partitioners.solve_min_cut

_PATCH_LOCK = threading.Lock()
_PATCHED = False


def restore_default_checkpointing():
    global _PATCHED
    with _PATCH_LOCK:
        _partitioners.solve_min_cut = _ORIGINAL_SOLVE_MIN_CUT
        _PATCHED = False


def is_checkpointing_patched():
    return _PATCHED


def _locate_ban_function(lines):
    start = None
    end = None
    for i, line in enumerate(lines):
        if line.startswith(_NEEDLE) or line.lstrip().startswith('def should_ban_recomputation('):
            if start is None:
                start = i
        elif start is not None and end is None and line.startswith('    def '):
            end = i
    return start, end


def _splice_source(src):
    lines = src.split('\n')
    start, end = _locate_ban_function(lines)
    if start is None or end is None:
        return None
    replacement = textwrap.indent(_CUSTOM_SHOULD_BAN, '    ')
    return '\n'.join(lines[:start]) + '\n' + replacement + '\n'.join(lines[end:])


def register_long_context_checkpointing():
    global _PATCHED
    with _PATCH_LOCK:
        if _PATCHED:
            return

        try:
            src = inspect.getsource(_partitioners.solve_min_cut)
        except (OSError, TypeError):
            logger.warning("AutoSP: could not retrieve source for solve_min_cut; "
                           "selective activation checkpointing disabled.")
            return

        if 'def should_ban_recomputation(' not in src:
            logger.warning(
                f"AutoSP: PyTorch {__import__('torch').__version__} changed "
                f"solve_min_cut signature. Selective activation checkpointing disabled.")
            return

        new_src = _splice_source(src)
        if new_src is None:
            logger.warning(
                "AutoSP: solve_min_cut structure does not match expected pattern; "
                "selective activation checkpointing disabled.")
            return

        try:
            exec(new_src, _partitioners.__dict__)
            _PATCHED = True
        except Exception as e:
            _partitioners.solve_min_cut = _ORIGINAL_SOLVE_MIN_CUT
            logger.warning(f"AutoSP: failed to inject custom checkpointing policy: {e}. "
                           "Falling back to default PyTorch checkpointing.")
