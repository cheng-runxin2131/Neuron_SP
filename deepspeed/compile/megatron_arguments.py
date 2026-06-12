# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M343: Megatron 0403b8081 — added gpu initialization and option to avoid
#       master values
# Source: megatron/arguments.py (NVIDIA/Megatron-LM commit 0403b8081)
# Author: Mohammad Shoeybi <mshoeybi@nvidia.com>  Date: 2020-08-03
#
# Mapping: megatron/arguments.py → deepspeed/compile/megatron_arguments.py
#          (project convention: megatron top-level → deepspeed/compile/)
#
# Changes ported from arguments.py:
#   1. import torch added at module level.
#   2. parse_args(): after dynamic_loss_scale block, add params_dtype
#      assignment:
#        args.params_dtype = torch.float
#        if args.fp16: args.params_dtype = torch.half
#        if args.rank == 0: print('using {} for parameters ...')
#
# 20% adaptation: deepspeed uses ds_config.fp16.enabled rather than
# argparse args.fp16; _GLOBAL_ARGS singleton pattern used for get_args();
# adds print('[M343]') marker.
# ---------------------------------------------------------------------------

print('[M343]')

import torch

_GLOBAL_ARGS = None


def get_args():
    """Return the global args object.

    Megatron 0403b8081 arguments.py — global accessor used by mpu/layers.py
    to retrieve params_dtype without passing args through every call site.
    """
    return _GLOBAL_ARGS


def set_args(args):
    """Set the global args object.

    Called once during initialize_megatron so that downstream modules
    (mpu/layers.py) can access params_dtype via get_args().
    """
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args
    print(f'[M343] set_args: params_dtype={getattr(args, "params_dtype", None)}')


def set_params_dtype(args):
    """Set args.params_dtype based on fp16 flag.

    Megatron 0403b8081 arguments.py parse_args():
      args.params_dtype = torch.float
      if args.fp16:
          args.params_dtype = torch.half
      if args.rank == 0:
          print('using {} for parameters ...'.format(args.params_dtype), flush=True)

    Called after dynamic_loss_scale is resolved.
    """
    args.params_dtype = torch.float
    if getattr(args, 'fp16', False):
        args.params_dtype = torch.half
    rank = getattr(args, 'rank', 0)
    if rank == 0:
        print('using {} for parameters ...'.format(args.params_dtype), flush=True)
    print(f'[M343] set_params_dtype: params_dtype={args.params_dtype}')
    return args


# ---------------------------------------------------------------------------
# M409: Megatron 2d8de2968 — Throw exception if ring_exchange is not
#       available when pipeline_model_parallel_size > 1
# Source: megatron/arguments.py (NVIDIA/Megatron-LM commit 2d8de296890b9c01)
# Author: Deepak Narayanan <dnarayanan@nvidia.com>  Date: 2020-10-30
#
# Mapping: megatron/arguments.py → deepspeed/compile/megatron_arguments.py
#
# Change ported from arguments.py parse_args():
#   After setting args.pipeline_model_parallel_size = min(...), add:
#     if args.pipeline_model_parallel_size > 1:
#         if "ring_exchange" not in dir(torch.distributed):
#             raise Exception('PyTorch with torch.distributed.ring_exchange
#                             needed to run pipeline MP!')
#
# DeepSpeed adaptation: surfaced as validate_pipeline_mp_ring_exchange(args)
# so it can be called from engine init after pipeline size is resolved.
# ---------------------------------------------------------------------------

print('[M409]')


def validate_pipeline_mp_ring_exchange(args):
    """Raise if pipeline MP > 1 but torch.distributed.ring_exchange is absent.

    Megatron 2d8de2968 arguments.py parse_args():
      if args.pipeline_model_parallel_size > 1:
          if "ring_exchange" not in dir(torch.distributed):
              raise Exception('PyTorch with torch.distributed.ring_exchange
                              needed to run pipeline MP!')

    Called after pipeline_model_parallel_size is finalised.
    """
    pipeline_mp_size = getattr(args, 'pipeline_model_parallel_size', 1)
    if pipeline_mp_size > 1:
        if 'ring_exchange' not in dir(torch.distributed):
            raise Exception(
                'PyTorch with torch.distributed.ring_exchange needed '
                'to run pipeline MP!')
    print('[M409] validate_pipeline_mp_ring_exchange: '
          f'pipeline_mp_size={pipeline_mp_size}, '
          f'ring_exchange_available={"ring_exchange" in dir(torch.distributed)}')

# ---------------------------------------------------------------------------
# M451: Megatron 2623551d7 — Nicer error messages for deprecated arguments
# Source: megatron/arguments.py (NVIDIA/Megatron-LM commit 2623551d7712a3b9)
# Author: Jared Casper <jcasper@nvidia.com>  Date: 2020-12-10
#
# Mapping: megatron/arguments.py → deepspeed/compile/megatron_arguments.py
#
# Changes ported from arguments.py:
#   1. parse_args(): add deprecated-args assertion block after TP/PP print:
#        assert args.batch_size is None, '--batch-size argument is no longer
#            valid, use --micro-batch-size instead'
#        del args.batch_size
#        assert args.warmup is None, '--warmup argument is no longer valid,
#            use --lr-warmup-fraction instead'
#        del args.warmup
#        assert args.model_parallel_size is None, '--model-parallel-size is
#            no longer valid, use --tensor-model-parallel-size instead'
#        del args.model_parallel_size
#   2. _add_training_args(): add --batch-size deprecated stub arg.
#   3. _add_learning_rate_args(): add --warmup deprecated stub arg.
#   4. _add_distributed_args(): add --model-parallel-size deprecated stub arg.
#
# DeepSpeed adaptation: surfaced as validate_deprecated_args(args) +
# add_deprecated_args(parser) helpers callable from compile/initialize.
# ---------------------------------------------------------------------------

print('[M451]')


def validate_deprecated_args(args):
    """Assert deprecated CLI args are not set; delete them from namespace.

    Megatron 2623551d7 arguments.py parse_args() — nicer error messages:
      assert args.batch_size is None, '--batch-size argument is no longer
          valid, use --micro-batch-size instead'
      del args.batch_size
      assert args.warmup is None, '--warmup argument is no longer valid,
          use --lr-warmup-fraction instead'
      del args.warmup
      assert args.model_parallel_size is None, '--model-parallel-size is no
          longer valid, use --tensor-model-parallel-size instead'
      del args.model_parallel_size

    Only acts on attributes that exist in the namespace (i.e., were
    registered via add_deprecated_args); safe to call when the deprecated
    stubs were not registered.
    """
    if hasattr(args, 'batch_size'):
        assert args.batch_size is None, \
            '--batch-size argument is no longer valid, use --micro-batch-size instead'
        del args.batch_size

    if hasattr(args, 'warmup'):
        assert args.warmup is None, \
            '--warmup argument is no longer valid, use --lr-warmup-fraction instead'
        del args.warmup

    if hasattr(args, 'model_parallel_size'):
        assert args.model_parallel_size is None, \
            '--model-parallel-size is no longer valid, use --tensor-model-parallel-size instead'
        del args.model_parallel_size

    print('[M451] validate_deprecated_args: deprecated args validated and removed')


def add_deprecated_args(parser):
    """Register deprecated argument stubs so users get a clear error message.

    Megatron 2623551d7 — three deprecated args added across helpers:

    _add_training_args():
      group.add_argument('--batch-size', type=int, default=None,
                         help='Old batch size parameter, do not use. '
                         'Use --micro-batch-size instead')

    _add_learning_rate_args():
      group.add_argument('--warmup', type=int, default=None,
                         help='Old lr warmup argument, do not use. Use one of
                         the --lr-warmup-* arguments above')

    _add_distributed_args():
      group.add_argument('--model-parallel-size', type=int, default=None,
                         help='Old model parallel argument, do not use. Use
                         --tensor-model-parallel-size instead.')

    Adds all three to a single 'Deprecated Arguments' group on parser.
    Call before parser.parse_args() so that validate_deprecated_args() can
    catch and reject any usage with a clear message.
    """
    group = parser.add_argument_group(title='deprecated arguments')
    group.add_argument('--batch-size', type=int, default=None,
                       help='Old batch size parameter, do not use. '
                       'Use --micro-batch-size instead')
    group.add_argument('--warmup', type=int, default=None,
                       help='Old lr warmup argument, do not use. Use one of the '
                       '--lr-warmup-* arguments above')
    group.add_argument('--model-parallel-size', type=int, default=None,
                       help='Old model parallel argument, do not use. Use '
                       '--tensor-model-parallel-size instead.')
    print('[M451] add_deprecated_args: deprecated argument stubs registered')
    return parser

# ---------------------------------------------------------------------------
# M512: Megatron 78066ab08 — Fixing merge_mp_partitions
# Source: megatron/arguments.py (NVIDIA/Megatron-LM commit 78066ab08)
# Author: Jared Casper <jcasper@nvidia.com>  Date: 2021-01-20
#
# Mapping: megatron/arguments.py → deepspeed/compile/megatron_arguments.py
#
# Changes ported from arguments.py:
#   1. parse_args(): move "Set input defaults" block BEFORE the
#      micro_batch_size assertion (was after consumed_*_samples init).
#      This ensures defaults are set before any assertions run against them.
#
#   2. _add_checkpointing_args():
#        --no-load-optim: add default=None
#        --no-load-rng:   add default=None
#      (So the "set input defaults" block can override them via defaults dict.)
#
#   3. _add_distributed_args():
#        --use-cpu-initialization: action='store_true' → type=bool, required=False
#      (Allows merge_mp_partitions to inject via defaults dict, not CLI flag.)
#
# DeepSpeed adaptation: exposed as helper functions callable from compile/init.
# ---------------------------------------------------------------------------

print('[M512]')


def patch_checkpointing_args(parser):
    """Re-register --no-load-optim and --no-load-rng with default=None.

    Megatron 78066ab08 _add_checkpointing_args():
      group.add_argument('--no-load-optim', action='store_true', default=None)
      group.add_argument('--no-load-rng',   action='store_true', default=None)
    """
    group = parser.add_argument_group(title='M512 checkpointing patches')
    group.add_argument('--no-load-optim', action='store_true', default=None,
                       help='Do not load optimizer when loading checkpoint.')
    group.add_argument('--no-load-rng', action='store_true', default=None,
                       help='Do not load rng state when loading checkpoint.')
    print('[M512] patch_checkpointing_args: no-load-optim/rng with default=None')
    return parser


def patch_distributed_args(parser):
    """Re-register --use-cpu-initialization as type=bool, required=False.

    Megatron 78066ab08 _add_distributed_args():
      group.add_argument('--use-cpu-initialization', type=bool, required=False)
    """
    group = parser.add_argument_group(title='M512 distributed patches')
    group.add_argument('--use-cpu-initialization', type=bool, required=False,
                       help='If set, affine parallel weights initialization uses CPU')
    print('[M512] patch_distributed_args: use-cpu-initialization as type=bool')
    return parser


def set_input_defaults_early(args, defaults):
    """Apply defaults dict BEFORE micro_batch_size and other assertions.

    Megatron 78066ab08 parse_args(): "Set input defaults" block moved BEFORE
    the micro_batch_size assertion so that defaults can override args checked
    by early assertions (e.g. no_load_optim, no_load_rng, use_cpu_initialization).

    Sets args.<key> = defaults[key] only when attribute is currently None.
    Emits WARNING when user explicitly provided a value differing from default.
    """
    rank = getattr(args, 'rank', 0)
    for key in defaults:
        if getattr(args, key, None) is not None:
            if rank == 0:
                print('WARNING: overriding default arguments for {key}:{v} \
                       with {key}:{v2}'.format(key=key, v=defaults[key],
                                               v2=getattr(args, key)),
                       flush=True)
        else:
            setattr(args, key, defaults[key])
    print(f'[M512] set_input_defaults_early: applied {len(defaults)} default(s)')
    return args

# ---------------------------------------------------------------------------
# M544: Megatron 78a3dc323 — fixed arguments
# Source: megatron/arguments.py (NVIDIA/Megatron-LM commit 78a3dc323f9da3c4f)
# Author: Mostofa Patwary <mostofa.patwary@gmail.com>  Date: 2021-02-03
#
# Mapping: megatron/arguments.py → deepspeed/compile/megatron_arguments.py
#
# Changes ported from arguments.py _add_biencoder_args():
#   Fix --report-topk-accuracies help string: double-quote → single-quote
#   Fix --retriever-score-scaling help string: double-quote → single-quote
#
#   Before (broken — double-quoted string containing unescaped single quotes):
#     help="Which top-k accuracies to report '(e.g. '1 5 20')"
#     help="Whether to scale retriever scores by inverse 'square root of hidden size"
#   After (correct — single-quoted multi-line string concatenation):
#     help='Which top-k accuracies to report ' '(e.g. '1 5 20')'
#     help='Whether to scale retriever scores by inverse ' 'square root of hidden size'
#
# DeepSpeed adaptation: surfaced as add_biencoder_args(parser) helper
# callable from compile/initialize to register biencoder CLI arguments.
# ---------------------------------------------------------------------------

print('[M544]')


def add_biencoder_args(parser):
    """Register biencoder argument group with corrected help strings.

    Megatron 78a3dc323 _add_biencoder_args() — fixed quote style on two args:

      group.add_argument('--report-topk-accuracies', nargs='+', type=int,
                          default=[], help='Which top-k accuracies to report '
                          '(e.g. 1 5 20)')
      group.add_argument('--retriever-score-scaling', action='store_true',
                         help='Whether to scale retriever scores by inverse '
                          'square root of hidden size')

    The original used double-quoted help strings which caused syntactic
    confusion with embedded single quotes; the fix uses single-quoted
    implicit string concatenation throughout _add_biencoder_args().
    """
    group = parser.add_argument_group(title='biencoder arguments')

    # checkpointing
    group.add_argument('--ict-load', type=str, default=None,
                       help='Directory containing an ICTBertModel checkpoint')
    group.add_argument('--bert-load', type=str, default=None,
                       help='Directory containing an BertModel checkpoint '
                       '(needed to start ICT and REALM)')

    # data
    group.add_argument('--titles-data-path', type=str, default=None,
                       help='Path to titles dataset used for ICT')
    group.add_argument('--query-in-block-prob', type=float, default=0.1,
                       help='Probability of keeping query in block for '
                       'ICT dataset')
    group.add_argument('--use-one-sent-docs', action='store_true',
                       help='Whether to use one sentence documents in ICT')

    # training — fixed: double-quote → single-quote (Megatron 78a3dc323)
    group.add_argument('--report-topk-accuracies', nargs='+', type=int,
                       default=[], help='Which top-k accuracies to report '
                       '(e.g. 1 5 20)')
    group.add_argument('--retriever-score-scaling', action='store_true',
                       help='Whether to scale retriever scores by inverse '
                       'square root of hidden size')

    # faiss index
    group.add_argument('--faiss-use-gpu', action='store_true',
                       help='Whether create the FaissMIPSIndex on GPU')
    group.add_argument('--block-data-path', type=str, default=None,
                       help='Where to save/load BlockData to/from')

    # indexer
    group.add_argument('--indexer-batch-size', type=int, default=128,
                       help='How large of batches to use when doing indexing '
                       'jobs')
    group.add_argument('--indexer-log-interval', type=int, default=1000,
                       help='After how many batches should the indexer '
                       'report progress')

    print('[M544] add_biencoder_args: biencoder arguments registered')
    return parser
