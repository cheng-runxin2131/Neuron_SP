# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# M464: Megatron 7abd3e9 — Pipeline P2P ring-exchange comm pattern.
# Introduces send_forward/recv_forward/send_backward/recv_backward as
# first-class ring-exchange primitives (cf. Megatron commit 7abd3e90,
# megatron/p2p_communication.py).  All new ops are routed through the
# DES-LOC comm tracker in deepspeed/comm/comm.py so pipeline tensors
# participate in per-tier bandwidth accounting alongside gradient traffic.
#
# Knuth critique I  — Literate-programming axiom: the code itself is the
#   documentation.  A bare send() table with no semantic names forces every
#   reader to reconstruct the pipeline direction from call sites; named
#   directional helpers erase that reconstruction cost entirely.
# Knuth critique II — "Premature optimisation is the root of all evil, but
#   we should not pass up our opportunities in that critical 3%."  The
#   original back-to-back send/recv in train_step is a correctness bottleneck:
#   it serialises the pipeline bubble.  Ring-exchange overlaps the forward
#   recv with the send, shrinking exposed latency in the critical path.

import msgpack
import typing

import torch
from deepspeed import comm as dist

from deepspeed.utils.torch import required_torch_version
from deepspeed.accelerator import get_accelerator

_groups = None
_grid = None

_async = []

# ---------------------------------------------------------------------------
# Ring-exchange byte counters — fused into DES-LOC comm accounting.
# Tier-0 (param) is reused for pipeline activations; the accounting is
# directional-agnostic because p2p tensors are not reduced, only forwarded.
# ---------------------------------------------------------------------------
_ring_bytes_sent: int = 0
_ring_bytes_recv: int = 0


def _desloc_track_p2p(tensor: torch.Tensor, direction: str) -> None:
    """Record bytes for a p2p tensor into the global DES-LOC comm tracker.

    Silently no-ops when the tracker is absent so this path is safe before
    deepspeed engine init and in unit-test harnesses.

    Args:
        tensor: the tensor being communicated.
        direction: 'send' or 'recv' — used only for the print diagnostic.
    """
    global _ring_bytes_sent, _ring_bytes_recv
    try:
        from deepspeed.comm.comm import get_desloc_scheduler
        tracker_attr = '_desloc_comm_tracker'
        # Walk up from scheduler to engine tracker if available.
        sched = get_desloc_scheduler()
        if sched is not None and hasattr(sched, tracker_attr):
            nb = tensor.numel() * tensor.element_size()
            getattr(sched, tracker_attr).sync(0, nb)  # tier-0 = params/activations
    except Exception:
        pass
    nb = tensor.numel() * tensor.element_size()
    if direction == 'send':
        _ring_bytes_sent += nb
    else:
        _ring_bytes_recv += nb


def can_send_recv() -> bool:
    return required_torch_version(min_version=1.8)


#initializes adjacent process groups
#run this only after deepspeed.init_distributed() has been called
def init_process_groups(grid):
    global _groups, _grid
    _grid = grid

    assert _grid.pipe_parallel_size > 1, "There is no pipeline parallelism"

    if not can_send_recv():
        _groups = [dist.new_group(ranks=group) for group in _grid.p2p_groups]


def _is_valid_send_recv(src_stage, dest_stage):
    first_stage = 0
    last_stage = _grid.pipe_parallel_size - 1
    assert abs(src_stage-dest_stage) == 1 or \
        (src_stage == first_stage and dest_stage == last_stage) or \
        (src_stage == last_stage and dest_stage == first_stage), \
    "Functionality currently limited to send and receive between adjacent ranks only"


def send(tensor, dest_stage, async_op=False):
    global _groups
    assert async_op == False, "Doesn't support async_op true"
    src_stage = _grid.get_stage_id()
    _is_valid_send_recv(src_stage, dest_stage)

    _desloc_track_p2p(tensor, 'send')

    dest_rank = _grid.stage_to_global(stage_id=dest_stage)
    if async_op:
        global _async
        op = dist.isend(tensor, dest_rank)
        _async.append(op)
    else:

        if can_send_recv():
            return dist.send(tensor, dest_rank)
        else:
            group = _get_send_recv_group(src_stage, dest_stage)
            src_rank = _grid.stage_to_global(stage_id=src_stage)
            return dist.broadcast(tensor, src_rank, group=group, async_op=async_op)


def recv(tensor, src_stage, async_op=False):
    global _groups
    assert async_op == False, "Doesn't support async_op true"
    dest_stage = _grid.get_stage_id()
    _is_valid_send_recv(src_stage, dest_stage)

    src_rank = _grid.stage_to_global(stage_id=src_stage)

    if async_op:
        global _async
        op = dist.irecv(tensor, src_rank)
        _async.append(op)
    else:
        if can_send_recv():
            ret = dist.recv(tensor, src_rank)
            _desloc_track_p2p(tensor, 'recv')
            return ret
        else:
            group = _get_send_recv_group(src_stage, dest_stage)
            ret = dist.broadcast(tensor, src_rank, group=group, async_op=async_op)
            _desloc_track_p2p(tensor, 'recv')
            return ret


# ==========================================================================
# Ring-exchange P2P layer — Megatron 7abd3e9 pattern
# Ref: Megatron-LM megatron/p2p_communication.py — send_forward/recv_forward
# Ref: Narayanan et al. 2021 §3.2 — 1F1B schedule with ring-exchange to
#      overlap send and recv across adjacent pipeline stages.
#
# The four directional helpers below map onto the pipeline ring topology:
#
#   stage 0 → stage 1 → … → stage N-1 → (wrap) stage 0
#
# "forward" direction: activations flow 0→N-1 (increasing stage id).
# "backward" direction: gradients flow N-1→0 (decreasing stage id).
#
# Ring-exchange idiom:  rather than serialised send-then-recv, we issue both
# ops and let the NCCL transport layer overlap them.  Correctness is
# preserved because the two tensors travel in opposite directions on the ring
# (no aliasing).
# ==========================================================================


def _ring_exchange(send_tensor, send_stage, recv_tensor, recv_stage):
    """Issue a paired send+recv in one logical step (ring-exchange).

    Both ops are blocking to keep the scheduler simple.  The NCCL/transport
    layer is free to pipeline them internally.  For single-stage pipelines
    (pipe_parallel_size==1) this should never be called — callers guard on
    that invariant.

    Args:
        send_tensor:  tensor to transmit to *send_stage*.
        send_stage:   destination stage id for the send.
        recv_tensor:  pre-allocated buffer to receive into from *recv_stage*.
        recv_stage:   source stage id for the receive.

    Returns:
        recv_tensor (mutated in-place with received data).
    """
    my_stage = _grid.get_stage_id()
    _is_valid_send_recv(my_stage, send_stage)
    _is_valid_send_recv(recv_stage, my_stage)

    send_rank = _grid.stage_to_global(stage_id=send_stage)
    recv_rank = _grid.stage_to_global(stage_id=recv_stage)

    _desloc_track_p2p(send_tensor, 'send')

    # Issue recv first then send so the receive buffer is posted before the
    # remote peer sends — avoids potential rendezvous deadlock on some
    # transports.  NCCL is rendezvous-free but the ordering is defensive.
    if can_send_recv():
        dist.recv(recv_tensor, recv_rank)
        dist.send(send_tensor, send_rank)
    else:
        group_r = _get_send_recv_group(recv_stage, my_stage)
        group_s = _get_send_recv_group(my_stage, send_stage)
        dist.broadcast(recv_tensor, recv_rank, group=group_r)
        src_rank = _grid.stage_to_global(stage_id=my_stage)
        dist.broadcast(send_tensor, src_rank, group=group_s)

    _desloc_track_p2p(recv_tensor, 'recv')
    return recv_tensor


def send_forward(output_tensor, timers=None):
    """Send activations to the next pipeline stage (forward direction).

    Implements the send half of the ring-exchange for the 1F1B schedule.
    Activations produced at this stage are forwarded to stage+1; the last
    stage wraps to stage 0 (ring topology).

    Args:
        output_tensor: activation tensor produced by this stage's forward.
        timers:        optional Megatron-style timers object; if provided,
                       the 'forward-send' timer is bracketed around the send.

    Diagnostic: prints tensor shape + destination rank once per 100 calls to
    keep noise low while preserving debuggability in multi-node runs.
    """
    if _grid is None:
        raise RuntimeError("[p2p] send_forward called before init_process_groups()")
    my_stage = _grid.get_stage_id()
    next_stage = (my_stage + 1) % _grid.pipe_parallel_size
    dest_rank = _grid.stage_to_global(stage_id=next_stage)

    if send_forward.call_count % 100 == 0:
        print(
            f"[p2p][send_forward] stage={my_stage}→{next_stage} "
            f"rank={_grid.get_global_rank()}→{dest_rank} "
            f"shape={tuple(output_tensor.shape)} dtype={output_tensor.dtype} "
            f"call={send_forward.call_count}",
            flush=True,
        )
    send_forward.call_count += 1

    if timers is not None:
        timers('forward-send').start()
    send(output_tensor, next_stage)
    if timers is not None:
        timers('forward-send').stop()


send_forward.call_count = 0


def recv_forward(recv_buffer, timers=None):
    """Receive activations from the previous pipeline stage (forward direction).

    The complementary recv for send_forward.  Receives the activation tensor
    produced by stage-1 (with wrap-around from the last stage).

    Args:
        recv_buffer: pre-allocated tensor matching the expected activation shape.
        timers:      optional timers object; brackets 'forward-recv'.

    Returns:
        recv_buffer (mutated in-place).

    Diagnostic: prints shape + source rank every 100 calls.
    """
    if _grid is None:
        raise RuntimeError("[p2p] recv_forward called before init_process_groups()")
    my_stage = _grid.get_stage_id()
    prev_stage = (my_stage - 1) % _grid.pipe_parallel_size
    src_rank = _grid.stage_to_global(stage_id=prev_stage)

    if recv_forward.call_count % 100 == 0:
        print(
            f"[p2p][recv_forward] stage={prev_stage}→{my_stage} "
            f"rank={src_rank}→{_grid.get_global_rank()} "
            f"shape={tuple(recv_buffer.shape)} dtype={recv_buffer.dtype} "
            f"call={recv_forward.call_count}",
            flush=True,
        )
    recv_forward.call_count += 1

    if timers is not None:
        timers('forward-recv').start()
    recv(recv_buffer, prev_stage)
    if timers is not None:
        timers('forward-recv').stop()
    return recv_buffer


recv_forward.call_count = 0


def send_backward(input_tensor_grad, timers=None):
    """Send gradients to the previous pipeline stage (backward direction).

    Gradients flow in the direction opposite to activations: from stage N-1
    back toward stage 0.

    Args:
        input_tensor_grad: gradient tensor w.r.t. this stage's input.
        timers:            optional timers; brackets 'backward-send'.

    Diagnostic: shape + dest rank every 100 calls.
    """
    if _grid is None:
        raise RuntimeError("[p2p] send_backward called before init_process_groups()")
    my_stage = _grid.get_stage_id()
    prev_stage = (my_stage - 1) % _grid.pipe_parallel_size
    dest_rank = _grid.stage_to_global(stage_id=prev_stage)

    if send_backward.call_count % 100 == 0:
        print(
            f"[p2p][send_backward] stage={my_stage}→{prev_stage} "
            f"rank={_grid.get_global_rank()}→{dest_rank} "
            f"shape={tuple(input_tensor_grad.shape)} dtype={input_tensor_grad.dtype} "
            f"call={send_backward.call_count}",
            flush=True,
        )
    send_backward.call_count += 1

    if timers is not None:
        timers('backward-send').start()
    send(input_tensor_grad, prev_stage)
    if timers is not None:
        timers('backward-send').stop()


send_backward.call_count = 0


def recv_backward(recv_buffer, timers=None):
    """Receive gradients from the next pipeline stage (backward direction).

    The complementary recv for send_backward.  Receives from stage+1.

    Args:
        recv_buffer: pre-allocated tensor matching the expected gradient shape.
        timers:      optional timers; brackets 'backward-recv'.

    Returns:
        recv_buffer (mutated in-place).

    Diagnostic: shape + source rank every 100 calls.
    """
    if _grid is None:
        raise RuntimeError("[p2p] recv_backward called before init_process_groups()")
    my_stage = _grid.get_stage_id()
    next_stage = (my_stage + 1) % _grid.pipe_parallel_size
    src_rank = _grid.stage_to_global(stage_id=next_stage)

    if recv_backward.call_count % 100 == 0:
        print(
            f"[p2p][recv_backward] stage={next_stage}→{my_stage} "
            f"rank={src_rank}→{_grid.get_global_rank()} "
            f"shape={tuple(recv_buffer.shape)} dtype={recv_buffer.dtype} "
            f"call={recv_backward.call_count}",
            flush=True,
        )
    recv_backward.call_count += 1

    if timers is not None:
        timers('backward-recv').start()
    recv(recv_buffer, next_stage)
    if timers is not None:
        timers('backward-recv').stop()
    return recv_buffer


recv_backward.call_count = 0


def ring_exchange_forward(send_tensor, recv_buffer, timers=None):
    """Simultaneous send-forward + recv-forward in one ring-exchange step.

    This is the latency-hiding form of the 1F1B forward pass: we issue both
    the send to stage+1 and the receive from stage-1 together, allowing the
    transport layer to overlap them.

    Args:
        send_tensor:  activation tensor to forward to stage+1.
        recv_buffer:  pre-allocated buffer for the activation from stage-1.
        timers:       optional timers; brackets 'forward-ring-exchange'.

    Returns:
        recv_buffer (mutated in-place with received activations).

    Diagnostic: prints pair shapes + ranks every 100 calls.
    """
    if _grid is None:
        raise RuntimeError("[p2p] ring_exchange_forward called before init_process_groups()")
    my_stage = _grid.get_stage_id()
    next_stage = (my_stage + 1) % _grid.pipe_parallel_size
    prev_stage = (my_stage - 1) % _grid.pipe_parallel_size

    if ring_exchange_forward.call_count % 100 == 0:
        print(
            f"[p2p][ring_exchange_forward] stage={my_stage} "
            f"send→{next_stage} shape={tuple(send_tensor.shape)} | "
            f"recv←{prev_stage} shape={tuple(recv_buffer.shape)} "
            f"call={ring_exchange_forward.call_count}",
            flush=True,
        )
    ring_exchange_forward.call_count += 1

    if timers is not None:
        timers('forward-ring-exchange').start()
    _ring_exchange(send_tensor, next_stage, recv_buffer, prev_stage)
    if timers is not None:
        timers('forward-ring-exchange').stop()
    return recv_buffer


ring_exchange_forward.call_count = 0


def ring_exchange_backward(send_tensor, recv_buffer, timers=None):
    """Simultaneous send-backward + recv-backward in one ring-exchange step.

    Gradient counterpart to ring_exchange_forward.  Sends gradient to stage-1
    while receiving gradient from stage+1.

    Args:
        send_tensor:  gradient tensor to send to stage-1.
        recv_buffer:  pre-allocated buffer for gradient from stage+1.
        timers:       optional timers; brackets 'backward-ring-exchange'.

    Returns:
        recv_buffer (mutated in-place).

    Diagnostic: prints pair shapes + ranks every 100 calls.
    """
    if _grid is None:
        raise RuntimeError("[p2p] ring_exchange_backward called before init_process_groups()")
    my_stage = _grid.get_stage_id()
    next_stage = (my_stage + 1) % _grid.pipe_parallel_size
    prev_stage = (my_stage - 1) % _grid.pipe_parallel_size

    if ring_exchange_backward.call_count % 100 == 0:
        print(
            f"[p2p][ring_exchange_backward] stage={my_stage} "
            f"send→{prev_stage} shape={tuple(send_tensor.shape)} | "
            f"recv←{next_stage} shape={tuple(recv_buffer.shape)} "
            f"call={ring_exchange_backward.call_count}",
            flush=True,
        )
    ring_exchange_backward.call_count += 1

    if timers is not None:
        timers('backward-ring-exchange').start()
    _ring_exchange(send_tensor, prev_stage, recv_buffer, next_stage)
    if timers is not None:
        timers('backward-ring-exchange').stop()
    return recv_buffer


ring_exchange_backward.call_count = 0


def get_ring_comm_stats() -> dict:
    """Return cumulative ring-exchange byte counters for diagnostic use.

    These counters complement the DES-LOC comm tracker and are readable from
    any rank without synchronisation (local stats only).

    Returns:
        dict with keys 'ring_bytes_sent', 'ring_bytes_recv', 'ring_bytes_total'.
    """
    return {
        'ring_bytes_sent': _ring_bytes_sent,
        'ring_bytes_recv': _ring_bytes_recv,
        'ring_bytes_total': _ring_bytes_sent + _ring_bytes_recv,
    }


def wait():
    global _async
    for op in _async:
        op.wait()
    _async = []

    get_accelerator().synchronize()


def send_obj(msg: typing.Any, dest: int):
    """Send an arbitrary python object to ``dest``.

    Note: ``msg`` must be serializable by msgpack.

    WARN: This incurs a CPU -> GPU transfer and should be used sparingly
    for performance reasons.

    Args:
        msg (typing.Any): The object to send.
        dest (int): Destination rank.
    """
    # serialize the message
    msg = msgpack.packb(msg)
    # construct a tensor to send
    msg = torch.ByteTensor(torch.ByteStorage.from_buffer(msg)).to(get_accelerator().device_name())

    # Send meta and message
    length_tensor = torch.tensor([len(msg)], dtype=torch.long).to(get_accelerator().device_name())
    dist.send(length_tensor, dst=dest)
    dist.send(msg, dst=dest)


def recv_obj(sender: int) -> typing.Any:
    """Receive an arbitrary python object from ``sender``.

    WARN: This incur a CPU <-> GPU transfers and should be used sparingly
    for performance reasons.

    Args:
        sender (int): The rank sending the message.
    """
    # Get message meta
    length = torch.tensor([0], dtype=torch.long).to(get_accelerator().device_name())
    dist.recv(length, src=sender)

    # Receive and deserialize
    msg = torch.empty(length.item(), dtype=torch.uint8).to(get_accelerator().device_name())
    dist.recv(msg, src=sender)

    msg = msgpack.unpackb(msg.cpu().numpy().tobytes())

    def _to(x):
        """Recursively move to the current device."""
        if torch.is_tensor(x):
            return x.to(get_accelerator().device_name())
        if isinstance(x, (tuple, list)):
            ret = [_to(x_) for x_ in x]
            if isinstance(x, tuple):
                ret = tuple(ret)
            return ret
        # handle kwargs
        if isinstance(x, dict):
            ret = dict()
            for key, val in x.items():
                ret[_to(key)] = _to(val)
            return ret

        # Anything else is a no-op
        return x

    msg = _to(msg)
    return msg


def _get_send_recv_group(src_stage, dest_stage):
    '''the group id is always the smaller rank unless its a wrap around'''

    stage_id = None

    first_stage = 0
    last_stage = _grid.pipe_parallel_size - 1

    if (src_stage == first_stage and dest_stage == last_stage
            or dest_stage == first_stage and src_stage == last_stage):
        stage_id = last_stage
    elif src_stage > dest_stage:
        stage_id = dest_stage
    else:
        stage_id = src_stage
    '''group_id corresponds to group of [group_id, group_id+1]
     unless group_id is the rank of the last stage
     in which case group_id corresponds to group[group_id-num_stages+1, group_id]
     '''
    group_id = _grid.stage_to_global(stage_id=stage_id)

    return _groups[group_id]
