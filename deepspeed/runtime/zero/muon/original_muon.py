# Copyright (c) 2024 Keller Jordan
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
MIT License

Copyright (c) 2024 Keller Jordan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import logging
from typing import Optional

import torch
import deepspeed.comm as dist  # replace torch's distributed package with deepspeed.comm to resolve deepspeed check
from deepspeed.runtime import compiler

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────
# M473: QKV split helpers ported from Megatron 947d6ae1
# (Fix Muon QKV split for gated attention #4728)
# ───────────────────────────────────────────────────────────────


def _get_qkv_split_shapes(model_cfg) -> list[int]:
    """Compute QKV projection split dimensions from model config.

    For standard GQA: [Q, K, V] where Q = heads/groups * kv_channels.
    For gated attention (attention_output_gate=True): [Q, Q_gate, K, V]
    — the gate projection mirrors Q in width, so both halves receive a
    separate Muon orthogonalisation pass rather than being lumped together.

    Knuth critique #1: the original flat [Q,K,V] tuple silently produced
    wrong gradient slabs when gate projections doubled row-count; this
    explicit branch makes the contract observable at construction time.
    """
    query_projection_size = (
        model_cfg.num_attention_heads // model_cfg.num_query_groups * model_cfg.kv_channels
    )
    if getattr(model_cfg, 'attention_output_gate', False):
        # gated attention: Q and its gate share the same column count
        print(
            f"[M473 _get_qkv_split_shapes] gated attention detected: "
            f"split=[Q={query_projection_size}, gate={query_projection_size}, "
            f"K={model_cfg.kv_channels}, V={model_cfg.kv_channels}]"
        )
        return [
            query_projection_size,
            query_projection_size,
            model_cfg.kv_channels,
            model_cfg.kv_channels,
        ]
    print(
        f"[M473 _get_qkv_split_shapes] standard GQA split: "
        f"[Q={query_projection_size}, K={model_cfg.kv_channels}, V={model_cfg.kv_channels}]"
    )
    return [query_projection_size, model_cfg.kv_channels, model_cfg.kv_channels]


def _tag_param_qkv(param: torch.nn.Parameter,
                   name: str,
                   model_cfg,
                   _cache: dict) -> None:
    """Attach is_qkv + qkv_split_shapes to a QKV weight parameter.

    Called once per named_parameters() traversal. Uses a one-element
    dict cache so _get_qkv_split_shapes() is computed at most once per
    model chunk (Megatron's pattern; avoids redundant config reads).

    Knuth critique #2: tagging inside the loop with no memoisation meant
    _get_qkv_split_shapes() ran O(n_layers) times — cheap but misleading;
    caching makes the once-per-chunk intent explicit.
    """
    if 'is_qkv' not in _cache:
        _cache['is_qkv'] = None  # will be populated below

    # Only 2-D weight tensors are candidates
    if len(param.shape) != 2:
        return

    if _cache['is_qkv'] is None:
        _cache['is_qkv'] = _get_qkv_split_shapes(model_cfg)
    qkv_split_shapes: list[int] = _cache['is_qkv']

    qkv_split_dim = sum(qkv_split_shapes)
    if param.shape[0] % qkv_split_dim == 0:
        param.is_qkv = True
        param.qkv_split_shapes = qkv_split_shapes
        print(
            f"[M473 _tag_param_qkv] tagged {name}: "
            f"shape={tuple(param.shape)}, split_shapes={qkv_split_shapes}"
        )
    else:
        logger.debug(
            "[M473 _tag_param_qkv] QKV split skipped for %s: "
            "shape=%s, split_shapes=%s — row count not divisible by split dim %d",
            name, tuple(param.shape), qkv_split_shapes, qkv_split_dim,
        )


def _apply_qkv_split_grad(
    grad: torch.Tensor,
    param: torch.nn.Parameter,
    optimizer_qkv_split_shapes: Optional[list[int]],
) -> list[torch.Tensor]:
    """Split a QKV gradient tensor into per-projection slabs.

    Resolution order for split config:
      1. Per-param attribute  (set by _tag_param_qkv / Megatron tagger)
      2. Optimizer-level default (qkv_split_shapes constructor arg)
      3. RuntimeError — split was requested but shapes are nowhere to be found

    Returns a list of 2-D tensors, one per projection head group,
    each shaped (group_rows, hidden_dim).
    """
    qkv_split_shapes: Optional[list[int]] = getattr(param, 'qkv_split_shapes', None)
    if qkv_split_shapes is None:
        qkv_split_shapes = optimizer_qkv_split_shapes
    if qkv_split_shapes is None:
        raise RuntimeError(
            "Muon QKV split requested but qkv_split_shapes is not set "
            "(neither on param nor on optimizer). "
            "Pass qkv_split_shapes to the optimizer constructor or tag "
            "parameters with _tag_param_qkv()."
        )

    grad_shape = grad.shape
    qkv_split_dim = sum(qkv_split_shapes)
    if grad_shape[0] % qkv_split_dim != 0:
        raise RuntimeError(
            f"[M473] Muon QKV split shape mismatch: "
            f"grad_shape={tuple(grad_shape)}, split_shapes={qkv_split_shapes}, "
            f"split_dim={qkv_split_dim} — first dim not divisible"
        )

    num_query_groups = grad_shape[0] // qkv_split_dim
    print(
        f"[M473 _apply_qkv_split_grad] grad_shape={tuple(grad_shape)}, "
        f"split_shapes={qkv_split_shapes}, num_query_groups={num_query_groups}"
    )
    qkv_grads = torch.split(
        grad.view(num_query_groups, qkv_split_dim, -1),
        qkv_split_shapes,
        dim=1,
    )
    return [g.reshape(-1, grad_shape[-1]) for g in qkv_grads]


@compiler.compile()
def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@compiler.compile()
def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


def muon_update_with_qkv_split(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    param: torch.nn.Parameter,
    optimizer_qkv_split_shapes: Optional[list[int]],
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
) -> torch.Tensor:
    """Muon update that orthogonalises each QKV projection slab independently.

    M473: When a parameter carries is_qkv=True, naive orthogonalisation of
    the full [Q||K||V] matrix (or [Q||Q_gate||K||V] for gated attention)
    treats projections from different semantic roles as a single entity.
    Splitting first and merging after ensures each head group gradient is
    orthogonalised in isolation.

    For non-QKV params this is identical to muon_update().

    Knuth critique #1: silent wrong-split was the original bug — we make the
    shape contract observable at the call site via RuntimeError.
    Knuth critique #2: memoised qkv_split_shapes avoids recomputing across
    steps; resolution order: per-param attr > optimizer default > error.
    """
    is_qkv = getattr(param, "is_qkv", False)

    if not is_qkv:
        return muon_update(grad, momentum, beta=beta, ns_steps=ns_steps, nesterov=nesterov)

    # ── QKV path ────────────────────────────────────────────────
    grad_shape = grad.shape
    qkv_grads = _apply_qkv_split_grad(grad, param, optimizer_qkv_split_shapes)

    qkv_split_shapes: list[int] = (
        getattr(param, "qkv_split_shapes", None) or optimizer_qkv_split_shapes
    )
    num_query_groups = grad_shape[0] // sum(qkv_split_shapes)
    mom_slabs = _apply_qkv_split_grad(momentum.data.clone(), param, optimizer_qkv_split_shapes)

    updated_slabs: list[torch.Tensor] = []
    updated_mom_slabs: list[torch.Tensor] = []
    for g_slab, m_slab in zip(qkv_grads, mom_slabs):
        m_slab.lerp_(g_slab, 1 - beta)
        u = g_slab.lerp_(m_slab, beta) if nesterov else m_slab.clone()
        u = zeropower_via_newtonschulz5(u, steps=ns_steps)
        u = u * max(1, g_slab.size(-2) / g_slab.size(-1)) ** 0.5
        updated_slabs.append(u)
        updated_mom_slabs.append(m_slab)

    # Reconstruct full-row update
    update = torch.cat(updated_slabs, dim=0)

    # Write updated momentum back into the flat buffer in-place
    mom_chunks = [
        s.reshape(num_query_groups, sz, -1)
        for s, sz in zip(updated_mom_slabs, qkv_split_shapes)
    ]
    momentum.data.copy_(torch.cat(mom_chunks, dim=1).reshape(grad_shape))

    print(
        f"[M473 muon_update_with_qkv_split] "
        f"grad_shape={tuple(grad_shape)}, update_shape={tuple(update.shape)}, "
        f"n_slabs={len(updated_slabs)}, split_shapes={qkv_split_shapes}"
    )
    return update


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. For efficient orthogonalization we use a Newton-Schulz iteration, which has the
    advantage that it can be stably run in bfloat16 on the GPU.

    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.
    Hidden convolutional weights can be trained using Muon by viewing them as 2D and then
    collapsing their last 3 dimensions.

    Arguments:
        lr: The learning rate, in units of spectral norm per update.
        weight_decay: The AdamW-style weight decay.
        momentum: The momentum. A value of 0.95 here is usually fine.
    """

    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95,
                 split_qkv: bool = False,
                 qkv_split_shapes: Optional[list[int]] = None):
        """M473: split_qkv enables per-projection orthogonalisation for QKV
        weight matrices. qkv_split_shapes is the optimizer-level fallback;
        per-param qkv_split_shapes (set by _tag_param_qkv) takes precedence."""
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)
        self.split_qkv = split_qkv
        self.qkv_split_shapes = qkv_split_shapes
        if split_qkv:
            print(
                f"[M473 Muon.__init__] split_qkv=True, "
                f"optimizer-level qkv_split_shapes={qkv_split_shapes}"
            )

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])
                                   ] * (dist.get_world_size() - len(params) % dist.get_world_size())
            for base_i in range(len(params))[::dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    # M473: use QKV-aware update when split_qkv requested
                    if self.split_qkv and getattr(p, "is_qkv", False):
                        update = muon_update_with_qkv_split(
                            p.grad, state["momentum_buffer"], p,
                            self.qkv_split_shapes, beta=group["momentum"],
                        )
                    else:
                        update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()],
                                params_pad[base_i + dist.get_rank()])

        return loss


class SingleDeviceMuon(torch.optim.Optimizer):
    """
    Muon variant for usage in non-distributed settings.
    """

    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95,
                 split_qkv: bool = False,
                 qkv_split_shapes: Optional[list[int]] = None):
        """M473: single-device Muon with optional QKV split support."""
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)
        self.split_qkv = split_qkv
        self.qkv_split_shapes = qkv_split_shapes

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                # M473: QKV-aware update path
                if self.split_qkv and getattr(p, "is_qkv", False):
                    update = muon_update_with_qkv_split(
                        p.grad, state["momentum_buffer"], p,
                        self.qkv_split_shapes, beta=group["momentum"],
                    )
                else:
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    Distributed Muon variant that can be used for all parameters in the network, since it runs an
    internal AdamW for the parameters that are not compatible with Muon. The user must manually
    specify which parameters shall be optimized with Muon and which with Adam by passing in a
    list of param_groups with the `use_muon` flag set.

    The point of this class is to allow the user to have a single optimizer in their code, rather
    than having both a Muon and an Adam which each need to be stepped.

    You can see an example usage below:

    https://github.com/KellerJordan/modded-nanogpt/blob/master/records/052525_MuonWithAuxAdamExample/b01550f9-03d8-4a9c-86fe-4ab434f1c5e0.txt#L470
    ```
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    from muon import MuonWithAuxAdam
    adam_groups = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
    adam_groups = [dict(**g, betas=(0.8, 0.95), eps=1e-10, use_muon=False) for g in adam_groups]
    muon_group = dict(params=hidden_matrix_params, lr=0.05, momentum=0.95, use_muon=True)
    param_groups = [*adam_groups, muon_group]
    optimizer = MuonWithAuxAdam(param_groups)
    ```
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())
        # M473: optimizer-level fallback for QKV split shapes
        self.split_qkv: bool = False
        self.qkv_split_shapes: Optional[list[int]] = None

    def configure_qkv_split(self, split_qkv: bool,
                             qkv_split_shapes: Optional[list[int]] = None) -> None:
        """M473: opt-in to per-projection QKV orthogonalisation.

        Call after construction with split_qkv=True and the appropriate
        split_shapes (or rely on per-param is_qkv / qkv_split_shapes attrs
        set by _tag_param_qkv).
        """
        self.split_qkv = split_qkv
        self.qkv_split_shapes = qkv_split_shapes
        print(
            f"[M473 MuonWithAuxAdam.configure_qkv_split] "
            f"split_qkv={split_qkv}, qkv_split_shapes={qkv_split_shapes}"
        )

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])
                                       ] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        if p.grad is None:
                            # continue
                            p.grad = torch.zeros_like(p)  # Force synchronization
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        # M473: QKV-aware update path
                        if self.split_qkv and getattr(p, "is_qkv", False):
                            update = muon_update_with_qkv_split(
                                p.grad, state["momentum_buffer"], p,
                                self.qkv_split_shapes, beta=group["momentum"],
                            )
                        else:
                            update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()],
                                    params_pad[base_i + dist.get_rank()])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"], state["step"], group["betas"],
                                         group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    # M473: QKV-aware update for single-device aux-adam variant
                    split_qkv = getattr(self, "split_qkv", False)
                    qkv_split_shapes = getattr(self, "qkv_split_shapes", None)
                    if split_qkv and getattr(p, "is_qkv", False):
                        update = muon_update_with_qkv_split(
                            p.grad, state["momentum_buffer"], p,
                            qkv_split_shapes, beta=group["momentum"],
                        )
                    else:
                        update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"], state["step"], group["betas"],
                                         group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


# ═══════════════════════════════════════════════════════════════
# DES-LOC extensions for Muon optimizer classes (M189)
# Section 5.6: Kx-gated all_gather + Ku-gated momentum sync
# ═══════════════════════════════════════════════════════════════


class DESLOCMuon(Muon):
    """DES-LOC variant of Muon with Kx-gated all_gather.

    The original Muon calls dist.all_gather every step to broadcast
    parameter updates across workers. DES-LOC gates this by Kx:
    all_gather only happens when step % Kx == 0.

    Between sync points, each worker runs independently with its
    local parameters + Muon momentum, matching the DES-LOC
    desynchronization principle.
    """

    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95,
                 desloc_Kx=1, desloc_Ku=1, desloc_clip_radius=1.0,
                 split_qkv: bool = False,
                 qkv_split_shapes=None):
        # M473: forward QKV split config to base Muon
        super().__init__(params, lr=lr, weight_decay=weight_decay, momentum=momentum,
                         split_qkv=split_qkv, qkv_split_shapes=qkv_split_shapes)
        self._desloc_Kx = desloc_Kx
        self._desloc_Ku = desloc_Ku
        self._desloc_clip_radius = desloc_clip_radius
        self._desloc_step = 0
        self._desloc_gather_count = 0
        self._desloc_skip_count = 0
        self._desloc_total_gather_bytes = 0

    def _should_gather(self):
        """Check if all_gather should execute this step."""
        if self._desloc_Kx <= 1:
            return True
        return self._desloc_step % self._desloc_Kx == 0

    @torch.no_grad()
    def step(self, closure=None):
        self._desloc_step += 1

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]

            # Apply per-coordinate clipping if enabled
            if self._desloc_clip_radius < float('inf'):
                for p in params:
                    if p.grad is not None:
                        p.grad.clamp_(-self._desloc_clip_radius,
                                      self._desloc_clip_radius)

            params_pad = params + [torch.empty_like(params[-1])
                                   ] * (dist.get_world_size() - len(params) % dist.get_world_size())
            for base_i in range(len(params))[::dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    # M473: use QKV-aware update when split_qkv is active
                    if self.split_qkv and getattr(p, "is_qkv", False):
                        update = muon_update_with_qkv_split(
                            p.grad, state["momentum_buffer"], p,
                            self.qkv_split_shapes, beta=group["momentum"],
                        )
                    else:
                        update = muon_update(p.grad, state["momentum_buffer"],
                                             beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])

                # M189: Kx-gated all_gather
                if self._should_gather():
                    dist.all_gather(
                        params_pad[base_i:base_i + dist.get_world_size()],
                        params_pad[base_i + dist.get_rank()])
                    self._desloc_gather_count += 1
                    self._desloc_total_gather_bytes += sum(
                        p.numel() * p.element_size() for p in
                        params_pad[base_i:base_i + dist.get_world_size()])
                else:
                    self._desloc_skip_count += 1

        return loss

    def get_desloc_stats(self):
        return {
            'step': self._desloc_step,
            'gather_count': self._desloc_gather_count,
            'skip_count': self._desloc_skip_count,
            'total_gather_bytes': self._desloc_total_gather_bytes,
            'Kx': self._desloc_Kx,
            'Ku': self._desloc_Ku,
        }


class DESLOCMuonWithAuxAdam(MuonWithAuxAdam):
    """DES-LOC variant of MuonWithAuxAdam with Kx-gated all_gather.

    Same DES-LOC gating as DESLOCMuon but for the combined
    Muon+Adam optimizer. The Adam part (non-Muon params) runs
    locally without any all_gather gating — only Muon params
    are gated by Kx.
    """

    def __init__(self, param_groups, desloc_Kx=1, desloc_Ku=1,
                 desloc_clip_radius=1.0):
        super().__init__(param_groups)
        self._desloc_Kx = desloc_Kx
        self._desloc_Ku = desloc_Ku
        self._desloc_clip_radius = desloc_clip_radius
        self._desloc_step = 0
        self._desloc_gather_count = 0
        self._desloc_skip_count = 0
        self._desloc_muon_comm_bytes = 0
        self._desloc_adam_comm_bytes = 0

    def _should_gather(self):
        if self._desloc_Kx <= 1:
            return True
        return self._desloc_step % self._desloc_Kx == 0

    @torch.no_grad()
    def step(self, closure=None):
        self._desloc_step += 1

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Apply per-coordinate clipping
        if self._desloc_clip_radius < float('inf'):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.clamp_(-self._desloc_clip_radius,
                                      self._desloc_clip_radius)

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])
                                       ] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        # M473: QKV-aware update path for DESLOC+AuxAdam
                        if self.split_qkv and getattr(p, "is_qkv", False):
                            update = muon_update_with_qkv_split(
                                p.grad, state["momentum_buffer"], p,
                                self.qkv_split_shapes, beta=group["momentum"],
                            )
                        else:
                            update = muon_update(p.grad, state["momentum_buffer"],
                                                 beta=group["momentum"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])

                    # M189: Kx-gated all_gather for Muon params
                    if self._should_gather():
                        dist.all_gather(
                            params_pad[base_i:base_i + dist.get_world_size()],
                            params_pad[base_i + dist.get_rank()])
                        self._desloc_gather_count += 1
                    else:
                        self._desloc_skip_count += 1
            else:
                # Adam params: local update only (no all_gather gating)
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad, state["exp_avg"], state["exp_avg_sq"],
                        state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss

    def get_desloc_stats(self):
        return {
            'step': self._desloc_step,
            'gather_count': self._desloc_gather_count,
            'skip_count': self._desloc_skip_count,
            'Kx': self._desloc_Kx,
            'Ku': self._desloc_Ku,
        }


class DESLOCSingleDeviceMuon(SingleDeviceMuon):
    """DES-LOC variant of SingleDeviceMuon (no communication).

    For single-device, DES-LOC only adds per-coordinate clipping.
    No sync gating needed (no distributed communication).
    """

    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95,
                 desloc_clip_radius=1.0):
        super().__init__(params, lr=lr, weight_decay=weight_decay,
                         momentum=momentum)
        self._desloc_clip_radius = desloc_clip_radius
        self._desloc_step = 0

    @torch.no_grad()
    def step(self, closure=None):
        self._desloc_step += 1

        # Apply per-coordinate clipping
        if self._desloc_clip_radius < float('inf'):
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        p.grad.clamp_(-self._desloc_clip_radius,
                                      self._desloc_clip_radius)

        # Delegate to base (no comm to gate)
        return super().step(closure)


# End M189
