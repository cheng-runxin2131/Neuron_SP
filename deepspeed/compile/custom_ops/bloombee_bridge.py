import os
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

_HIVEMIND_AVAILABLE = False
try:
    import hivemind
    from hivemind.dht import DHT, DHTValue
    from hivemind.utils import get_dht_time
    _HIVEMIND_AVAILABLE = True
except ImportError:
    pass

DHT_NEURONSP_PREFIX = "neuronsp_gpu_caps"
DHT_CAPABILITY_EXPIRY_SECS = 120


@dataclass
class PeerCapability:
    peer_id: str
    rank: int
    device_name: str
    compute_capability: Tuple[int, int]
    memory_total_gb: float
    memory_bandwidth_gbps: float
    tier: int
    nvlink_available: bool
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d['compute_capability'] = list(d['compute_capability'])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'PeerCapability':
        d['compute_capability'] = tuple(d['compute_capability'])
        return cls(**d)


def announce_capability_via_dht(dht, capability):
    if not _HIVEMIND_AVAILABLE:
        logger.warning("[DHTBridge] hivemind not available, skipping DHT announcement")
        return False
    capability.timestamp = time.time()
    key = f"{DHT_NEURONSP_PREFIX}:{capability.peer_id}"
    try:
        dht.store(
            key=key,
            value=json.dumps(capability.to_dict()),
            expiration_time=get_dht_time() + DHT_CAPABILITY_EXPIRY_SECS,
        )
        logger.info(f"[DHTBridge] Announced capability: {capability.device_name} "
                     f"tier={capability.tier} rank={capability.rank}")
        return True
    except Exception as e:
        logger.warning(f"[DHTBridge] Failed to announce: {e}")
        return False


def discover_peers_via_dht(dht, timeout_secs=10.0):
    if not _HIVEMIND_AVAILABLE:
        return []
    try:
        result = dht.get(f"{DHT_NEURONSP_PREFIX}:*", latest=True)
        if result is None:
            return []
        peers = []
        if isinstance(result, dict):
            for key, (value, _expiry) in result.items():
                try:
                    cap = PeerCapability.from_dict(json.loads(value))
                    if time.time() - cap.timestamp < DHT_CAPABILITY_EXPIRY_SECS:
                        peers.append(cap)
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
        return peers
    except Exception as e:
        logger.warning(f"[DHTBridge] Discovery failed: {e}")
        return []


@dataclass
class RemoteA2AConfig:
    use_compression: bool = True
    compression_codec: str = "fp16"
    max_chunk_bytes: int = 64 * 1024 * 1024
    timeout_secs: float = 30.0
    retry_count: int = 3


def _compress_fp16(tensor, meta):
    import numpy as np
    compressed = tensor.half().cpu().numpy().tobytes()
    return compressed, meta


def _compress_int8(tensor, meta):
    import torch
    import numpy as np
    orig_shape = tensor.shape
    flat = tensor.reshape(-1, orig_shape[-1])
    absmax = flat.abs().amax(dim=0)
    scales = absmax / 127.0
    scales = scales.clamp(min=1e-8)
    quantized = (flat / scales.unsqueeze(0)).clamp(-127, 127).to(torch.int8)
    compressed = quantized.cpu().numpy().tobytes()
    meta["scales"] = scales.cpu().tolist()
    meta["orig_shape"] = list(orig_shape)
    meta["flat_shape"] = list(flat.shape)
    return compressed, meta


def _compress_none(tensor, meta):
    import numpy as np
    meta["codec"] = "none"
    return tensor.cpu().numpy().tobytes(), meta


def _decompress_fp16(data, meta, device):
    import torch
    import numpy as np
    shape = meta["shape"]
    original_dtype = getattr(torch, meta["dtype"].replace("torch.", ""))
    arr = np.frombuffer(data, dtype=np.float16).reshape(shape)
    return torch.from_numpy(arr.copy()).to(dtype=original_dtype, device=device)


def _decompress_int8(data, meta, device):
    import torch
    import numpy as np
    orig_shape = meta.get("orig_shape", meta["shape"])
    hidden_dim = orig_shape[-1]
    flat_shape = tuple(meta["flat_shape"]) if "flat_shape" in meta else (-1, hidden_dim)
    arr = np.frombuffer(data, dtype=np.int8).reshape(flat_shape)
    tensor = torch.from_numpy(arr.copy()).to(dtype=torch.float32, device=device)
    scales = torch.tensor(meta["scales"], dtype=torch.float32, device=device)
    tensor = tensor * scales.unsqueeze(0)
    original_dtype = getattr(torch, meta["dtype"].replace("torch.", ""))
    return tensor.reshape(orig_shape).to(original_dtype)


def _decompress_none(data, meta, device):
    import torch
    import numpy as np
    shape = meta["shape"]
    original_dtype = getattr(torch, meta["dtype"].replace("torch.", ""))
    arr = np.frombuffer(data, dtype=np.float32).reshape(shape)
    return torch.from_numpy(arr.copy()).to(dtype=original_dtype, device=device)


_COMPRESS_DISPATCH = {
    "fp16": _compress_fp16,
    "int8": _compress_int8,
}

_DECOMPRESS_DISPATCH = {
    "fp16": _decompress_fp16,
    "int8": _decompress_int8,
    "none": _decompress_none,
}


class TensorCodec:

    def __init__(self, codec="fp16"):
        self._codec = codec

    def compress(self, tensor):
        import torch
        meta = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "codec": self._codec,
        }
        compress_fn = _COMPRESS_DISPATCH.get(self._codec)
        if compress_fn is None:
            return _compress_none(tensor, meta)
        if self._codec == "fp16" and tensor.dtype != torch.float32:
            return _compress_none(tensor, meta)
        return compress_fn(tensor, meta)

    def decompress(self, data, meta, device):
        codec = meta.get("codec", "none")
        decompress_fn = _DECOMPRESS_DISPATCH.get(codec, _decompress_none)
        return decompress_fn(data, meta, device)


class RemoteA2AProxy:

    def __init__(self, config=None):
        self.config = config or RemoteA2AConfig()
        self._codec = TensorCodec(self.config.compression_codec)
        self._stats = {"sends": 0, "recvs": 0, "bytes_sent": 0, "bytes_recv": 0}

    def compress_tensor(self, tensor):
        return self._codec.compress(tensor)

    def decompress_tensor(self, data, meta, device):
        return self._codec.decompress(data, meta, device)

    def get_stats(self):
        return dict(self._stats)


@dataclass
class HeteroSyncConfig:
    base_Kx: int = 1
    min_Kx: int = 1
    max_Kx: int = 32
    straggler_threshold: float = 2.0
    straggler_Kx_multiplier: float = 2.0
    unreachable_grace_steps: int = 5
    adaptive_enabled: bool = False


class HeteroSyncGate:

    def __init__(self, config=None):
        self.config = config or HeteroSyncConfig()
        self._effective_Kx = self.config.base_Kx
        self._step = 0
        self._peer_last_seen: Dict[int, float] = {}
        self._grad_norms: List[float] = []

    def should_sync(self, step, is_last_pass=False):
        self._step = step
        if is_last_pass:
            return True
        if self._effective_Kx <= 1:
            return True
        return (step % self._effective_Kx) == 0

    def update_peer_status(self, rank, reachable):
        if reachable:
            self._peer_last_seen[rank] = time.time()

    def update_grad_norm(self, grad_norm):
        self._grad_norms.append(grad_norm)
        if len(self._grad_norms) > 100:
            self._grad_norms = self._grad_norms[-100:]

    def _detect_grad_explosion(self):
        if len(self._grad_norms) < 10:
            return False
        first_half = sum(self._grad_norms[-10:-5]) / 5
        second_half = sum(self._grad_norms[-5:]) / 5
        return second_half > first_half * 2.0

    def _compute_grad_cv(self):
        if len(self._grad_norms) < 20:
            return None
        recent = self._grad_norms[-20:]
        mean_norm = sum(recent) / len(recent)
        variance = sum((x - mean_norm) ** 2 for x in recent) / len(recent)
        return (variance ** 0.5) / (mean_norm + 1e-8)

    def adapt_Kx(self, tier_infos=None):
        if not self.config.adaptive_enabled:
            return

        new_Kx = self._effective_Kx

        if tier_infos and len(tier_infos) > 1:
            scores = [t.compute_score() if hasattr(t, 'compute_score') else 0.0 for t in tier_infos.values()]
            if min(scores) > 0 and max(scores) / min(scores) > self.config.straggler_threshold:
                new_Kx = int(new_Kx * self.config.straggler_Kx_multiplier)

        cv = self._compute_grad_cv()
        if cv is not None:
            if cv < 0.1:
                new_Kx = min(new_Kx * 2, self.config.max_Kx)
            elif cv > 0.5:
                new_Kx = max(new_Kx // 2, self.config.min_Kx)

        if self._detect_grad_explosion():
            new_Kx = self.config.min_Kx
            logger.warning(
                f"[HeteroSync] Gradient explosion detected. Forcing Kx={new_Kx} for recovery.")

        self._effective_Kx = max(self.config.min_Kx, min(new_Kx, self.config.max_Kx))

    @property
    def effective_Kx(self):
        return self._effective_Kx


class RemoteA2ADoubleBuffer:

    def __init__(self, config=None):
        self._config = config or RemoteA2AConfig()
        self._proxy = RemoteA2AProxy(self._config)
        self.selector = 0
        self._send_buffers = [None, None]
        self._recv_buffers = [None, None]
        self._swap_count = 0

    def allocate(self, shape, dtype):
        import torch as _t
        for i in range(2):
            self._send_buffers[i] = _t.empty(shape, dtype=dtype, device='cpu')
            self._recv_buffers[i] = _t.empty(shape, dtype=dtype, device='cpu')

    def current_send(self):
        return self._send_buffers[self.selector]

    def alternate_send(self):
        return self._send_buffers[self.selector ^ 1]

    def current_recv(self):
        return self._recv_buffers[self.selector]

    def alternate_recv(self):
        return self._recv_buffers[self.selector ^ 1]

    def swap(self):
        self.selector ^= 1
        self._swap_count += 1

    def swap_count(self):
        return self._swap_count

    def compress_and_stage(self, tensor):
        return self._proxy.compress_tensor(tensor)

    def decompress_from_stage(self, data, meta, device):
        return self._proxy.decompress_tensor(data, meta, device)

    def free(self):
        self._send_buffers = [None, None]
        self._recv_buffers = [None, None]
        self.selector = 0
        self._swap_count = 0
