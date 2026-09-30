"""Seed normalization helpers shared by ComfyUI and OpenAI-compatible callers.

The native llama.cpp sampler accepts a 32-bit seed while ComfyUI exposes an
unsigned 64-bit seed.  Keep that conversion in one place and give automatically
randomized API requests a small recent-history guard so a long-running service
cannot accidentally recycle the same *effective* sampler seed even when two
64-bit request seeds hash to the same 32-bit value.
"""
from __future__ import annotations

from collections import deque
import secrets
import threading
from typing import Callable

MAX_REQUEST_SEED = 0xFFFFFFFFFFFFFFFF
LLAMA_DEFAULT_SEED = 0xFFFFFFFF
MAX_DETERMINISTIC_LLAMA_SEED = 0xFFFFFFFE


def llama_seed_from_u64(seed: int) -> int:
    """Map an unsigned 64-bit request seed to llama.cpp's deterministic uint32 seed.

    SplitMix64 uses all 64 input bits instead of simply truncating high bits.
    ``0xFFFFFFFF`` is llama.cpp's reserved random/default sentinel, so remap that
    one output to the highest deterministic value.
    """
    x = int(seed) & MAX_REQUEST_SEED
    z = (x + 0x9E3779B97F4A7C15) & MAX_REQUEST_SEED
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MAX_REQUEST_SEED
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MAX_REQUEST_SEED
    z ^= z >> 31
    out = int(z & 0xFFFFFFFF)
    if out == LLAMA_DEFAULT_SEED:
        out = MAX_DETERMINISTIC_LLAMA_SEED
    return out


class RecentSamplerSeedAllocator:
    """Allocate random request seeds without recently repeating sampler seeds.

    llama.cpp ultimately consumes only 32 bits.  A random 64-bit request seed can
    therefore collide after the 64->32 conversion even though the 64-bit values
    differ.  The birthday probability becomes non-trivial on a long-running API
    service, so automatically-random requests keep a bounded recent set of actual
    sampler seeds and redraw on collision.

    Explicit user-provided seeds remain fully deterministic/reproducible; their
    effective sampler seed is merely remembered so a following auto-random request
    does not accidentally select the same recent native seed.
    """

    def __init__(
        self,
        *,
        recent_limit: int = 65536,
        randbits: Callable[[int], int] | None = None,
        mapper: Callable[[int], int] = llama_seed_from_u64,
    ) -> None:
        self.recent_limit = max(1, int(recent_limit))
        self._randbits = randbits or secrets.randbits
        self._mapper = mapper
        self._recent: deque[int] = deque()
        self._recent_set: set[int] = set()
        self._lock = threading.Lock()

    def allocate(self) -> tuple[int, int]:
        with self._lock:
            # At 65,536 retained values in a ~2^32 deterministic seed space, a
            # redraw almost always succeeds immediately. Bound the retry loop and
            # fail loudly rather than silently violating the no-repeat invariant if
            # the entropy source is broken.
            for _ in range(256):
                request_seed = int(self._randbits(64)) & MAX_REQUEST_SEED
                sampler_seed = int(self._mapper(request_seed)) & 0xFFFFFFFF
                if sampler_seed == LLAMA_DEFAULT_SEED:
                    sampler_seed = MAX_DETERMINISTIC_LLAMA_SEED
                if sampler_seed not in self._recent_set:
                    self._remember_locked(sampler_seed)
                    return request_seed, sampler_seed
            raise RuntimeError("Unable to allocate a fresh llama.cpp sampler seed")

    def _remember_locked(self, sampler_seed: int) -> None:
        sampler_seed = int(sampler_seed) & 0xFFFFFFFF
        if sampler_seed in self._recent_set:
            return
        if len(self._recent) >= self.recent_limit:
            old = self._recent.popleft()
            self._recent_set.discard(old)
        self._recent.append(sampler_seed)
        self._recent_set.add(sampler_seed)

    def remember(self, sampler_seed: int) -> None:
        """Reserve a recently used explicit sampler seed from auto-random reuse."""
        with self._lock:
            self._remember_locked(sampler_seed)


def parse_openai_seed(raw, allocator: RecentSamplerSeedAllocator) -> tuple[int, int, bool]:
    """Resolve OpenAI/local-client seed semantics.

    ``None`` and ``-1`` mean automatic randomization.  Explicit non-negative
    integers, including zero, remain deterministic.  The returned tuple is
    ``(request_seed_u64, llama_sampler_seed_u32, randomized)``.
    """
    if raw is None:
        request_seed, sampler_seed = allocator.allocate()
        return request_seed, sampler_seed, True
    if isinstance(raw, bool):
        raise ValueError("OpenAI seed must be an integer, null, or -1 for random")
    if isinstance(raw, float) and not raw.is_integer():
        raise ValueError("OpenAI seed must be an integer, null, or -1 for random")
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("OpenAI seed must be an integer, null, or -1 for random") from exc
    if value == -1:
        request_seed, sampler_seed = allocator.allocate()
        return request_seed, sampler_seed, True
    if value < 0:
        raise ValueError("OpenAI seed must be non-negative, null, or -1 for random")
    if value > MAX_REQUEST_SEED:
        raise ValueError(f"OpenAI seed must be <= {MAX_REQUEST_SEED}")
    sampler_seed = llama_seed_from_u64(value)
    allocator.remember(sampler_seed)
    return value, sampler_seed, False
