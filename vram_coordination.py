"""GPU ownership / VRAM lease coordination for the native llama.cpp runtime.

This module deliberately keeps native llama.cpp outside ComfyUI's LoadedModel /
ModelPatcher system.  A llama.cpp context is an external CUDA owner from
ComfyUI's point of view: it cannot be partially offloaded by ComfyUI/AIMDO.

The acquisition policy is therefore simple and explicit:

1. Quiesce ComfyUI's transient CUDA/AIMDO buffers and flush the PyTorch cache.
2. Compute one semantic free-VRAM target:
       llama runtime requirement + device headroom
3. Ask ComfyUI to make that much room.
4. Quiesce/flush again and verify *raw driver-visible* free VRAM.
5. If cooperative eviction cannot establish the lease, perform one explicit
   target-device exclusive eviction and verify again.
6. Only then may llama.cpp construct a native context.

There are no stacked request cushions or verification tolerances.  Requests are
rounded upward to the allocator page granularity, while verification is against
the unrounded semantic target.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, asdict
from typing import Any, Optional

_MIB = 1024 * 1024
_GIB = 1024 * 1024 * 1024

# llama.cpp's current --fit default is 1024 MiB per device.  We use the same
# safety floor for an external native runtime unless ComfyUI is configured to
# reserve more.  This is a single post-runtime headroom policy, not an addition
# to separate arbitrary margins.
LLAMA_DEFAULT_DEVICE_MARGIN_BYTES = 1024 * _MIB

# AIMDO VBAR pages are 32 MiB.  Aligning the *request* avoids treating a
# sub-page request difference as a special verification tolerance.
AIMDO_REQUEST_GRANULARITY_BYTES = 32 * _MIB
DEFAULT_REQUEST_GRANULARITY_BYTES = 1 * _MIB

log = logging.getLogger(__name__)


def _device_index(device) -> Optional[int]:
    try:
        idx = getattr(device, "index", None)
        if idx is not None:
            return int(idx)
    except Exception:
        pass
    try:
        text = str(device)
        if ":" in text:
            return int(text.rsplit(":", 1)[1])
    except Exception:
        pass
    return None


def _round_up(value: int, granularity: int) -> int:
    value = max(0, int(value or 0))
    granularity = max(1, int(granularity or 1))
    return int(math.ceil(value / granularity) * granularity) if value else 0


def _sync_cuda(device) -> None:
    try:
        import torch
        if not torch.cuda.is_available():
            return
        idx = _device_index(device)
        torch.cuda.synchronize(idx if idx is not None else device)
    except Exception:
        # The caller will still verify free memory.  Keep CPU/older builds usable.
        return


def _raw_cuda_free(device) -> Optional[int]:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        idx = _device_index(device)
        free, _ = torch.cuda.mem_get_info(idx if idx is not None else device)
        return int(free)
    except Exception:
        return None


def _comfy_free(mm, device) -> int:
    fn = getattr(mm, "get_free_memory", None)
    if not callable(fn):
        return 0
    try:
        value = fn(device)
        if isinstance(value, (tuple, list)):
            value = value[0]
        return max(0, int(value or 0))
    except Exception:
        return 0


def _configured_headroom_bytes(mm, aimdo_enabled: bool) -> tuple[int, dict[str, int]]:
    """Return the one post-runtime margin that the native lease must preserve.

    Sources:
      * llama.cpp's 1024 MiB default fit margin (safety floor)
      * ComfyUI's OS/other-application reserve (`--reserve-vram` or default)
      * ComfyUI DynamicVRAM's explicit `--vram-headroom` extra, when AIMDO is on

    The ComfyUI reserve + DynamicVRAM extra expresses the user's configured
    desired free space.  llama.cpp's fit margin is a floor, so the final margin
    is max(llama default, configured ComfyUI total), not a sum of unrelated
    safety constants.
    """
    comfy_reserve = 0
    fn = getattr(mm, "extra_reserved_memory", None)
    if callable(fn):
        try:
            comfy_reserve = max(0, int(fn() or 0))
        except Exception:
            comfy_reserve = 0

    dynamic_extra = 0
    if aimdo_enabled:
        try:
            from comfy.cli_args import args
            dynamic_extra = max(0, int(float(getattr(args, "vram_headroom", 0) or 0) * _GIB))
        except Exception:
            dynamic_extra = 0

    configured = comfy_reserve + dynamic_extra
    headroom = max(LLAMA_DEFAULT_DEVICE_MARGIN_BYTES, configured)
    return int(headroom), {
        "llama_default_margin_bytes": int(LLAMA_DEFAULT_DEVICE_MARGIN_BYTES),
        "comfy_reserved_bytes": int(comfy_reserve),
        "dynamic_vram_extra_headroom_bytes": int(dynamic_extra),
        "configured_comfy_headroom_bytes": int(configured),
    }


def runtime_target_bytes(estimated_bytes: int, observed_bytes: int = 0) -> tuple[int, str]:
    """Runtime requirement used before a native load.

    The estimator remains the first-load floor.  A measured native high-water can
    only raise it.  Safety headroom is *not* mixed into this number; the lease
    planner owns that separately.
    """
    estimate = max(0, int(estimated_bytes or 0))
    observed = max(0, int(observed_bytes or 0))
    if observed <= 0:
        return estimate, "first-load-estimate"
    target = max(estimate, observed)
    if observed > estimate:
        source = "observed-highwater>estimate"
    elif estimate > observed:
        source = "estimate>observed-highwater"
    else:
        source = "estimate=observed-highwater"
    return int(target), source


@dataclass
class GPULeaseResult:
    runtime_target_bytes: int
    headroom_bytes: int
    free_target_bytes: int
    request_target_bytes: int
    request_granularity_bytes: int
    raw_free_before_bytes: Optional[int] = None
    raw_free_after_cleanup_bytes: Optional[int] = None
    raw_free_after_cooperative_bytes: Optional[int] = None
    raw_free_after_bytes: Optional[int] = None
    comfy_free_after_bytes: int = 0
    satisfied: bool = False
    strategy: str = "none"
    cooperative_eviction_called: bool = False
    exclusive_eviction_called: bool = False
    transient_cleanup_called: bool = False
    soft_cache_flush_called: bool = False
    aimdo: bool = False
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    headroom_sources: Optional[dict[str, int]] = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class GPUMemoryLeaseManager:
    """Acquire physical VRAM ownership for an all-or-nothing native runtime."""

    def __init__(self, mm, *, aimdo_enabled: bool = False):
        self.mm = mm
        self.aimdo_enabled = bool(aimdo_enabled)

    def plan(self, runtime_required: int) -> GPULeaseResult:
        runtime_required = max(0, int(runtime_required or 0))
        headroom, sources = _configured_headroom_bytes(self.mm, self.aimdo_enabled)
        free_target = runtime_required + headroom if runtime_required > 0 else 0
        granularity = AIMDO_REQUEST_GRANULARITY_BYTES if self.aimdo_enabled else DEFAULT_REQUEST_GRANULARITY_BYTES
        request_target = _round_up(free_target, granularity)
        return GPULeaseResult(
            runtime_target_bytes=runtime_required,
            headroom_bytes=headroom if runtime_required > 0 else 0,
            free_target_bytes=free_target,
            request_target_bytes=request_target,
            request_granularity_bytes=granularity,
            satisfied=(runtime_required <= 0),
            aimdo=self.aimdo_enabled,
            headroom_sources=sources,
        )

    def _cleanup_transients(self, device, result: GPULeaseResult) -> None:
        """Mirror the stable parts of ComfyUI's own AIMDO node-boundary cleanup."""
        result.transient_cleanup_called = True

        if self.aimdo_enabled:
            # Current ComfyUI calls reset_cast_buffers() and prefetch cleanup at
            # every node boundary under AIMDO.  Use the public module functions
            # when present, but do not depend on comfy_aimdo's temporary
            # VRAMBuffer API directly.
            reset = getattr(self.mm, "reset_cast_buffers", None)
            if callable(reset):
                try:
                    reset()
                except Exception as e:
                    # Continue to the explicit cache flush and raw verification;
                    # a cleanup failure should be visible but not silently abort
                    # before ComfyUI gets a chance to evict models.
                    result.error = f"reset_cast_buffers: {e}"
            try:
                import comfy.model_prefetch as model_prefetch
                cleanup = getattr(model_prefetch, "cleanup_prefetch_queues", None)
                if callable(cleanup):
                    cleanup()
            except Exception as e:
                if result.error is None:
                    result.error = f"cleanup_prefetch_queues: {e}"

        # AIMDO explicitly recommends fully flushing the PyTorch caching allocator
        # before a new model run.  Do this on every fresh native lease, not only
        # when a heuristic thinks the cache is large enough to matter.
        flush = getattr(self.mm, "soft_empty_cache", None)
        if callable(flush):
            try:
                flush()
                result.soft_cache_flush_called = True
            except Exception as e:
                if result.error is None:
                    result.error = f"soft_empty_cache: {e}"
        _sync_cuda(device)

    @staticmethod
    def _effective_free(device, mm) -> tuple[int, Optional[int], int]:
        raw = _raw_cuda_free(device)
        comfy = _comfy_free(mm, device)
        # Native llama.cpp cares about raw driver-visible room.  Fall back to the
        # Comfy number only when CUDA mem_get_info is unavailable.
        effective = int(raw if raw is not None else comfy)
        return effective, raw, comfy

    def acquire(self, runtime_required: int, device) -> dict[str, Any]:
        result = self.plan(runtime_required)
        if result.satisfied:
            return result.as_dict()

        started = time.perf_counter()
        _, raw_initial, _ = self._effective_free(device, self.mm)
        result.raw_free_before_bytes = raw_initial

        # Every fresh native model is a GPU ownership transition, even if a raw
        # snapshot initially appears sufficient.
        self._cleanup_transients(device, result)
        effective, raw, comfy = self._effective_free(device, self.mm)
        result.raw_free_after_cleanup_bytes = raw
        if effective >= result.free_target_bytes:
            result.raw_free_after_bytes = raw
            result.comfy_free_after_bytes = comfy
            result.satisfied = True
            result.strategy = "clean-fast-path"
            result.elapsed_seconds = time.perf_counter() - started
            return result.as_dict()

        # Cooperative acquisition: ask ComfyUI's own model manager to make room.
        # Because the allocator cache was just flushed, ComfyUI's free-memory
        # accounting is much closer to physical driver-visible room here.
        result.cooperative_eviction_called = True
        result.strategy = "cooperative-comfy-eviction"
        try:
            self.mm.free_memory(result.request_target_bytes, device)
        except Exception as e:
            result.error = f"cooperative free_memory: {e}"
        self._cleanup_transients(device, result)
        effective, raw, comfy = self._effective_free(device, self.mm)
        result.raw_free_after_cooperative_bytes = raw
        if effective >= result.free_target_bytes:
            result.raw_free_after_bytes = raw
            result.comfy_free_after_bytes = comfy
            result.satisfied = True
            result.elapsed_seconds = time.perf_counter() - started
            return result.as_dict()

        # Exclusive fallback: the external runtime cannot share partially-loaded
        # ComfyUI models safely when the cooperative lease is still short.  Make
        # that ownership transition explicit instead of adding more magic margins.
        result.exclusive_eviction_called = True
        result.strategy = "exclusive-target-gpu-eviction"
        try:
            self.mm.free_memory(1e30, device)
        except Exception as e:
            result.error = f"exclusive free_memory: {e}"
        self._cleanup_transients(device, result)
        effective, raw, comfy = self._effective_free(device, self.mm)
        result.raw_free_after_bytes = raw
        result.comfy_free_after_bytes = comfy
        result.satisfied = bool(effective >= result.free_target_bytes)
        result.elapsed_seconds = time.perf_counter() - started
        return result.as_dict()

    def acquire_exclusive(self, device, *, minimum_runtime_bytes: int = 0) -> dict[str, Any]:
        """Fully clear one target GPU, used when multi-GPU distribution is unknown.

        llama.cpp's initial Layer/Row/Tensor split has a per-device allocation
        distribution that our GGUF-level estimator cannot determine exactly.
        Rather than inventing a per-device split estimate, first-load multi-GPU
        operation takes an explicit exclusive lease on each participating device.
        """
        minimum_runtime = max(0, int(minimum_runtime_bytes or 0))
        result = self.plan(minimum_runtime)
        if minimum_runtime <= 0:
            headroom, sources = _configured_headroom_bytes(self.mm, self.aimdo_enabled)
            granularity = AIMDO_REQUEST_GRANULARITY_BYTES if self.aimdo_enabled else DEFAULT_REQUEST_GRANULARITY_BYTES
            result.headroom_bytes = int(headroom)
            result.free_target_bytes = int(headroom)
            result.request_target_bytes = _round_up(headroom, granularity)
            result.request_granularity_bytes = int(granularity)
            result.headroom_sources = sources
        started = time.perf_counter()
        _, raw_initial, _ = self._effective_free(device, self.mm)
        result.raw_free_before_bytes = raw_initial
        result.exclusive_eviction_called = True
        result.strategy = "exclusive-target-gpu-eviction"
        try:
            self.mm.free_memory(1e30, device)
        except Exception as e:
            result.error = f"exclusive free_memory: {e}"
        self._cleanup_transients(device, result)
        effective, raw, comfy = self._effective_free(device, self.mm)
        result.raw_free_after_bytes = raw
        result.comfy_free_after_bytes = comfy
        # With no known per-device runtime requirement, require at least the
        # configured headroom to remain physically available after cleanup.
        verify_target = result.free_target_bytes
        result.satisfied = bool(effective >= verify_target)
        result.elapsed_seconds = time.perf_counter() - started
        return result.as_dict()


def lease_failure_message(room: dict[str, Any]) -> str:
    runtime = int((room or {}).get("runtime_target_bytes") or 0)
    headroom = int((room or {}).get("headroom_bytes") or 0)
    free_target = int((room or {}).get("free_target_bytes") or 0)
    request_target = int((room or {}).get("request_target_bytes") or 0)
    raw = (room or {}).get("raw_free_after_bytes")
    raw_text = f"{float(raw) / _MIB:.1f}" if raw is not None else "unknown"
    shortfall = max(0, free_target - int(raw or 0)) if raw is not None else free_target
    return (
        "Local GGUF LLM could not acquire a safe GPU memory lease before native load: "
        f"runtime={runtime / _MIB:.1f} MiB, headroom={headroom / _MIB:.1f} MiB, "
        f"free_target={free_target / _MIB:.1f} MiB, request_target={request_target / _MIB:.1f} MiB, "
        f"raw_free_after={raw_text} MiB, shortfall={shortfall / _MIB:.1f} MiB, "
        f"strategy={(room or {}).get('strategy', 'unknown')}. "
        "ComfyUI/AIMDO could not establish the driver-visible room required by the native llama.cpp lease; "
        "the native model was not loaded."
    )
