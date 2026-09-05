"""GPU ownership / VRAM lease coordination for the native llama.cpp runtime.

The native llama.cpp context is an external CUDA owner from ComfyUI's point of
view.  It is intentionally not registered as a ComfyUI LoadedModel because
ComfyUI/AIMDO cannot partially migrate its model/context/compute allocations.

Admission is deliberately staged so the common case is cheap:

1. If raw driver-visible VRAM already satisfies the lease, do nothing.
2. If PyTorch's *unused cache alone* can satisfy the shortfall, release only
   that cache.
3. Otherwise ask ComfyUI for a *logical-free* target that is compensated by
   its current PyTorch allocator slack. ComfyUI intentionally stops eviction on
   (raw CUDA free + allocator slack); llama.cpp requires raw CUDA free. Adding
   the current allocator slack to the request aligns those two metrics without
   inventing a tolerance or safety margin.
4. Re-measure after cooperative eviction. If allocator slack changed enough to
   leave raw VRAM short, issue one corrected cooperative request from the new
   measurement.
5. Under AIMDO only, if cooperative admission still cannot establish the lease,
   clear stale node-boundary cast/prefetch state and retry once.
6. Only then take an explicit exclusive target-device lease.
7. Before declaring failure, synchronize once and make a final raw-memory check.

The semantic lease remains simple:
    native runtime requirement + one device headroom = raw-free target

No verification tolerances or stacked safety margins are used.  AIMDO requests
are only rounded upward to its physical VBAR page granularity.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, asdict
from typing import Any, Optional

_MIB = 1024 * 1024
_GIB = 1024 * 1024 * 1024

# llama.cpp's current --fit default target margin is 1024 MiB per device.
LLAMA_DEFAULT_DEVICE_MARGIN_BYTES = 1024 * _MIB

# AIMDO VBAR pages are 32 MiB; only the request is aligned, never the semantic
# target used for admission verification.
AIMDO_REQUEST_GRANULARITY_BYTES = 32 * _MIB
DEFAULT_REQUEST_GRANULARITY_BYTES = 1 * _MIB

# Avoid a cache operation for microscopic allocator bookkeeping.  This is not a
# safety margin: the cache path is selected only when reclaimable cache already
# covers the entire semantic shortfall.
MIN_CACHE_RECLAIM_BYTES = 8 * _MIB

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


def _raw_cuda_total(device) -> Optional[int]:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        idx = _device_index(device)
        _free, total = torch.cuda.mem_get_info(idx if idx is not None else device)
        return int(total)
    except Exception:
        return None


def _safe_model_label(item) -> str:
    """Best-effort label for diagnostics only; never calls model lifecycle methods."""
    try:
        wrapped = getattr(item, "model", None)
        patcher_name = type(wrapped).__name__ if wrapped is not None else type(item).__name__
        inner = getattr(wrapped, "model", None) if wrapped is not None else None
        inner_name = type(inner).__name__ if inner is not None else ""
        device = getattr(item, "device", None)
        if device is None and wrapped is not None:
            device = getattr(wrapped, "load_device", None)
        core = f"{patcher_name}/{inner_name}" if inner_name else patcher_name
        return f"{core}@{device}" if device is not None else core
    except Exception:
        return type(item).__name__


def _loaded_models_snapshot(mm, device, *, limit: int = 8) -> dict[str, Any]:
    """Read-only ComfyUI loaded-model inventory for lease diagnostics."""
    try:
        loaded = list(getattr(mm, "current_loaded_models", []) or [])
    except Exception:
        loaded = []
    target_idx = _device_index(device)
    same = []
    labels = []
    for item in loaded:
        label = _safe_model_label(item)
        labels.append(label)
        try:
            item_device = getattr(item, "device", None)
            if item_device is None:
                wrapped = getattr(item, "model", None)
                item_device = getattr(wrapped, "load_device", None) if wrapped is not None else None
            if target_idx is not None and _device_index(item_device) == target_idx:
                same.append(label)
        except Exception:
            pass
    return {
        "count": len(loaded),
        "same_device_count": len(same),
        "labels": labels[:limit],
        "same_device_labels": same[:limit],
        "truncated": max(0, len(labels) - limit),
        "same_device_truncated": max(0, len(same) - limit),
    }


def _memory_snapshot(mm, device) -> dict[str, Any]:
    raw = _raw_cuda_free(device)
    total = _raw_cuda_total(device)
    comfy, reclaimable = _comfy_memory(mm, device)
    return {
        "raw_free_bytes": raw,
        "total_bytes": total,
        "raw_used_bytes": (max(0, int(total) - int(raw)) if total is not None and raw is not None else None),
        "comfy_logical_free_bytes": int(comfy),
        "torch_reclaimable_bytes": int(reclaimable),
        "logical_minus_raw_bytes": (max(0, int(comfy) - int(raw)) if raw is not None else None),
        "loaded_models": _loaded_models_snapshot(mm, device),
    }


def _unloaded_labels(items, *, limit: int = 8) -> list[str]:
    try:
        values = list(items or [])
    except Exception:
        return []
    labels = [_safe_model_label(item) for item in values[:limit]]
    if len(values) > limit:
        labels.append(f"+{len(values) - limit} more")
    return labels


def _comfy_memory(mm, device) -> tuple[int, int]:
    """Return (Comfy logical-free, inactive PyTorch allocator slack) bytes."""
    fn = getattr(mm, "get_free_memory", None)
    if not callable(fn):
        return 0, 0
    try:
        value = fn(device, torch_free_too=True)
        if isinstance(value, (tuple, list)) and len(value) >= 2:
            return max(0, int(value[0] or 0)), max(0, int(value[1] or 0))
        return max(0, int(value or 0)), 0
    except TypeError:
        try:
            value = fn(device)
            if isinstance(value, (tuple, list)):
                value = value[0]
            return max(0, int(value or 0)), 0
        except Exception:
            return 0, 0
    except Exception:
        return 0, 0


def _configured_headroom_bytes(mm, aimdo_enabled: bool) -> tuple[int, dict[str, int]]:
    """Return the single post-runtime free-space margin for the native lease."""
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
    """Runtime requirement before a native load, excluding lease headroom."""
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
    cooperative_request_target_bytes: int = 0
    cooperative_torch_slack_bytes: int = 0
    cooperative_retry_request_target_bytes: int = 0
    raw_free_before_bytes: Optional[int] = None
    raw_free_after_cache_bytes: Optional[int] = None
    raw_free_after_cooperative_bytes: Optional[int] = None
    raw_free_after_aimdo_cleanup_bytes: Optional[int] = None
    raw_free_after_bytes: Optional[int] = None
    comfy_free_before_bytes: int = 0
    torch_reclaimable_before_bytes: int = 0
    comfy_free_after_bytes: int = 0
    torch_reclaimable_after_bytes: int = 0
    stage_memory: Optional[dict[str, Any]] = None
    stage_timings_seconds: Optional[dict[str, float]] = None
    unloaded_models: Optional[dict[str, list[str]]] = None
    satisfied: bool = False
    strategy: str = "none"
    cache_reclaim_called: bool = False
    cache_reclaim_stage: str = "none"
    cooperative_eviction_called: bool = False
    cooperative_unloaded_count: int = 0
    aimdo_cleanup_called: bool = False
    aimdo_retry_called: bool = False
    exclusive_eviction_called: bool = False
    final_sync_called: bool = False
    aimdo: bool = False
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    headroom_sources: Optional[dict[str, int]] = None
    reduced_headroom_fallback: bool = False
    available_headroom_bytes: int = 0
    preferred_headroom_shortfall_bytes: int = 0
    admission_warning: Optional[str] = None

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

    def _cooperative_request(self, result: GPULeaseResult, torch_slack_bytes: int) -> int:
        """Translate a raw-driver lease target into ComfyUI's logical-free metric.

        ComfyUI's get_free_memory() reports raw CUDA free plus inactive bytes in
        PyTorch's allocator. free_memory() uses that combined value as its stop
        condition. llama.cpp is outside PyTorch and can only consume raw CUDA
        free, so requesting the raw target directly lets ComfyUI stop early by
        exactly the allocator slack. Compensating the request by the *measured*
        slack aligns ComfyUI's stop condition with our raw target.
        """
        slack = max(0, int(torch_slack_bytes or 0))
        logical_target = int(result.free_target_bytes) + slack
        return _round_up(logical_target, int(result.request_granularity_bytes or 1))

    @staticmethod
    def _effective_free(device, mm) -> tuple[int, Optional[int], int, int]:
        raw = _raw_cuda_free(device)
        comfy, reclaimable = _comfy_memory(mm, device)
        effective = int(raw if raw is not None else comfy)
        return effective, raw, comfy, reclaimable

    def _flush_cache(self, result: GPULeaseResult, *, stage: str) -> None:
        flush = getattr(self.mm, "soft_empty_cache", None)
        if not callable(flush):
            return
        try:
            started = time.perf_counter()
            flush()
            if result.stage_timings_seconds is not None:
                result.stage_timings_seconds[f"cache_flush_{stage}"] = time.perf_counter() - started
            result.cache_reclaim_called = True
            result.cache_reclaim_stage = str(stage or "unspecified")
        except Exception as e:
            if result.error is None:
                result.error = f"soft_empty_cache: {e}"

    def _cleanup_aimdo_transients(self, device, result: GPULeaseResult) -> None:
        """Use ComfyUI's own node-boundary AIMDO cleanup, only on failed admission.

        ComfyUI already executes these operations after every node.  Repeating
        them on the normal LLM path wastes work and discards warmed prefetch/CUDA
        graph state, so this is intentionally a recovery tier only.
        """
        if not self.aimdo_enabled:
            return
        result.aimdo_cleanup_called = True
        reset = getattr(self.mm, "reset_cast_buffers", None)
        if callable(reset):
            try:
                reset()
            except Exception as e:
                if result.error is None:
                    result.error = f"reset_cast_buffers: {e}"
        try:
            import comfy.model_prefetch as model_prefetch
            cleanup = getattr(model_prefetch, "cleanup_prefetch_queues", None)
            if callable(cleanup):
                cleanup()
        except Exception as e:
            if result.error is None:
                result.error = f"cleanup_prefetch_queues: {e}"
        try:
            import comfy_aimdo.model_vbar as model_vbar
            reset_limits = getattr(model_vbar, "vbars_reset_watermark_limits", None)
            if callable(reset_limits):
                reset_limits()
        except Exception as e:
            if result.error is None:
                result.error = f"vbars_reset_watermark_limits: {e}"
        # This recovery tier may have touched side streams / VBAR mappings.
        _sync_cuda(device)

    def _finish_if_satisfied(self, result: GPULeaseResult, device, strategy: str) -> bool:
        effective, raw, comfy, reclaimable = self._effective_free(device, self.mm)
        result.raw_free_after_bytes = raw
        result.comfy_free_after_bytes = comfy
        result.torch_reclaimable_after_bytes = reclaimable
        if effective >= result.free_target_bytes:
            result.satisfied = True
            result.strategy = strategy
            return True
        return False

    def _admit_reduced_headroom_if_runtime_fits(self, result: GPULeaseResult, raw_free: Optional[int]) -> bool:
        """Last-resort admission when the runtime fits but preferred headroom does not.

        This is deliberately evaluated only after the normal cooperative/AIMDO/
        exclusive reclamation path has been exhausted.  It never lowers the
        runtime requirement itself.  If the driver-visible free memory is at
        least the full runtime target, llama.cpp is allowed to attempt the load
        and the caller receives a prominent warning that the preferred lease
        margin could not be preserved.
        """
        if result.satisfied or raw_free is None:
            return False
        runtime = max(0, int(result.runtime_target_bytes or 0))
        raw = max(0, int(raw_free or 0))
        if runtime <= 0 or raw < runtime:
            return False

        available_headroom = max(0, raw - runtime)
        preferred_headroom = max(0, int(result.headroom_bytes or 0))
        shortfall = max(0, preferred_headroom - available_headroom)
        result.satisfied = True
        result.reduced_headroom_fallback = True
        result.available_headroom_bytes = int(available_headroom)
        result.preferred_headroom_shortfall_bytes = int(shortfall)
        result.strategy = "reduced-headroom-runtime-fit"
        result.admission_warning = (
            "Local GGUF LLM could not preserve the preferred GPU lease headroom, "
            "but the full estimated native runtime still fits in driver-visible VRAM. "
            f"Proceeding with reduced headroom: runtime={runtime / _MIB:.1f} MiB, "
            f"raw_free={raw / _MIB:.1f} MiB, available_headroom={available_headroom / _MIB:.1f} MiB, "
            f"preferred_headroom={preferred_headroom / _MIB:.1f} MiB, "
            f"preferred_shortfall={shortfall / _MIB:.1f} MiB. "
            "The native load will still fail normally if the model cannot actually fit."
        )
        return True

    def acquire(self, runtime_required: int, device) -> dict[str, Any]:
        result = self.plan(runtime_required)
        if result.satisfied:
            return result.as_dict()

        started = time.perf_counter()
        result.stage_memory = {}
        result.stage_timings_seconds = {}
        result.unloaded_models = {}

        def snap(stage: str) -> dict[str, Any]:
            t0 = time.perf_counter()
            value = _memory_snapshot(self.mm, device)
            result.stage_memory[str(stage)] = value
            result.stage_timings_seconds[f"snapshot_{stage}"] = time.perf_counter() - t0
            return value

        # Cheapest possible admission probe: raw CUDA free memory requires no
        # allocator-stat scan and no synchronization.  If it already satisfies
        # the lease, there is nothing ComfyUI or PyTorch can usefully reclaim.
        probe_started = time.perf_counter()
        raw = _raw_cuda_free(device)
        result.stage_timings_seconds["raw_probe"] = time.perf_counter() - probe_started
        result.raw_free_before_bytes = raw
        if raw is not None and raw >= result.free_target_bytes:
            result.stage_memory["before"] = {
                "raw_free_bytes": raw, "total_bytes": _raw_cuda_total(device),
                "raw_used_bytes": None, "comfy_logical_free_bytes": 0,
                "torch_reclaimable_bytes": 0, "logical_minus_raw_bytes": 0,
                "loaded_models": _loaded_models_snapshot(self.mm, device),
            }
            result.raw_free_after_bytes = raw
            result.satisfied = True
            result.strategy = "raw-fast-path"
            result.elapsed_seconds = time.perf_counter() - started
            return result.as_dict()

        # Only inspect PyTorch allocator state after physical VRAM is actually
        # short.  On backends where raw mem_get_info is unavailable, ComfyUI's
        # total-free value remains the compatibility fallback.
        comfy_probe_started = time.perf_counter()
        comfy, reclaimable = _comfy_memory(self.mm, device)
        result.stage_timings_seconds["comfy_probe"] = time.perf_counter() - comfy_probe_started
        result.comfy_free_before_bytes = comfy
        result.torch_reclaimable_before_bytes = reclaimable
        total_before = _raw_cuda_total(device)
        result.stage_memory["before"] = {
            "raw_free_bytes": raw,
            "total_bytes": total_before,
            "raw_used_bytes": (max(0, int(total_before) - int(raw)) if total_before is not None and raw is not None else None),
            "comfy_logical_free_bytes": int(comfy),
            "torch_reclaimable_bytes": int(reclaimable),
            "logical_minus_raw_bytes": (max(0, int(comfy) - int(raw)) if raw is not None else None),
            "loaded_models": _loaded_models_snapshot(self.mm, device),
        }
        if raw is None and comfy >= result.free_target_bytes:
            result.comfy_free_after_bytes = comfy
            result.torch_reclaimable_after_bytes = reclaimable
            result.satisfied = True
            result.strategy = "comfy-free-fast-path"
            result.elapsed_seconds = time.perf_counter() - started
            return result.as_dict()

        # If PyTorch allocator slack nominally covers the entire physical shortfall,
        # try ComfyUI's cache release before evicting a model. Some inactive
        # split blocks cannot be returned to CUDA; failure simply falls through
        # to the cooperative metric-compensated path.
        if raw is not None:
            shortfall = max(0, result.free_target_bytes - raw)
            if reclaimable >= shortfall and reclaimable >= MIN_CACHE_RECLAIM_BYTES:
                self._flush_cache(result, stage="pre-cooperative")
                effective, raw, comfy, reclaimable = self._effective_free(device, self.mm)
                result.raw_free_after_cache_bytes = raw
                snap("after_pre_cache")
                if effective >= result.free_target_bytes:
                    result.raw_free_after_bytes = raw
                    result.comfy_free_after_bytes = comfy
                    result.torch_reclaimable_after_bytes = reclaimable
                    result.satisfied = True
                    result.strategy = "pytorch-cache-reclaim"
                    result.elapsed_seconds = time.perf_counter() - started
                    return result.as_dict()

        # Normal cooperative path. ComfyUI intentionally uses logical free
        # memory (raw CUDA free + inactive PyTorch allocator bytes) as its stop
        # condition. Translate our raw-driver target into that metric so ComfyUI
        # does not stop one allocator-slack amount short of what llama.cpp can
        # actually consume.
        result.cooperative_eviction_called = True
        cooperative_request = self._cooperative_request(result, reclaimable)
        result.cooperative_request_target_bytes = int(cooperative_request)
        result.cooperative_torch_slack_bytes = int(reclaimable)
        try:
            coop_started = time.perf_counter()
            unloaded = self.mm.free_memory(cooperative_request, device)
            result.stage_timings_seconds["cooperative_free_memory"] = time.perf_counter() - coop_started
            try:
                result.cooperative_unloaded_count = len(unloaded or [])
            except Exception:
                result.cooperative_unloaded_count = 0
            result.unloaded_models["cooperative"] = _unloaded_labels(unloaded)
        except Exception as e:
            result.error = f"cooperative free_memory: {e}"
        effective, raw, comfy, reclaimable = self._effective_free(device, self.mm)
        result.raw_free_after_cooperative_bytes = raw
        snap("after_cooperative")
        if raw is not None and raw >= result.free_target_bytes:
            result.raw_free_after_bytes = raw
            result.comfy_free_after_bytes = comfy
            result.torch_reclaimable_after_bytes = reclaimable
            result.satisfied = True
            result.strategy = "cooperative-comfy-eviction"
            result.elapsed_seconds = time.perf_counter() - started
            return result.as_dict()
        if raw is None and comfy >= result.free_target_bytes:
            result.raw_free_after_bytes = raw
            result.comfy_free_after_bytes = comfy
            result.torch_reclaimable_after_bytes = reclaimable
            result.satisfied = True
            result.strategy = "cooperative-comfy-eviction"
            result.elapsed_seconds = time.perf_counter() - started
            return result.as_dict()

        # Eviction/cache release can change PyTorch's inactive-reserved amount.
        # Recompute the logical target once from the new measurement rather than
        # flushing caches blindly or escalating to a full-device eviction.
        corrected_request = self._cooperative_request(result, reclaimable)
        if corrected_request > cooperative_request:
            result.cooperative_retry_request_target_bytes = int(corrected_request)
            try:
                retry_started = time.perf_counter()
                more_unloaded = self.mm.free_memory(corrected_request, device)
                result.stage_timings_seconds["cooperative_retry_free_memory"] = time.perf_counter() - retry_started
                try:
                    result.cooperative_unloaded_count += len(more_unloaded or [])
                except Exception:
                    pass
                result.unloaded_models["cooperative_retry"] = _unloaded_labels(more_unloaded)
            except Exception as e:
                if result.error is None:
                    result.error = f"cooperative corrected free_memory: {e}"
            effective, raw, comfy, reclaimable = self._effective_free(device, self.mm)
            result.raw_free_after_cooperative_bytes = raw
            snap("after_cooperative_retry")
            if raw is not None and raw >= result.free_target_bytes:
                result.raw_free_after_bytes = raw
                result.comfy_free_after_bytes = comfy
                result.torch_reclaimable_after_bytes = reclaimable
                result.satisfied = True
                result.strategy = "cooperative-comfy-eviction-corrected"
                result.elapsed_seconds = time.perf_counter() - started
                return result.as_dict()
            if raw is None and comfy >= result.free_target_bytes:
                result.raw_free_after_bytes = raw
                result.comfy_free_after_bytes = comfy
                result.torch_reclaimable_after_bytes = reclaimable
                result.satisfied = True
                result.strategy = "cooperative-comfy-eviction-corrected"
                result.elapsed_seconds = time.perf_counter() - started
                return result.as_dict()

        # AIMDO recovery tier.  Stale pinned/prefetched state can survive a
        # cooperative pressure request.  Only now discard that state, matching
        # ComfyUI's own node-boundary cleanup, then retry normal free_memory once.
        if self.aimdo_enabled:
            aimdo_cleanup_started = time.perf_counter()
            self._cleanup_aimdo_transients(device, result)
            result.stage_timings_seconds["aimdo_transient_cleanup"] = time.perf_counter() - aimdo_cleanup_started
            effective, raw, comfy, reclaimable = self._effective_free(device, self.mm)
            result.raw_free_after_aimdo_cleanup_bytes = raw
            snap("after_aimdo_cleanup")
            if effective >= result.free_target_bytes:
                result.raw_free_after_bytes = raw
                result.comfy_free_after_bytes = comfy
                result.torch_reclaimable_after_bytes = reclaimable
                result.satisfied = True
                result.strategy = "aimdo-transient-cleanup"
                result.elapsed_seconds = time.perf_counter() - started
                return result.as_dict()
            result.aimdo_retry_called = True
            _effective, _raw, _comfy, aimdo_reclaimable = self._effective_free(device, self.mm)
            aimdo_request = self._cooperative_request(result, aimdo_reclaimable)
            try:
                aimdo_retry_started = time.perf_counter()
                aimdo_unloaded = self.mm.free_memory(aimdo_request, device)
                result.stage_timings_seconds["aimdo_retry_free_memory"] = time.perf_counter() - aimdo_retry_started
                result.unloaded_models["aimdo_retry"] = _unloaded_labels(aimdo_unloaded)
            except Exception as e:
                result.error = f"aimdo retry free_memory: {e}"
            snap("after_aimdo_retry")
            if self._finish_if_satisfied(result, device, "aimdo-cleanup+cooperative-retry"):
                result.elapsed_seconds = time.perf_counter() - started
                return result.as_dict()

        # Exclusive fallback is deliberately last.  It is reliable but expensive
        # because it destroys ComfyUI residency that may otherwise be reusable.
        result.exclusive_eviction_called = True
        try:
            exclusive_started = time.perf_counter()
            exclusive_unloaded = self.mm.free_memory(1e30, device)
            result.stage_timings_seconds["exclusive_free_memory"] = time.perf_counter() - exclusive_started
            result.unloaded_models["exclusive"] = _unloaded_labels(exclusive_unloaded)
        except Exception as e:
            result.error = f"exclusive free_memory: {e}"
        snap("after_exclusive")
        if self._finish_if_satisfied(result, device, "exclusive-target-gpu-eviction"):
            result.elapsed_seconds = time.perf_counter() - started
            return result.as_dict()

        # Rare failure path only: wait for any deferred releases before refusing
        # the native load.  Synchronizing here cannot hurt the common path.
        result.final_sync_called = True
        final_sync_started = time.perf_counter()
        _sync_cuda(device)
        result.stage_timings_seconds["final_sync"] = time.perf_counter() - final_sync_started
        snap("after_final_sync")
        self._finish_if_satisfied(result, device, "exclusive-target-gpu-eviction+final-sync")
        if not result.satisfied:
            # All normal reclamation has been exhausted.  Preserve the runtime
            # requirement as the hard floor, but do not turn llama.cpp's preferred
            # fit margin into an absolute ban on smaller GPUs.  If the runtime
            # itself fits, allow one guarded native load attempt with a warning.
            self._admit_reduced_headroom_if_runtime_fits(result, result.raw_free_after_bytes)
        result.elapsed_seconds = time.perf_counter() - started
        return result.as_dict()

    def acquire_exclusive(self, device, *, minimum_runtime_bytes: int = 0) -> dict[str, Any]:
        """Fully clear one target GPU when the per-device split is unknown."""
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
        result.stage_memory = {}
        result.stage_timings_seconds = {}
        result.unloaded_models = {}
        result.stage_memory["before"] = _memory_snapshot(self.mm, device)
        effective, raw, comfy, reclaimable = self._effective_free(device, self.mm)
        result.raw_free_before_bytes = raw
        result.comfy_free_before_bytes = comfy
        result.torch_reclaimable_before_bytes = reclaimable
        result.exclusive_eviction_called = True
        result.strategy = "exclusive-target-gpu-eviction"
        try:
            exclusive_started = time.perf_counter()
            exclusive_unloaded = self.mm.free_memory(1e30, device)
            result.stage_timings_seconds["exclusive_free_memory"] = time.perf_counter() - exclusive_started
            result.unloaded_models["exclusive"] = _unloaded_labels(exclusive_unloaded)
        except Exception as e:
            result.error = f"exclusive free_memory: {e}"
        result.stage_memory["after_exclusive"] = _memory_snapshot(self.mm, device)
        effective, raw, comfy, reclaimable = self._effective_free(device, self.mm)
        result.raw_free_after_bytes = raw
        result.comfy_free_after_bytes = comfy
        result.torch_reclaimable_after_bytes = reclaimable
        result.satisfied = bool(effective >= result.free_target_bytes)
        if not result.satisfied:
            result.final_sync_called = True
            final_sync_started = time.perf_counter()
            _sync_cuda(device)
            result.stage_timings_seconds["final_sync"] = time.perf_counter() - final_sync_started
            result.stage_memory["after_final_sync"] = _memory_snapshot(self.mm, device)
            effective, raw, comfy, reclaimable = self._effective_free(device, self.mm)
            result.raw_free_after_bytes = raw
            result.comfy_free_after_bytes = comfy
            result.torch_reclaimable_after_bytes = reclaimable
            result.satisfied = bool(effective >= result.free_target_bytes)
        if minimum_runtime > 0 and not result.satisfied:
            self._admit_reduced_headroom_if_runtime_fits(result, result.raw_free_after_bytes)
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
