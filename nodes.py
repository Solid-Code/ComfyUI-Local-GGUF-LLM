import base64
import copy
import gc
import hashlib
import inspect
import io
import importlib
import json
import logging
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import folder_paths

from .gguf_meta import read_gguf_metadata, detect_family, recommended_model_preset, available_model_presets
from .presets import MODEL_PRESETS, MODEL_PRESET_META, MEMORY_PRESETS, capabilities_for_family, public_presets

LLM_DIR = os.path.join(folder_paths.models_dir, "llm")
os.makedirs(LLM_DIR, exist_ok=True)
folder_paths.add_model_folder_path("llm", LLM_DIR, is_default=True)
try:
    folder_paths.folder_names_and_paths["llm"][1].add(".gguf")
except Exception:
    pass

_NONE = "None"
_AUTO_VISION = "Auto (matching mmproj)"
_MODEL_CACHE = {"key": None, "llm": None, "metadata": None, "family": None, "base_chat_handler": None, "managed_adapter": None, "load_diagnostics": None, "mode": None}
_MODEL_LOCK = threading.RLock()
_META_CACHE = {}


# ComfyUI/AIMDO integration state. The native llama.cpp allocation is deliberately
# kept OUTSIDE ComfyUI's LoadedModel/ModelPatcher list. A thin free_memory hook
# gives the all-or-nothing native context first right of refusal under real
# ComfyUI memory pressure, then delegates back to ComfyUI unchanged.
_COMFY_YIELD_HOOK_LOCK = threading.RLock()
_COMFY_YIELD_HOOK_TLS = threading.local()


def _device_index(device):
    try:
        if getattr(device, "type", None) != "cuda":
            return None
        idx = getattr(device, "index", None)
        if idx is not None:
            return int(idx)
        import torch
        return int(torch.cuda.current_device())
    except Exception:
        return None


def _same_cuda_device(a, b):
    ai = _device_index(a)
    bi = _device_index(b)
    return ai is not None and bi is not None and ai == bi


def _comfy_free_bytes(mm, device):
    """Use ComfyUI's free-memory accounting when available.

    Unlike torch.cuda.mem_get_info(), ComfyUI can account for reclaimable PyTorch
    allocator cache, which avoids initiating model eviction just because CUDA's
    raw driver counter looks temporarily tight.
    """
    fn = getattr(mm, "get_free_memory", None)
    if callable(fn):
        try:
            return int(fn(device))
        except Exception:
            pass
    try:
        import torch
        idx = _device_index(device)
        free, _total = torch.cuda.mem_get_info(idx if idx is not None else device)
        return int(free)
    except Exception:
        return 0


def _sync_cuda_device(device):
    """Finish queued CUDA work before changing native/ComfyUI residency."""
    try:
        import torch
        idx = _device_index(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize(idx if idx is not None else device)
    except Exception:
        # Synchronization is a safety assist; older/non-CUDA builds must remain usable.
        pass


def _raw_cuda_free_bytes(device):
    """Fast, non-synchronizing driver free-memory sample for lifecycle diagnostics."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        idx = _device_index(device)
        free, _total = torch.cuda.mem_get_info(idx if idx is not None else device)
        return int(free)
    except Exception:
        return None


def _comfy_memory_breakdown(mm, device):
    """Return ComfyUI total-free, reclaimable torch cache, and raw driver free bytes.

    Current ComfyUI exposes ``get_free_memory(..., torch_free_too=True)`` where
    total-free includes allocator cache and torch-free is the reclaimable portion.
    Older builds fall back to CUDA allocator stats. This lets the native llama.cpp
    path decide whether a cache flush can *actually* satisfy a driver-VRAM shortage
    before paying for ComfyUI's synchronized ``soft_empty_cache()``.
    """
    total_free = None
    torch_reclaimable = None
    fn = getattr(mm, "get_free_memory", None)
    if callable(fn):
        try:
            value = fn(device, torch_free_too=True)
            if isinstance(value, (tuple, list)) and len(value) >= 2:
                total_free = int(value[0])
                torch_reclaimable = max(0, int(value[1]))
            else:
                total_free = int(value)
        except TypeError:
            try:
                total_free = int(fn(device))
            except Exception:
                pass
        except Exception:
            pass

    raw_free = _raw_cuda_free_bytes(device)
    if torch_reclaimable is None:
        try:
            import torch
            idx = _device_index(device)
            if torch.cuda.is_available():
                stats = torch.cuda.memory_stats(idx if idx is not None else device)
                reserved = int(stats.get("reserved_bytes.all.current", 0))
                active = int(stats.get("active_bytes.all.current", stats.get("allocated_bytes.all.current", 0)))
                torch_reclaimable = max(0, reserved - active)
        except Exception:
            torch_reclaimable = 0
    if torch_reclaimable is None:
        torch_reclaimable = 0
    if total_free is None:
        if raw_free is not None:
            total_free = int(raw_free + torch_reclaimable)
        else:
            total_free = 0
    return {
        "comfy_free_bytes": int(max(0, total_free)),
        "torch_reclaimable_bytes": int(max(0, torch_reclaimable)),
        "raw_free_bytes": raw_free,
    }


def _host_load_snapshot():
    """Cheap Linux/WSL page-cache + process fault snapshot for GGUF load profiling."""
    out = {}
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF)
        out.update({
            "minor_faults": int(getattr(r, "ru_minflt", 0)),
            "major_faults": int(getattr(r, "ru_majflt", 0)),
            "block_inputs": int(getattr(r, "ru_inblock", 0)),
        })
    except Exception:
        pass
    try:
        mem = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if ":" not in line:
                    continue
                key, rest = line.split(":", 1)
                if key in {"Cached", "SReclaimable", "MemAvailable"}:
                    parts = rest.strip().split()
                    if parts:
                        mem[key] = int(parts[0]) * 1024
        if mem:
            out["page_cache_bytes"] = int(mem.get("Cached", 0) + mem.get("SReclaimable", 0))
            out["mem_available_bytes"] = int(mem.get("MemAvailable", 0))
    except Exception:
        pass
    try:
        kernel = Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8").strip().lower()
        out["wsl"] = "microsoft" in kernel or "wsl" in kernel
    except Exception:
        pass
    return out


def _load_mode_name(load_kwargs):
    mmap = bool(load_kwargs.get("use_mmap", True))
    mlock = bool(load_kwargs.get("use_mlock", False))
    if mmap and mlock:
        return "mmap+mlock"
    if mmap:
        return "mmap"
    if mlock:
        return "no-mmap+mlock"
    return "no-mmap"


def _host_load_delta(before, after, load_mode="unknown"):
    before = before or {}
    after = after or {}
    out = {
        "major_faults_delta": max(0, int(after.get("major_faults", 0)) - int(before.get("major_faults", 0))),
        "minor_faults_delta": max(0, int(after.get("minor_faults", 0)) - int(before.get("minor_faults", 0))),
        "block_inputs_delta": max(0, int(after.get("block_inputs", 0)) - int(before.get("block_inputs", 0))),
        "page_cache_before_bytes": before.get("page_cache_bytes"),
        "page_cache_after_bytes": after.get("page_cache_bytes"),
        "mem_available_before_bytes": before.get("mem_available_bytes"),
        "mem_available_after_bytes": after.get("mem_available_bytes"),
        "wsl": bool(after.get("wsl", before.get("wsl", False))),
    }
    # This is a hint, not proof: mmap-backed loads with no major faults are very
    # likely being satisfied from RAM/page cache rather than storage.
    if out["major_faults_delta"] > 0 or out["block_inputs_delta"] > 0:
        out["page_cache_hint"] = "storage-backed-faults-observed"
    elif str(load_mode).startswith("mmap"):
        out["page_cache_hint"] = "warm-or-memory-backed"
    else:
        out["page_cache_hint"] = "unknown"
    return out


_RELOAD_VRAM_MARGIN = 256 * 1024 * 1024


def _perf_log(event, **fields):
    """Compact always-on lifecycle timings for KoboldCpp-style comparisons."""
    parts = []
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, float):
            parts.append(f"{key}={value:.3f}")
        else:
            parts.append(f"{key}={value}")
    suffix = " • " + " • ".join(parts) if parts else ""
    _LOGGER.info("[Local GGUF LLM PERF] %s%s", event, suffix)


def _load_signature_id(load_key):
    """Stable short id for the complete native-allocation signature."""
    try:
        return hashlib.sha1(repr(load_key).encode("utf-8", "replace")).hexdigest()[:10]
    except Exception:
        return "unknown"


def _aimdo_enabled():
    try:
        import comfy.memory_management as cmm
        return bool(getattr(cmm, "aimdo_enabled", False))
    except Exception:
        return False


def _install_comfy_llm_yield_hook(mm):
    """Thin ComfyUI -> native-LLM yield coordination.

    The llama.cpp context is intentionally *not* registered as a ComfyUI model.
    Instead, intercept ComfyUI's real memory-shortfall path and close the complete
    native context before ComfyUI starts evicting/partially unloading its own
    models.  This keeps llama.cpp all-or-nothing and avoids VBAR/AIMDO partial
    residency semantics for an allocation ComfyUI cannot actually migrate.
    """
    with _COMFY_YIELD_HOOK_LOCK:
        current = getattr(mm, "free_memory", None)
        if not callable(current):
            return False
        if getattr(current, "_local_gguf_llm_yield_hook", False):
            if getattr(current, "_local_gguf_llm_hook_version", None) == 5:
                return True
            # Replace an older wrapper instead of stacking wrappers. This matters
            # during development/hot reloads where model_management can outlive
            # the custom-node module instance.
            original = getattr(current, "_local_gguf_llm_original", current)
        else:
            original = current

        def free_memory_with_llm_yield(memory_required, device, *args, **kwargs):
            if getattr(_COMFY_YIELD_HOOK_TLS, "active", False):
                return original(memory_required, device, *args, **kwargs)
            hook_started = time.perf_counter()
            log_original_timing = False
            try:
                # `managed_adapter` is retained as the cache field name for
                # service/backward compatibility; the object is now only a native
                # residency controller and is never inserted into loaded_models().
                resident_ctl = _MODEL_CACHE.get("managed_adapter")
                resident = resident_ctl is not None and getattr(resident_ctl, "llm", None) is not None
                same_device = resident and (device is None or _same_cuda_device(device, resident_ctl.load_device))
                needed = max(0, int(float(memory_required or 0)))
                target_device = resident_ctl.load_device if device is None and resident else device
                comfy_free = _comfy_free_bytes(mm, target_device) if target_device is not None else 0
                raw_free = _raw_cuda_free_bytes(target_device) if target_device is not None else None
                # Native llama.cpp needs driver-visible VRAM. Use raw CUDA free
                # memory when available so reclaimable PyTorch allocator cache
                # does not hide a real shortfall while the LLM is resident.
                effective_free = raw_free if raw_free is not None else comfy_free
                if same_device and needed > int(effective_free or 0):
                    log_original_timing = True
                    _COMFY_YIELD_HOOK_TLS.active = True
                    try:
                        _perf_log(
                            "ComfyUI memory pressure -> native LLM yield",
                            signature=resident_ctl.signature_id,
                            required_mib=needed / _MIB,
                            raw_free_before_mib=(raw_free / _MIB) if raw_free is not None else None,
                            comfy_free_before_mib=comfy_free / _MIB,
                        )
                        sync_started = time.perf_counter()
                        _sync_cuda_device(resident_ctl.load_device)
                        sync_seconds = time.perf_counter() - sync_started
                        freed = int(resident_ctl._unload_native(
                            reason="comfy-memory-pressure",
                            required_free_bytes=needed,
                        ) or 0)
                        if freed > 0:
                            _LOGGER.info(
                                "[Local GGUF LLM] Auto Yield fully closed native llama.cpp context "
                                "(accounted %.1f MiB) before ComfyUI eviction.",
                                freed / _MIB,
                            )
                        _perf_log(
                            "native LLM pre-yield complete",
                            signature=resident_ctl.signature_id,
                            sync_s=sync_seconds,
                            accounted_mib=freed / _MIB,
                            hook_s=time.perf_counter() - hook_started,
                        )
                    finally:
                        _COMFY_YIELD_HOOK_TLS.active = False
            except Exception as e:
                _LOGGER.warning("[Local GGUF LLM] Auto Yield pre-eviction hook failed: %s", e)
            original_started = time.perf_counter()
            result = original(memory_required, device, *args, **kwargs)
            if log_original_timing:
                _perf_log(
                    "ComfyUI free_memory complete after native LLM yield",
                    original_s=time.perf_counter() - original_started,
                    total_hook_s=time.perf_counter() - hook_started,
                )
            return result

        free_memory_with_llm_yield._local_gguf_llm_yield_hook = True
        free_memory_with_llm_yield._local_gguf_llm_hook_version = 5
        free_memory_with_llm_yield._local_gguf_llm_original = original
        mm.free_memory = free_memory_with_llm_yield
        return True

def _request_comfyui_room(mm, required_bytes, device):
    """Make driver-visible room for native llama.cpp with minimum synchronization.

    The fast path only calls ``soft_empty_cache()`` when ComfyUI reports enough
    reclaimable allocator cache to cover the *entire* raw-driver shortfall. If
    cache cannot possibly solve it, skip that synchronized flush and go directly
    to model eviction. AIMDO full-unloads only the target GPU via ``free_memory``
    with an intentionally huge target instead of globally unloading every device.
    """
    required = max(0, int(required_bytes or 0))
    result = {
        "required_free_bytes": required,
        "free_before_bytes": None,
        "raw_free_before_bytes": None,
        "torch_reclaimable_before_bytes": 0,
        "free_after_bytes": None,
        "raw_free_after_bytes": None,
        "strategy": "none",
        "release_called": False,
        "cache_probe_called": False,
        "cache_probe_skipped": False,
        "aimdo": _aimdo_enabled(),
    }
    if required <= 0:
        return result

    total_started = time.perf_counter()
    before = _comfy_memory_breakdown(mm, device)
    free_before = before["comfy_free_bytes"]
    raw_before = before["raw_free_bytes"]
    reclaimable_before = before["torch_reclaimable_bytes"]
    result["free_before_bytes"] = free_before
    result["raw_free_before_bytes"] = raw_before
    result["torch_reclaimable_before_bytes"] = reclaimable_before
    fit_before = raw_before if raw_before is not None else free_before
    shortfall = max(0, required - int(fit_before or 0))
    result["raw_shortfall_before_bytes"] = shortfall
    if fit_before >= required:
        result["free_after_bytes"] = free_before
        result["raw_free_after_bytes"] = raw_before
        result["total_seconds"] = time.perf_counter() - total_started
        _perf_log(
            "LLM VRAM room check: already fits",
            required_mib=required / _MIB,
            comfy_free_mib=free_before / _MIB,
            raw_free_mib=(raw_before / _MIB) if raw_before is not None else None,
            reclaimable_mib=reclaimable_before / _MIB,
            total_s=result["total_seconds"],
        )
        return result

    # Only pay for ComfyUI's synchronized cache flush if the reclaimable cache
    # is large enough to satisfy the complete raw-driver shortage by itself.
    cache_can_satisfy = raw_before is None or reclaimable_before >= shortfall
    result["cache_can_satisfy_shortfall"] = bool(cache_can_satisfy)
    if cache_can_satisfy and reclaimable_before >= 16 * _MIB:
        result["cache_probe_called"] = True
        cache_probe_started = time.perf_counter()
        try:
            mm.soft_empty_cache()
        except Exception:
            pass
        result["cache_probe_seconds"] = time.perf_counter() - cache_probe_started
        after_cache = _comfy_memory_breakdown(mm, device)
        raw_after_cache = after_cache["raw_free_bytes"]
        comfy_after_cache = after_cache["comfy_free_bytes"]
        result["raw_free_after_cache_bytes"] = raw_after_cache
        result["free_after_cache_bytes"] = comfy_after_cache
        result["torch_reclaimable_after_cache_bytes"] = after_cache["torch_reclaimable_bytes"]
        fit_after_cache = raw_after_cache if raw_after_cache is not None else comfy_after_cache
        if fit_after_cache >= required:
            result["strategy"] = "soft-cache-only"
            result["free_after_bytes"] = comfy_after_cache
            result["raw_free_after_bytes"] = raw_after_cache
            result["total_seconds"] = time.perf_counter() - total_started
            _perf_log(
                "LLM VRAM handoff complete",
                strategy=result["strategy"],
                required_mib=required / _MIB,
                shortfall_mib=shortfall / _MIB,
                reclaimable_mib=reclaimable_before / _MIB,
                raw_before_mib=(raw_before / _MIB) if raw_before is not None else None,
                raw_after_mib=(raw_after_cache / _MIB) if raw_after_cache is not None else None,
                cache_probe_s=result["cache_probe_seconds"],
                total_s=result["total_seconds"],
            )
            return result
    else:
        result["cache_probe_skipped"] = True
        result["cache_probe_skip_reason"] = (
            "reclaimable-cache-too-small" if raw_before is not None else "reclaimable-cache-unavailable"
        )
        _perf_log(
            "skip speculative PyTorch cache flush",
            required_mib=required / _MIB,
            raw_free_mib=(raw_before / _MIB) if raw_before is not None else None,
            shortfall_mib=shortfall / _MIB,
            reclaimable_mib=reclaimable_before / _MIB,
        )

    result["release_called"] = True
    sync_started = time.perf_counter()
    _sync_cuda_device(device)
    result["pre_release_sync_seconds"] = time.perf_counter() - sync_started

    release_started = time.perf_counter()
    if result["aimdo"]:
        # Full unload semantics, target GPU only. This avoids VBAR being walked
        # toward watermark 0 while also avoiding unload_all_models() on unrelated
        # accelerators. The free_memory hook is safe here because the LLM is not
        # resident during its own pre-load handoff.
        result["strategy"] = "aimdo-target-gpu-full-unload"
        mm.free_memory(1e30, device)
    else:
        result["strategy"] = "targeted-free-memory"
        mm.free_memory(required, device)
    result["release_seconds"] = time.perf_counter() - release_started

    after_release = _comfy_memory_breakdown(mm, device)
    raw_after_release = after_release["raw_free_bytes"]
    result["raw_free_after_release_bytes"] = raw_after_release
    result["torch_reclaimable_after_release_bytes"] = after_release["torch_reclaimable_bytes"]
    post_cache_seconds = 0.0
    # ComfyUI normally flushes allocator cache itself after an actual model
    # unload. Only make one final attempt if driver-visible room is still short
    # *and* the remaining torch cache can cover that exact shortfall.
    fit_after_release = raw_after_release if raw_after_release is not None else after_release["comfy_free_bytes"]
    remaining_shortfall = max(0, required - int(fit_after_release or 0))
    if remaining_shortfall > 0 and after_release["torch_reclaimable_bytes"] >= remaining_shortfall:
        cache_started = time.perf_counter()
        try:
            mm.soft_empty_cache()
        except Exception:
            pass
        post_cache_seconds = time.perf_counter() - cache_started
    result["soft_empty_cache_seconds"] = post_cache_seconds
    result["post_release_sync_seconds"] = 0.0

    final = _comfy_memory_breakdown(mm, device)
    result["free_after_bytes"] = final["comfy_free_bytes"]
    result["raw_free_after_bytes"] = final["raw_free_bytes"]
    result["torch_reclaimable_after_bytes"] = final["torch_reclaimable_bytes"]
    result["total_seconds"] = time.perf_counter() - total_started
    _perf_log(
        "LLM VRAM handoff complete",
        strategy=result["strategy"],
        required_mib=required / _MIB,
        shortfall_mib=shortfall / _MIB,
        reclaimable_mib=reclaimable_before / _MIB,
        comfy_before_mib=free_before / _MIB,
        raw_before_mib=(raw_before / _MIB) if raw_before is not None else None,
        comfy_after_mib=(result["free_after_bytes"] or 0) / _MIB,
        raw_after_mib=(result["raw_free_after_bytes"] / _MIB) if result["raw_free_after_bytes"] is not None else None,
        cache_probe_s=result.get("cache_probe_seconds", 0.0),
        cache_skipped=result.get("cache_probe_skipped"),
        pre_sync_s=result.get("pre_release_sync_seconds"),
        release_s=result.get("release_seconds"),
        cache_s=result.get("soft_empty_cache_seconds"),
        total_s=result.get("total_seconds"),
    )
    return result


def _llama_perf_snapshot(llm):
    """Best-effort llama.cpp context timing snapshot.

    Newer llama-cpp-python builds expose llama_perf_context(), which separates
    prompt evaluation from sampled-token evaluation.  Keep this optional so the
    node remains compatible with older bindings.
    """
    try:
        import llama_cpp
        fn = getattr(llama_cpp, "llama_perf_context", None)
        ctx_obj = getattr(llm, "_ctx", None)
        ctx = getattr(ctx_obj, "ctx", None)
        if fn is None or ctx is None:
            return None
        perf = fn(ctx)
        def value(name, default=0.0):
            try:
                return getattr(perf, name)
            except Exception:
                return default
        return {
            "prompt_ms": float(value("t_p_eval_ms", 0.0) or 0.0),
            "prompt_tokens": int(value("n_p_eval", 0) or 0),
            "eval_ms": float(value("t_eval_ms", 0.0) or 0.0),
            "eval_tokens": int(value("n_eval", 0) or 0),
        }
    except Exception:
        return None


def _perf_delta(after, before):
    if not after:
        return None
    before = before or {}

    # Some llama.cpp / binding versions reset perf counters at the start of a
    # completion. If a counter moved backwards, the current value is already the
    # per-request value; otherwise subtract our pre-call snapshot.
    def delta(name, integer=False):
        a = after.get(name, 0) or 0
        b = before.get(name, 0) or 0
        a = int(a) if integer else float(a)
        b = int(b) if integer else float(b)
        return a - b if a >= b else a

    return {
        "prompt_ms": max(0.0, delta("prompt_ms")),
        "prompt_tokens": max(0, delta("prompt_tokens", True)),
        "eval_ms": max(0.0, delta("eval_ms")),
        "eval_tokens": max(0, delta("eval_tokens", True)),
    }


def _scan_gguf():
    try:
        names = folder_paths.get_filename_list("llm")
    except Exception:
        names = []
    return sorted([n for n in names if n.lower().endswith(".gguf")], key=str.lower)


def _model_lists():
    all_files = _scan_gguf()
    vision = [n for n in all_files if any(x in os.path.basename(n).lower() for x in ("mmproj", "vision-proj", "projector"))]
    models = [n for n in all_files if n not in vision]
    return models or ["No GGUF models found"], [_NONE, _AUTO_VISION] + vision


def _gpu_choices():
    """Return logical accelerator indices with human-readable device names.

    Prefer PyTorch because its logical device numbering follows the exact CUDA/ROCm
    visibility seen by ComfyUI (including CUDA_VISIBLE_DEVICES remapping). Fall back
    to nvidia-smi only when torch cannot enumerate an accelerator.
    """
    devices = []
    try:
        import torch
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            for i in range(int(torch.cuda.device_count())):
                try:
                    props = torch.cuda.get_device_properties(i)
                    name = str(getattr(props, "name", "") or torch.cuda.get_device_name(i) or f"GPU {i}").strip()
                    total = int(getattr(props, "total_memory", 0) or 0)
                    if total > 0:
                        gib = total / (1024 ** 3)
                        devices.append(f"{i} — {name} ({gib:.1f} GiB)")
                    else:
                        devices.append(f"{i} — {name}")
                except Exception:
                    try:
                        devices.append(f"{i} — {torch.cuda.get_device_name(i)}")
                    except Exception:
                        devices.append(f"{i} — GPU {i}")
    except Exception:
        pass

    if devices:
        return devices

    # Fallback is mainly useful when ComfyUI is using a non-standard Python
    # environment where torch enumeration is unavailable but NVIDIA tools are.
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=2,
        )
        for line in output.splitlines():
            parts = [x.strip() for x in line.split(",", 2)]
            if len(parts) >= 2 and parts[0].isdigit():
                idx, name = int(parts[0]), parts[1]
                mem = parts[2] if len(parts) >= 3 else ""
                try:
                    gib = float(mem) / 1024.0
                    devices.append(f"{idx} — {name} ({gib:.1f} GiB)")
                except Exception:
                    devices.append(f"{idx} — {name}")
    except Exception:
        pass

    return devices or ["0 — Default GPU (backend device 0)"]


def _gpu_index(value):
    """Convert a labeled GPU combo value (or legacy integer) to llama.cpp index."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return max(0, int(value))
    text = str(value or "").strip()
    # New labels start with the logical device index: `0 — NVIDIA ...`.
    m = re.match(r"^\s*(\d+)\b", text)
    if m:
        return int(m.group(1))
    raise ValueError(f"Invalid Main GPU selection: {value!r}")


def _full_path(name):
    p = folder_paths.get_full_path("llm", name)
    if p is None:
        p = os.path.join(LLM_DIR, name)
    return p


def _metadata_for(name):
    path = _full_path(name)
    try:
        stat = os.stat(path)
        key = (path, stat.st_mtime_ns, stat.st_size)
        cached = _META_CACHE.get(key)
        if cached is not None:
            return cached
        md = read_gguf_metadata(path)
        _META_CACHE.clear()
        _META_CACHE[key] = md
        return md
    except Exception:
        return {}


def _name_tokens(name):
    base = os.path.basename(str(name or "")).lower()
    stop = {
        "gguf", "mmproj", "vision", "proj", "projector", "model", "clip", "image",
        "f16", "f32", "bf16", "q2", "q3", "q4", "q5", "q6", "q8", "iq", "km", "ks",
        "k", "m", "s", "instruct", "chat",
    }
    return [t for t in re.split(r"[^a-z0-9.]+", base) if len(t) >= 2 and t not in stop]


def _projector_profile(projector_name):
    if not projector_name or projector_name in (_NONE, _AUTO_VISION):
        return {"name": projector_name, "family": "unknown", "metadata": {}}
    md = _metadata_for(projector_name)
    return {
        "name": projector_name,
        "family": detect_family(md, projector_name),
        "metadata": md,
    }


def _validate_mmproj_pair(model_name, projector_name):
    """Best-effort model/mmproj compatibility result without blocking unknown future VLMs."""
    md = _metadata_for(model_name)
    family = detect_family(md, model_name)
    caps = capabilities_for_family(family)
    profile = _projector_profile(projector_name)
    pfamily = profile.get("family") or "unknown"

    if caps.get("vision") is False:
        return {
            "compatible": False,
            "model_family": family,
            "projector_family": pfamily,
            "reason": f"Detected model family '{family}' is text-only in the capability registry.",
        }

    known_model = caps.get("capability_confidence") == "known"
    pcaps = capabilities_for_family(pfamily)
    known_proj = pcaps.get("capability_confidence") == "known"
    if known_model and known_proj and family != pfamily:
        return {
            "compatible": False,
            "model_family": family,
            "projector_family": pfamily,
            "reason": f"Projector appears to target '{pfamily}', not detected model family '{family}'.",
        }

    model_tokens = set(_name_tokens(model_name))
    proj_tokens = set(_name_tokens(projector_name))
    overlap = sorted(model_tokens & proj_tokens)
    compatible = True if family == pfamily and known_model else None
    reason = (
        f"Family match: {family}." if compatible is True else
        "No conflicting family metadata was detected; compatibility will be verified by llama.cpp at load time."
    )
    return {
        "compatible": compatible,
        "model_family": family,
        "projector_family": pfamily,
        "name_token_overlap": overlap,
        "reason": reason,
    }


def _find_matching_mmproj(model_name):
    _models, vision = _model_lists()
    candidates = [x for x in vision if x not in (_NONE, _AUTO_VISION)]
    if not candidates:
        return None
    model_md = _metadata_for(model_name)
    family = detect_family(model_md, model_name)
    caps = capabilities_for_family(family)
    if caps.get("vision") is False:
        return None

    model_tokens = set(_name_tokens(model_name))
    model_dir = os.path.dirname(str(model_name or "")).lower()
    scored = []
    for c in candidates:
        check = _validate_mmproj_pair(model_name, c)
        if check.get("compatible") is False:
            continue
        cb_tokens = set(_name_tokens(c))
        overlap = model_tokens & cb_tokens
        score = len(overlap) * 4
        if os.path.dirname(c).lower() == model_dir:
            score += 8
        if check.get("model_family") == check.get("projector_family") and check.get("model_family") not in {"generic", "unknown", "clip"}:
            score += 100
        # A weak filename-only match is still useful for unknown/new VLM families,
        # but require more than a single generic size token.
        scored.append((score, len(overlap), c))
    scored.sort(key=lambda x: (-x[0], -x[1], x[2].lower()))
    if not scored:
        return None
    best_score, overlap_count, best = scored[0]
    best_check = _validate_mmproj_pair(model_name, best)
    exact_family = (
        best_check.get("compatible") is True
        and best_check.get("model_family") not in {"generic", "unknown", "clip"}
    )
    # Directory proximity is only a tie-breaker, never sufficient evidence by
    # itself. Picking the wrong projector is worse than asking the user to select
    # one explicitly. Unknown/new families therefore need at least two meaningful
    # filename tokens in common.
    return best if exact_family or overlap_count >= 2 else None


def _resolve_kv_type(llama_cpp, name):
    """Resolve a llama.cpp GGML KV-cache enum across binding layout changes.

    Newer/alternate llama-cpp-python builds do not always re-export every GGML
    enum at the package root even though the low-level ``llama_cpp.llama_cpp``
    module and native backend still support it.  The enum values are ABI-stable:
    upstream ggml only appends new entries and preserves existing numeric values.
    """
    if name == "Auto":
        return None

    spec = {
        "f32":   ("GGML_TYPE_F32", 0),
        "f16":   ("GGML_TYPE_F16", 1),
        "q4_0":  ("GGML_TYPE_Q4_0", 2),
        "q4_1":  ("GGML_TYPE_Q4_1", 3),
        "q5_0":  ("GGML_TYPE_Q5_0", 6),
        "q5_1":  ("GGML_TYPE_Q5_1", 7),
        "q8_0":  ("GGML_TYPE_Q8_0", 8),
        "iq4_nl":("GGML_TYPE_IQ4_NL", 20),
        "bf16":  ("GGML_TYPE_BF16", 30),
    }.get(str(name).lower())
    if spec is None:
        raise RuntimeError(f"Unknown KV cache type '{name}'.")

    attr, stable_value = spec

    # 1) Traditional llama-cpp-python package-level re-export.
    value = getattr(llama_cpp, attr, None)
    if value is not None:
        return value

    # 2) Current/alternate bindings may expose the enum only in the low-level
    #    ctypes module rather than at ``import llama_cpp`` package scope.
    low_level = getattr(llama_cpp, "llama_cpp", None)
    if low_level is None:
        try:
            low_level = importlib.import_module("llama_cpp.llama_cpp")
        except Exception:
            low_level = None
    value = getattr(low_level, attr, None) if low_level is not None else None
    if value is not None:
        _LOGGER.info("[Local GGUF LLM] KV type %s resolved from low-level llama_cpp binding (%s=%s).", name, attr, value)
        return value

    # 3) ggml enum values are intentionally stable for backward compatibility.
    #    Only use this fallback when the high-level Llama constructor still
    #    exposes type_k/type_v; a truly incompatible wrapper will then be rejected
    #    by _require_init_option before context creation.
    _LOGGER.warning(
        "[Local GGUF LLM] %s is not exported by this llama-cpp-python package; "
        "using upstream stable GGML enum value %d for KV cache type '%s'.",
        attr, stable_value, name,
    )
    return stable_value


def _split_mode(llama_cpp, name):
    attrs = {
        "None (single GPU)": ("LLAMA_SPLIT_MODE_NONE", 0),
        "Layer": ("LLAMA_SPLIT_MODE_LAYER", 1),
        "Row": ("LLAMA_SPLIT_MODE_ROW", 2),
        "Tensor": ("LLAMA_SPLIT_MODE_TENSOR", None),
    }
    attr, fallback = attrs[name]
    if hasattr(llama_cpp, attr):
        return getattr(llama_cpp, attr)
    if fallback is not None:
        return fallback
    raise RuntimeError(
        "Tensor split mode is not available in the installed llama-cpp-python/llama.cpp build. "
        "Select Layer or Row, or update llama-cpp-python."
    )


def _parse_tensor_split(text):
    text = (text or "").strip()
    if not text:
        return None
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals or any(x < 0 for x in vals) or sum(vals) <= 0:
        raise ValueError("Tensor Split must be comma-separated non-negative numbers, e.g. 1,1 or 0.7,0.3")
    return vals


def _parse_stop(value):
    """Normalize UI pipe-delimited or API list stop sequences.

    The ComfyUI widget uses ``|`` as a separator and ``\\|`` for a literal pipe.
    OpenAI clients, including SillyTavern, natively send an array; keep those
    entries intact so special tokens such as ``<|im_end|>`` are never split.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        out = [str(x).replace("\\n", "\n") for x in value if x is not None and str(x) != ""]
        return out or None

    text = str(value)
    if not text.strip():
        return None

    parts = []
    buf = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt == "|":
                buf.append("|")
                i += 2
                continue
            if nxt == "\\":
                buf.append("\\")
                i += 2
                continue
        if ch == "|":
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    parts.append("".join(buf))
    out = [x.replace("\\n", "\n") for x in parts if x != ""]
    return out or None


def _signature_info(callable_obj):
    """Return (parameters, accepts **kwargs) without assuming inspect always works."""
    try:
        params = inspect.signature(callable_obj).parameters
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        return params, has_var_kw
    except Exception:
        return {}, True


def _filter_supported_kwargs(callable_obj, kwargs):
    """Drop None and parameters definitely unsupported by an older binding."""
    params, has_var_kw = _signature_info(callable_obj)
    clean = {k: v for k, v in kwargs.items() if v is not None}
    if has_var_kw or not params:
        return clean, []
    unsupported = [k for k in clean if k not in params]
    return {k: v for k, v in clean.items() if k in params}, unsupported


def _require_init_option(llama_init_params, has_var_kw, option, reason):
    if option in llama_init_params or has_var_kw:
        return
    raise RuntimeError(
        f"The installed llama-cpp-python build does not expose '{option}', required for {reason}. "
        "Update/rebuild llama-cpp-python with a current llama.cpp backend."
    )



_LOGGER = logging.getLogger(__name__)
_MIB = 1024 * 1024
_GIB = 1024 * 1024 * 1024


def _close_llama_object(llm):
    """Close llama.cpp + multimodal contexts without assuming a binding version."""
    if llm is None:
        return
    try:
        chat_handler = getattr(llm, "chat_handler", None)
        exit_stack = getattr(chat_handler, "_exit_stack", None)
        if exit_stack is not None and hasattr(exit_stack, "close"):
            exit_stack.close()
    except Exception:
        pass
    try:
        close = getattr(llm, "close", None)
        if callable(close):
            close()
    except Exception:
        pass


def _torch_cuda_snapshot():
    """Device-wide free-memory snapshot; includes native llama.cpp CUDA allocations."""
    out = {}
    try:
        import torch
        if not torch.cuda.is_available():
            return out
        for i in range(int(torch.cuda.device_count())):
            try:
                torch.cuda.synchronize(i)
            except Exception:
                pass
            try:
                free, total = torch.cuda.mem_get_info(i)
                out[i] = {
                    "free": int(free),
                    "total": int(total),
                    "name": str(torch.cuda.get_device_name(i)),
                }
            except Exception:
                pass
    except Exception:
        pass
    return out


def _snapshot_deltas(before, after):
    deltas = {}
    for i in sorted(set(before) | set(after)):
        if i in before and i in after:
            # Positive means device-wide free VRAM decreased while llama.cpp loaded.
            deltas[i] = max(0, int(before[i]["free"]) - int(after[i]["free"]))
    return deltas


def _llama_system_info(llama_cpp):
    for owner in (llama_cpp, getattr(llama_cpp, "llama_cpp", None)):
        fn = getattr(owner, "llama_print_system_info", None) if owner is not None else None
        if callable(fn):
            try:
                value = fn()
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                return str(value)
            except Exception:
                pass
    return "unavailable"


def _llama_gpu_offload_hint(llama_cpp):
    for owner in (llama_cpp, getattr(llama_cpp, "llama_cpp", None)):
        fn = getattr(owner, "llama_supports_gpu_offload", None) if owner is not None else None
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                pass
    return None


def _llama_backend_libraries(llama_cpp):
    """Return a small diagnostic list; never use this alone to decide GPU support."""
    try:
        root = Path(llama_cpp.__file__).resolve().parent
        found = set()
        for sub in (root, root / "lib"):
            if not sub.exists():
                continue
            for pattern in ("*ggml-cuda*", "*ggml-hip*", "*ggml-vulkan*", "*cuda*.dll", "*cuda*.so*"):
                for f in sub.glob(pattern):
                    if f.is_file():
                        found.add(f.name)
        return sorted(found)[:32]
    except Exception:
        return []


def _load_llama_verified(llama_cpp, Llama, load_kwargs, gpu_layers):
    """Load a model and verify requested GPU offload by measuring device-wide VRAM."""
    total_started = time.perf_counter()
    before_started = time.perf_counter()
    before = _torch_cuda_snapshot()
    snapshot_before_seconds = time.perf_counter() - before_started
    load_mode = _load_mode_name(load_kwargs)
    host_before = _host_load_snapshot()
    started = time.perf_counter()
    llm = Llama(**load_kwargs)
    elapsed = time.perf_counter() - started
    host_after = _host_load_snapshot()
    host_delta = _host_load_delta(host_before, host_after, load_mode)
    after_started = time.perf_counter()
    after = _torch_cuda_snapshot()
    snapshot_after_seconds = time.perf_counter() - after_started
    deltas = _snapshot_deltas(before, after)

    diag_started = time.perf_counter()
    requested_gpu = int(gpu_layers) != 0
    total_delta = sum(deltas.values())
    support_hint = _llama_gpu_offload_hint(llama_cpp)
    system_info = _llama_system_info(llama_cpp)
    diagnostics = {
        "requested_gpu_layers": int(gpu_layers),
        "gpu_requested": requested_gpu,
        "gpu_offload_verified": (total_delta >= 8 * _MIB) if requested_gpu else False,
        "vram_delta_bytes_by_gpu": {str(i): int(v) for i, v in deltas.items()},
        "vram_delta_mib_by_gpu": {str(i): round(v / _MIB, 1) for i, v in deltas.items()},
        "total_vram_delta_mib": round(total_delta / _MIB, 1),
        "llama_supports_gpu_offload_hint": support_hint,
        "llama_system_info": system_info,
        "backend_libraries": _llama_backend_libraries(llama_cpp),
        "native_load_seconds": round(elapsed, 3),
        "load_path": "verified-first-load",
        "verification_skipped": False,
        "snapshot_before_seconds": round(snapshot_before_seconds, 4),
        "snapshot_after_seconds": round(snapshot_after_seconds, 4),
        "load_mode": load_mode,
        "model_file_size_bytes": int(os.path.getsize(load_kwargs.get("model_path"))) if load_kwargs.get("model_path") and os.path.exists(load_kwargs.get("model_path")) else 0,
        **host_delta,
    }
    try:
        import torch
        diagnostics["torch_cuda_version"] = str(getattr(torch.version, "cuda", None))
        diagnostics["torch_hip_version"] = str(getattr(torch.version, "hip", None))
        diagnostics["torch_cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        pass
    diagnostics["diagnostics_seconds"] = round(time.perf_counter() - diag_started, 4)
    diagnostics["verified_load_total_seconds"] = round(time.perf_counter() - total_started, 4)
    _perf_log(
        "native GGUF verified load",
        native_s=elapsed,
        snapshot_before_s=snapshot_before_seconds,
        snapshot_after_s=snapshot_after_seconds,
        diagnostics_s=float(diagnostics["diagnostics_seconds"]),
        total_s=float(diagnostics["verified_load_total_seconds"]),
        vram_delta_mib=total_delta / _MIB,
        load_mode=load_mode,
        major_faults=host_delta.get("major_faults_delta"),
        minor_faults=host_delta.get("minor_faults_delta"),
        block_inputs=host_delta.get("block_inputs_delta"),
        page_cache_mib=(host_delta.get("page_cache_after_bytes") / _MIB) if host_delta.get("page_cache_after_bytes") is not None else None,
        cache_hint=host_delta.get("page_cache_hint"),
        wsl=host_delta.get("wsl"),
    )

    if requested_gpu and total_delta < 8 * _MIB:
        _close_llama_object(llm)
        lib_text = ", ".join(diagnostics["backend_libraries"]) or "none detected"
        raise RuntimeError(
            "GPU offload was requested (gpu_layers != 0), but loading the GGUF changed visible GPU VRAM "
            f"by only {diagnostics['total_vram_delta_mib']:.1f} MiB. The model is almost certainly running on CPU.\n\n"
            "Most common cause: ComfyUI's Python environment has a CPU-only llama-cpp-python build, or its "
            "llama.cpp CUDA backend does not match the installed CUDA/runtime. n_gpu_layers=-1 cannot enable GPU "
            "offload in a CPU-only build.\n\n"
            f"llama-cpp-python GPU-offload hint: {support_hint}\n"
            f"Detected backend libraries: {lib_text}\n"
            f"llama.cpp system info: {system_info[:1200]}\n\n"
            "Install/rebuild a CUDA-capable llama-cpp-python in the SAME Python environment that starts ComfyUI, "
            "then restart ComfyUI. Set gpu_layers=0 only when CPU inference is intentional."
        )

    if requested_gpu:
        _LOGGER.info(
            "[Local GGUF LLM] GPU offload verified: %.1f MiB native VRAM allocated across visible GPUs.",
            total_delta / _MIB,
        )
    return llm, diagnostics


def _load_llama_fast(Llama, load_kwargs, prior_diagnostics=None, reload_count=0):
    """Reload a previously verified model without heavyweight GPU diagnostics."""
    device = None
    free_before = None
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda", int(load_kwargs.get("main_gpu", 0) or 0))
            free_before = _raw_cuda_free_bytes(device)
    except Exception:
        pass

    load_mode = _load_mode_name(load_kwargs)
    host_before = _host_load_snapshot()
    started = time.perf_counter()
    llm = Llama(**load_kwargs)
    elapsed = time.perf_counter() - started
    host_after = _host_load_snapshot()
    host_delta = _host_load_delta(host_before, host_after, load_mode)
    free_after = _raw_cuda_free_bytes(device) if device is not None else None

    diagnostics = dict(prior_diagnostics or {})
    diagnostics.update({
        "native_load_seconds": round(elapsed, 3),
        "load_path": "fast-reload",
        "verification_skipped": True,
        "reload_count": int(reload_count),
        "fast_reload_free_before_mib": round(free_before / _MIB, 1) if free_before is not None else None,
        "fast_reload_free_after_mib": round(free_after / _MIB, 1) if free_after is not None else None,
        "fast_reload_immediate_vram_delta_mib": (
            round(max(0, free_before - free_after) / _MIB, 1)
            if free_before is not None and free_after is not None else None
        ),
        "load_mode": load_mode,
        "model_file_size_bytes": int(os.path.getsize(load_kwargs.get("model_path"))) if load_kwargs.get("model_path") and os.path.exists(load_kwargs.get("model_path")) else 0,
        **host_delta,
    })
    _perf_log(
        "native GGUF FAST reload",
        reload_count=reload_count,
        native_s=elapsed,
        free_before_mib=(free_before / _MIB) if free_before is not None else None,
        free_after_mib=(free_after / _MIB) if free_after is not None else None,
        load_mode=load_mode,
        major_faults=host_delta.get("major_faults_delta"),
        minor_faults=host_delta.get("minor_faults_delta"),
        block_inputs=host_delta.get("block_inputs_delta"),
        page_cache_mib=(host_delta.get("page_cache_after_bytes") / _MIB) if host_delta.get("page_cache_after_bytes") is not None else None,
        cache_hint=host_delta.get("page_cache_hint"),
        wsl=host_delta.get("wsl"),
    )
    if host_delta.get("wsl") and host_delta.get("page_cache_hint") == "storage-backed-faults-observed":
        _LOGGER.warning(
            "[Local GGUF LLM PERF] Fast GGUF reload under WSL required storage-backed page faults; "
            "the model was not fully warm in Linux page cache. OS memory reclaim may be limiting reload speed."
        )
    return llm, diagnostics


def _meta_suffix_value(metadata, suffix, default=0):
    for k, v in metadata.items():
        if str(k).endswith(suffix):
            try:
                return int(v)
            except Exception:
                return default
    return default


def _kv_scalar_bytes(kind):
    # GGML block-format effective payload sizes. These are planning estimates,
    # not exact allocator sizes; actual native VRAM replaces the estimate after load.
    return {
        "f32": 4.0,
        "f16": 2.0,
        "bf16": 2.0,
        "q8_0": 34.0 / 32.0,
        "q5_1": 24.0 / 32.0,
        "q5_0": 22.0 / 32.0,
        "q4_1": 20.0 / 32.0,
        "q4_0": 18.0 / 32.0,
        "iq4_nl": 18.0 / 32.0,
        "Auto": 2.0,
    }.get(kind, 2.0)


def _estimate_kv_vram(metadata, n_ctx, type_k, type_v, location):
    if location != "GPU" or n_ctx <= 0:
        return 0
    blocks = _meta_suffix_value(metadata, ".block_count")
    emb = _meta_suffix_value(metadata, ".embedding_length")
    heads = _meta_suffix_value(metadata, ".attention.head_count")
    kv_heads = _meta_suffix_value(metadata, ".attention.head_count_kv") or heads
    if not blocks or not emb or not heads or not kv_heads:
        return 0
    head_dim = emb / heads
    key_len = _meta_suffix_value(metadata, ".attention.key_length") or head_dim
    value_len = _meta_suffix_value(metadata, ".attention.value_length") or head_dim
    k_dim = kv_heads * key_len
    v_dim = kv_heads * value_len
    return int(n_ctx * blocks * (k_dim * _kv_scalar_bytes(type_k) + v_dim * _kv_scalar_bytes(type_v)))


def _estimate_native_vram(model_path, mmproj_path, metadata, gpu_layers, n_ctx, kv_k, kv_v, kv_location):
    """Conservative first-load estimate for ComfyUI pressure planning."""
    try:
        model_bytes = os.path.getsize(model_path)
    except Exception:
        model_bytes = 0
    blocks = _meta_suffix_value(metadata, ".block_count")
    if int(gpu_layers) == 0:
        weight_gpu = 0
    elif int(gpu_layers) < 0:
        weight_gpu = model_bytes
    elif blocks > 0:
        weight_gpu = int(model_bytes * min(1.0, max(0.0, float(gpu_layers) / float(blocks))))
    else:
        # No architecture metadata: do not under-reserve an explicitly GPU-loaded model.
        weight_gpu = model_bytes
    kv_gpu = _estimate_kv_vram(metadata, int(n_ctx), kv_k, kv_v, kv_location)
    vision_gpu = 0
    if mmproj_path:
        try:
            vision_gpu = os.path.getsize(mmproj_path)
        except Exception:
            pass
    core = weight_gpu + kv_gpu + vision_gpu
    overhead = (384 * _MIB) if core else 0
    overhead += int(min(1536 * _MIB, core * 0.04)) if core else 0
    return max(0, int(core + overhead))


class _NativeLLMResident:
    """All-or-nothing native llama.cpp residency controller.

    This object is deliberately *not* a ComfyUI ModelPatcher/LoadedModel.  It
    caches the exact native load signature and kwargs, owns one Llama context,
    closes that context completely when yielding, and recreates it directly on
    demand.  ComfyUI coordination lives outside this object in the pre-load room
    check and free_memory yield hook.
    """
    def __init__(self, llama_cpp, Llama, load_key, load_kwargs, metadata, family,
                 estimated_vram, model_file_size, main_gpu_index, gpu_layers):
        import torch
        self._llama_cpp = llama_cpp
        self._Llama = Llama
        self._load_key = load_key
        self.signature_id = _load_signature_id(load_key)
        self._load_kwargs = dict(load_kwargs)
        self._metadata = metadata
        self._family = family
        self._estimated_vram = int(max(0, estimated_vram))
        self._model_file_size = int(max(0, model_file_size))
        gpu_expected = int(gpu_layers) != 0 or bool(load_kwargs.get("offload_kqv", False))
        if gpu_expected and torch.cuda.is_available():
            self.load_device = torch.device("cuda", int(main_gpu_index))
        else:
            self.load_device = torch.device("cpu")
        self._llm = None
        self._base_chat_handler = None
        self._diagnostics = None
        self._accounted_size = self._estimated_vram if self.load_device.type != "cpu" else self._model_file_size
        self._observed_vram_bytes = 0
        self._gpu_layers = int(gpu_layers)
        self._verified_once = False
        self._reload_count = 0
        self._unload_count = 0
        self._last_unload_diagnostics = None

    @property
    def llm(self):
        return self._llm

    @property
    def diagnostics(self):
        return self._diagnostics or {}

    @property
    def load_key(self):
        return self._load_key

    @property
    def observed_vram_bytes(self):
        return int(max(0, self._observed_vram_bytes))

    def preload_target_bytes(self):
        """Use measured native allocation for warm reloads; estimate only first load.

        The first-load estimator already includes a substantial fixed/proportional
        overhead. After one verified allocation, a small fixed driver margin is
        safer and usually much less wasteful than reapplying the conservative
        estimate plus ComfyUI's unrelated inference reserve.
        """
        if self.load_device.type == "cpu":
            return 0, "cpu"
        if self._observed_vram_bytes > 0:
            return int(self._observed_vram_bytes + _RELOAD_VRAM_MARGIN), "observed+256MiB"
        return int(self._estimated_vram), "first-load-estimate"

    def _load_native(self):
        with _MODEL_LOCK:
            if self._llm is not None:
                _perf_log(
                    "native resident signature hit; load is no-op",
                    signature=self.signature_id,
                    reload_count=self._reload_count,
                )
                return 0
            total_started = time.perf_counter()
            _perf_log(
                "native resident load begin",
                signature=self.signature_id,
                verified_once=self._verified_once,
                reload_count=self._reload_count,
                estimated_mib=self._estimated_vram / _MIB,
            )
            if self._verified_once:
                self._reload_count += 1
                llm, diag = _load_llama_fast(
                    self._Llama,
                    self._load_kwargs,
                    prior_diagnostics=self._diagnostics,
                    reload_count=self._reload_count,
                )
                path = "direct-fast-reload"
            else:
                llm, diag = _load_llama_verified(
                    self._llama_cpp, self._Llama, self._load_kwargs, self._gpu_layers
                )
                self._verified_once = True
                path = "verified-first-load"
            self._llm = llm
            self._base_chat_handler = getattr(llm, "chat_handler", None)
            diag = dict(diag or {})
            diag["load_path"] = path
            diag["signature_id"] = self.signature_id
            diag["native_residency_policy"] = "all-or-nothing"
            diag["comfyui_loaded_model_registration"] = False
            self._diagnostics = diag

            if self.load_device.type != "cpu":
                verified_delta = int((diag.get("vram_delta_bytes_by_gpu") or {}).get(str(self.load_device.index), 0))
                fast_delta_mib = diag.get("fast_reload_immediate_vram_delta_mib")
                try:
                    fast_delta = int(float(fast_delta_mib) * _MIB) if fast_delta_mib is not None else 0
                except Exception:
                    fast_delta = 0
                delta = max(0, verified_delta, fast_delta)
                if delta > 0:
                    self._observed_vram_bytes = max(self._observed_vram_bytes, delta)
                    self._accounted_size = self._observed_vram_bytes
                diag["observed_vram_bytes"] = int(self._observed_vram_bytes)
                diag["observed_vram_mib"] = round(self._observed_vram_bytes / _MIB, 1)
                diag["next_reload_target_bytes"] = int(self.preload_target_bytes()[0])
                diag["next_reload_target_mib"] = round(self.preload_target_bytes()[0] / _MIB, 1)
                diag["next_reload_target_source"] = self.preload_target_bytes()[1]
            else:
                self._accounted_size = max(self._accounted_size, self._model_file_size)

            if _MODEL_CACHE.get("managed_adapter") is self:
                _MODEL_CACHE.update({
                    "llm": llm,
                    "base_chat_handler": self._base_chat_handler,
                    "load_diagnostics": diag,
                    "metadata": self._metadata,
                    "family": self._family,
                })
            _perf_log(
                "native resident load complete",
                signature=self.signature_id,
                path=path,
                reload_count=self._reload_count,
                native_s=diag.get("native_load_seconds"),
                total_s=time.perf_counter() - total_started,
                accounted_mib=self._accounted_size / _MIB,
                observed_mib=self._observed_vram_bytes / _MIB if self._observed_vram_bytes else None,
                next_target_mib=self.preload_target_bytes()[0] / _MIB if self.load_device.type != "cpu" else None,
                target_source=self.preload_target_bytes()[1],
            )
            return int(self._accounted_size)

    def _unload_native(self, reason="auto-yield", heavy_cleanup=False, required_free_bytes=None):
        with _MODEL_LOCK:
            if self._llm is None:
                _perf_log("native resident unload no-op", signature=self.signature_id, reason=reason)
                return 0
            total_started = time.perf_counter()
            before = int(max(0, self._accounted_size))
            free_before = _raw_cuda_free_bytes(self.load_device) if self.load_device.type == "cuda" else None

            # Break every strong reference we own *before* closing.  The local
            # `llm` variable is then the only intentional owner of the context.
            llm = self._llm
            self._llm = None
            self._base_chat_handler = None
            if _MODEL_CACHE.get("managed_adapter") is self:
                _MODEL_CACHE["llm"] = None
                _MODEL_CACHE["base_chat_handler"] = None

            close_started = time.perf_counter()
            _close_llama_object(llm)
            close_seconds = time.perf_counter() - close_started
            del llm

            free_after_close = _raw_cuda_free_bytes(self.load_device) if self.load_device.type == "cuda" else None
            gc_seconds = 0.0
            cache_seconds = 0.0
            fallback_gc_used = False
            fallback_cache_used = False

            if heavy_cleanup:
                gc_started = time.perf_counter()
                gc.collect()
                gc_seconds = time.perf_counter() - gc_started
                try:
                    import comfy.model_management as mm
                    cache_started = time.perf_counter()
                    mm.soft_empty_cache()
                    cache_seconds = time.perf_counter() - cache_started
                except Exception:
                    pass
            elif self.load_device.type == "cuda" and free_after_close is not None:
                # Normal Auto Yield is intentionally just close()+drop refs. Only
                # invoke Python GC/PyTorch cache cleanup when VRAM demonstrably did
                # not return enough for the caller's requested target (or barely
                # changed at all when no target was supplied).
                target = max(0, int(required_free_bytes or 0))
                delta = max(0, free_after_close - (free_before or 0)) if free_before is not None else 0
                release_failed = (target > 0 and free_after_close < target)
                if target <= 0 and free_before is not None and before >= 256 * _MIB:
                    expected_floor = min(256 * _MIB, max(32 * _MIB, int(before * 0.02)))
                    release_failed = delta < expected_floor
                if release_failed:
                    fallback_gc_used = True
                    gc_started = time.perf_counter()
                    gc.collect()
                    gc_seconds = time.perf_counter() - gc_started
                    free_after_gc = _raw_cuda_free_bytes(self.load_device)
                    # PyTorch cache is unrelated to the native llama allocation.
                    # Only pay for ComfyUI's synchronized cache flush if its
                    # reclaimable allocator blocks can actually cover the remaining
                    # driver-visible shortfall; otherwise the outer ComfyUI
                    # free_memory() call will evict a model anyway.
                    if target > 0 and free_after_gc is not None and free_after_gc < target:
                        try:
                            import comfy.model_management as mm
                            remaining = max(0, target - free_after_gc)
                            breakdown = _comfy_memory_breakdown(mm, self.load_device)
                            reclaimable = int(breakdown.get("torch_reclaimable_bytes") or 0)
                            if reclaimable >= remaining and reclaimable >= 16 * _MIB:
                                fallback_cache_used = True
                                cache_started = time.perf_counter()
                                mm.soft_empty_cache()
                                cache_seconds = time.perf_counter() - cache_started
                            else:
                                _perf_log(
                                    "skip post-LLM-close PyTorch cache flush",
                                    signature=self.signature_id,
                                    remaining_shortfall_mib=remaining / _MIB,
                                    reclaimable_mib=reclaimable / _MIB,
                                )
                        except Exception:
                            pass

            free_after = _raw_cuda_free_bytes(self.load_device) if self.load_device.type == "cuda" else None
            self._unload_count += 1
            total_seconds = time.perf_counter() - total_started
            self._last_unload_diagnostics = {
                "reason": str(reason),
                "signature_id": self.signature_id,
                "unload_count": self._unload_count,
                "released_accounted_bytes": before,
                "close_seconds": round(close_seconds, 4),
                "gc_seconds": round(gc_seconds, 4),
                "soft_empty_cache_seconds": round(cache_seconds, 4),
                "heavy_cleanup": bool(heavy_cleanup),
                "fallback_gc_used": bool(fallback_gc_used),
                "fallback_cache_used": bool(fallback_cache_used),
                "required_free_mib": round(int(required_free_bytes or 0) / _MIB, 1),
                "total_seconds": round(total_seconds, 4),
                "free_before_mib": round(free_before / _MIB, 1) if free_before is not None else None,
                "free_after_close_mib": round(free_after_close / _MIB, 1) if free_after_close is not None else None,
                "free_after_mib": round(free_after / _MIB, 1) if free_after is not None else None,
                "driver_free_delta_mib": (
                    round((free_after - free_before) / _MIB, 1)
                    if free_before is not None and free_after is not None else None
                ),
            }
            _perf_log(
                "native resident full close",
                signature=self.signature_id,
                reason=reason,
                unload_count=self._unload_count,
                close_s=close_seconds,
                gc_s=gc_seconds,
                cache_s=cache_seconds,
                fallback_gc=fallback_gc_used,
                fallback_cache=fallback_cache_used,
                total_s=total_seconds,
                accounted_mib=before / _MIB,
                free_before_mib=(free_before / _MIB) if free_before is not None else None,
                free_after_close_mib=(free_after_close / _MIB) if free_after_close is not None else None,
                free_after_mib=(free_after / _MIB) if free_after is not None else None,
            )
            return before


def _cleanup_llm():
    """Explicitly destroy the native context and clear all cached load state."""
    total_started = time.perf_counter()
    _perf_log("explicit cleanup begin", mode=_MODEL_CACHE.get("mode"))
    resident_ctl = _MODEL_CACHE.get("managed_adapter")
    cleanup_done = False
    if resident_ctl is not None:
        try:
            resident_ctl._unload_native(reason="explicit-cleanup", heavy_cleanup=True)
            cleanup_done = True
        except Exception as e:
            _LOGGER.warning("[Local GGUF LLM] Native resident explicit cleanup failed: %s", e)
    else:
        _close_llama_object(_MODEL_CACHE.get("llm"))

    with _MODEL_LOCK:
        _MODEL_CACHE.update({
            "key": None,
            "llm": None,
            "metadata": None,
            "family": None,
            "base_chat_handler": None,
            "managed_adapter": None,
            "load_diagnostics": None,
            "mode": None,
        })

    # Direct/unmanaged contexts do not have the resident controller's explicit
    # heavy cleanup. Keep Stop/Reload/model-change teardown conservative there.
    gc_seconds = 0.0
    cache_seconds = 0.0
    if not cleanup_done:
        gc_started = time.perf_counter()
        gc.collect()
        gc_seconds = time.perf_counter() - gc_started
        try:
            import comfy.model_management as mm
            cache_started = time.perf_counter()
            mm.soft_empty_cache()
            cache_seconds = time.perf_counter() - cache_started
        except Exception:
            pass
    _perf_log(
        "explicit cleanup complete",
        gc_s=gc_seconds,
        cache_s=cache_seconds,
        total_s=time.perf_counter() - total_started,
    )

def _tensor_batch(image):
    if image is None:
        return None
    arr = image.detach().cpu().numpy() if hasattr(image, "detach") else image
    try:
        ndim = arr.ndim
    except Exception:
        return None
    if ndim == 3:
        arr = arr[None, ...]
    if getattr(arr, "ndim", 0) != 4:
        raise ValueError("Vision IMAGE input must be an HWC image or BHWC image batch.")
    return arr


def _encode_image_samples(arr, indices, max_edge=1536):
    try:
        import numpy as np
        from PIL import Image
    except Exception as e:
        raise RuntimeError("Vision input requires Pillow and NumPy (normally included with ComfyUI).") from e
    out = []
    for idx in indices:
        sample = arr[int(idx)]
        sample = np.clip(sample * 255.0, 0, 255).astype(np.uint8)
        pil = Image.fromarray(sample)
        if max(pil.size) > max_edge:
            scale = max_edge / max(pil.size)
            pil = pil.resize((max(1, int(pil.width * scale)), max(1, int(pil.height * scale))), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        pil.save(buf, format="PNG", optimize=True)
        out.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"))
    return out


def _image_to_data_uris(image, max_images=4, max_edge=1536):
    """Encode the first N images of a ComfyUI IMAGE batch in stable input order."""
    arr = _tensor_batch(image)
    if arr is None:
        return []
    count = min(len(arr), max(1, int(max_images)))
    return _encode_image_samples(arr, range(count), max_edge)


def _video_frames_to_data_uris(video_frames, max_frames=24, max_edge=1536):
    """Evenly sample a ComfyUI IMAGE batch interpreted as ordered video frames."""
    arr = _tensor_batch(video_frames)
    if arr is None or len(arr) == 0:
        return []
    n = len(arr)
    take = min(n, max(1, int(max_frames)))
    if take == 1:
        indices = [0]
    else:
        indices = []
        for i in range(take):
            idx = int(round(i * (n - 1) / (take - 1)))
            if not indices or idx != indices[-1]:
                indices.append(idx)
    return _encode_image_samples(arr, indices, max_edge)


def _vision_modules():
    """Return multimodal modules exposed by current and older llama-cpp-python builds."""
    modules = []
    for modname in ("llama_cpp.llama_multimodal", "llama_cpp.llama_chat_format", "llama_cpp"):
        try:
            mod = __import__(modname, fromlist=["*"])
            if mod not in modules:
                modules.append(mod)
        except Exception:
            pass
    return modules


def _handler_class(names):
    for name in names:
        for mod in _vision_modules():
            cls = getattr(mod, name, None)
            if cls is not None:
                return cls
    return None


def _handler_params(cls):
    try:
        return dict(inspect.signature(cls.__init__).parameters)
    except Exception:
        try:
            return dict(inspect.signature(cls).parameters)
        except Exception:
            return {}


def _has_var_keyword(params):
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def _mmproj_keyword_for_handler(cls):
    """Pick mmproj_path vs clip_model_path across upstream/fork API generations."""
    params = _handler_params(cls)
    if "mmproj_path" in params:
        return "mmproj_path"
    if "clip_model_path" in params and not _has_var_keyword(params):
        return "clip_model_path"

    # Several model-specific handlers expose only **kwargs and forward the path
    # to MTMDChatHandler. Inspect the installed MTMD base class before choosing.
    if _has_var_keyword(params):
        mtmd = _handler_class(["MTMDChatHandler"])
        if mtmd is not None and mtmd is not cls:
            base_params = _handler_params(mtmd)
            if "mmproj_path" in base_params:
                return "mmproj_path"
            if "clip_model_path" in base_params:
                return "clip_model_path"

    if "clip_model_path" in params:
        return "clip_model_path"
    # Upstream abetlen releases historically use clip_model_path; newer JamePeng
    # MTMD releases explicitly advertise mmproj_path and are caught above.
    return "clip_model_path"


def _instantiate_handler(cls, mmproj_path, extra):
    if cls is None:
        return None
    params = _handler_params(cls)
    accepts_var_kw = _has_var_keyword(params)
    path_key = _mmproj_keyword_for_handler(cls)
    base = {path_key: mmproj_path}

    # Try the richest supported configuration first. Do not blindly send all
    # extras into old handlers unless they accept **kwargs.
    supported_extra = extra if accepts_var_kw else {k: v for k, v in extra.items() if k in params}
    candidates = [
        {**base, **supported_extra},
        {**base, **{k: v for k, v in supported_extra.items() if k in ("verbose", "extra_template_arguments")}},
        {**base, **({"verbose": extra.get("verbose", False)} if (accepts_var_kw or "verbose" in params) else {})},
        base,
    ]
    # Deduplicate candidate dictionaries while preserving order.
    unique = []
    seen = set()
    for kw in candidates:
        key = tuple(sorted((k, repr(v)) for k, v in kw.items()))
        if key not in seen:
            seen.add(key)
            unique.append(kw)

    last = None
    for kw in unique:
        try:
            return cls(**kw)
        except TypeError as e:
            last = e
    if last:
        raise last
    return None


def _available_vision_handlers():
    names = (
        "GenericMTMDChatHandler", "MTMDChatHandler", "Qwen35ChatHandler",
        "Qwen3VLChatHandler", "Qwen25VLChatHandler", "Gemma4ChatHandler",
        "Gemma3ChatHandler", "Llava16ChatHandler", "Llava15ChatHandler",
        "MiniCPMV46ChatHandler", "MiniCPMv45ChatHandler", "MiniCPMv26ChatHandler",
        "GLM46VChatHandler", "GLM41VChatHandler", "LFM2VLChatHandler", "LFM25VLChatHandler",
        "GraniteDoclingChatHandler", "PaddleOCRChatHandler", "Step3VLChatHandler",
        "MoondreamChatHandler", "NanoLlavaChatHandler", "Llama3VisionAlphaChatHandler",
    )
    found = []
    for name in names:
        if _handler_class([name]) is not None:
            found.append(name)
    return found


def _vision_handler(family, mmproj_path, thinking_mode, preserve_thinking, reasoning_effort, verbose=False):
    extra_template = _thinking_template_kwargs(
        family, thinking_mode, preserve_thinking, reasoning_effort
    )
    caps = capabilities_for_family(family)
    preferred = list(caps.get("preferred_chat_handlers") or [])

    # Prefer a family-specific handler when the installed binding provides one.
    # These handlers can expose model-specific controls (e.g. Qwen force_reasoning)
    # that generic MTMD cannot always infer. Generic MTMD remains the compatibility
    # fallback for new/unknown VLMs.
    names = [*preferred, "GenericMTMDChatHandler", "MTMDChatHandler", "Llava16ChatHandler", "Llava15ChatHandler"]
    cls = _handler_class(names)
    if cls is None:
        try:
            import llama_cpp
            version = getattr(llama_cpp, "__version__", "unknown")
        except Exception:
            version = "unknown"
        found = _available_vision_handlers()
        raise RuntimeError(
            "No compatible multimodal handler was found in the installed llama-cpp-python "
            f"(version {version}). Vision input requires an MTMD/vision-capable build. "
            f"Detected handlers: {', '.join(found) if found else 'none'}."
        )

    extra = {
        "verbose": verbose,
        "extra_template_arguments": extra_template,
    }
    # Model-specific handler switches mirrored from current llama-cpp VLM handlers.
    if family == "qwen3-vl" and thinking_mode in {"Enabled", "Disabled"}:
        extra["force_reasoning"] = thinking_mode == "Enabled"
    elif family in {"qwen3.5", "qwen3.6", "qwen3.8", "glm4.6v", "minicpm-v4.5", "minicpm-v4.6"} and thinking_mode in {"Enabled", "Disabled"}:
        extra["enable_thinking"] = thinking_mode == "Enabled"
    elif "enable_thinking" in extra_template:
        extra["enable_thinking"] = extra_template["enable_thinking"]
    if "preserve_thinking" in extra_template:
        extra["preserve_thinking"] = extra_template["preserve_thinking"]

    return _instantiate_handler(cls, mmproj_path, extra)




def _embedded_template_chat_handler(llm, template_kwargs):
    """Wrap the GGUF embedded Jinja template and inject template kwargs.

    llama-cpp-python's direct Llama.create_chat_completion API still does not
    expose arbitrary chat-template kwargs per request.  This wrapper makes
    model controls such as Qwen enable_thinking and GPT-OSS/Qwen3.8
    reasoning_effort actually reach the embedded GGUF template.
    """
    if not template_kwargs:
        return None
    try:
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter, chat_formatter_to_chat_completion_handler
    except Exception:
        return None
    metadata = getattr(llm, "metadata", {}) or {}
    chat_template = metadata.get("tokenizer.chat_template")
    if not chat_template:
        return None
    model_obj = getattr(llm, "_model", None)

    def token_text(token_id):
        if token_id is None or token_id == -1 or model_obj is None or not hasattr(model_obj, "token_get_text"):
            return ""
        try:
            return model_obj.token_get_text(token_id)
        except Exception:
            return ""

    try:
        eos_id = llm.token_eos()
        bos_id = llm.token_bos()
        eot_id = llm.token_eot() if hasattr(llm, "token_eot") else -1
        stop_ids = [x for x in (eos_id, eot_id) if x is not None and x != -1] or None
        formatter = Jinja2ChatFormatter(
            template=chat_template,
            eos_token=token_text(eos_id),
            bos_token=token_text(bos_id),
            stop_token_ids=stop_ids,
        )
    except Exception:
        return None

    def controlled_formatter(*, messages, **kwargs):
        kwargs.update(template_kwargs)
        return formatter(messages=messages, **kwargs)

    try:
        return chat_formatter_to_chat_completion_handler(controlled_formatter)
    except Exception:
        return None


def _thinking_template_kwargs(family, thinking_mode, preserve_thinking, reasoning_effort):
    """Return only template arguments the detected family is known to use."""
    x = {}

    # Qwen3-family templates implement a real hard thinking switch.
    if family in ("qwen3", "qwen3-vl", "qwen3.5", "qwen3.6", "qwen3.8"):
        if thinking_mode == "Enabled":
            x["enable_thinking"] = True
        elif thinking_mode == "Disabled":
            x["enable_thinking"] = False

    # Qwen3.8 uniquely documents preserved historical reasoning and trained
    # low/medium/xhigh effort levels.
    if family == "qwen3.8":
        x["preserve_thinking"] = bool(preserve_thinking)
        if reasoning_effort not in ("Auto", ""):
            x["reasoning_effort"] = reasoning_effort.lower()

    # GPT-OSS consumes low/medium/high reasoning effort through Harmony/template
    # arguments. It does not use Qwen's enable_thinking switch.
    if family == "gpt-oss" and reasoning_effort not in ("Auto", ""):
        x["reasoning_effort"] = reasoning_effort.lower()

    # Current Nemotron templates expose an enable_thinking-style control in
    # supported llama.cpp conversions. Do not invent reasoning-effort levels.
    if family == "nemotron3":
        if thinking_mode == "Enabled":
            x["enable_thinking"] = True
        elif thinking_mode == "Disabled":
            x["enable_thinking"] = False

    return x



_REASONING_FIELD_NAMES = (
    "reasoning_content",
    "reasoning",
    "thinking_content",
    "thinking",
    "analysis",
    "thoughts",
    "thought",
)

_REASONING_BLOCK_TYPES = {
    "reasoning", "thinking", "analysis", "thought", "reasoning_text",
    "thinking_text", "analysis_text", "summary_text",
}


def _as_mapping(value):
    """Best-effort conversion for dict / pydantic / OpenAI-style response objects."""
    if isinstance(value, dict):
        return value
    for attr in ("model_dump", "dict"):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                out = fn()
                if isinstance(out, dict):
                    return out
            except Exception:
                pass
    try:
        d = vars(value)
        if isinstance(d, dict):
            return d
    except Exception:
        pass
    return {}


def _text_value(value):
    """Extract readable text without turning structured objects into Python reprs."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return str(value)
    if isinstance(value, (list, tuple)):
        return "\n".join(x for x in (_text_value(v) for v in value) if x).strip()
    m = _as_mapping(value)
    if m:
        for key in ("text", "content", "value", "thinking", "reasoning", "analysis", "summary"):
            if key in m:
                text = _text_value(m.get(key))
                if text:
                    return text
        return ""
    return str(value)


def _merge_reasoning(parts):
    """Join reasoning fragments while suppressing exact duplicates."""
    out = []
    seen = set()
    for part in parts:
        text = _text_value(part).strip()
        if not text:
            continue
        # Do not duplicate the same trace when a backend returns it both in a
        # structured field and inline in <think> tags.
        key = re.sub(r"\s+", " ", text).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return "\n\n".join(out).strip()


def _content_and_reasoning(value):
    """Split OpenAI-style multimodal/content arrays into final text and thoughts."""
    if value is None:
        return "", ""
    if isinstance(value, str):
        return value, ""
    if not isinstance(value, (list, tuple)):
        return _text_value(value), ""

    content_parts = []
    reasoning_parts = []
    for block in value:
        m = _as_mapping(block)
        if not m:
            text = _text_value(block)
            if text:
                content_parts.append(text)
            continue

        block_type = str(m.get("type") or m.get("role") or "").strip().lower()
        if block_type in _REASONING_BLOCK_TYPES or any(k in m for k in _REASONING_FIELD_NAMES):
            reason = ""
            for key in _REASONING_FIELD_NAMES + ("text", "content", "value", "summary"):
                if key in m:
                    reason = _text_value(m.get(key))
                    if reason:
                        break
            if reason:
                reasoning_parts.append(reason)
            continue

        text = ""
        for key in ("text", "content", "value"):
            if key in m:
                text = _text_value(m.get(key))
                if text:
                    break
        if text:
            content_parts.append(text)

    return "".join(content_parts).strip(), _merge_reasoning(reasoning_parts)


def _split_reasoning_markup(text):
    """Return (clean final response, inline reasoning, extraction modes).

    Supports the formats currently encountered in llama.cpp / GGUF chat models:
    - <think>...</think>, <thinking>, <reasoning>, <analysis>
    - Qwen-style delimiter-only `reasoning ... </think> final`
    - an unfinished opening reasoning tag when generation hits max_tokens
    - Gemma-style `<|channel>thought ... <channel|>`
    - Harmony/GPT-OSS style analysis/reasoning channels

    The clean response never contains reasoning delimiters.
    """
    raw = str(text or "")
    cleaned = raw
    reasoning_parts = []
    modes = []

    # XML-ish reasoning blocks. Extract all occurrences, not only the first.
    xml_pattern = re.compile(
        r"<(?P<tag>think|thinking|reasoning|analysis)(?:\s[^>]*)?>\s*(?P<body>.*?)\s*</(?P=tag)\s*>",
        flags=re.I | re.S,
    )

    def _xml_repl(match):
        body = (match.group("body") or "").strip()
        if body:
            reasoning_parts.append(body)
        modes.append("xml_tag")
        return ""

    cleaned = xml_pattern.sub(_xml_repl, cleaned)

    # Gemma-family native thought channel, seen in llama-cpp-python where the
    # Python chat handler has not converted it to a structured field yet.
    gemma_channel = re.compile(
        r"<\|channel>\s*(?:thought|thinking|analysis|reasoning)\s*(.*?)\s*<channel\|>",
        flags=re.I | re.S,
    )

    def _gemma_repl(match):
        body = (match.group(1) or "").strip()
        if body:
            reasoning_parts.append(body)
        modes.append("native_thought_channel")
        return ""

    cleaned = gemma_channel.sub(_gemma_repl, cleaned)

    # GPT-OSS Harmony-like channels. Keep final/content channels, extract only
    # analysis/reasoning/thought channels. This is intentionally permissive so
    # it also handles variants using <|end|> or another following channel token.
    harmony_reason = re.compile(
        r"<\|channel\|>\s*(?:analysis|reasoning|thought|thinking)\s*<\|message\|>"
        r"(.*?)(?=<\|end\|>|<\|start\|>|<\|channel\|>|$)",
        flags=re.I | re.S,
    )

    def _harmony_repl(match):
        body = (match.group(1) or "").strip()
        if body:
            reasoning_parts.append(body)
        modes.append("harmony_reasoning_channel")
        return ""

    cleaned = harmony_reason.sub(_harmony_repl, cleaned)

    # If the backend/template prefilled the opening <think>, generated content
    # can legitimately contain only `reasoning text </think> final answer`.
    # This is common in Qwen-family templates and was the main hole in v0.5.0.
    close_only = re.search(r"</(?:think|thinking|reasoning|analysis)\s*>", cleaned, flags=re.I)
    open_any = re.search(r"<(?:think|thinking|reasoning|analysis)(?:\s[^>]*)?>", cleaned, flags=re.I)
    if close_only and (not open_any or open_any.start() > close_only.start()):
        prefix = cleaned[:close_only.start()]
        suffix = cleaned[close_only.end():]
        prefix = re.sub(r"^\s*<(?:think|thinking|reasoning|analysis)(?:\s[^>]*)?>", "", prefix, flags=re.I).strip()
        if prefix:
            reasoning_parts.append(prefix)
        cleaned = suffix
        modes.append("closing_delimiter")

    # If generation stops during the thinking phase, preserve the partial trace
    # on the thinking output and remove it from the final response.
    orphan_open = re.search(r"<(?:think|thinking|reasoning|analysis)(?:\s[^>]*)?>", cleaned, flags=re.I)
    if orphan_open:
        before = cleaned[:orphan_open.start()]
        after = cleaned[orphan_open.end():]
        if after.strip():
            reasoning_parts.append(after.strip())
        cleaned = before
        modes.append("unfinished_reasoning")

    # Strip any leftover standalone reasoning delimiters from final response.
    cleaned = re.sub(r"</?(?:think|thinking|reasoning|analysis)(?:\s[^>]*)?>", "", cleaned, flags=re.I)
    cleaned = re.sub(r"<\|channel>\s*(?:thought|thinking|analysis|reasoning)\s*|<channel\|>", "", cleaned, flags=re.I)

    # Harmony final channel wrappers should never leak into the response text.
    # Remove role framing as a unit first so a bare `assistant` label does not
    # survive after stripping the special tokens.
    cleaned = re.sub(r"<\|end\|>\s*<\|start\|>\s*assistant\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"<\|start\|>\s*assistant\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"<\|channel\|>\s*(?:final|content)\s*<\|message\|>", "", cleaned, flags=re.I)
    cleaned = re.sub(r"<\|(?:end|start)\|>", "", cleaned, flags=re.I)

    return cleaned.strip(), _merge_reasoning(reasoning_parts), modes


def _extract_response(result):
    """Normalize llama-cpp-python completion variants.

    Returns raw response text, structured reasoning text, and source labels.
    Inline reasoning markup is deliberately handled separately by
    `_split_reasoning_markup` so response Strip/Keep remains a local UI choice.
    """
    root = _as_mapping(result)
    choices = root.get("choices") or []
    choice = _as_mapping(choices[0]) if choices else {}
    msg = _as_mapping(choice.get("message"))

    content_value = msg.get("content") if "content" in msg else choice.get("text", "")
    content, content_reasoning = _content_and_reasoning(content_value)

    reasoning_parts = []
    sources = []
    if content_reasoning:
        reasoning_parts.append(content_reasoning)
        sources.append("content_blocks")

    # Different llama.cpp / OpenAI-compatible layers have used different field
    # names and nesting levels over time. Search all plausible locations.
    for label, container in (("message", msg), ("choice", choice), ("result", root)):
        for key in _REASONING_FIELD_NAMES:
            if key not in container:
                continue
            value = _text_value(container.get(key)).strip()
            if value:
                reasoning_parts.append(value)
                sources.append(f"{label}.{key}")

        # Some clients expose arrays of thinking blocks instead of a single
        # reasoning_content string.
        for key in ("thinking_blocks", "reasoning_blocks", "analysis_blocks"):
            if key in container:
                _, block_reasoning = _content_and_reasoning(container.get(key))
                if not block_reasoning:
                    block_reasoning = _text_value(container.get(key)).strip()
                if block_reasoning:
                    reasoning_parts.append(block_reasoning)
                    sources.append(f"{label}.{key}")

    return str(content or ""), _merge_reasoning(reasoning_parts), sources


def _llama_seed_from_comfy(seed):
    """Map ComfyUI's standard unsigned 64-bit seed to llama.cpp's uint32 seed.

    ComfyUI exposes 0..2^64-1 seeds, while llama.cpp's sampler seed is uint32_t
    and reserves 0xFFFFFFFF as LLAMA_DEFAULT_SEED (random). SplitMix64 gives a
    stable, well-distributed 32-bit value from the full Comfy seed. Avoiding the
    reserved all-ones value guarantees that every explicit Comfy seed remains
    deterministic, including 0xFFFFFFFF and larger 64-bit values.
    """
    x = int(seed) & 0xFFFFFFFFFFFFFFFF
    z = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z ^= z >> 31
    out = int(z & 0xFFFFFFFF)
    if out == 0xFFFFFFFF:
        out = 0xFFFFFFFE
    return out


class LocalGGUFLLMAPI:
    """Stable facade passed between ComfyUI custom nodes.

    The facade intentionally does not expose a raw llama_cpp.Llama pointer.  The
    native context may be reloaded, evicted, or replaced as workflows execute, so
    downstream nodes talk to this object instead.  Query methods return copies of
    the effective configuration and ``generate`` re-enters the normal Local GGUF
    execution path, which safely reuses the persistent model when its load settings
    still match.
    """

    API_TYPE = "LOCAL_GGUF_LLM_API"
    API_VERSION = 1

    _GENERATION_OVERRIDES = {
        "thinking_mode", "reasoning_effort", "preserve_thinking", "thinking_output",
        "temperature", "top_p", "top_k", "min_p", "typical_p", "repeat_penalty",
        "presence_penalty", "frequency_penalty", "max_tokens", "seed",
        "last_n_tokens", "tfs_z", "mirostat_mode", "mirostat_tau", "mirostat_eta",
        "penalize_newline", "stop_sequences", "vision_max_images", "vision_max_frames", "vision_max_edge",
        "verbose",
    }

    def __init__(self, settings, call_config, load_key):
        self._settings = copy.deepcopy(settings)
        self._call_config = copy.deepcopy(call_config)
        self._load_key = load_key
        self._call_lock = threading.RLock()

    def __repr__(self):
        model = self._settings.get("model", {}).get("name", "unknown")
        return f"<LocalGGUFLLMAPI v{self.API_VERSION} model={model!r} loaded={self.is_loaded()}>"

    def get_settings(self):
        """Return a read-only-by-copy snapshot of all effective source settings."""
        return copy.deepcopy(self._settings)

    def get_model_settings(self):
        return copy.deepcopy(self._settings.get("model", {}))

    def get_generation_settings(self):
        return copy.deepcopy(self._settings.get("generation", {}))

    def get_memory_settings(self):
        return copy.deepcopy(self._settings.get("memory", {}))

    def get_prompting_settings(self):
        return copy.deepcopy(self._settings.get("prompting", {}))

    def get_advanced_settings(self):
        return copy.deepcopy(self._settings.get("advanced", {}))

    def get_model_info(self):
        """Alias intended for downstream nodes that only need model identity/capabilities."""
        out = self.get_model_settings()
        out["capabilities"] = copy.deepcopy(
            capabilities_for_family(out.get("detected_family", "generic"))
        )
        return out

    def get(self, path, default=None):
        """Query a setting with ``section.key`` notation, e.g. memory.context_size."""
        if not path:
            return default
        value = self._settings
        for part in str(path).split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return copy.deepcopy(value)

    def status(self):
        """Return live residency status without exposing the native llama object."""
        with _MODEL_LOCK:
            cache_key_matches = _MODEL_CACHE.get("key") == self._load_key
            adapter = _MODEL_CACHE.get("managed_adapter")
            managed_llm = getattr(adapter, "llm", None) if adapter is not None else None
            native_llm = _MODEL_CACHE.get("llm")
            resident = managed_llm if managed_llm is not None else native_llm
            loaded = bool(cache_key_matches and resident is not None)
            cache_mode = _MODEL_CACHE.get("mode")
            cache_family = _MODEL_CACHE.get("family")

        model = self._settings.get("model", {})
        memory = self._settings.get("memory", {})
        return {
            "api_type": self.API_TYPE,
            "api_version": self.API_VERSION,
            "loaded": loaded,
            "configured_model": model.get("name"),
            "configured_family": model.get("detected_family"),
            "configured_vision": model.get("resolved_vision"),
            "configured_management": memory.get("model_retention"),
            "active_cache_mode": cache_mode,
            "active_cache_family": cache_family,
            "active_cache_matches": bool(cache_key_matches),
        }

    def is_loaded(self):
        return bool(self.status()["loaded"])

    def generate(self, prompt=None, system_prompt=None, image=None, video_frames=None, messages=None, **overrides):
        """Generate with the configured LLM while preserving source-node settings.

        Only request-time/generation controls may be overridden.  Model and memory
        allocation controls remain owned by the source Local GGUF LLM node so a
        downstream node cannot unexpectedly swap models or change KV allocation.
        Overrides are temporary and never mutate the source node or this API's
        stored settings.

        Returns a dictionary with ``response``, ``thinking``, parsed ``info``, and
        ``tokens``.  The nested API object produced by the internal execution is
        intentionally discarded.
        """
        unknown = sorted(set(overrides) - self._GENERATION_OVERRIDES)
        if unknown:
            raise ValueError(
                "Unsupported LocalGGUFLLMAPI override(s): " + ", ".join(unknown) + ". "
                "Model/memory settings are read-only through the API; configure them on the source node."
            )

        args = copy.deepcopy(self._call_config)
        # The source has already resolved both preset layers.  Force Custom for
        # programmatic calls so a temporary sampler override is not overwritten by
        # preset reapplication inside LocalGGUFLLM.generate().
        args["model_preset"] = "Custom"
        args["memory_preset"] = "Custom"
        if prompt is not None:
            args["prompt"] = str(prompt)
        if system_prompt is not None:
            args["system_prompt"] = str(system_prompt)
        args.update(overrides)
        if image is not None:
            args["image"] = image
        if video_frames is not None:
            args["video_frames"] = video_frames
        if messages is not None:
            if not isinstance(messages, (list, tuple)):
                raise TypeError("messages must be a list of OpenAI-style chat message dictionaries.")
            args["messages_override"] = copy.deepcopy(list(messages))

        # Local lock gives a clear single-service call contract for linked nodes;
        # the global model lock additionally serializes access across all facades.
        with self._call_lock:
            response, thinking, info_json, tokens, _api = LocalGGUFLLM().generate(**args)
        try:
            info = json.loads(info_json)
        except Exception:
            info = {"raw": info_json}
        return {
            "response": response,
            "thinking": thinking,
            "info": info,
            "tokens": int(tokens),
        }

    def query(self, prompt=None, system_prompt=None, image=None, video_frames=None, messages=None, **overrides):
        """Convenience alias for ``generate`` used by downstream processing nodes."""
        return self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            image=image,
            video_frames=video_frames,
            messages=messages,
            **overrides,
        )

    def generate_text(self, prompt=None, system_prompt=None, image=None, video_frames=None, messages=None, **overrides):
        """Convenience helper returning only final response text."""
        return self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            image=image,
            video_frames=video_frames,
            messages=messages,
            **overrides,
        )["response"]


class LocalGGUFLLM:
    @classmethod
    def INPUT_TYPES(cls):
        models, vision = _model_lists()
        model_presets = ["Auto (Detected)", "Custom"] + list(MODEL_PRESETS.keys())
        memory_presets = ["Custom"] + list(MEMORY_PRESETS.keys())
        gpu_choices = _gpu_choices()
        return {
            "required": {
                # Model + generation
                "model": (models,),
                "vision_model": (vision,),
                "model_preset": (model_presets, {"default": "Auto (Detected)", "tooltip": "Applies model-recommended sampling and chat-template behavior only. Editing an owned setting switches this to Custom."}),
                "thinking_mode": (["Auto", "Enabled", "Disabled"], {"default": "Auto"}),
                "reasoning_effort": (["Auto", "Low", "Medium", "High", "XHigh"], {"default": "Auto"}),
                "preserve_thinking": ("BOOLEAN", {"default": True, "tooltip": "Qwen3.8 history setting: preserve prior assistant reasoning in future chat turns. This single-shot node normally has no assistant history, so it matters only if history support is added/used. Distinct from response Strip/Keep."}),
                "thinking_output": (["Strip", "Keep"], {"default": "Strip", "tooltip": "Controls only the response output. The thinking output ALWAYS contains extracted reasoning when the model/backend returns it. Strip removes reasoning tags/channels from response; Keep leaves the raw response untouched."}),
                "chat_format": (["Auto (GGUF embedded)", "chatml", "llama-3", "mistral-instruct", "gemma", "qwen", "Custom"], {"default": "Auto (GGUF embedded)"}),
                "custom_chat_format": ("STRING", {"default": "", "multiline": False}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 5.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 10000}),
                "min_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "typical_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 4096, "min": -1, "max": 262144, "tooltip": "Task-specific generation limit; intentionally not changed by Model Preset."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True, "tooltip": "Standard ComfyUI seed. Use control after generate for fixed, increment, decrement, or randomize behavior. Intentionally not changed by Model Preset."}),

                # Memory
                "memory_preset": (memory_presets, {"default": "Balanced"}),
                "context_size": ("INT", {"default": 32768, "min": 0, "max": 1048576, "step": 1024}),
                "kv_cache_k": (["Auto", "f32", "f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"], {"default": "q8_0"}),
                "kv_cache_v": (["Auto", "f32", "f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"], {"default": "q5_0"}),
                "kv_cache_location": (["GPU", "CPU"], {"default": "GPU"}),
                "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 999}),
                "flash_attention": ("BOOLEAN", {"default": True}),
                "prompt_batch_size": ("INT", {"default": 2048, "min": 1, "max": 32768}),
                "memory_batch_size": ("INT", {"default": 512, "min": 1, "max": 32768}),
                "use_mmap": ("BOOLEAN", {"default": True}),
                "use_mlock": ("BOOLEAN", {"default": False}),
                "model_retention": (["Persistent (Driver Managed)", "ComfyUI Managed", "Unload After Run"], {"default": "Persistent (Driver Managed)", "tooltip": "Persistent keeps llama.cpp resident until settings change or explicit unload. ComfyUI Managed/Auto Yield also keeps a native all-or-nothing context, but closes it first when ComfyUI needs VRAM and reloads directly from the cached signature on the next request. Unload After Run closes after every generation."}),

                # Prompting
                "system_prompt": ("STRING", {"default": "You are a helpful assistant.", "multiline": True, "dynamicPrompts": False}),
                "prompt": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),

                # Advanced / diagnostics
                "split_mode": (["None (single GPU)", "Layer", "Row", "Tensor"], {"default": "None (single GPU)", "tooltip": "Single GPU is simplest. Multi-GPU Layer/Row/Tensor split is supported in Persistent and Unload modes; ComfyUI Managed requires single-GPU accounting when multiple accelerators are visible."}),
                "main_gpu": (gpu_choices, {"default": gpu_choices[0], "tooltip": "Logical accelerator used as llama.cpp main_gpu. Names follow the GPU numbering visible to ComfyUI."}),
                "tensor_split": ("STRING", {"default": "", "multiline": False}),
                "threads": ("INT", {"default": 0, "min": 0, "max": 512}),
                "threads_batch": ("INT", {"default": 0, "min": 0, "max": 512}),
                "last_n_tokens": ("INT", {"default": 64, "min": 0, "max": 32768}),
                "tfs_z": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mirostat_mode": (["Off", "v1", "v2"], {"default": "Off"}),
                "mirostat_tau": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "mirostat_eta": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "penalize_newline": ("BOOLEAN", {"default": True}),
                "op_offload": (["Auto", "Enabled", "Disabled"], {"default": "Auto"}),
                "swa_full": (["Auto", "Enabled", "Disabled"], {"default": "Auto"}),
                "rope_freq_base": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000000.0, "step": 1.0}),
                "rope_freq_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 16.0, "step": 0.01}),
                "yarn_ext_factor": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 16.0, "step": 0.01}),
                "yarn_attn_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 16.0, "step": 0.01}),
                "yarn_beta_fast": ("FLOAT", {"default": 32.0, "min": 0.0, "max": 128.0, "step": 0.1}),
                "yarn_beta_slow": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 128.0, "step": 0.1}),
                "yarn_orig_ctx": ("INT", {"default": 0, "min": 0, "max": 1048576, "step": 1024}),
                "stop_sequences": ("STRING", {"default": "", "multiline": False}),
                "vision_max_images": ("INT", {"default": 4, "min": 1, "max": 32, "tooltip": "Maximum still images accepted from the IMAGE batch."}),
                "vision_max_frames": ("INT", {"default": 24, "min": 1, "max": 1024, "tooltip": "Evenly sample at most this many frames from the optional Video Frames IMAGE batch."}),
                "vision_max_edge": ("INT", {"default": 1536, "min": 256, "max": 4096, "step": 64, "tooltip": "Downscale vision inputs so their longest edge does not exceed this value."}),
                "verbose": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "One image or an IMAGE batch. Multiple images are preserved in batch order."}),
                "video_frames": ("IMAGE", {"tooltip": "Ordered video frames as an IMAGE batch. Frames are evenly sampled using Vision Max Frames."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "LOCAL_GGUF_LLM_API")
    RETURN_NAMES = ("response", "thinking", "info", "tokens", "api")
    FUNCTION = "generate"
    CATEGORY = "LLM/Local GGUF"
    DESCRIPTION = "Run local GGUF LLMs from models/llm and expose a live API facade for linked custom nodes."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Supplement ComfyUI's normal input cache key with external GGUF state.

        ComfyUI already includes every ordinary node input in its cache signature,
        including the standard seed widget. The frontend's normal "control after
        generate" behavior is therefore authoritative: Fixed keeps the same seed
        and remains cacheable, while Randomize/Increment/Decrement changes the seed
        and naturally invalidates the cache.

        This method only fingerprints external files that ComfyUI cannot observe as
        normal inputs, so replacing a GGUF/mmproj under the same filename invalidates
        the previous output.
        """
        def file_state(name):
            if not name or name in (_NONE, _AUTO_VISION, "No GGUF models found"):
                return "none"
            try:
                path = _full_path(str(name))
                st = os.stat(path)
                # mtime_ns + size catches normal model replacement/update without
                # hashing multi-gigabyte GGUF files on every queue.
                return f"{os.path.realpath(path)}|{int(st.st_size)}|{int(st.st_mtime_ns)}"
            except Exception as e:
                # A missing/replaced file must also invalidate a prior successful
                # cache entry, while remaining stable if the failure state persists.
                return f"missing:{name}:{type(e).__name__}"

        model = kwargs.get("model")
        vision_selection = kwargs.get("vision_model", _NONE)
        resolved_vision = None
        if vision_selection == _AUTO_VISION:
            try:
                resolved_vision = _find_matching_mmproj(str(model or ""))
            except Exception:
                resolved_vision = None
        elif vision_selection not in (None, "", _NONE):
            resolved_vision = vision_selection

        # Keep the fingerprint a plain JSON-safe string. Normal ComfyUI input
        # hashing still carries every explicit setting and prompt alongside it.
        return "local-gguf-cache-v1|model=" + file_state(model) + "|vision=" + file_state(resolved_vision)

    def generate(self, model, vision_model, model_preset, thinking_mode, reasoning_effort,
                 preserve_thinking, thinking_output, chat_format, custom_chat_format,
                 temperature, top_p, top_k, min_p, typical_p, repeat_penalty,
                 presence_penalty, frequency_penalty, max_tokens, seed,
                 memory_preset, context_size, kv_cache_k, kv_cache_v, kv_cache_location,
                 gpu_layers, flash_attention, prompt_batch_size, memory_batch_size,
                 use_mmap, use_mlock, model_retention,
                 system_prompt, prompt,
                 split_mode, main_gpu, tensor_split, threads, threads_batch, last_n_tokens,
                 tfs_z, mirostat_mode, mirostat_tau, mirostat_eta, penalize_newline,
                 op_offload, swa_full, rope_freq_base, rope_freq_scale, yarn_ext_factor,
                 yarn_attn_factor, yarn_beta_fast, yarn_beta_slow, yarn_orig_ctx,
                 stop_sequences, vision_max_images, vision_max_frames, vision_max_edge, verbose, image=None, video_frames=None, messages_override=None, progress_callback=None, token_callback=None):
        if model == "No GGUF models found":
            raise FileNotFoundError(f"Put GGUF files in {LLM_DIR}")
        try:
            import llama_cpp
            from llama_cpp import Llama
        except Exception as e:
            raise RuntimeError(
                "llama-cpp-python is not installed in ComfyUI's Python environment. "
                "Install a build matching your CPU/GPU backend; this node intentionally does not auto-install a CPU-only wheel."
            ) from e

        model_path = _full_path(model)
        metadata = _metadata_for(model)
        family = detect_family(metadata, model)
        if model_preset == "Auto (Detected)":
            model_preset_resolved = recommended_model_preset(metadata, model)
        else:
            model_preset_resolved = model_preset

        # Presets are authoritative at execution time too. The frontend mirrors these
        # values for editability; if a user changes one in the UI it flips to Custom.
        # Enforcing here also keeps API/headless execution deterministic.
        if model_preset_resolved in MODEL_PRESETS and model_preset != "Custom":
            mp = MODEL_PRESETS[model_preset_resolved]
            thinking_mode = mp.get("thinking_mode", thinking_mode)
            reasoning_effort = mp.get("reasoning_effort", reasoning_effort)
            preserve_thinking = mp.get("preserve_thinking", preserve_thinking)
            chat_format = mp.get("chat_format", chat_format)
            temperature = mp.get("temperature", temperature)
            top_p = mp.get("top_p", top_p)
            top_k = mp.get("top_k", top_k)
            min_p = mp.get("min_p", min_p)
            typical_p = mp.get("typical_p", typical_p)
            repeat_penalty = mp.get("repeat_penalty", repeat_penalty)
            presence_penalty = mp.get("presence_penalty", presence_penalty)
            frequency_penalty = mp.get("frequency_penalty", frequency_penalty)
            tfs_z = mp.get("tfs_z", tfs_z)
            mirostat_mode = mp.get("mirostat_mode", mirostat_mode)
            mirostat_tau = mp.get("mirostat_tau", mirostat_tau)
            mirostat_eta = mp.get("mirostat_eta", mirostat_eta)

        if memory_preset in MEMORY_PRESETS and memory_preset != "Custom":
            mem = MEMORY_PRESETS[memory_preset]
            context_size = mem.get("context_size", context_size)
            kv_cache_k = mem.get("kv_cache_k", kv_cache_k)
            kv_cache_v = mem.get("kv_cache_v", kv_cache_v)
            kv_cache_location = mem.get("kv_cache_location", kv_cache_location)
            gpu_layers = mem.get("gpu_layers", gpu_layers)
            flash_attention = mem.get("flash_attention", flash_attention)
            prompt_batch_size = mem.get("prompt_batch_size", prompt_batch_size)
            memory_batch_size = mem.get("memory_batch_size", memory_batch_size)
            use_mmap = mem.get("use_mmap", use_mmap)
            use_mlock = mem.get("use_mlock", use_mlock)

        resolved_vision = None
        if vision_model == _AUTO_VISION:
            resolved_vision = _find_matching_mmproj(model)
        elif vision_model != _NONE:
            resolved_vision = vision_model
        selected_mmproj_path = _full_path(resolved_vision) if resolved_vision else None

        # Selecting/auto-detecting an mmproj should not force a text-only request
        # to require a multimodal build. Activate the projector when either a
        # ComfyUI IMAGE is connected OR an OpenAI-style messages_override payload
        # already contains image_url blocks (e.g. SillyTavern vision requests).
        messages_have_images = False
        if isinstance(messages_override, (list, tuple)):
            for _msg in messages_override:
                _content = _msg.get("content") if isinstance(_msg, dict) else None
                if isinstance(_content, list) and any(isinstance(_b, dict) and _b.get("type") == "image_url" for _b in _content):
                    messages_have_images = True
                    break
        caps = capabilities_for_family(family)
        ignored_vision_inputs = []
        if caps.get("vision") is False:
            if image is not None:
                image = None
                ignored_vision_inputs.append("image(s)")
            if video_frames is not None:
                video_frames = None
                ignored_vision_inputs.append("video_frames")
            if ignored_vision_inputs:
                _LOGGER.warning(
                    "[Local GGUF LLM] Ignoring %s input for detected text-only model family '%s'.",
                    " and ".join(ignored_vision_inputs),
                    family,
                )
        vision_active = image is not None or video_frames is not None or messages_have_images
        if vision_active and caps.get("vision") is False:
            raise RuntimeError(
                f"Vision content was provided through chat messages, but detected model family '{family}' is marked text-only. "
                "Choose a vision-capable GGUF model."
            )
        if vision_active and resolved_vision:
            mmproj_check = _validate_mmproj_pair(model, resolved_vision)
            if mmproj_check.get("compatible") is False:
                raise RuntimeError(
                    "The selected vision projector does not match the model: " + str(mmproj_check.get("reason") or "incompatible model/mmproj pair")
                )
        else:
            mmproj_check = None
        mmproj_path = selected_mmproj_path if vision_active else None

        chat_fmt = None
        if chat_format == "Custom":
            chat_fmt = custom_chat_format.strip() or None
        elif chat_format != "Auto (GGUF embedded)":
            chat_fmt = chat_format

        type_k = _resolve_kv_type(llama_cpp, kv_cache_k)
        type_v = _resolve_kv_type(llama_cpp, kv_cache_v)
        tensor_split_values = _parse_tensor_split(tensor_split)
        if split_mode == "None (single GPU)":
            tensor_split_values = None
        main_gpu_index = _gpu_index(main_gpu)

        template_kwargs = _thinking_template_kwargs(family, thinking_mode, preserve_thinking, reasoning_effort)
        llama_init_params, llama_init_var_kw = _signature_info(Llama.__init__)

        # Memory controls must never silently degrade. If a user selected a
        # quantized KV type or CPU KV cache, verify the installed Python binding
        # exposes the corresponding llama.cpp controls before model allocation.
        if kv_cache_k != "Auto":
            _require_init_option(llama_init_params, llama_init_var_kw, "type_k", f"KV cache K type {kv_cache_k}")
        if kv_cache_v != "Auto":
            _require_init_option(llama_init_params, llama_init_var_kw, "type_v", f"KV cache V type {kv_cache_v}")
        if kv_cache_location == "CPU":
            _require_init_option(llama_init_params, llama_init_var_kw, "offload_kqv", "CPU KV cache placement")

        load_kwargs = dict(
            model_path=model_path,
            n_gpu_layers=gpu_layers,
            split_mode=_split_mode(llama_cpp, split_mode),
            main_gpu=main_gpu_index,
            tensor_split=tensor_split_values,
            use_mmap=use_mmap,
            use_mlock=use_mlock,
            n_ctx=context_size,
            n_batch=prompt_batch_size,
            n_ubatch=memory_batch_size,
            n_threads=(None if threads == 0 else threads),
            n_threads_batch=(None if threads_batch == 0 else threads_batch),
            offload_kqv=(kv_cache_location == "GPU"),
            flash_attn=flash_attention,
            last_n_tokens_size=last_n_tokens,
            chat_format=chat_fmt,
            rope_freq_base=rope_freq_base,
            rope_freq_scale=rope_freq_scale,
            yarn_ext_factor=yarn_ext_factor,
            yarn_attn_factor=yarn_attn_factor,
            yarn_beta_fast=yarn_beta_fast,
            yarn_beta_slow=yarn_beta_slow,
            yarn_orig_ctx=yarn_orig_ctx,
            verbose=verbose,
        )
        if type_k is not None:
            load_kwargs["type_k"] = type_k
        if type_v is not None:
            load_kwargs["type_v"] = type_v
        if op_offload != "Auto":
            load_kwargs["op_offload"] = (op_offload == "Enabled")
        if swa_full != "Auto":
            load_kwargs["swa_full"] = (swa_full == "Enabled")
        # Keep text chat-template controls out of Llama construction so switching
        # Qwen thinking mode/reasoning effort does not reload the model. They are
        # injected into the embedded Jinja handler per request below. Multimodal
        # handlers are different: their template behavior is commonly fixed when
        # the handler is constructed, so it is included in the stable vision cache
        # key and rebuilt only when those controls change.
        handler = None
        if mmproj_path:
            if "mmproj_path" in llama_init_params or "clip_model_path" in llama_init_params:
                path_arg = "mmproj_path" if "mmproj_path" in llama_init_params else "clip_model_path"
                load_kwargs[path_arg] = mmproj_path
                if "chat_template_kwargs" in llama_init_params:
                    load_kwargs["chat_template_kwargs"] = template_kwargs
                elif "chat_handler_kwargs" in llama_init_params:
                    load_kwargs["chat_handler_kwargs"] = {
                        "extra_template_arguments": template_kwargs,
                        "verbose": verbose,
                    }
            else:
                handler = _vision_handler(family, mmproj_path, thinking_mode, preserve_thinking, reasoning_effort, verbose)
                load_kwargs["chat_handler"] = handler

            # Newer MTMD implementations recommend disabling context checkpoints
            # for single-turn hybrid Qwen vision models; only pass it when the
            # installed binding explicitly supports the parameter.
            if family in {"qwen3.5", "qwen3.6", "qwen3.8"} and "ctx_checkpoints" in llama_init_params:
                load_kwargs["ctx_checkpoints"] = 0

        # llama-cpp-python changes quickly. Pass only options the installed
        # constructor advertises instead of failing on a newly-added/removed knob.
        # Critical memory options were validated above, so filtering cannot hide a
        # requested KV behavior.
        load_kwargs, unsupported_load_options = _filter_supported_kwargs(Llama.__init__, load_kwargs)

        # Only parameters that affect model/context allocation belong in the cache key.
        def freeze(v):
            if isinstance(v, list):
                return tuple(v)
            if isinstance(v, dict):
                return tuple(sorted((k, freeze(x)) for k, x in v.items()))
            return v
        # Explicit chat-handler objects are not stable cache-key values: dedicated
        # multimodal handlers are recreated on each execution, so including the
        # object itself would force a full model reload every run.  Text-only
        # thinking/reasoning controls are applied by a lightweight handler below
        # and therefore do not belong in the allocation key.  For vision handlers,
        # template behavior is part of the handler construction, so include a
        # stable behavior tuple to reload only when that behavior actually changes.
        allocation_kwargs = {
            k: v for k, v in load_kwargs.items()
            if k not in {"verbose", "chat_handler", "chat_handler_kwargs", "chat_template_kwargs"}
        }
        vision_behavior_key = (
            "vision_behavior",
            resolved_vision or "",
            tuple(sorted(template_kwargs.items())),
        ) if mmproj_path else ("text_behavior",)
        def allocation_file_state(path):
            if not path:
                return None
            try:
                st = os.stat(path)
                return (os.path.realpath(path), int(st.st_size), int(st.st_mtime_ns))
            except Exception:
                return (os.path.realpath(path), "missing")

        # Include backing-file state in the native allocation key too.  IS_CHANGED
        # already invalidates ComfyUI's output cache when a GGUF is replaced under
        # the same filename; this additionally prevents the persistent native cache
        # from silently continuing to serve the old mmap/model instance.
        load_key = (
            tuple(sorted((k, freeze(v)) for k, v in allocation_kwargs.items())),
            ("model_file_state", allocation_file_state(model_path)),
            ("vision_file_state", allocation_file_state(mmproj_path)),
            ("custom_chat_format", custom_chat_format if chat_format == "Custom" else ""),
            vision_behavior_key,
        )

        # Legacy retention labels all map to the persistent native-context mode.
        # This is the KoboldCPP-like behavior: llama.cpp owns its allocations and
        # remains hot until a real load-setting change or explicit unload.
        if model_retention in {"Keep Loaded", "Keep Loaded / Pinned"}:
            model_retention = "Persistent (Driver Managed)"

        # ComfyUI can only account an external native allocation against one
        # device at a time. Keep managed mode exact and safe by requiring a
        # single llama.cpp GPU when more than one accelerator is visible.
        if model_retention == "ComfyUI Managed" and split_mode != "None (single GPU)" and int(gpu_layers) != 0:
            try:
                import torch
                if torch.cuda.is_available() and int(torch.cuda.device_count()) > 1:
                    raise RuntimeError(
                        "ComfyUI Managed/Auto Yield currently requires 'None (single GPU)' when multiple GPUs are visible. "
                        "The thin yield hook coordinates one native CUDA device at a time, so single-GPU mode guarantees "
                        "that a ComfyUI pressure event can release the complete llama.cpp allocation. "
                        "For llama.cpp multi-GPU Layer/Row/Tensor split, choose 'Persistent (Driver Managed)' or "
                        "'Unload After Run' instead."
                    )
            except ImportError:
                pass

        try:
            model_file_size = int(os.path.getsize(model_path))
        except Exception:
            model_file_size = 0
        estimated_vram = _estimate_native_vram(
            model_path, mmproj_path, metadata, gpu_layers, context_size,
            kv_cache_k, kv_cache_v, kv_cache_location,
        )

        # A change in allocation settings OR management mode invalidates the old
        # native context and its cached direct-reload signature.
        with _MODEL_LOCK:
            stale_cache = (
                _MODEL_CACHE.get("key") is not None
                and (_MODEL_CACHE.get("key") != load_key or _MODEL_CACHE.get("mode") != model_retention)
            )
        if stale_cache:
            _cleanup_llm()

        with _MODEL_LOCK:
            if model_retention == "ComfyUI Managed":
                _existing_adapter = _MODEL_CACHE.get("managed_adapter")
                _load_needed = _existing_adapter is None or getattr(_existing_adapter, "llm", None) is None
            else:
                _load_needed = _MODEL_CACHE.get("llm") is None
        if _load_needed and progress_callback is not None:
            try:
                progress_callback({"phase": "loading", "completion_tokens": 0, "tokens_per_second": 0.0, "elapsed": 0.0})
            except Exception:
                pass

        started_load = time.perf_counter()
        _perf_log(
            "request load phase begin",
            mode=model_retention,
            load_needed=_load_needed,
            estimated_vram_mib=estimated_vram / _MIB,
            gpu_layers=gpu_layers,
            main_gpu=main_gpu_index,
        )
        if model_retention == "ComfyUI Managed":
            try:
                import comfy.model_management as mm
            except Exception as e:
                raise RuntimeError("ComfyUI memory manager could not be imported.") from e

            # Auto Yield now uses a thin coordinator around an ordinary native
            # llama.cpp context. The context is NOT registered with
            # load_models_gpu()/LoadedModel and never participates in ComfyUI
            # partial-loading semantics.
            _install_comfy_llm_yield_hook(mm)
            with _MODEL_LOCK:
                resident_ctl = _MODEL_CACHE.get("managed_adapter")
                if resident_ctl is None:
                    resident_ctl = _NativeLLMResident(
                        llama_cpp=llama_cpp,
                        Llama=Llama,
                        load_key=load_key,
                        load_kwargs=load_kwargs,
                        metadata=metadata,
                        family=family,
                        estimated_vram=estimated_vram,
                        model_file_size=model_file_size,
                        main_gpu_index=main_gpu_index,
                        gpu_layers=gpu_layers,
                    )
                    _MODEL_CACHE.update({
                        "key": load_key,
                        "metadata": metadata,
                        "family": family,
                        "managed_adapter": resident_ctl,
                        "mode": model_retention,
                    })
                elif resident_ctl.load_key != load_key:
                    # Normally stale_cache above handles this. Keep an explicit
                    # guard here so a signature mismatch can never reuse native
                    # residency accidentally.
                    raise RuntimeError(
                        "Internal native LLM load-signature mismatch. The stale context must be cleaned before reuse."
                    )
                was_resident = resident_ctl.llm is not None

            managed_preload_memory = {
                "policy": "thin-native-auto-yield-v5",
                "signature_id": resident_ctl.signature_id,
                "requested_release_bytes": 0,
                "free_before_bytes": None,
                "free_after_bytes": None,
                "comfyui_release_called": False,
                "strategy": "none",
                "aimdo": _aimdo_enabled(),
                "registered_with_comfyui_loaded_models": False,
            }
            handoff_started = time.perf_counter()
            if not was_resident and int(gpu_layers) != 0:
                try:
                    import torch
                    if torch.cuda.is_available():
                        target = torch.device("cuda", int(main_gpu_index))
                        reserve, reserve_source = resident_ctl.preload_target_bytes()
                        managed_preload_memory["requested_release_bytes"] = reserve
                        managed_preload_memory["target_source"] = reserve_source
                        managed_preload_memory["estimated_vram_bytes"] = int(estimated_vram)
                        managed_preload_memory["observed_vram_bytes"] = int(resident_ctl.observed_vram_bytes)
                        _perf_log(
                            "native preload VRAM target",
                            signature=resident_ctl.signature_id,
                            source=reserve_source,
                            target_mib=reserve / _MIB,
                            estimated_mib=estimated_vram / _MIB,
                            observed_mib=resident_ctl.observed_vram_bytes / _MIB if resident_ctl.observed_vram_bytes else None,
                        )
                        room = _request_comfyui_room(mm, reserve, target)
                        managed_preload_memory.update({
                            "free_before_bytes": room.get("free_before_bytes"),
                            "free_after_bytes": room.get("free_after_bytes"),
                            "comfyui_release_called": bool(room.get("release_called")),
                            "strategy": room.get("strategy", "none"),
                            "aimdo": bool(room.get("aimdo")),
                            "room_details": copy.deepcopy(room),
                        })
                except Exception as e:
                    managed_preload_memory["error"] = str(e)
                    _LOGGER.warning(
                        "[Local GGUF LLM] native pre-load ComfyUI memory handoff failed: %s", e
                    )
            managed_preload_memory["handoff_seconds"] = round(time.perf_counter() - handoff_started, 4)

            if was_resident:
                # Exact signature + resident context = strict no-op. No ComfyUI
                # memory calls, no loader registration checks, no CUDA snapshots.
                _perf_log(
                    "native resident reuse",
                    signature=resident_ctl.signature_id,
                    handoff_s=0.0,
                    comfy_loader_s=0.0,
                )
            else:
                resident_ctl._load_native()

            llm = resident_ctl.llm
            if llm is None:
                raise RuntimeError("Native GGUF residency controller returned without loading the model.")
            reused = was_resident
            load_diagnostics = dict(resident_ctl.diagnostics)
            if was_resident:
                load_diagnostics["last_native_load_seconds"] = load_diagnostics.get("native_load_seconds")
                load_diagnostics["native_load_seconds"] = 0.0
                load_diagnostics["load_path"] = "resident-reuse"
                load_diagnostics["major_faults_delta"] = 0
                load_diagnostics["minor_faults_delta"] = 0
                load_diagnostics["block_inputs_delta"] = 0
                load_diagnostics["page_cache_hint"] = "not-loaded"
            load_diagnostics["preload_memory"] = managed_preload_memory
            load_diagnostics["comfy_load_models_gpu_seconds"] = 0.0
            load_diagnostics["comfyui_loaded_model_registration"] = False
            load_diagnostics["coordination_mode"] = "thin-free-memory-hook-v5"
            load_diagnostics["last_unload"] = copy.deepcopy(resident_ctl._last_unload_diagnostics)
        else:
            with _MODEL_LOCK:
                cached = _MODEL_CACHE.get("llm")
            if cached is None:
                # Persistent/native mode deliberately stays OUTSIDE ComfyUI's
                # current_loaded_models list. This mirrors a long-running external
                # llama.cpp/KoboldCPP process: once loaded, the model/context and KV
                # allocations remain alive and ComfyUI simply observes less
                # device-wide free VRAM. We only ask ComfyUI to release its own
                # cached models before the FIRST native allocation when the current
                # free-VRAM snapshot is clearly below our conservative estimate.
                preload_memory = {
                    "policy": "conditional-first-load-only-v3",
                    "requested_release_bytes": 0,
                    "free_before_bytes": None,
                    "free_after_bytes": None,
                    "comfyui_release_called": False,
                    "strategy": "none",
                    "aimdo": _aimdo_enabled(),
                }
                if int(gpu_layers) != 0 and model_retention == "Persistent (Driver Managed)":
                    try:
                        import torch
                        import comfy.model_management as mm
                        if torch.cuda.is_available():
                            target = torch.device("cuda", int(main_gpu_index))
                            # The estimator already includes native/KV/vision overhead.
                            # Do not stack ComfyUI's inference reserve on top of an
                            # unrelated llama.cpp native allocation.
                            reserve = int(estimated_vram)
                            preload_memory["requested_release_bytes"] = reserve
                            preload_memory["target_source"] = "first-load-estimate"
                            _perf_log(
                                "native preload VRAM target",
                                source="first-load-estimate",
                                target_mib=reserve / _MIB,
                                estimated_mib=estimated_vram / _MIB,
                            )
                            room = _request_comfyui_room(mm, reserve, target)
                            preload_memory.update({
                                "free_before_bytes": room.get("free_before_bytes"),
                                "free_after_bytes": room.get("free_after_bytes"),
                                "comfyui_release_called": bool(room.get("release_called")),
                                "strategy": room.get("strategy", "none"),
                                "aimdo": bool(room.get("aimdo")),
                                "room_details": copy.deepcopy(room),
                            })
                    except Exception as e:
                        preload_memory["error"] = str(e)
                        _LOGGER.warning("[Local GGUF LLM] conditional pre-load ComfyUI memory release failed: %s", e)

                llm, load_diagnostics = _load_llama_verified(llama_cpp, Llama, load_kwargs, gpu_layers)
                if model_retention == "Persistent (Driver Managed)":
                    load_diagnostics = dict(load_diagnostics or {})
                    load_diagnostics["preload_memory"] = preload_memory
                with _MODEL_LOCK:
                    _MODEL_CACHE.update({
                        "key": load_key,
                        "llm": llm,
                        "metadata": metadata,
                        "family": family,
                        "base_chat_handler": getattr(llm, "chat_handler", None),
                        "managed_adapter": None,
                        "load_diagnostics": load_diagnostics,
                        "mode": model_retention,
                    })
                reused = False
            else:
                llm = cached
                reused = True
                load_diagnostics = dict(_MODEL_CACHE.get("load_diagnostics") or {})
                load_diagnostics["last_native_load_seconds"] = load_diagnostics.get("native_load_seconds")
                load_diagnostics["native_load_seconds"] = 0.0
                load_diagnostics["load_path"] = "resident-reuse"
                load_diagnostics["major_faults_delta"] = 0
                load_diagnostics["minor_faults_delta"] = 0
                load_diagnostics["block_inputs_delta"] = 0
                load_diagnostics["page_cache_hint"] = "not-loaded"

        with _MODEL_LOCK:
            # Direct llama-cpp-python does not currently expose arbitrary Jinja
            # template kwargs through Llama.create_chat_completion. Refresh a
            # lightweight embedded-template handler per run instead of reloading
            # the model when enable_thinking/reasoning_effort changes.
            if not mmproj_path and chat_format == "Auto (GGUF embedded)":
                base_handler = _MODEL_CACHE.get("base_chat_handler")
                if template_kwargs:
                    embedded_handler = _embedded_template_chat_handler(llm, template_kwargs)
                    if embedded_handler is None:
                        raise RuntimeError(
                            "This model preset needs chat-template controls, but its embedded GGUF "
                            "template could not be wrapped by the installed llama-cpp-python. "
                            "Update llama-cpp-python or use a compatible GGUF with tokenizer.chat_template metadata."
                        )
                    llm.chat_handler = embedded_handler
                else:
                    llm.chat_handler = base_handler
        load_seconds = time.perf_counter() - started_load
        _perf_log(
            "request load phase complete",
            mode=model_retention,
            reused=reused,
            total_s=load_seconds,
            load_path=(load_diagnostics or {}).get("load_path"),
            native_s=(load_diagnostics or {}).get("native_load_seconds"),
            comfy_load_models_s=(load_diagnostics or {}).get("comfy_load_models_gpu_seconds"),
            handoff_s=((load_diagnostics or {}).get("preload_memory") or {}).get("handoff_seconds"),
        )

        effective_system = system_prompt.strip()

        image_uris = _image_to_data_uris(image, vision_max_images, vision_max_edge) if image is not None else []
        video_uris = _video_frames_to_data_uris(video_frames, vision_max_frames, vision_max_edge) if video_frames is not None else []
        data_uris = image_uris + video_uris
        if (data_uris or messages_have_images) and not mmproj_path:
            raise RuntimeError(
                "A vision request was received, but no usable vision projector is selected. "
                "Choose a matching mmproj GGUF in Vision, or use Auto when a matching projector is present in models/llm."
            )

        if messages_override is not None:
            if not isinstance(messages_override, (list, tuple)):
                raise TypeError("messages_override must be a list of OpenAI-style chat message dictionaries.")
            messages = copy.deepcopy(list(messages_override))
            for item in messages:
                if not isinstance(item, dict) or not item.get("role"):
                    raise ValueError("Each messages_override item must be a dictionary with a role field.")
            if data_uris:
                # Attach ComfyUI IMAGE input to the final user message while preserving
                # the rest of the external conversation history.
                user_index = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"), None)
                if user_index is None:
                    messages.append({"role": "user", "content": ""})
                    user_index = len(messages) - 1
                existing = messages[user_index].get("content", "")
                blocks = [{"type": "image_url", "image_url": {"url": u}} for u in data_uris]
                if isinstance(existing, list):
                    blocks.extend(existing)
                elif existing not in (None, ""):
                    blocks.append({"type": "text", "text": str(existing)})
                messages[user_index]["content"] = blocks
        else:
            messages = []
            if effective_system:
                messages.append({"role": "system", "content": effective_system})
            if data_uris:
                content = [{"type": "image_url", "image_url": {"url": u}} for u in data_uris]
                content.append({"type": "text", "text": prompt})
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "user", "content": prompt})

        llama_seed = _llama_seed_from_comfy(seed)

        completion_kwargs = dict(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            typical_p=typical_p,
            repeat_penalty=repeat_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens,
            seed=llama_seed,
            stop=_parse_stop(stop_sequences),
            tfs_z=tfs_z,
            mirostat_mode={"Off": 0, "v1": 1, "v2": 2}[mirostat_mode],
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            penalize_nl=penalize_newline,
        )
        # Filter kwargs for older llama-cpp-python chat handlers/signatures.
        try:
            call_sig = inspect.signature(llm.create_chat_completion)
            if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in call_sig.parameters.values()):
                completion_kwargs = {k: v for k, v in completion_kwargs.items() if k in call_sig.parameters}
        except Exception:
            pass

        if progress_callback is not None:
            try:
                progress_callback({"phase": "processing", "completion_tokens": 0, "tokens_per_second": 0.0, "elapsed": 0.0})
            except Exception:
                pass
        started = time.perf_counter()
        perf_before = _llama_perf_snapshot(llm)
        first_output_at = None
        last_output_at = None
        prompt_perf_at_first = None
        # Keep the native context alive for the entire generation. Manual unloads
        # or concurrent ComfyUI eviction requests wait until llama.cpp returns.
        with _MODEL_LOCK:
            # Some older chat handlers do not expose a per-call seed argument.
            # Set the sampler seed directly as a compatibility fallback as well as
            # passing it in completion_kwargs when supported.
            try:
                llm.set_seed(llama_seed)
            except Exception:
                pass

            if progress_callback is None:
                result = llm.create_chat_completion(**completion_kwargs)
            else:
                # The global service consumes llama.cpp's streaming iterator even
                # when the external client requested a buffered response.  This
                # gives us live throughput/status without changing the final API
                # semantics or exposing partial response text.
                stream_kwargs = dict(completion_kwargs)
                stream_kwargs["stream"] = True
                stream_kwargs["stream_options"] = {"include_usage": True}
                parts = []
                reasoning_parts = []
                usage = {}
                token_events = 0
                emitted_any = False

                def consume_stream(iterator):
                    nonlocal usage, token_events, emitted_any, first_output_at, last_output_at, prompt_perf_at_first
                    for chunk in iterator:
                        emitted_any = True
                        chunk_map = _as_mapping(chunk)
                        if not chunk_map:
                            continue
                        usage_map = _as_mapping(chunk_map.get("usage"))
                        if usage_map:
                            usage = dict(usage_map)
                        choices = chunk_map.get("choices") or []
                        if not choices:
                            continue
                        choice = _as_mapping(choices[0])
                        if not choice:
                            continue
                        delta = _as_mapping(choice.get("delta"))
                        content = delta.get("content") if delta else None
                        # A few llama-cpp-python/chat-handler combinations still
                        # stream legacy ``choice.text`` instead of ``delta.content``.
                        # Accept it so the service cannot silently consume generated
                        # text while returning an empty OpenAI response.
                        if content in (None, "") and choice.get("text") not in (None, ""):
                            content = choice.get("text")
                        reasoning_delta = (
                            (delta.get("reasoning_content") if delta else None)
                            or (delta.get("reasoning") if delta else None)
                            or (delta.get("analysis") if delta else None)
                        )
                        event_has_text = False
                        if content not in (None, ""):
                            content_text = str(content)
                            parts.append(content_text)
                            event_has_text = True
                            if token_callback is not None:
                                try:
                                    token_callback({"type": "content", "text": content_text})
                                except Exception:
                                    pass
                        if reasoning_delta not in (None, ""):
                            reasoning_text = str(reasoning_delta)
                            reasoning_parts.append(reasoning_text)
                            event_has_text = True
                            if token_callback is not None:
                                try:
                                    token_callback({"type": "reasoning", "text": reasoning_text})
                                except Exception:
                                    pass
                        if event_has_text:
                            # The first streamed output marks the prompt->decode
                            # transition.  Throughput from this point forward must
                            # not include prompt evaluation / time-to-first-token.
                            now_out = time.perf_counter()
                            if first_output_at is None:
                                first_output_at = now_out
                                prompt_perf_at_first = _perf_delta(_llama_perf_snapshot(llm), perf_before)
                            last_output_at = now_out
                            token_events += 1
                            decode_elapsed = max(now_out - first_output_at, 0.0)
                            # At the first token there is no measurable inter-token
                            # interval yet. Native perf data, when available, gives
                            # a meaningful decode rate immediately; otherwise show 0
                            # until a second token establishes a real interval.
                            native_now = _perf_delta(_llama_perf_snapshot(llm), perf_before)
                            native_eval_ms = float((native_now or {}).get("eval_ms") or 0.0)
                            native_eval_tokens = int((native_now or {}).get("eval_tokens") or 0)
                            if native_eval_ms > 0 and native_eval_tokens > 0:
                                live_speed = native_eval_tokens / (native_eval_ms / 1000.0)
                            elif token_events > 1 and decode_elapsed > 0:
                                live_speed = (token_events - 1) / decode_elapsed
                            else:
                                live_speed = 0.0
                            pp = prompt_perf_at_first or {}
                            prompt_ms = float(pp.get("prompt_ms") or 0.0)
                            prompt_count = int(pp.get("prompt_tokens") or 0)
                            prompt_speed = (prompt_count / (prompt_ms / 1000.0)) if prompt_count and prompt_ms > 0 else 0.0
                            try:
                                progress_callback({
                                    "phase": "generating",
                                    "completion_tokens": token_events,
                                    "tokens_per_second": live_speed,
                                    "elapsed": decode_elapsed,
                                    "prompt_tokens": prompt_count,
                                    "prompt_tokens_per_second": prompt_speed,
                                })
                            except Exception:
                                pass

                try:
                    streamed = llm.create_chat_completion(**stream_kwargs)
                    streamed_map = _as_mapping(streamed)
                    if streamed_map and "choices" in streamed_map:
                        result = streamed_map
                    else:
                        consume_stream(streamed)
                        msg = {"role": "assistant", "content": "".join(parts)}
                        if reasoning_parts:
                            msg["reasoning_content"] = "".join(reasoning_parts)
                        result = {"choices": [{"index": 0, "message": msg, "finish_reason": "stop"}], "usage": usage}
                except TypeError:
                    # Older llama-cpp-python builds may support stream=True but not
                    # stream_options. Retry only when no output was emitted, so a
                    # partially generated response can never be generated twice.
                    if emitted_any:
                        raise
                    stream_kwargs.pop("stream_options", None)
                    streamed = llm.create_chat_completion(**stream_kwargs)
                    streamed_map = _as_mapping(streamed)
                    if streamed_map and "choices" in streamed_map:
                        result = streamed_map
                    else:
                        consume_stream(streamed)
                        msg = {"role": "assistant", "content": "".join(parts)}
                        if reasoning_parts:
                            msg["reasoning_content"] = "".join(reasoning_parts)
                        result = {"choices": [{"index": 0, "message": msg, "finish_reason": "stop"}], "usage": usage}

                # Some binding versions do not return usage in streaming mode.
                # Recover an exact-ish completion count with the model tokenizer
                # once at the end; live reporting above intentionally stays cheap.
                if isinstance(result, dict) and not (result.get("usage") or {}).get("completion_tokens"):
                    joined = "".join(reasoning_parts) + "".join(parts)
                    if joined:
                        try:
                            ct = len(llm.tokenize(joined.encode("utf-8"), add_bos=False))
                        except TypeError:
                            ct = len(llm.tokenize(joined.encode("utf-8")))
                        except Exception:
                            ct = token_events
                    else:
                        ct = token_events
                    result.setdefault("usage", {})["completion_tokens"] = int(ct)
                    result["usage"].setdefault("prompt_tokens", 0)
                    result["usage"]["total_tokens"] = int(result["usage"].get("prompt_tokens") or 0) + int(ct)

        request_generation_seconds = time.perf_counter() - started
        perf_after = _perf_delta(_llama_perf_snapshot(llm), perf_before)
        raw_response, structured_thinking, thinking_sources = _extract_response(result)
        stripped_response, inline_thinking, inline_modes = _split_reasoning_markup(raw_response)

        # The `thinking` output is ALWAYS populated with every reasoning trace we
        # can recover, independent of whether the response keeps or strips tags.
        # This prevents the response-format control from accidentally hiding CoT.
        thinking = _merge_reasoning((structured_thinking, inline_thinking))
        response = stripped_response if thinking_output == "Strip" else raw_response
        thinking_extraction = {
            "structured_sources": thinking_sources,
            "inline_modes": inline_modes,
            "found": bool(thinking),
            "response_tags_stripped": thinking_output == "Strip",
        }

        usage = result.get("usage") or {}
        total_tokens = int(usage.get("total_tokens") or 0)
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        # Prefer llama.cpp's native separated perf counters.  They exclude prompt
        # evaluation from decode tok/s.  Fall back to the streamed first->last
        # output interval on older bindings.
        native_prompt_ms = float((perf_after or {}).get("prompt_ms") or 0.0)
        native_prompt_tokens = int((perf_after or {}).get("prompt_tokens") or 0)
        native_eval_ms = float((perf_after or {}).get("eval_ms") or 0.0)
        native_eval_tokens = int((perf_after or {}).get("eval_tokens") or 0)
        prompt_eval_tokens = native_prompt_tokens or prompt_tokens
        prompt_eval_seconds = native_prompt_ms / 1000.0 if native_prompt_ms > 0 else 0.0
        prompt_tok_s = (prompt_eval_tokens / prompt_eval_seconds) if prompt_eval_tokens and prompt_eval_seconds > 0 else 0.0

        if native_eval_ms > 0 and native_eval_tokens > 0:
            decode_seconds = native_eval_ms / 1000.0
            tok_s = native_eval_tokens / decode_seconds
            perf_source = "llama.cpp"
        elif first_output_at is not None and last_output_at is not None and completion_tokens > 1:
            decode_seconds = max(last_output_at - first_output_at, 1e-9)
            tok_s = (completion_tokens - 1) / decode_seconds
            perf_source = "stream_interval"
        else:
            decode_seconds = 0.0
            tok_s = 0.0
            perf_source = "unavailable"

        # Effective configuration snapshot backing the live API facade.  It is
        # no longer emitted as a separate ComfyUI output; linked nodes query it
        # through api.get_settings()/get_*_settings().
        settings = {
            "schema": "local_gguf_llm_settings",
            "schema_version": 3,
            "api_type": "LOCAL_GGUF_LLM_API",
            "api_version": LocalGGUFLLMAPI.API_VERSION,
            "model": {
                "name": model,
                "path": os.path.realpath(model_path),
                "vision_selection": vision_model,
                "resolved_vision": resolved_vision or "None",
                "resolved_vision_path": os.path.realpath(selected_mmproj_path) if selected_mmproj_path else None,
                "detected_family": family,
                "architecture": metadata.get("general.architecture", "unknown"),
                "capabilities": capabilities_for_family(family),
                "model_preset": model_preset,
                "resolved_model_preset": model_preset_resolved,
            },
            "generation": {
                "thinking_mode": thinking_mode,
                "reasoning_effort": reasoning_effort,
                "preserve_thinking": preserve_thinking,
                "thinking_output": thinking_output,
                "chat_format": chat_format,
                "custom_chat_format": custom_chat_format,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "typical_p": typical_p,
                "repeat_penalty": repeat_penalty,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "max_tokens": max_tokens,
                "seed": seed,
                "llama_seed": llama_seed,
            },
            "memory": {
                "memory_preset": memory_preset,
                "context_size": context_size,
                "kv_cache_k": kv_cache_k,
                "kv_cache_v": kv_cache_v,
                "kv_cache_location": kv_cache_location,
                "gpu_layers": gpu_layers,
                "flash_attention": flash_attention,
                "prompt_batch_size": prompt_batch_size,
                "memory_batch_size": memory_batch_size,
                "use_mmap": use_mmap,
                "use_mlock": use_mlock,
                "model_retention": model_retention,
            },
            "prompting": {
                "system_prompt": system_prompt,
                "prompt": prompt,
            },
            "advanced": {
                "split_mode": split_mode,
                "main_gpu": str(main_gpu),
                "main_gpu_index": main_gpu_index,
                "tensor_split": tensor_split,
                "tensor_split_values": tensor_split_values,
                "threads": threads,
                "threads_batch": threads_batch,
                "last_n_tokens": last_n_tokens,
                "tfs_z": tfs_z,
                "mirostat_mode": mirostat_mode,
                "mirostat_tau": mirostat_tau,
                "mirostat_eta": mirostat_eta,
                "penalize_newline": penalize_newline,
                "op_offload": op_offload,
                "swa_full": swa_full,
                "rope_freq_base": rope_freq_base,
                "rope_freq_scale": rope_freq_scale,
                "yarn_ext_factor": yarn_ext_factor,
                "yarn_attn_factor": yarn_attn_factor,
                "yarn_beta_fast": yarn_beta_fast,
                "yarn_beta_slow": yarn_beta_slow,
                "yarn_orig_ctx": yarn_orig_ctx,
                "stop_sequences": stop_sequences,
                "vision_max_images": vision_max_images,
                "vision_max_frames": vision_max_frames,
                "vision_max_edge": vision_max_edge,
                "verbose": verbose,
            },
        }

        info = {
            "model": model,
            "vision": resolved_vision or "None",
            "vision_active": bool(data_uris or messages_have_images),
            "vision_inputs": {
                "still_images": len(image_uris),
                "sampled_video_frames": len(video_uris),
                "total_local_images": len(data_uris),
                "ignored_inputs": ignored_vision_inputs,
            },
            "mmproj_validation": mmproj_check,
            "detected_family": family,
            "architecture": metadata.get("general.architecture", "unknown"),
            "model_preset": model_preset,
            "resolved_model_preset": model_preset_resolved,
            "model_capabilities": capabilities_for_family(family),
            "effective_model_settings": {
                "thinking_mode": thinking_mode,
                "reasoning_effort": reasoning_effort,
                "preserve_thinking": preserve_thinking,
                "chat_format": chat_format,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "typical_p": typical_p,
                "repeat_penalty": repeat_penalty,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
            },
            "independent_output_controls": {
                "thinking_output": thinking_output,
                "max_tokens": max_tokens,
                "seed": seed,
                "llama_seed": llama_seed,
            },
            "thinking_extraction": thinking_extraction,
            "memory_preset": memory_preset,
            "memory_management": {
                "mode": model_retention,
                "persistent_native": model_retention == "Persistent (Driver Managed)",
                "comfyui_managed": model_retention == "ComfyUI Managed",
                "estimated_native_vram_mib": round(estimated_vram / _MIB, 1),
                "whole_model_eviction": model_retention == "ComfyUI Managed",
                "note": (
                    "Persistent mode keeps llama.cpp's native model/context alive outside ComfyUI's eviction list; "
                    "ComfyUI observes the remaining device-wide free VRAM and the OS/GPU driver controls residency."
                    if model_retention == "Persistent (Driver Managed)" else
                    "Auto Yield keeps llama.cpp as an all-or-nothing native context outside ComfyUI loaded_models; a thin free-memory hook closes it before ComfyUI eviction and the next request reloads it directly from the cached signature."
                    if model_retention == "ComfyUI Managed" else
                    "Unload After Run destroys the native llama.cpp model/context after each generation."
                ),
            },
            "context_size": context_size,
            "kv_cache": {"k": kv_cache_k, "v": kv_cache_v, "location": kv_cache_location},
            "gpu_layers": gpu_layers,
            "split_mode": split_mode,
            "main_gpu": {"selection": str(main_gpu), "index": main_gpu_index},
            "gpu_backend": load_diagnostics,
            "model_reused": reused,
            "load_seconds": round(load_seconds, 3),
            "generation_seconds": round(decode_seconds, 3),
            "request_generation_seconds": round(request_generation_seconds, 3),
            "prompt_eval_seconds": round(prompt_eval_seconds, 3),
            "prompt_tokens": prompt_tokens,
            "prompt_eval_tokens": prompt_eval_tokens,
            "prompt_tokens_per_second": round(prompt_tok_s, 2),
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "tokens_per_second": round(tok_s, 2),
            "performance_source": perf_source,
            "llama_cpp_version": getattr(llama_cpp, "__version__", "unknown"),
            "ignored_unsupported_load_options": unsupported_load_options,
        }

        # Store only scalar/configuration inputs in the API.  Do not retain the
        # source IMAGE tensor: keeping it inside a cached API output could pin a
        # large CPU/GPU tensor indefinitely.  A linked node can pass image=... to
        # api.generate/query when it needs multimodal inference.
        api_call_config = {
            "model": model,
            "vision_model": vision_model,
            "model_preset": "Custom",
            "thinking_mode": thinking_mode,
            "reasoning_effort": reasoning_effort,
            "preserve_thinking": preserve_thinking,
            "thinking_output": thinking_output,
            "chat_format": chat_format,
            "custom_chat_format": custom_chat_format,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "typical_p": typical_p,
            "repeat_penalty": repeat_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "max_tokens": max_tokens,
            "seed": seed,
            "memory_preset": "Custom",
            "context_size": context_size,
            "kv_cache_k": kv_cache_k,
            "kv_cache_v": kv_cache_v,
            "kv_cache_location": kv_cache_location,
            "gpu_layers": gpu_layers,
            "flash_attention": flash_attention,
            "prompt_batch_size": prompt_batch_size,
            "memory_batch_size": memory_batch_size,
            "use_mmap": use_mmap,
            "use_mlock": use_mlock,
            "model_retention": model_retention,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "split_mode": split_mode,
            "main_gpu": main_gpu,
            "tensor_split": tensor_split,
            "threads": threads,
            "threads_batch": threads_batch,
            "last_n_tokens": last_n_tokens,
            "tfs_z": tfs_z,
            "mirostat_mode": mirostat_mode,
            "mirostat_tau": mirostat_tau,
            "mirostat_eta": mirostat_eta,
            "penalize_newline": penalize_newline,
            "op_offload": op_offload,
            "swa_full": swa_full,
            "rope_freq_base": rope_freq_base,
            "rope_freq_scale": rope_freq_scale,
            "yarn_ext_factor": yarn_ext_factor,
            "yarn_attn_factor": yarn_attn_factor,
            "yarn_beta_fast": yarn_beta_fast,
            "yarn_beta_slow": yarn_beta_slow,
            "yarn_orig_ctx": yarn_orig_ctx,
            "stop_sequences": stop_sequences,
            "vision_max_images": vision_max_images,
            "vision_max_frames": vision_max_frames,
            "vision_max_edge": vision_max_edge,
            "verbose": verbose,
        }
        api_facade = LocalGGUFLLMAPI(settings=settings, call_config=api_call_config, load_key=load_key)

        if model_retention == "Unload After Run":
            _cleanup_llm()

        return (response, thinking, json.dumps(info, indent=2), total_tokens, api_facade)


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


# Optional HTTP endpoints used by the frontend extension for preset/model refresh.
try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/local_gguf_llm/presets")
    async def local_gguf_presets(request):
        return web.json_response(public_presets())

    @PromptServer.instance.routes.get("/local_gguf_llm/models")
    async def local_gguf_models(request):
        models, vision = _model_lists()
        return web.json_response({"models": models, "vision": vision, "gpus": _gpu_choices()})

    @PromptServer.instance.routes.get("/local_gguf_llm/model_info")
    async def local_gguf_model_info(request):
        name = request.query.get("model", "")
        if not name or name == "No GGUF models found":
            return web.json_response({"metadata": {}, "family": "unknown", "recommended_preset": "Generic Chat", "available_presets": ["Generic Chat"], "capabilities": capabilities_for_family("generic")})
        md = _metadata_for(name)
        family = detect_family(md, name)
        matching_vision = _find_matching_mmproj(name)
        return web.json_response({
            "metadata": {k: (v[:2000] + "…" if isinstance(v, str) and len(v) > 2000 else v) for k, v in md.items()},
            "family": family,
            "recommended_preset": recommended_model_preset(md, name),
            "available_presets": available_model_presets(md, name),
            "capabilities": capabilities_for_family(family),
            "matching_vision": matching_vision,
            "matching_vision_validation": (_validate_mmproj_pair(name, matching_vision) if matching_vision else None),
        })

    @PromptServer.instance.routes.post("/local_gguf_llm/unload")
    async def local_gguf_unload(request):
        _cleanup_llm()
        return web.json_response({"ok": True})
except Exception:
    pass
