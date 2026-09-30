"""CPU topology and automatic llama.cpp tuning helpers.

The helpers are deliberately dependency-free so CPU-only installations do not
need PyTorch, psutil, or a GPU runtime merely to resolve sensible thread counts.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path


CPU_PROFILE_VERSION = 1


def _parse_cpu_list(text: str) -> set[int]:
    out: set[int] = set()
    for part in str(text or "").strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_text, hi_text = part.split("-", 1)
            lo, hi = int(lo_text), int(hi_text)
            if hi < lo:
                lo, hi = hi, lo
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return out


def available_cpu_ids() -> set[int]:
    """Return CPUs available to this process, respecting Linux/WSL affinity."""
    try:
        get_affinity = getattr(os, "sched_getaffinity", None)
        if callable(get_affinity):
            ids = {int(x) for x in get_affinity(0)}
            if ids:
                return ids
    except Exception:
        pass
    count = max(1, int(os.cpu_count() or 1))
    return set(range(count))


def _linux_physical_cores(cpu_ids: set[int]) -> int | None:
    sys_cpu = Path("/sys/devices/system/cpu")
    if not sys_cpu.exists():
        return None
    cores: set[tuple[str, str]] = set()
    for cpu in sorted(cpu_ids):
        topo = sys_cpu / f"cpu{cpu}" / "topology"
        try:
            package_id = (topo / "physical_package_id").read_text(encoding="utf-8").strip()
            core_id = (topo / "core_id").read_text(encoding="utf-8").strip()
        except Exception:
            continue
        cores.add((package_id, core_id))
    return len(cores) or None


def _windows_physical_cores() -> int | None:
    if platform.system().lower() != "windows":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        RelationProcessorCore = 0
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        fn = kernel32.GetLogicalProcessorInformationEx
        fn.argtypes = [wintypes.DWORD, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
        fn.restype = wintypes.BOOL
        needed = wintypes.DWORD(0)
        fn(RelationProcessorCore, None, ctypes.byref(needed))
        if needed.value <= 0:
            return None
        buf = ctypes.create_string_buffer(needed.value)
        if not fn(RelationProcessorCore, buf, ctypes.byref(needed)):
            return None
        offset = 0
        count = 0
        # SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX begins with DWORD Relationship,
        # DWORD Size. With RelationProcessorCore the API emits one record/core.
        while offset + 8 <= needed.value:
            relationship = int.from_bytes(buf.raw[offset:offset + 4], "little")
            size = int.from_bytes(buf.raw[offset + 4:offset + 8], "little")
            if size < 8 or offset + size > needed.value:
                break
            if relationship == RelationProcessorCore:
                count += 1
            offset += size
        return count or None
    except Exception:
        return None


def physical_core_count(cpu_ids: set[int] | None = None) -> int:
    ids = set(cpu_ids or available_cpu_ids())
    logical = max(1, len(ids))
    detected = _linux_physical_cores(ids)
    if detected is None:
        detected = _windows_physical_cores()
    if detected:
        return max(1, min(logical, int(detected)))
    # Match llama-cpp-python's long-standing automatic decode-thread default on
    # SMT systems. This gives 64 decode threads on a 3990X (128 logical CPUs).
    return max(1, logical // 2 if logical > 1 else 1)


def numa_node_count(cpu_ids: set[int] | None = None) -> int:
    """Best-effort count of NUMA nodes that contain CPUs available to this process."""
    ids = set(cpu_ids or available_cpu_ids())
    node_root = Path("/sys/devices/system/node")
    if node_root.exists():
        count = 0
        for node in node_root.glob("node[0-9]*"):
            try:
                cpus = _parse_cpu_list((node / "cpulist").read_text(encoding="utf-8"))
            except Exception:
                continue
            if cpus & ids:
                count += 1
        if count:
            return count
    if platform.system().lower() == "windows":
        try:
            import ctypes
            from ctypes import wintypes

            highest = wintypes.ULONG(0)
            fn = ctypes.WinDLL("kernel32", use_last_error=True).GetNumaHighestNodeNumber
            fn.argtypes = [ctypes.POINTER(wintypes.ULONG)]
            fn.restype = wintypes.BOOL
            if fn(ctypes.byref(highest)):
                return max(1, int(highest.value) + 1)
        except Exception:
            pass
    return 1


def resolve_cpu_threads(threads: int | None = 0, threads_batch: int | None = 0) -> dict:
    """Resolve Auto (0/None) to physical-core decode and logical-core prompt threads."""
    ids = available_cpu_ids()
    logical = max(1, len(ids))
    physical = physical_core_count(ids)
    try:
        requested_decode = int(threads or 0)
    except Exception:
        requested_decode = 0
    try:
        requested_batch = int(threads_batch or 0)
    except Exception:
        requested_batch = 0
    decode = requested_decode if requested_decode > 0 else physical
    batch = requested_batch if requested_batch > 0 else logical
    return {
        "logical_cpus": logical,
        "physical_cores": physical,
        "threads": max(1, decode),
        "threads_batch": max(1, batch),
        "threads_auto": requested_decode <= 0,
        "threads_batch_auto": requested_batch <= 0,
        "numa_nodes": numa_node_count(ids),
    }


def resolve_numa_mode(mode: str | None, *, cpu_only: bool, nodes: int | None = None) -> str:
    """Resolve user-facing NUMA mode to a stable semantic strategy name."""
    value = str(mode or "Auto").strip()
    allowed = {"Auto", "Disabled", "Distribute", "Isolate", "Numactl"}
    if value not in allowed:
        value = "Auto"
    if value != "Auto":
        return value
    if not cpu_only:
        return "Disabled"
    count = int(nodes if nodes is not None else numa_node_count())
    return "Distribute" if count > 1 else "Disabled"


def cpu_runtime_profile(threads: int | None = 0, threads_batch: int | None = 0,
                        numa_mode: str | None = "Auto", *, cpu_only: bool = True) -> dict:
    profile = resolve_cpu_threads(threads, threads_batch)
    profile["numa_requested"] = str(numa_mode or "Auto")
    profile["numa_effective"] = resolve_numa_mode(
        numa_mode, cpu_only=cpu_only, nodes=profile["numa_nodes"]
    )
    profile["profile_version"] = CPU_PROFILE_VERSION
    return profile
