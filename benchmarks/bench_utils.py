"""Shared benchmark helpers: wall-clock timing + Windows process memory peak."""
import gc
import time
import ctypes
import ctypes.wintypes


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.wintypes.DWORD),
        ("PageFaultCount", ctypes.wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _get_mem_counters() -> PROCESS_MEMORY_COUNTERS:
    kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
    kernel32.K32GetProcessMemoryInfo.argtypes = [
        ctypes.wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
        ctypes.wintypes.DWORD,
    ]
    pmc = PROCESS_MEMORY_COUNTERS()
    pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    handle = kernel32.GetCurrentProcess()
    ok = kernel32.K32GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb)
    if not ok:
        raise OSError(f"GetProcessMemoryInfo failed, err={ctypes.get_last_error()}")
    return pmc


def rss_bytes() -> int:
    """Current working set of this process (bytes)."""
    return int(_get_mem_counters().WorkingSetSize)


def peak_rss_bytes() -> int:
    return int(_get_mem_counters().PeakWorkingSetSize)


def timed(fn, warmup=3, iters=10):
    """Median wall time per call (seconds) after warmup."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def reset_memory():
    gc.collect()


class RssTracker:
    """Tracks RSS delta around a code block."""

    def __enter__(self):
        reset_memory()
        self.before = rss_bytes()
        self.peak_before = peak_rss_bytes()
        return self

    def __exit__(self, *exc):
        self.after = rss_bytes()
        self.delta = self.after - self.before
