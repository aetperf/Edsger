"""
Memory usage of a phase of a benchmark: graph building or algorithm run.

The memory is the resident set size (RSS) of the process. For a phase, two
values are measured, relative to the RSS before the phase:

- retained: the RSS increase after the phase, e.g. the memory held by a graph
  once it is built;
- peak: the increase at the peak of the phase, e.g. the working memory of an
  algorithm, its temporary arrays and its result included.

On Linux, the peak is exact: the peak RSS of the process (VmHWM) is reset
before the phase, by writing to /proc/self/clear_refs. Elsewhere, the RSS is
sampled by a thread every millisecond with psutil, which misses the peaks
shorter than that, and the peaks of code that holds the GIL.

With glibc, the memory freed by the process but kept by the allocator is
returned to the system (malloc_trim) before and after the phase: otherwise the
phase could reuse it without any RSS increase, and the retained memory would
count it.
"""

import ctypes
import ctypes.util
import gc
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

MB = 1024**2

_STATUS = "/proc/self/status"
_CLEAR_REFS = "/proc/self/clear_refs"


def _read_status(key):
    """A value of /proc/self/status, in bytes."""
    with open(_STATUS) as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) * 1024  # kB
    raise KeyError(key)


def _linux_peak_resettable():
    """Whether the peak RSS can be read and reset (Linux 4.0+)."""
    try:
        _read_status("VmHWM")
        with open(_CLEAR_REFS, "w") as f:
            f.write("5")
        return True
    except OSError:
        return False
    except KeyError:
        return False


class _Sampler:
    """Peak RSS sampled by a thread, with psutil."""

    def __init__(self, process, interval=0.001):
        self._process = process
        self._interval = interval
        self._peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self._peak = max(self._peak, self._process.memory_info().rss)
            time.sleep(self._interval)

    def start(self):
        self._peak = self._process.memory_info().rss
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        return max(self._peak, self._process.memory_info().rss)


def _load_malloc_trim():
    """glibc's malloc_trim, or None."""
    name = ctypes.util.find_library("c")
    if name is None:
        return None
    try:
        return ctypes.CDLL(name).malloc_trim
    except (OSError, AttributeError):
        return None


_malloc_trim = _load_malloc_trim()


def _release_free_memory():
    """Collect the garbage and return the free memory of the allocator."""
    gc.collect()
    if _malloc_trim is not None:
        _malloc_trim(0)


if _linux_peak_resettable():
    METHOD = "VmHWM"
    _process = None
else:
    import psutil

    METHOD = "sampling"
    _process = psutil.Process(os.getpid())


def rss():
    """Current RSS of the process, in bytes."""
    if METHOD == "VmHWM":
        return _read_status("VmRSS")
    assert _process is not None
    return _process.memory_info().rss


@dataclass
class PhaseMemory:
    """Memory of a phase, in bytes, relative to the RSS before it."""

    retained: int = 0
    peak: int = 0

    @property
    def retained_mb(self):
        return self.retained / MB

    @property
    def peak_mb(self):
        return self.peak / MB


@contextmanager
def measure():
    """Measure the memory of the code run in the context.

    The garbage is collected before and after the phase, so that the retained
    memory only counts what the phase keeps.
    """
    memory = PhaseMemory()
    _release_free_memory()
    before = rss()
    if METHOD == "VmHWM":
        with open(_CLEAR_REFS, "w") as f:
            f.write("5")  # reset the peak RSS to the current RSS
        sampler = None
    else:
        assert _process is not None
        sampler = _Sampler(_process)
        sampler.start()
    try:
        yield memory
    finally:
        peak = _read_status("VmHWM") if sampler is None else sampler.stop()
        _release_free_memory()
        memory.retained = rss() - before
        memory.peak = peak - before


def nbytes(obj):
    """Size of the data of a NumPy array or a pandas DataFrame, in bytes."""
    if obj is None:
        return 0
    if hasattr(obj, "memory_usage"):
        return int(obj.memory_usage(index=True, deep=True).sum())
    return int(obj.nbytes)
