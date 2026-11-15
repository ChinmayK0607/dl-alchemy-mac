"""Portable MLX runtime policies (safe to import without MLX or Metal)."""
import math
import os

MEMORY_FRACTION = 0.75


def validate_memory_gb(memory_gb):
    if memory_gb is not None and (not math.isfinite(memory_gb) or memory_gb <= 0):
        raise ValueError("memory-gb must be positive and finite")


def configure_memory(mx, memory_gb=None):
    """Apply the same allocation/wired cap; return the effective byte limit.

    Default to 75% of Metal's recommended working set, leaving headroom for
    the OS and other applications. Explicit overrides are measured in GiB.
    Call only after selecting the GPU device.
    """
    validate_memory_gb(memory_gb)
    if memory_gb is None:
        recommended = mx.device_info().get("max_recommended_working_set_size", 0)
        if not isinstance(recommended, (int, float)) or not math.isfinite(recommended) or recommended <= 0:
            raise RuntimeError("MLX did not report a recommended GPU working set; pass --memory-gb explicitly")
        limit = int(recommended * MEMORY_FRACTION)
    else:
        limit = int(memory_gb * 1024**3)
    if limit < 1:
        raise ValueError("memory-gb must allow at least one byte")
    mx.set_memory_limit(limit)
    mx.set_wired_limit(limit)
    return limit


def configure_test_device(mx):
    """Use normal MLX's default device unless MLX_DEVICE=cpu/gpu is requested."""
    device = os.environ.get("MLX_DEVICE")
    if device is not None:
        if device not in ("cpu", "gpu"):
            raise ValueError("MLX_DEVICE must be cpu or gpu")
        mx.set_default_device(mx.cpu if device == "cpu" else mx.gpu)
    return str(mx.default_device())
