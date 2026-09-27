"""Shared numerical boundary checks."""
import math
from numbers import Integral, Real
import numpy as np


def positive_int(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def positive_float(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def finite_array(value, name, complex_ok=True):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iufc" or (not complex_ok and np.iscomplexobj(raw)):
            raise ValueError()
        arr = np.asarray(raw, dtype=np.complex128 if np.iscomplexobj(raw) else np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values")
    return arr
