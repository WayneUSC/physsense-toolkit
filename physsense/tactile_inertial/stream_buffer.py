"""Bounded, lock-protected sensor buffer; no real-time timing guarantee."""
import threading
import numpy as np
from physsense._validation import positive_int, finite_array


class StreamBuffer:
    """One clock, monotonically nondecreasing timestamps, exact channel counts.

    Append and reads use the same lock. Windows are detached copies in insertion
    order. Equal timestamps are allowed; clock resets need a new buffer.
    """
    def __init__(self, capacity=10000, num_channels=6):
        self.capacity = positive_int(capacity, "capacity")
        self.num_channels = positive_int(num_channels, "num_channels")
        self._timestamps = np.zeros(self.capacity, dtype=np.float64)
        self._data = np.zeros((self.capacity, self.num_channels), dtype=np.float64)
        self._head = self._count = 0
        self._last_timestamp = None
        self._lock = threading.Lock()

    def append(self, timestamp, values):
        stamp = finite_array(timestamp, "timestamp", complex_ok=False)
        data = finite_array(values, "values", complex_ok=False)
        if stamp.ndim != 0:
            raise ValueError("timestamp must be a scalar")
        if data.shape != (self.num_channels,):
            raise ValueError(f"values must contain exactly {self.num_channels} channels")
        stamp = float(stamp)
        with self._lock:
            if self._last_timestamp is not None and stamp < self._last_timestamp:
                raise ValueError("timestamps must be monotonically nondecreasing")
            self._timestamps[self._head] = stamp
            self._data[self._head] = data
            self._head = (self._head + 1) % self.capacity
            self._count = min(self._count + 1, self.capacity)
            self._last_timestamp = stamp

    def get_latest_window(self, window_size):
        window_size = positive_int(window_size, "window_size", minimum=0)
        with self._lock:
            n = min(window_size, self._count)
            indices = (self._head - n + np.arange(n)) % self.capacity
            return self._timestamps[indices].copy(), self._data[indices].copy()

    @property
    def sample_count(self):
        with self._lock:
            return self._count
