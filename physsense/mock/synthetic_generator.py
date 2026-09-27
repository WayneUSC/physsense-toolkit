"""Seeded synthetic fixtures, not a physical sensor/material simulation."""
import math
import random
import struct
import warnings
from physsense._validation import positive_float, positive_int


class SyntheticSignalGenerator:
    def __init__(self, sample_rate_hz=1000.0, seed=0):
        self.sample_rate_hz = positive_float(sample_rate_hz, "sample_rate_hz")
        self.t = 0.0
        self._sample = 0
        self._rng = random.Random(seed)

    def generate_tactile_sample(self, is_contact=False, burst=False, *, is_slip=None):
        """Toy noise/tones/burst in arbitrary units; frequencies below Nyquist.

        is_contact and burst are generator settings, not detector output or
        ground-truth evidence of real contact/slip. Legacy is_slip maps to burst.
        """
        if is_slip is not None:
            warnings.warn("is_slip is a synthetic burst setting, not measured slip; use burst",
                          DeprecationWarning, stacklevel=2)
            burst = is_slip
        self.t = self._sample / self.sample_rate_hz
        self._sample += 1
        noise = [self._rng.gauss(0, 0.02) for _ in range(3)]
        if not is_contact:
            return noise
        phase = 2 * math.pi * self.sample_rate_hz * self.t
        vibration = 0.2 * math.sin(0.12 * phase) + 0.1 * math.sin(0.35 * phase)
        if burst:
            vibration += 0.5 * math.sin(0.42 * phase) + self._rng.gauss(0, 0.15)
        return [noise[0] + vibration * 0.5, noise[1] + vibration * 0.3, noise[2] + vibration]

    def generate_mock_radar_bytes(self, num_chirps=16, num_samples=256, num_rx=4):
        """Random little-endian adjacent IQ file fixture; NOT UDP/DCA1000 packets."""
        count = (positive_int(num_chirps, "num_chirps") *
                 positive_int(num_samples, "num_samples") * positive_int(num_rx, "num_rx") * 2)
        words = [self._rng.randint(-1000, 1000) for _ in range(count)]
        return struct.pack(f"<{count}h", *words)
