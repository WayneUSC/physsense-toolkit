"""NumPy window statistics for uniformly sampled, calibrated sensor data.

Thresholds use the caller's input units. They are illustrative activity measures,
not calibrated contact/slip detectors. Spectra use channel-wise mean removal,
a Hann window, and summed one-sided power; vector magnitudes are NOT rectified
before the FFT (which would double a sinusoid's apparent frequency).
"""
import numpy as np
from physsense._validation import positive_float, finite_array


class TactileFeatureExtractor:
    def __init__(self, sample_rate=1000.0, contact_threshold_rms=0.15,
                 high_frequency_cutoff=200.0):
        self.sample_rate = positive_float(sample_rate, "sample_rate")
        self.contact_threshold_rms = positive_float(contact_threshold_rms, "contact_threshold_rms")
        self.high_frequency_cutoff = positive_float(high_frequency_cutoff, "high_frequency_cutoff")
        if self.high_frequency_cutoff >= self.sample_rate / 2:
            raise ValueError("high_frequency_cutoff must be below the Nyquist frequency")

    @staticmethod
    def _channels(signal):
        arr = finite_array(signal, "signal", complex_ok=False)
        if arr.ndim == 1:
            return arr[:, None]
        if arr.ndim == 2 and arr.shape[1] > 0:
            return arr
        raise ValueError("signal must have shape (samples,) or (samples, channels)")

    def compute_rms(self, signal):
        """RMS of the vector norm, including DC, over all supplied channels."""
        arr = self._channels(signal)
        if len(arr) == 0:
            return 0.0
        scale = np.max(np.abs(arr))
        if scale == 0:
            return 0.0
        result = float(scale * np.sqrt(np.mean(np.sum((arr / scale) ** 2, axis=1))))
        if not np.isfinite(result):
            raise ValueError("signal magnitude overflows RMS")
        return result

    def compute_spectral_features(self, signal):
        """Power-weighted centroid, power fraction above cutoff, and peak Hz.

        Windows shorter than 16 samples or containing only DC return zeros.
        All channels contribute; the caller must select/calibrate channels.
        """
        arr = self._channels(signal)
        zero = {"spectral_centroid": 0.0, "hfer": 0.0, "dominant_frequency": 0.0}
        n = len(arr)
        if n < 16:
            return zero
        scale = np.max(np.abs(arr))
        if scale == 0:
            return zero
        normalized = arr / scale
        centered = normalized - normalized.mean(axis=0)
        fft = np.fft.rfft(centered * np.hanning(n)[:, None], axis=0)
        power = np.sum(np.abs(fft) ** 2, axis=1)
        # Account for the omitted negative-frequency half of a real spectrum.
        power[1:-1 if n % 2 == 0 else None] *= 2
        total = power.sum()
        if total <= np.finfo(float).eps:
            return zero
        frequencies = np.fft.rfftfreq(n, 1 / self.sample_rate)
        return {
            "spectral_centroid": float(np.dot(frequencies, power) / total),
            "hfer": float(power[frequencies > self.high_frequency_cutoff].sum() / total),
            "dominant_frequency": float(frequencies[np.argmax(power)]),
        }

    def extract_features(self, sensor_window):
        rms = self.compute_rms(sensor_window)
        return {"rms": rms, **self.compute_spectral_features(sensor_window),
                "in_contact": float(rms >= self.contact_threshold_rms)}
