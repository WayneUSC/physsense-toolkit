"""Signed slow-time STFT magnitude; no target separation or velocity calibration."""
import numpy as np
from physsense._validation import positive_float, positive_int, finite_array


class MicroDopplerExtractor:
    def __init__(self, chirp_rate_hz=1000.0, n_fft=128, hop_length=16):
        self.chirp_rate_hz = positive_float(chirp_rate_hz, "chirp_rate_hz")
        self.n_fft = positive_int(n_fft, "n_fft", minimum=3)
        self.hop_length = positive_int(hop_length, "hop_length")
        if self.hop_length > self.n_fft:
            raise ValueError("hop_length must not exceed n_fft")

    def extract_spectrogram(self, slow_time_signal):
        """Return frame-center seconds, signed Hz, and (frequency, time) magnitude.

        Input is uniformly sampled complex IQ (or real samples) from one selected
        range bin/channel. Full two-sided FFT preserves motion direction. Values
        are divided by the Hann window sum; they are not PSD or dB. Incomplete
        trailing windows are omitted and no padding is applied.
        """
        signal = finite_array(slow_time_signal, "slow_time_signal")
        if signal.ndim != 1:
            raise ValueError("slow_time_signal must be one-dimensional")
        freq = np.fft.fftshift(np.fft.fftfreq(self.n_fft, 1 / self.chirp_rate_hz))
        if len(signal) < self.n_fft:
            return np.empty(0), freq, np.empty((self.n_fft, 0))
        starts = np.arange(0, len(signal) - self.n_fft + 1, self.hop_length)
        window = np.hanning(self.n_fft)
        frames = np.stack([signal[start:start + self.n_fft] for start in starts], axis=1)
        spectrum = np.fft.fftshift(np.fft.fft(frames * window[:, None], axis=0), axes=0)
        magnitude = np.abs(spectrum) / window.sum()
        if not np.all(np.isfinite(magnitude)):
            raise ValueError("signal magnitude overflows STFT")
        times = (starts + (self.n_fft - 1) / 2) / self.chirp_rate_hz
        return times, freq, magnitude
