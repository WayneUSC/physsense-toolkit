"""Offline ADC file layouts; not a DCA1000 UDP packet decoder or device driver.

The caller must remove transport headers, reorder packets, resolve packet loss,
and select a layout from the actual capture configuration. No format detection,
ADC bit-depth correction, IQ swap, or TDM-MIMO deinterleaving is performed.
"""
import warnings
import numpy as np
from physsense._validation import positive_int, finite_array


class ADCFileParser:
    """Decode little-endian signed 16-bit samples into (chirp, rx, sample).

    Layouts (each receiver's samples are contiguous within a chirp):
    * iq_interleaved: I0,Q0,I1,Q1,... (generic fixture layout).
    * ti_two_lane_iq: I0,I1,Q0,Q1,... (TI SWRA581B Figure 11 offline layout).
    * real: S0,S1,... .

    The TI layout is limited to even sample counts and 1, 2, or 4 receivers.
    It has synthetic fixture coverage only, not physical-device validation.
    Empty input returns a shaped empty cube; truncated chirps raise ValueError.
    """
    def __init__(self, num_rx=4, num_samples_per_chirp=256, layout="iq_interleaved"):
        self.num_rx = positive_int(num_rx, "num_rx")
        self.num_samples_per_chirp = positive_int(num_samples_per_chirp, "num_samples_per_chirp", 3)
        if layout not in {"iq_interleaved", "ti_two_lane_iq", "real"}:
            raise ValueError("unsupported ADC layout")
        if layout == "ti_two_lane_iq" and (self.num_rx not in (1, 2, 4) or self.num_samples_per_chirp % 2):
            raise ValueError("ti_two_lane_iq requires 1, 2, or 4 receivers and an even sample count")
        self.layout = layout
        self.is_complex = layout != "real"

    def parse_adc_bytes(self, raw_bytes):
        if not isinstance(raw_bytes, (bytes, bytearray, memoryview)):
            raise ValueError("raw_bytes must be a bytes-like buffer")
        raw_bytes = bytes(raw_bytes)
        words_per_chirp = self.num_rx * self.num_samples_per_chirp * (2 if self.is_complex else 1)
        if len(raw_bytes) % (2 * words_per_chirp):
            raise ValueError("input must contain complete chirps; truncated bytes are not discarded")
        words = np.frombuffer(raw_bytes, dtype="<i2").astype(np.float64)
        if self.layout == "ti_two_lane_iq":
            groups = words.reshape(-1, 4)
            decoded = (groups[:, :2] + 1j * groups[:, 2:]).reshape(-1)
        elif self.layout == "iq_interleaved":
            decoded = words[0::2] + 1j * words[1::2]
        else:
            decoded = words.astype(np.complex128)
        return decoded.reshape(-1, self.num_rx, self.num_samples_per_chirp)

    def compute_range_profile(self, adc_cube, positive_only=False):
        """Hann-windowed FFT magnitude; full unshifted bins by default.

        Positive-only output selects bins 0..N//2-1 explicitly. IQ frequency signs
        are otherwise preserved; distance in meters requires sensor calibration.
        This is unnormalized FFT magnitude, not power, range, or point clouds.
        """
        cube = finite_array(adc_cube, "adc_cube")
        if cube.ndim != 3 or cube.shape[1:] != (self.num_rx, self.num_samples_per_chirp):
            raise ValueError("adc_cube shape must be (chirps, num_rx, num_samples_per_chirp)")
        if not isinstance(positive_only, bool):
            raise ValueError("positive_only must be a bool")
        bins = self.num_samples_per_chirp // 2 if positive_only else self.num_samples_per_chirp
        if len(cube) == 0:
            return np.empty((0, self.num_rx, bins))
        transformed = np.fft.fft(cube * np.hanning(self.num_samples_per_chirp), axis=-1)
        result = np.abs(transformed[..., :bins])
        if not np.all(np.isfinite(result)):
            raise ValueError("ADC magnitude overflows FFT")
        return result


class DCA1000Parser(ADCFileParser):
    """Deprecated prototype name; only offline file layouts are supported."""
    def __init__(self, num_rx=4, num_samples_per_chirp=256, is_complex=True, *, layout=None):
        if not isinstance(is_complex, bool):
            raise ValueError("is_complex must be a bool")
        warnings.warn("Use ADCFileParser and an explicit offline layout; this is not a packet parser",
                      DeprecationWarning, stacklevel=2)
        if layout is None:
            layout = "iq_interleaved" if is_complex else "real"
        super().__init__(num_rx, num_samples_per_chirp, layout)
