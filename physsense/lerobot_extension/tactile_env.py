"""Plain dictionary augmentation; not a Gym or LeRobot policy adapter."""
from collections.abc import Mapping
from physsense._validation import positive_int
from physsense.tactile_inertial.features import TactileFeatureExtractor
from physsense.tactile_inertial.stream_buffer import StreamBuffer


class TactileObservationWrapper:
    """Add summary scalars without changing the input mapping.

    Caller owns time alignment, missing/stale data handling, observation schema,
    feature normalization and model training. No policy compatibility is implied.
    Values assume uniform sampling at sample_rate; timestamps are not resampled.
    """
    KEYS = ("tactile_rms", "tactile_spectral_centroid", "tactile_hfer", "contact_state",
            "tactile_sample_count", "tactile_latest_timestamp")

    def __init__(self, stream_buffer: StreamBuffer, window_size=64, sample_rate=1000.0):
        if not isinstance(stream_buffer, StreamBuffer):
            raise ValueError("stream_buffer must be a StreamBuffer")
        self.stream_buffer = stream_buffer
        self.window_size = positive_int(window_size, "window_size")
        self.extractor = TactileFeatureExtractor(sample_rate=sample_rate)

    def augment_observation(self, base_obs):
        if not isinstance(base_obs, Mapping):
            raise ValueError("base_obs must be a mapping")
        if set(base_obs).intersection(self.KEYS):
            raise ValueError("base_obs already contains reserved tactile keys")
        timestamps, window = self.stream_buffer.get_latest_window(self.window_size)
        features = self.extractor.extract_features(window)
        return {
            **base_obs,
            "tactile_rms": features["rms"],
            "tactile_spectral_centroid": features["spectral_centroid"],
            "tactile_hfer": features["hfer"],
            "contact_state": int(features["in_contact"]),
            "tactile_sample_count": len(window),
            "tactile_latest_timestamp": float(timestamps[-1]) if len(window) else None,
        }
