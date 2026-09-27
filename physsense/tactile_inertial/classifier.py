"""Unvalidated toy rules for experimenting with vibration features."""
import warnings
from physsense.tactile_inertial.features import TactileFeatureExtractor


class HeuristicContentClassifier:
    """Illustrative thresholds only: no training data, benchmark, or accuracy claim.

    Labels are hypotheses, not measurements of container contents. RMS thresholds
    assume arbitrary demo units. Return values are the label and measured features,
    never confidence scores or probabilities. Do not use for operational decisions.
    """
    CLASSES = ("empty", "liquid", "granular", "rigid_solid")

    def __init__(self, sample_rate=1000.0):
        self.extractor = TactileFeatureExtractor(sample_rate=sample_rate)

    def classify_shaking_episode(self, vibration_signal):
        arr = self.extractor._channels(vibration_signal)
        if len(arr) < 16:
            raise ValueError("classification requires at least 16 samples")
        features = self.extractor.extract_features(arr)
        if features["rms"] < 0.12:
            label = "empty"
        elif features["spectral_centroid"] < 160 and features["hfer"] < 0.25:
            label = "liquid"
        elif features["spectral_centroid"] <= 480:
            label = "granular"
        else:
            label = "rigid_solid"
        return label, features


class ShakeSortClassifier(HeuristicContentClassifier):
    """Deprecated prototype name; not a research implementation.

    The second tuple element now contains measured features, not probabilities.
    """
    def __init__(self, *args, **kwargs):
        warnings.warn("Use HeuristicContentClassifier; returns features, not probabilities",
                      DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)
