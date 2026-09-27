import numpy as np
import pytest
from physsense import TactileFeatureExtractor, HeuristicContentClassifier, ShakeSortClassifier


def test_rms_list_array_and_integer_inputs_agree():
    extractor = TactileFeatureExtractor()
    for signal in ([30000, -30000], np.array([30000, -30000], dtype=np.int16)):
        assert extractor.compute_rms(signal) == pytest.approx(30000)
    assert extractor.compute_rms([[3, 4], [3, 4]]) == pytest.approx(5)


def test_multichannel_spectrum_preserves_frequency_without_rectification():
    t = np.arange(1000) / 1000
    signal = np.sin(2 * np.pi * 120 * t)
    extractor = TactileFeatureExtractor()
    features = extractor.extract_features(np.column_stack([signal + 9, signal, signal]))
    assert features['dominant_frequency'] == 120
    assert features['spectral_centroid'] == pytest.approx(120, abs=.01)
    assert extractor.extract_features(signal.tolist())['dominant_frequency'] == 120


def test_hfer_is_power_fraction_not_amplitude_fraction():
    t = np.arange(1000) / 1000
    signal = np.sin(2 * np.pi * 100 * t) + 2 * np.sin(2 * np.pi * 300 * t)
    features = TactileFeatureExtractor().compute_spectral_features(signal)
    assert features['hfer'] == pytest.approx(.8, abs=.001)
    assert features['spectral_centroid'] == pytest.approx(260, abs=.01)
    assert features['dominant_frequency'] == 300


def test_empty_short_and_dc_windows():
    extractor = TactileFeatureExtractor()
    assert extractor.extract_features([]) == dict(rms=0, spectral_centroid=0, hfer=0, dominant_frequency=0, in_contact=0)
    for signal in ([2] * 8, [2] * 64, np.zeros((0, 3))):
        assert extractor.compute_spectral_features(signal)['hfer'] == 0
    assert extractor.extract_features([.3] * 64)['in_contact'] == 1


@pytest.mark.parametrize('signal', [[np.nan], [np.inf], [1j], [[1, 2], [3]], np.zeros((2, 2, 2)), np.empty((10, 0)), 1, ['1']])
def test_invalid_signal_rejected(signal):
    with pytest.raises(ValueError):
        TactileFeatureExtractor().extract_features(signal)


@pytest.mark.parametrize('kwargs', [dict(sample_rate=0), dict(sample_rate=np.nan), dict(sample_rate=True), dict(sample_rate=400), dict(contact_threshold_rms=-1), dict(high_frequency_cutoff=500)])
def test_invalid_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        TactileFeatureExtractor(**kwargs)


def test_heuristic_reports_features_not_probabilities_and_rejects_no_data():
    classifier = HeuristicContentClassifier()
    label, features = classifier.classify_shaking_episode(np.zeros(64))
    assert label == 'empty'
    assert set(features) == {'rms', 'hfer', 'spectral_centroid', 'dominant_frequency', 'in_contact'}
    with pytest.raises(ValueError):
        classifier.classify_shaking_episode([])
    with pytest.warns(DeprecationWarning):
        legacy = ShakeSortClassifier()
    assert legacy.classify_shaking_episode(np.zeros(64)) == (label, features)
