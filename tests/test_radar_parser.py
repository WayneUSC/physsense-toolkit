import numpy as np
import pytest
from physsense import ADCFileParser, DCA1000Parser, MicroDopplerExtractor


@pytest.mark.parametrize('layout', ['iq_interleaved', 'ti_two_lane_iq'])
def test_iq_golden_vector_receivers_and_chirps(layout):
    expected = (np.arange(16).reshape(2, 2, 4) - 8) + 1j * (np.arange(16).reshape(2, 2, 4) + 100)
    flattened = expected.reshape(-1)
    if layout == 'iq_interleaved':
        words = np.column_stack([flattened.real, flattened.imag])
    else:
        pairs = flattened.reshape(-1, 2)
        words = np.column_stack([pairs.real, pairs.imag])
    parser = ADCFileParser(2, 4, layout)
    cube = parser.parse_adc_bytes(words.astype('<i2').tobytes())
    np.testing.assert_array_equal(cube, expected)


def test_real_signed_little_endian_boundary_values():
    raw = bytes.fromhex('0080ff7f0100ffff')
    parser = ADCFileParser(1, 4, 'real')
    np.testing.assert_array_equal(parser.parse_adc_bytes(raw), [[[-32768, 32767, 1, -1]]])


@pytest.mark.parametrize('length', [1, 2, 3, 7, 15, 17, 30])
def test_incomplete_data_never_silently_dropped(length):
    with pytest.raises(ValueError, match='complete chirps'):
        ADCFileParser(1, 4).parse_adc_bytes(b'\x00' * length)


def test_empty_cube_and_profile_shapes():
    parser = ADCFileParser(2, 16)
    cube = parser.parse_adc_bytes(b'')
    assert cube.shape == (0, 2, 16)
    assert parser.compute_range_profile(cube).shape == (0, 2, 16)
    assert parser.compute_range_profile(cube, positive_only=True).shape == (0, 2, 8)


@pytest.mark.parametrize('frequency', [5, -5])
def test_range_fft_complex_tone_preserves_sign(frequency):
    n = 64
    cube = np.exp(2j * np.pi * frequency * np.arange(n) / n)[None, None, :]
    parser = ADCFileParser(1, n)
    profile = parser.compute_range_profile(cube)
    assert profile.shape == (1, 1, n)
    assert np.argmax(profile[0, 0]) == frequency % n
    assert profile[0, 0, frequency % n] == pytest.approx(np.hanning(n).sum())


@pytest.mark.parametrize('kwargs', [dict(num_rx=0), dict(num_samples_per_chirp=0), dict(num_rx=3, layout='ti_two_lane_iq'), dict(num_samples_per_chirp=5, layout='ti_two_lane_iq'), dict(layout='automatic')])
def test_invalid_adc_configuration(kwargs):
    with pytest.raises(ValueError):
        ADCFileParser(**kwargs)


def test_adc_invalid_values_shapes_and_legacy_name():
    parser = ADCFileParser(1, 4)
    for cube in (np.zeros((4,)), np.ones((1, 1, 4)) * np.inf):
        with pytest.raises(ValueError):
            parser.compute_range_profile(cube)
    with pytest.warns(DeprecationWarning):
        legacy = DCA1000Parser(1, 4, is_complex=False)
    assert legacy.layout == 'real'


@pytest.mark.parametrize('frequency', [128, -128])
def test_microdoppler_complex_tone_signed_peak_and_amplitude(frequency):
    signal = np.exp(2j * np.pi * frequency * np.arange(256) / 1024)
    times, frequencies, spectrum = MicroDopplerExtractor(1024, 128, 64).extract_spectrogram(signal.tolist())
    assert spectrum.shape == (128, 3)
    np.testing.assert_array_equal(frequencies[np.argmax(spectrum, axis=0)], [frequency] * 3)
    np.testing.assert_allclose(spectrum.max(axis=0), 1)
    np.testing.assert_allclose(times, np.array([63.5, 127.5, 191.5]) / 1024)


def test_microdoppler_empty_and_invalid():
    extractor = MicroDopplerExtractor()
    times, freq, spec = extractor.extract_spectrogram([])
    assert len(times) == 0 and len(freq) == 128 and spec.shape == (128, 0)
    for signal in ([[1, 2]], [np.nan], [np.inf]):
        with pytest.raises(ValueError):
            extractor.extract_spectrogram(signal)
    for kwargs in (dict(n_fft=2), dict(hop_length=0), dict(hop_length=129), dict(chirp_rate_hz=np.nan)):
        with pytest.raises(ValueError):
            MicroDopplerExtractor(**kwargs)
