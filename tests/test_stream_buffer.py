from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from physsense import StreamBuffer, TactileObservationWrapper


def test_ring_order_and_detached_snapshots():
    buffer = StreamBuffer(5, 2)
    for i in range(12):
        buffer.append(i / 1000, [i, -i])
    ts, data = buffer.get_latest_window(20)
    np.testing.assert_allclose(ts, np.arange(7, 12) / 1000)
    np.testing.assert_array_equal(data[:, 0], np.arange(7, 12))
    ts[:] = 999
    data[:] = 999
    assert buffer.get_latest_window(1)[1][0, 0] == 11
    assert buffer.sample_count == 5
    assert buffer.get_latest_window(0)[1].shape == (0, 2)


@pytest.mark.parametrize('kwargs', [dict(capacity=0), dict(capacity=-1), dict(capacity=1.2), dict(capacity=True), dict(num_channels=0)])
def test_invalid_buffer_config(kwargs):
    with pytest.raises(ValueError):
        StreamBuffer(**kwargs)


def test_append_validation_is_atomic():
    buffer = StreamBuffer(5, 2)
    buffer.append(1, [2, 3])
    for stamp, values in ((0, [1, 2]), (np.nan, [1, 2]), (2, [1]), (2, [1, 2, 3]), (2, [1, np.inf])):
        with pytest.raises(ValueError):
            buffer.append(stamp, values)
    assert buffer.sample_count == 1
    np.testing.assert_array_equal(buffer.get_latest_window(5)[1], [[2, 3]])
    for size in (-1, 1.5, True):
        with pytest.raises(ValueError):
            buffer.get_latest_window(size)


def test_concurrent_writer_and_readers_never_observe_partial_samples():
    buffer = StreamBuffer(64, 2)
    def write():
        for i in range(2000):
            buffer.append(i, [i, -i])
    def read():
        for _ in range(1000):
            ts, data = buffer.get_latest_window(64)
            assert len(ts) <= 64
            np.testing.assert_array_equal(ts, data[:, 0])
            np.testing.assert_array_equal(-ts, data[:, 1])
            assert np.all(np.diff(ts) >= 0)
    with ThreadPoolExecutor(max_workers=3) as executor:
        tasks = [executor.submit(write), executor.submit(read), executor.submit(read)]
        for task in tasks:
            task.result()
    assert buffer.sample_count == 64


def test_dictionary_wrapper_missing_data_metadata_and_collision():
    buffer = StreamBuffer(8, 3)
    wrapper = TactileObservationWrapper(buffer)
    original = {'position': [1, 2]}
    empty = wrapper.augment_observation(original)
    assert empty['tactile_sample_count'] == 0
    assert empty['tactile_latest_timestamp'] is None
    assert original == {'position': [1, 2]}
    buffer.append(12.5, [3, 4, 0])
    result = wrapper.augment_observation(original)
    assert result['tactile_rms'] == 5
    assert result['tactile_sample_count'] == 1
    assert result['tactile_latest_timestamp'] == 12.5
    with pytest.raises(ValueError, match='reserved'):
        wrapper.augment_observation({'tactile_rms': 99})
    with pytest.raises(ValueError):
        TactileObservationWrapper(buffer, window_size=0)
