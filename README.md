# PhysSense-Toolkit

Small, testable building blocks for vibration signals and offline radar IQ analysis.

[中文说明](README.zh-CN.md) · [Contributing](CONTRIBUTING.md)

**Status: experimental signal-processing alpha.** The included demos use synthetic data. This repository does not include sensor acquisition drivers, a validated slip detector, a trained material classifier, a measured real-time pipeline, or a LeRobot policy integration.

## Included building blocks

| Component | Implemented scope |
| --- | --- |
| Stream buffer | Bounded timestamped windows for multi-channel samples |
| Vibration features | RMS, spectral centroid, dominant frequency and high-frequency energy ratio |
| ADC reader | Strict decoding of explicitly supported offline sample layouts |
| Range spectrum | Windowed FFT of supplied ADC samples; no automatic metric range calibration |
| Micro-Doppler | Slow-time STFT with signed-frequency handling for complex IQ |
| Observation helper | Adds feature values to a Python observation dictionary |
| Rerun logger | Optional scalar and supplied point-cloud logging |
| Content heuristic | Illustrative hand-written rules returning a label and measured features, without confidence probabilities |

NumPy is required. Supplying a sample rate of 1000 Hz describes the signal's time base; it does not guarantee acquisition or processing at that wall-clock rate. Thresholds depend on sensor units, mounting and the experiment.

## Quickstart

Use Python 3.10 or newer:

```sh
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e '.[dev]'
python -m physsense.cli --mode tactile --duration 3
```

This is a synthetic demonstration. Displayed generator states are ground truth for that generator, not proof of a contact/slip estimator's accuracy on hardware.

For optional visualization:

```sh
python -m pip install -e '.[rerun]'
python -m physsense.cli --mode rerun --duration 3
```

The optional dependency specifies the supported Rerun SDK range. A GUI viewer needs a suitable desktop environment. Scalar logging and logging supplied points do not imply reconstruction of a 3D radar point cloud or synchronization across physical devices.

## Extract features

```python
import numpy as np
from physsense.tactile_inertial.features import TactileFeatureExtractor

sample_rate = 1000.0
t = np.arange(1000) / sample_rate
signal = np.column_stack([np.sin(2 * np.pi * 100 * t), np.zeros_like(t), np.zeros_like(t)])
features = TactileFeatureExtractor(sample_rate=sample_rate).extract_features(signal)
print(features)
```

Choose sample rate, window and units from your actual sensor data. Frequencies above Nyquist cannot be inferred from undersampled measurements. Feature thresholds and heuristic labels must be validated against labeled recordings before being used to draw physical conclusions.

## Radar input boundary

The ADC reader accepts **offline sample bytes** in a declared layout. It does not reassemble DCA1000 UDP packets, recover packet loss, infer TI capture configuration, or support arbitrary radar boards. Raw TI captures have device/configuration-specific packing; verify layout and byte count before analysis. The legacy module name `dca1000_parser` does not imply complete DCA1000 hardware support. Use `ADCFileParser` with an explicit layout; the legacy `DCA1000Parser` name is deprecated.

```python
import numpy as np
from physsense.radar_mmwave.dca1000_parser import ADCFileParser

parser = ADCFileParser(num_rx=1, num_samples_per_chirp=4, layout="iq_interleaved")
fixture = np.array([100, -20, 50, 30, -100, 20, -50, -30], dtype="<i2").tobytes()
cube = parser.parse_adc_bytes(fixture)  # shape (1, 1, 4)
spectrum = parser.compute_range_profile(cube)
```

Micro-Doppler consumes an already selected slow-time signal. Selecting an appropriate range bin and removing clutter remain caller responsibilities.

## Observation augmentation

```python
from physsense import StreamBuffer, TactileObservationWrapper

buffer = StreamBuffer(capacity=5000, num_channels=3)
buffer.append(0.0, [0.0, 0.0, 0.0])
augmented = TactileObservationWrapper(stream_buffer=buffer).augment_observation({"joint_positions": [0.0]})
```

This helper adds dictionary fields. It does not modify a model's input schema, train a policy, align asynchronous clocks, or prove compatibility with a particular LeRobot release. The host application must configure and test those steps explicitly.

## Validation and contributions

```sh
python -m pytest
python -m pip install build
python -m build
```

Synthetic unit tests can establish numerical and API behavior; they cannot establish material-classification accuracy, physical slip detection, radar-board compatibility or processing latency under sensor load. Contributions are most useful with an anonymized input fixture, capture configuration, expected result and reproducible environment.

Licensed under [Apache 2.0](LICENSE). Maintained by [@WayneUSC](https://github.com/WayneUSC).
