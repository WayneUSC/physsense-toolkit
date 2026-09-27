"""Deterministic synthetic signal demo and local throughput measurement."""
import argparse
import math
import time
from physsense.mock.synthetic_generator import SyntheticSignalGenerator
from physsense.tactile_inertial.features import TactileFeatureExtractor
from physsense.tactile_inertial.stream_buffer import StreamBuffer
from physsense.visualization.rerun_dashboard import RerunPhysicalDashboard


def main(argv=None):
    parser = argparse.ArgumentParser(description="PhysSense: offline/synthetic signal-processing alpha")
    parser.add_argument("--mode", choices=["tactile", "rerun", "benchmark"], default="tactile")
    parser.add_argument("--duration", type=float, default=3.0, help="Simulated seconds, not wall-clock duration")
    parser.add_argument("--realtime", action="store_true", help="Best-effort pacing; no real-time guarantee")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration) or args.duration <= 0 or args.duration > 3600:
        parser.error("duration must be finite and in (0, 3600] seconds")
    rate = 1000.0
    count = max(1, int(args.duration * rate))
    generator = SyntheticSignalGenerator(rate, seed=args.seed)
    buffer = StreamBuffer(capacity=5000, num_channels=3)
    extractor = TactileFeatureExtractor(sample_rate=rate)
    try:
        dashboard = RerunPhysicalDashboard(spawn=True) if args.mode == "rerun" else None
    except ImportError as exc:
        parser.error(str(exc))
    print(f"PhysSense synthetic demo: {count} samples on a {rate:g} Hz signal clock.")
    print("Generator burst labels are not measured slip. Contact is an uncalibrated RMS threshold.")
    start = time.perf_counter()
    for step in range(count):
        signal_time = step / rate
        active = (step // 400) % 2 == 1
        burst = active and (step // 800) % 2 == 1
        sample = generator.generate_tactile_sample(is_contact=active, burst=burst)
        buffer.append(signal_time, sample)
        if step >= 63 and step % 50 == 0:
            _, window = buffer.get_latest_window(64)
            feats = extractor.extract_features(window)
            if args.mode != "benchmark":
                print(f"t={signal_time:.3f}s generated_burst={int(burst)} "
                      f"threshold_active={int(feats['in_contact'])} RMS={feats['rms']:.3f} "
                      f"centroid={feats['spectral_centroid']:.1f}Hz HFER={feats['hfer']:.3f}")
            if dashboard:
                dashboard.log_tactile_sample(signal_time, sample, feats)
        if args.realtime:
            remaining = start + (step + 1) / rate - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
    elapsed = time.perf_counter() - start
    print(f"Processed {count} synthetic samples in {elapsed:.4f}s ({count / max(elapsed, 1e-12):.0f} samples/s).")
    print("This measures this process only; hardware acquisition and timing are not validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
