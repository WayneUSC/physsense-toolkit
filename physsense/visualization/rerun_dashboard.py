"""Optional Rerun scalar/point logger, tested against rerun-sdk 0.23.2."""
from collections.abc import Mapping
import numpy as np
from physsense._validation import finite_array


class RerunPhysicalDashboard:
    """Log caller-supplied samples; does not derive radar points or synchronize clocks."""
    def __init__(self, app_id="PhysSense-Demo", spawn=False):
        try:
            import rerun as rr
        except ImportError as exc:
            raise ImportError('Install the optional visualizer with pip install "physsense[rerun]"') from exc
        self.rr = rr
        self.has_rerun = True
        rr.init(app_id, spawn=spawn)

    def _time(self, timestamp):
        stamp = finite_array(timestamp, "timestamp", complex_ok=False)
        if stamp.ndim != 0:
            raise ValueError("timestamp must be scalar elapsed seconds")
        self.rr.set_time("sensor_time", duration=float(stamp))

    def log_tactile_sample(self, timestamp, acc_xyz, features):
        """Log one 3-axis sample and summary features at elapsed seconds."""
        xyz = finite_array(acc_xyz, "acc_xyz", complex_ok=False)
        if xyz.shape != (3,):
            raise ValueError("acc_xyz must contain three values")
        if not isinstance(features, Mapping):
            raise ValueError("features must be a mapping")
        names = ("rms", "spectral_centroid", "hfer", "in_contact")
        values = finite_array([features.get(key, 0.0) for key in names], "features", complex_ok=False)
        if values.shape != (4,):
            raise ValueError("feature values must be scalars")
        self._time(timestamp)
        for axis, value in zip("xyz", xyz):
            self.rr.log(f"tactile/acc_{axis}", self.rr.Scalars(float(value)))
        for name, value in zip(names, values):
            self.rr.log(f"features/{name}", self.rr.Scalars(float(value)))

    def log_radar_points(self, timestamp, point_cloud):
        """Log already-computed Nx3 points; no ADC-to-point-cloud conversion."""
        points = finite_array(point_cloud, "point_cloud", complex_ok=False)
        if points.size == 0:
            points = np.empty((0, 3))
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("point_cloud must have shape (N, 3)")
        self._time(timestamp)
        self.rr.log("radar/points", self.rr.Points3D(points, colors=[0, 200, 255]))
