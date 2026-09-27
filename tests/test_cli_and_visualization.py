from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from physsense.cli import main
from physsense.mock.synthetic_generator import SyntheticSignalGenerator
from physsense.visualization.rerun_dashboard import RerunPhysicalDashboard


def test_synthetic_generator_is_seeded_and_cli_reports_simulation(capsys):
    first = SyntheticSignalGenerator(seed=123)
    second = SyntheticSignalGenerator(seed=123)
    assert first.generate_tactile_sample(True, True) == second.generate_tactile_sample(True, True)
    assert main(['--duration', '.15', '--mode', 'benchmark']) == 0
    output = capsys.readouterr().out
    assert '150 samples' in output
    assert 'hardware acquisition and timing are not validated' in output
    for duration in ('nan', 'inf', '-1', '0'):
        with pytest.raises(SystemExit) as exc:
            main(['--duration', duration])
        assert exc.value.code == 2


def test_dashboard_uses_current_api_and_validates_before_logging(monkeypatch):
    calls = []
    fake = SimpleNamespace(init=lambda *a, **kw: calls.append(('init', a, kw)),
                           set_time=lambda *a, **kw: calls.append(('time', a, kw)),
                           Scalars=lambda value: value,
                           Points3D=lambda value, **kw: value,
                           log=lambda *a, **kw: calls.append(('log', a, kw)))
    monkeypatch.setitem(sys.modules, 'rerun', fake)
    dashboard = RerunPhysicalDashboard()
    dashboard.log_tactile_sample(1.25, [1, 2, 3], {'rms': 2})
    assert ('time', ('sensor_time',), {'duration': 1.25}) in calls
    assert len([c for c in calls if c[0] == 'log']) == 7
    before = len(calls)
    with pytest.raises(ValueError):
        dashboard.log_tactile_sample(2, [1, 2, np.nan], {})
    assert len(calls) == before
    dashboard.log_radar_points(2, [])


def test_real_rerun_sdk_headless_recording(tmp_path):
    rr = pytest.importorskip('rerun', reason='Optional SDK: run pip install .[dev,rerun]')
    dashboard = RerunPhysicalDashboard(app_id='physsense-tests', spawn=False)
    destination = tmp_path / 'smoke.rrd'
    rr.save(str(destination))
    dashboard.log_tactile_sample(0, [1, 2, 3], {'rms': 1, 'hfer': .2})
    dashboard.log_radar_points(.1, [[0, 1, 2]])
    rr.get_global_data_recording().flush()
    assert destination.exists()
    assert destination.stat().st_size > 100
