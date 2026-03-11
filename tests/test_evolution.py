"""Tests for scripts/analysis/evolution.py — cross-race evolution analysis."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.analysis.evolution import (
    load_all_races,
    load_all_laptimes,
    prepare_raceline_data,
)


def _create_race_dir(base_dir, name, track, n_laps=5, has_gps=True):
    """Create a minimal race directory with telemetry and metadata."""
    race_dir = os.path.join(base_dir, name)
    os.makedirs(race_dir, exist_ok=True)

    # race.yaml
    meta = {"track": track, "date": name[:10], "session_type": "Practice"}
    with open(os.path.join(race_dir, "race.yaml"), "w") as f:
        yaml.dump(meta, f)

    # telemetry.csv — circular track
    rows_per_lap = 50
    n = rows_per_lap * n_laps
    angle = np.tile(np.linspace(0, 2 * np.pi, rows_per_lap), n_laps)
    lap_nums = np.repeat(np.arange(1, n_laps + 1), rows_per_lap)
    elapsed = np.linspace(0, 60 * n_laps, n)

    lat = 52.5 + 0.001 * np.sin(angle)
    lon = -7.5 + 0.001 * np.cos(angle)
    speed = 10 + 2 * np.sin(4 * angle)

    lines = []
    if has_gps:
        lines.append("timestamp,elapsed_time,lap_number,latitude,longitude,speed")
        lines.append("s,s,,deg,deg,m/s")
        lines.append(",,,,, ")
        for i in range(n):
            lines.append(f"{elapsed[i]:.3f},{elapsed[i]:.3f},{int(lap_nums[i])},{lat[i]:.8f},{lon[i]:.8f},{speed[i]:.3f}")
    else:
        lines.append("timestamp,elapsed_time,lap_number,speed")
        lines.append("s,s,,m/s")
        lines.append(",,,")
        for i in range(n):
            lines.append(f"{elapsed[i]:.3f},{elapsed[i]:.3f},{int(lap_nums[i])},{speed[i]:.3f}")

    with open(os.path.join(race_dir, "telemetry.csv"), "w") as f:
        f.write("\n".join(lines))

    return race_dir


class TestPrepareRacelineData:
    def test_returns_correct_structure(self, tmp_path):
        _create_race_dir(str(tmp_path), "2026-03-08-TestTrack", "TestTrack")
        result = prepare_raceline_data("TestTrack", str(tmp_path / "2026-03-08-TestTrack"), data_dir=str(tmp_path))
        assert result is not None
        assert "sessions" in result
        assert len(result["sessions"]) == 1
        session = result["sessions"][0]
        assert session["date"] == "2026-03-08"
        assert session["is_current"] is True
        assert len(session["laps"]) > 0

    def test_is_current_flag(self, tmp_path):
        _create_race_dir(str(tmp_path), "2026-03-01-TestTrack", "TestTrack")
        _create_race_dir(str(tmp_path), "2026-03-08-TestTrack", "TestTrack")
        result = prepare_raceline_data("TestTrack", str(tmp_path / "2026-03-08-TestTrack"), data_dir=str(tmp_path))
        assert result is not None
        current = [s for s in result["sessions"] if s["is_current"]]
        other = [s for s in result["sessions"] if not s["is_current"]]
        assert len(current) == 1
        assert len(other) == 1
        assert current[0]["date"] == "2026-03-08"

    def test_downsample_limit(self, tmp_path):
        _create_race_dir(str(tmp_path), "2026-03-08-TestTrack", "TestTrack", n_laps=3)
        result = prepare_raceline_data("TestTrack", str(tmp_path / "2026-03-08-TestTrack"), data_dir=str(tmp_path))
        assert result is not None
        for session in result["sessions"]:
            for lap in session["laps"]:
                assert len(lap["x"]) <= 200
                assert len(lap["y"]) <= 200

    def test_no_gps_excluded(self, tmp_path):
        _create_race_dir(str(tmp_path), "2026-03-08-TestTrack", "TestTrack", has_gps=False)
        result = prepare_raceline_data("TestTrack", str(tmp_path / "2026-03-08-TestTrack"), data_dir=str(tmp_path))
        assert result is None

    def test_different_track_excluded(self, tmp_path):
        _create_race_dir(str(tmp_path), "2026-03-08-TrackA", "TrackA")
        _create_race_dir(str(tmp_path), "2026-03-08-TrackB", "TrackB")
        result = prepare_raceline_data("TrackA", str(tmp_path / "2026-03-08-TrackA"), data_dir=str(tmp_path))
        assert result is not None
        assert len(result["sessions"]) == 1
        assert result["sessions"][0]["date"] == "2026-03-08"

    def test_lap_fields(self, tmp_path):
        _create_race_dir(str(tmp_path), "2026-03-08-TestTrack", "TestTrack")
        result = prepare_raceline_data("TestTrack", str(tmp_path / "2026-03-08-TestTrack"), data_dir=str(tmp_path))
        lap = result["sessions"][0]["laps"][0]
        assert "lap" in lap
        assert "time_fmt" in lap
        assert "seconds" in lap
        assert "is_best" in lap
        assert "is_outlier" in lap
        assert "x" in lap
        assert "y" in lap

    def test_has_best_lap(self, tmp_path):
        _create_race_dir(str(tmp_path), "2026-03-08-TestTrack", "TestTrack", n_laps=5)
        result = prepare_raceline_data("TestTrack", str(tmp_path / "2026-03-08-TestTrack"), data_dir=str(tmp_path))
        best_laps = [l for l in result["sessions"][0]["laps"] if l["is_best"]]
        assert len(best_laps) == 1

    def test_empty_dir_returns_none(self, tmp_path):
        result = prepare_raceline_data("NoTrack", str(tmp_path / "nonexistent"), data_dir=str(tmp_path))
        assert result is None
