"""Tests for scripts/analysis/utils.py."""

import numpy as np
import pandas as pd

from scripts.analysis.utils import format_laptime, project_to_meters, detect_corners_with_positions


class TestFormatLaptime:
    def test_with_minutes(self):
        assert format_laptime(69.742) == "1:09.742"

    def test_under_minute(self):
        assert format_laptime(45.123) == "45.123"

    def test_exact_minute(self):
        assert format_laptime(60.0) == "1:00.000"

    def test_large(self):
        assert format_laptime(125.5) == "2:05.500"


class TestProjectToMeters:
    def test_origin(self):
        lat = np.array([52.5, 52.5, 52.5])
        lon = np.array([-7.5, -7.5, -7.5])
        x, y = project_to_meters(lat, lon)
        np.testing.assert_allclose(x, 0, atol=1e-10)
        np.testing.assert_allclose(y, 0, atol=1e-10)

    def test_scale(self):
        lat = np.array([52.5, 52.501])
        lon = np.array([-7.5, -7.5])
        x, y = project_to_meters(lat, lon)
        # 0.001 degree lat ~ 110.54m
        assert abs(y[1] - y[0]) > 100
        assert abs(y[1] - y[0]) < 120

    def test_single_point(self):
        x, y = project_to_meters(np.array([52.5]), np.array([-7.5]))
        assert float(x[0]) == 0.0
        assert float(y[0]) == 0.0


class TestDetectCornersWithPositions:
    def test_returns_labels(self, telemetry_df):
        result = detect_corners_with_positions(telemetry_df, best_lap=1)
        assert isinstance(result, list)
        if len(result) > 0:
            assert result[0]["label"].startswith("T")

    def test_empty_when_no_corners(self, telemetry_df_minimal):
        result = detect_corners_with_positions(telemetry_df_minimal)
        assert result == []

    def test_has_distance(self, telemetry_df):
        result = detect_corners_with_positions(telemetry_df, best_lap=1)
        for corner in result:
            assert "distance" in corner
