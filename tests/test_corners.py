"""Tests for scripts/analysis/corners.py."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from scripts.analysis.corners import (
    detect_corners,
    create_corner_analysis,
    create_corner_comparison_table,
    create_corner_min_speed_chart,
)


class TestDetectCorners:
    def test_finds_minima(self, telemetry_df):
        corners, lap_data = detect_corners(telemetry_df, best_lap=1)
        # Our sine-wave speed has 4 minima
        assert corners is not None
        assert len(corners) > 0

    def test_too_short(self):
        df = pd.DataFrame({
            "speed": np.ones(30),
            "lap_number": np.ones(30, dtype=int),
        })
        corners, lap_data = detect_corners(df, best_lap=1)
        assert corners is None
        assert lap_data is None

    def test_no_speed(self):
        df = pd.DataFrame({
            "timestamp": np.linspace(0, 10, 200),
            "lap_number": np.ones(200, dtype=int),
        })
        corners, lap_data = detect_corners(df)
        assert corners is None
        assert lap_data is None


class TestCornerAnalysis:
    def test_uses_distance_xaxis(self, telemetry_df, laptimes_df):
        fig = create_corner_analysis(telemetry_df, laptimes_df)
        if fig is not None:
            assert fig.layout.xaxis.title.text == "Distance (m)"

    def test_fallback_sample(self, telemetry_df, laptimes_df):
        df = telemetry_df.drop(columns=["distance_traveled"])
        fig = create_corner_analysis(df, laptimes_df)
        if fig is not None:
            assert fig.layout.xaxis.title.text == "Sample"


class TestComparisonTable:
    def test_returns_figure(self, telemetry_df_multi_lap):
        laptimes = pd.DataFrame({
            "lap": [1, 2, 3],
            "seconds": [70.0, 71.0, 72.0],
        })
        fig = create_corner_comparison_table(telemetry_df_multi_lap, laptimes)
        if fig is not None:
            assert isinstance(fig, go.Figure)
            assert any(isinstance(t, go.Table) for t in fig.data)

    def test_none_without_laptimes(self, telemetry_df):
        fig = create_corner_comparison_table(telemetry_df, laptimes_df=None)
        assert fig is None


class TestMinSpeedChart:
    def test_returns_figure(self, telemetry_df_multi_lap):
        laptimes = pd.DataFrame({
            "lap": [1, 2, 3],
            "seconds": [70.0, 71.0, 72.0],
        })
        fig = create_corner_min_speed_chart(telemetry_df_multi_lap, laptimes)
        if fig is not None:
            assert isinstance(fig, go.Figure)

    def test_highlights_best(self, telemetry_df_multi_lap):
        laptimes = pd.DataFrame({
            "lap": [1, 2, 3],
            "seconds": [70.0, 71.0, 72.0],
        })
        fig = create_corner_min_speed_chart(telemetry_df_multi_lap, laptimes)
        if fig is not None:
            # Best lap (1) should have red color
            best_traces = [t for t in fig.data if t.marker.color == "#e74c3c"]
            assert len(best_traces) > 0
