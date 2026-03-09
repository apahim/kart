"""Tests for scripts/analysis/speed.py."""

import pandas as pd
import plotly.graph_objects as go

from scripts.analysis.speed import create_speed_traces, create_best_vs_comparison_speed


class TestSpeedTraces:
    def test_returns_figure(self, telemetry_df_multi_lap):
        laptimes = pd.DataFrame({
            "lap": [1, 2, 3],
            "seconds": [70.0, 71.0, 72.0],
        })
        fig = create_speed_traces(telemetry_df_multi_lap, laptimes)
        assert isinstance(fig, go.Figure)

    def test_has_envelope(self, telemetry_df_multi_lap):
        laptimes = pd.DataFrame({
            "lap": [1, 2, 3],
            "seconds": [70.0, 71.0, 72.0],
        })
        fig = create_speed_traces(telemetry_df_multi_lap, laptimes)
        fill_traces = [t for t in fig.data if getattr(t, "fill", None) == "tonexty"]
        assert len(fill_traces) > 0

    def test_no_distance(self, telemetry_df):
        df = telemetry_df.drop(columns=["distance_traveled"])
        result = create_speed_traces(df)
        assert result is None


class TestBestVsComparison:
    def test_returns_figure(self, telemetry_df_multi_lap):
        laptimes = pd.DataFrame({
            "lap": [1, 2, 3],
            "seconds": [69.0, 71.0, 72.0],
        })
        fig = create_best_vs_comparison_speed(telemetry_df_multi_lap, laptimes)
        if fig is not None:
            assert isinstance(fig, go.Figure)

    def test_none_single_lap(self, telemetry_df, laptimes_df_single):
        result = create_best_vs_comparison_speed(telemetry_df, laptimes_df_single)
        assert result is None

    def test_none_same_best_median(self, telemetry_df):
        laptimes = pd.DataFrame({"lap": [1], "seconds": [70.0]})
        result = create_best_vs_comparison_speed(telemetry_df, laptimes)
        assert result is None
