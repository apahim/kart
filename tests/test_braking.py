"""Tests for scripts/analysis/braking.py."""

import plotly.graph_objects as go

from scripts.analysis.braking import create_braking_track_map, create_braking_consistency_chart


class TestBrakingMap:
    def test_returns_figure(self, telemetry_df):
        fig = create_braking_track_map(telemetry_df, best_lap=1)
        assert isinstance(fig, go.Figure)

    def test_axes_hidden(self, telemetry_df):
        fig = create_braking_track_map(telemetry_df, best_lap=1)
        assert fig.layout.xaxis.showticklabels is False
        assert fig.layout.yaxis.showticklabels is False

    def test_no_latlon(self, telemetry_df_minimal):
        result = create_braking_track_map(telemetry_df_minimal)
        assert result is None


class TestBrakingConsistency:
    def test_returns_figure_or_none(self, telemetry_df_multi_lap):
        import pandas as pd
        laptimes = pd.DataFrame({
            "lap": [1, 2, 3],
            "seconds": [70.0, 71.0, 72.0],
        })
        result = create_braking_consistency_chart(telemetry_df_multi_lap, laptimes)
        assert result is None or isinstance(result, go.Figure)

    def test_no_longitudinal(self, telemetry_df_minimal):
        result = create_braking_consistency_chart(telemetry_df_minimal)
        assert result is None
