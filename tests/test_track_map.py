"""Tests for scripts/analysis/track_map.py."""

import plotly.graph_objects as go

from scripts.analysis.track_map import create_speed_track_map, create_lateral_g_track_map


class TestSpeedMap:
    def test_returns_figure(self, telemetry_df):
        fig = create_speed_track_map(telemetry_df, best_lap=1)
        assert isinstance(fig, go.Figure)

    def test_axes_hidden(self, telemetry_df):
        fig = create_speed_track_map(telemetry_df, best_lap=1)
        assert fig.layout.xaxis.showticklabels is False
        assert fig.layout.yaxis.showticklabels is False

    def test_no_latlon(self, telemetry_df_minimal):
        result = create_speed_track_map(telemetry_df_minimal)
        assert result is None


class TestLateralGMap:
    def test_returns_figure(self, telemetry_df):
        fig = create_lateral_g_track_map(telemetry_df, best_lap=1)
        assert isinstance(fig, go.Figure)

    def test_colorscale(self, telemetry_df):
        fig = create_lateral_g_track_map(telemetry_df, best_lap=1)
        # Plotly expands "RdYlBu" to a list of [fraction, rgb] pairs
        cs = fig.data[0].marker.colorscale
        assert cs is not None
        # Should contain blue tones (from RdYlBu), not red-only (RdBu)
        cs_str = str(cs)
        assert "49,54,149" in cs_str  # deep blue from RdYlBu
