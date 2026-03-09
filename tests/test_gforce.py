"""Tests for scripts/analysis/gforce.py."""

import plotly.graph_objects as go

from scripts.analysis.gforce import create_gg_diagram


class TestGGDiagram:
    def test_returns_figure(self, telemetry_df):
        fig = create_gg_diagram(telemetry_df, best_lap=1)
        assert isinstance(fig, go.Figure)

    def test_has_quadrant_traces(self, telemetry_df):
        fig = create_gg_diagram(telemetry_df, best_lap=1)
        # 4 quadrant scatter traces (excluding reference circle lines)
        scatter_traces = [t for t in fig.data if t.name and "+" in t.name]
        assert len(scatter_traces) == 4

    def test_has_annotations(self, telemetry_df):
        fig = create_gg_diagram(telemetry_df, best_lap=1)
        texts = [a.text for a in fig.layout.annotations]
        assert any("Braking" in t for t in texts)
        assert any("Acceleration" in t for t in texts)
        assert any("Left" in t for t in texts)
        assert any("Right" in t for t in texts)

    def test_none_missing_columns(self, telemetry_df_minimal):
        result = create_gg_diagram(telemetry_df_minimal)
        assert result is None
