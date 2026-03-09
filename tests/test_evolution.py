"""Tests for scripts/analysis/evolution.py — cross-race evolution charts."""

import pandas as pd
import plotly.graph_objects as go
import pytest

from scripts.analysis.evolution import (
    enrich_races_with_quartiles,
    create_lap_distribution,
    create_session_overlay,
    create_laptime_progression,
    create_improvement_summary,
    create_consistency_trend,
    create_speed_gforce_trends,
)


class TestEnrichQuartiles:
    def test_adds_columns(self, races_df, all_laps_df):
        enriched = enrich_races_with_quartiles(races_df, all_laps_df)
        assert "q1" in enriched.columns
        assert "q3" in enriched.columns
        assert enriched["q1"].notna().all()
        assert enriched["q3"].notna().all()

    def test_empty_laps(self, races_df):
        result = enrich_races_with_quartiles(races_df, pd.DataFrame())
        assert "q1" not in result.columns


class TestLapDistribution:
    def test_returns_figure(self, all_laps_df):
        fig = create_lap_distribution(all_laps_df)
        assert isinstance(fig, go.Figure)

    def test_empty_returns_none(self):
        assert create_lap_distribution(pd.DataFrame()) is None
        assert create_lap_distribution(None) is None

    def test_trace_count(self, all_laps_df):
        fig = create_lap_distribution(all_laps_df)
        n_races = all_laps_df["race_dir"].nunique()
        assert len(fig.data) == n_races


class TestSessionOverlay:
    def test_returns_figure(self, all_laps_df):
        fig = create_session_overlay(all_laps_df)
        assert isinstance(fig, go.Figure)

    def test_empty_returns_none(self):
        assert create_session_overlay(pd.DataFrame()) is None
        assert create_session_overlay(None) is None

    def test_excludes_outliers(self, all_laps_df):
        fig = create_session_overlay(all_laps_df)
        # Scatter traces only (exclude hline shapes)
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        all_y = []
        for t in scatter_traces:
            all_y.extend(list(t.y))
        # The outlier value 72.8 should not appear
        assert 72.8 not in all_y


class TestLaptimeProgression:
    def test_returns_figure(self, races_df):
        fig = create_laptime_progression(races_df)
        assert isinstance(fig, go.Figure)

    def test_has_median(self, races_df):
        fig = create_laptime_progression(races_df)
        trace_names = [t.name for t in fig.data if t.name]
        assert "Median" in trace_names

    def test_has_iqr_band(self, races_df, all_laps_df):
        enriched = enrich_races_with_quartiles(races_df, all_laps_df)
        fig = create_laptime_progression(enriched)
        fills = [t.fill for t in fig.data if t.fill]
        assert "tonexty" in fills

    def test_empty_returns_none(self):
        assert create_laptime_progression(pd.DataFrame()) is None


class TestImprovementSummary:
    def test_returns_figure(self, races_df):
        fig = create_improvement_summary(races_df)
        assert isinstance(fig, go.Figure)

    def test_none_single_race(self, races_df):
        single = races_df.iloc[:1]
        assert create_improvement_summary(single) is None

    def test_colors(self, races_df):
        fig = create_improvement_summary(races_df)
        # All races improve (lower times), so colors should be green
        for trace in fig.data:
            if isinstance(trace, go.Bar):
                for color in trace.marker.color:
                    assert color == "#2ecc71"


class TestConsistencyTrend:
    def test_returns_figure(self, races_df):
        fig = create_consistency_trend(races_df)
        assert isinstance(fig, go.Figure)

    def test_yaxis_narrowed(self, races_df):
        fig = create_consistency_trend(races_df)
        # Y-axis should not span 0-100; range should be close to the data
        y_range = fig.layout.yaxis.range
        assert y_range is not None
        assert y_range[0] > 90  # consistency is ~99%, lower bound should be near


class TestSpeedGforceTrends:
    def test_returns_figure(self, races_df):
        fig = create_speed_gforce_trends(races_df)
        assert isinstance(fig, go.Figure)

    def test_empty_returns_none(self):
        assert create_speed_gforce_trends(pd.DataFrame()) is None
