"""Tests for scripts/analysis/laptimes.py."""

import plotly.graph_objects as go

from scripts.analysis.laptimes import (
    create_laptime_bar_chart,
    create_delta_to_best_chart,
    create_rolling_average_chart,
    create_laptime_histogram,
)


class TestLaptimeBarChart:
    def test_returns_figure(self, laptimes_df):
        fig = create_laptime_bar_chart(laptimes_df)
        assert isinstance(fig, go.Figure)

    def test_formatted_text(self, laptimes_df):
        fig = create_laptime_bar_chart(laptimes_df)
        bar = fig.data[0]
        # All times are >60s so formatted as M:SS.mmm
        assert any(":" in str(t) for t in bar.text)

    def test_y_range_narrow(self, laptimes_df):
        fig = create_laptime_bar_chart(laptimes_df)
        y_range = fig.layout.yaxis.range
        assert y_range is not None
        assert y_range[0] > 0


class TestDeltaChart:
    def test_returns_figure(self, laptimes_df):
        fig = create_delta_to_best_chart(laptimes_df)
        assert isinstance(fig, go.Figure)


class TestRollingAverage:
    def test_returns_figure(self, laptimes_df):
        fig = create_rolling_average_chart(laptimes_df)
        assert isinstance(fig, go.Figure)

    def test_segments(self, laptimes_df):
        fig = create_rolling_average_chart(laptimes_df)
        # 10 laps >= 6, so there should be vrect shapes
        shapes = fig.layout.shapes
        assert shapes is not None
        assert len(shapes) >= 3


class TestHistogram:
    def test_returns_figure(self, laptimes_df):
        fig = create_laptime_histogram(laptimes_df)
        assert isinstance(fig, go.Figure)

    def test_has_vlines(self, laptimes_df):
        fig = create_laptime_histogram(laptimes_df)
        shapes = fig.layout.shapes
        assert shapes is not None
        assert len(shapes) >= 2  # best + mean vertical lines
