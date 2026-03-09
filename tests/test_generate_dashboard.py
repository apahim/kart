"""Tests for scripts/generate_dashboard.py."""

import numpy as np
import plotly.graph_objects as go

from scripts.analysis.utils import fig_to_html, safe_chart


class TestFigToHtml:
    def test_none(self):
        assert fig_to_html(None) is None

    def test_returns_html_string(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.array([1.0, 2.0, 3.0]),
            y=np.array([4.0, 5.0, 6.0]),
        ))
        html = fig_to_html(fig)
        assert html is not None
        assert isinstance(html, str)
        assert "plotly" in html.lower() or "scatter" in html.lower()


class TestSafeChart:
    def test_catches_exception(self):
        def bad_func():
            raise ValueError("boom")
        result = safe_chart("test", bad_func)
        assert result is None

    def test_passes_through(self):
        def good_func():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[1], y=[1]))
            return fig
        result = safe_chart("test", good_func)
        assert result is not None
        assert isinstance(result, str)
