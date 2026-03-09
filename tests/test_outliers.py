"""Tests for scripts/analysis/outliers.py."""

import pandas as pd

from scripts.analysis.outliers import detect_outliers


class TestDetectOutliers:
    def test_no_outliers(self):
        df = pd.DataFrame({"lap": range(1, 11), "seconds": [70.0 + i * 0.1 for i in range(10)]})
        clean, excluded = detect_outliers(df)
        assert len(clean) == 10
        assert excluded == []

    def test_removes_slow_outlier(self):
        times = [70.0] * 9 + [200.0]
        df = pd.DataFrame({"lap": range(1, 11), "seconds": times})
        clean, excluded = detect_outliers(df)
        assert len(clean) < 10
        assert any(e["lap"] == 10 for e in excluded)

    def test_removes_fast_outlier(self):
        times = [70.0] * 9 + [10.0]
        df = pd.DataFrame({"lap": range(1, 11), "seconds": times})
        clean, excluded = detect_outliers(df)
        assert len(clean) < 10
        assert any(e["lap"] == 10 for e in excluded)

    def test_excluded_dict_format(self):
        times = [70.0] * 9 + [200.0]
        df = pd.DataFrame({"lap": range(1, 11), "seconds": times})
        _, excluded = detect_outliers(df)
        assert len(excluded) > 0
        for e in excluded:
            assert "lap" in e
            assert "time" in e
            assert "reason" in e

    def test_single_lap(self, laptimes_df_single):
        clean, excluded = detect_outliers(laptimes_df_single)
        assert len(clean) == 1
        assert excluded == []

    def test_empty_dataframe(self):
        df = pd.DataFrame({"lap": [], "seconds": []})
        clean, excluded = detect_outliers(df)
        assert len(clean) == 0
        assert excluded == []

    def test_custom_iqr_multiplier(self):
        times = [70.0, 70.1, 70.2, 70.3, 70.4, 70.5, 71.0, 71.5, 72.0, 75.0]
        df = pd.DataFrame({"lap": range(1, 11), "seconds": times})
        _, excluded_tight = detect_outliers(df, iqr_multiplier=1.0)
        _, excluded_loose = detect_outliers(df, iqr_multiplier=3.0)
        assert len(excluded_tight) >= len(excluded_loose)
