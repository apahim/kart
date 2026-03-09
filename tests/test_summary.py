"""Tests for scripts/analysis/summary.py."""

import os
import yaml

from scripts.analysis.summary import generate_summary, write_summary


class TestGenerateSummary:
    def test_summary_keys(self, laptimes_df):
        s = generate_summary(laptimes_df)
        expected = {
            "generated", "total_laps", "clean_laps", "excluded_laps",
            "best_lap", "worst_clean_lap", "average", "median", "std_dev",
            "consistency_pct", "best_lap_time_fmt", "worst_clean_time_fmt",
            "average_fmt", "median_fmt",
        }
        assert expected.issubset(set(s.keys()))

    def test_summary_formatted_times(self, laptimes_df):
        s = generate_summary(laptimes_df)
        assert ":" in s["best_lap_time_fmt"]
        assert ":" in s["average_fmt"]
        assert ":" in s["median_fmt"]

    def test_summary_with_telemetry(self, laptimes_df, telemetry_df):
        s = generate_summary(laptimes_df, telemetry_df)
        assert "top_speed_kmh" in s
        assert "max_lateral_g" in s
        assert "max_braking_g" in s
        assert "max_acceleration_g" in s

    def test_summary_without_telemetry(self, laptimes_df):
        s = generate_summary(laptimes_df)
        assert "top_speed_kmh" not in s
        assert "max_lateral_g" not in s

    def test_best_lap_correct(self, laptimes_df):
        s = generate_summary(laptimes_df)
        assert s["best_lap"]["time"] == min(laptimes_df["seconds"])


class TestWriteSummary:
    def test_write_summary(self, laptimes_df, tmp_path):
        s = generate_summary(laptimes_df)
        out = tmp_path / "summary.yaml"
        write_summary(s, str(out))
        assert out.exists()
        loaded = yaml.safe_load(out.read_text())
        assert "best_lap" in loaded
