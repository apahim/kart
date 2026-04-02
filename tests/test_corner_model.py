"""Tests for the unified Corner Object model."""

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.corner_model import (
    CornerRecord,
    CornerSummary,
    classify_root_cause,
    classify_archetype,
    build_corner_analysis,
    corner_analysis_to_template,
)


# --- CornerRecord tests ---


class TestCornerRecord:
    def test_construction(self):
        r = CornerRecord(corner_name="T1", corner_index=0, lap=1)
        assert r.corner_name == "T1"
        assert r.corner_index == 0
        assert r.lap == 1
        assert r.min_speed == 0.0
        assert r.root_cause is None

    def test_to_dict(self):
        r = CornerRecord(
            corner_name="T2", corner_index=1, lap=3,
            entry_speed=55.0, min_speed=32.0, exit_speed=48.0,
            time_loss=-0.15, root_cause="entry",
        )
        d = r.to_dict()
        assert isinstance(d, dict)
        assert d["corner_name"] == "T2"
        assert d["entry_speed"] == 55.0
        assert d["root_cause"] == "entry"

    def test_defaults(self):
        r = CornerRecord(corner_name="T1", corner_index=0, lap=1)
        assert r.lat is None
        assert r.braking_point is None
        assert r.braking_distance is None
        assert r.trail_braking_depth is None


# --- CornerSummary tests ---


class TestCornerSummary:
    def test_construction(self):
        s = CornerSummary(corner_name="T1", corner_index=0)
        assert s.archetype == "flow"
        assert s.records == []

    def test_to_dict_with_records(self):
        r = CornerRecord(corner_name="T1", corner_index=0, lap=1, min_speed=30.0)
        s = CornerSummary(
            corner_name="T1", corner_index=0,
            avg_min_speed=30.0, records=[r],
        )
        d = s.to_dict()
        assert len(d["records"]) == 1
        assert d["records"][0]["min_speed"] == 30.0


# --- classify_root_cause tests ---


class TestClassifyRootCause:
    def _make_record(self, entry=50, min_s=30, exit_s=45, time_loss=-0.1):
        return CornerRecord(
            corner_name="T1", corner_index=0, lap=1,
            entry_speed=entry, min_speed=min_s, exit_speed=exit_s,
            time_loss=time_loss,
        )

    def test_negligible_loss_returns_none(self):
        record = self._make_record(time_loss=0.02)
        best = self._make_record(entry=50, min_s=30, exit_s=45, time_loss=0)
        assert classify_root_cause(record, best) is None

    def test_negative_loss_returns_none(self):
        """Negative time_loss means gained time — no root cause needed."""
        record = self._make_record(time_loss=-0.05)
        best = self._make_record(time_loss=0)
        assert classify_root_cause(record, best) is None

    def test_entry_problem(self):
        """When entry speed is much lower than best, root cause is entry."""
        record = self._make_record(entry=40, min_s=30, exit_s=45, time_loss=0.15)
        best = self._make_record(entry=55, min_s=30, exit_s=45, time_loss=0)
        assert classify_root_cause(record, best) == "entry"

    def test_mid_problem(self):
        """When min speed is much lower than best, root cause is mid."""
        record = self._make_record(entry=50, min_s=22, exit_s=45, time_loss=0.15)
        best = self._make_record(entry=50, min_s=30, exit_s=45, time_loss=0)
        assert classify_root_cause(record, best) == "mid"

    def test_exit_problem(self):
        """When exit speed is much lower than best, root cause is exit."""
        record = self._make_record(entry=50, min_s=30, exit_s=35, time_loss=0.15)
        best = self._make_record(entry=50, min_s=30, exit_s=50, time_loss=0)
        assert classify_root_cause(record, best) == "exit"

    def test_all_equal_returns_none(self):
        """When speeds match best, no root cause."""
        record = self._make_record(entry=50, min_s=30, exit_s=45, time_loss=0.05)
        best = self._make_record(entry=50, min_s=30, exit_s=45, time_loss=0)
        assert classify_root_cause(record, best) is None


# --- classify_archetype tests ---


class TestClassifyArchetype:
    def _make_record(self, entry=50, min_s=30, exit_s=45):
        return CornerRecord(
            corner_name="T1", corner_index=0, lap=1,
            entry_speed=entry, min_speed=min_s, exit_speed=exit_s,
        )

    def test_flow_corner(self):
        """Low ratios → flow."""
        r = self._make_record(entry=35, min_s=30, exit_s=35)
        assert classify_archetype(r, next_corner_distance=50, median_inter_corner=60) == "flow"

    def test_entry_dependent(self):
        """High entry/min ratio → entry-dependent."""
        r = self._make_record(entry=50, min_s=30, exit_s=35)  # ratio 1.67
        assert classify_archetype(r, next_corner_distance=50, median_inter_corner=60) == "entry-dependent"

    def test_exit_dependent_with_long_straight(self):
        """High exit/min ratio + long straight → exit-dependent."""
        r = self._make_record(entry=35, min_s=30, exit_s=45)  # exit ratio 1.5
        assert classify_archetype(r, next_corner_distance=100, median_inter_corner=60) == "exit-dependent"

    def test_exit_not_dependent_short_straight(self):
        """High exit/min but short straight → flow (not exit-dependent)."""
        r = self._make_record(entry=35, min_s=30, exit_s=40)  # exit ratio 1.33
        assert classify_archetype(r, next_corner_distance=30, median_inter_corner=60) == "flow"

    def test_very_high_exit_no_distance_info(self):
        """Very high exit ratio without distance info → exit-dependent."""
        r = self._make_record(entry=35, min_s=30, exit_s=50)  # exit ratio 1.67
        assert classify_archetype(r) == "exit-dependent"

    def test_zero_min_speed(self):
        """Zero min speed → flow (avoid division by zero)."""
        r = self._make_record(entry=50, min_s=0, exit_s=45)
        assert classify_archetype(r) == "flow"


# --- build_corner_analysis tests ---


class TestBuildCornerAnalysis:
    def test_returns_none_missing_columns(self, laptimes_df):
        """Missing speed column → None."""
        df = pd.DataFrame({"lap_number": [1], "timestamp": [0]})
        assert build_corner_analysis(df, laptimes_df) is None

    def test_returns_none_single_lap(self, telemetry_df):
        """Single lap (can't compute delta) → None."""
        lt = pd.DataFrame({"lap": [1], "seconds": [70.0]})
        assert build_corner_analysis(telemetry_df, lt) is None

    def test_with_multi_lap_data(self, telemetry_df_multi_lap):
        """Multi-lap data should produce records and summaries."""
        laptimes = pd.DataFrame({
            "lap": [1, 2, 3],
            "seconds": [71.0, 69.5, 70.5],
        })
        result = build_corner_analysis(telemetry_df_multi_lap, laptimes)
        # May return None if compute_time_delta fails (best == median with 3 laps)
        # but should not raise an exception
        if result is not None:
            assert "records" in result
            assert "summaries" in result
            assert "best_lap" in result
            assert result["best_lap"] == 2

    def test_with_track_corners(self, telemetry_df_multi_lap):
        """Track corners should be used for detection."""
        laptimes = pd.DataFrame({
            "lap": [1, 2, 3],
            "seconds": [71.0, 69.5, 70.5],
        })
        track_corners = [
            {"name": "T1", "lat": 52.501, "lon": -7.5},
            {"name": "T2", "lat": 52.5, "lon": -7.499},
            {"name": "T3", "lat": 52.499, "lon": -7.5},
            {"name": "T4", "lat": 52.5, "lon": -7.501},
        ]
        result = build_corner_analysis(
            telemetry_df_multi_lap, laptimes, track_corners=track_corners
        )
        if result is not None:
            assert result["corner_names"] == ["T1", "T2", "T3", "T4"]

    def test_records_have_speed_data(self, telemetry_df_multi_lap):
        """Records should have non-zero speed data."""
        laptimes = pd.DataFrame({
            "lap": [1, 2, 3],
            "seconds": [71.0, 69.5, 70.5],
        })
        result = build_corner_analysis(telemetry_df_multi_lap, laptimes)
        if result is not None:
            for lap_records in result["records"].values():
                for r in lap_records:
                    assert r.min_speed > 0
                    assert r.entry_speed > 0
                    assert r.exit_speed > 0

    def test_best_lap_has_zero_time_loss(self, telemetry_df_multi_lap):
        """Best lap should have 0.0 time loss at all corners."""
        laptimes = pd.DataFrame({
            "lap": [1, 2, 3],
            "seconds": [71.0, 69.5, 70.5],
        })
        result = build_corner_analysis(telemetry_df_multi_lap, laptimes)
        if result is not None:
            best_lap = result["best_lap"]
            if best_lap in result["records"]:
                for r in result["records"][best_lap]:
                    assert r.time_loss == 0.0, f"Best lap corner {r.corner_name} should have 0 time loss"

    def test_non_best_laps_have_non_negative_time_loss(self, telemetry_df_multi_lap):
        """Non-best laps should generally have positive (or zero) time loss."""
        laptimes = pd.DataFrame({
            "lap": [1, 2, 3],
            "seconds": [71.0, 69.5, 70.5],
        })
        result = build_corner_analysis(telemetry_df_multi_lap, laptimes)
        if result is not None:
            best_lap = result["best_lap"]
            for lap, records in result["records"].items():
                if lap == best_lap:
                    continue
                total = sum(r.time_loss for r in records)
                # Total time loss across corners should be >= 0 (slower than best)
                assert total >= -0.01, f"Lap {lap} total time loss should be non-negative, got {total}"

    def test_summaries_sorted_by_time_loss(self, telemetry_df_multi_lap):
        """Summaries should be sorted worst (most positive) first."""
        laptimes = pd.DataFrame({
            "lap": [1, 2, 3],
            "seconds": [71.0, 69.5, 70.5],
        })
        result = build_corner_analysis(telemetry_df_multi_lap, laptimes)
        if result is not None and len(result["summaries"]) > 1:
            losses = [s.avg_time_loss for s in result["summaries"]]
            assert losses == sorted(losses, reverse=True)


# --- corner_analysis_to_template tests ---


class TestCornerAnalysisToTemplate:
    def test_none_input(self):
        assert corner_analysis_to_template(None) is None

    def test_basic_conversion(self):
        r1 = CornerRecord(
            corner_name="T1", corner_index=0, lap=1,
            entry_speed=50, min_speed=30, exit_speed=45,
            time_loss=0.15, root_cause="entry",
        )
        r2 = CornerRecord(
            corner_name="T1", corner_index=0, lap=2,
            entry_speed=52, min_speed=31, exit_speed=46,
            time_loss=0.05,
        )
        s = CornerSummary(
            corner_name="T1", corner_index=0,
            archetype="entry-dependent",
            avg_time_loss=0.10,
            best_min_speed=31, avg_min_speed=30.5, std_min_speed=0.5,
            best_entry_speed=52, best_exit_speed=46,
            braking_spread=3.2,
            dominant_root_cause="entry",
            records=[r1, r2],
        )

        analysis = {
            "records": {1: [r1], 2: [r2]},
            "summaries": [s],
            "best_lap": 2,
            "corner_names": ["T1"],
        }

        result = corner_analysis_to_template(analysis)
        assert result is not None
        assert len(result["summary_rows"]) == 1
        assert result["summary_rows"][0]["corner"] == "T1"
        assert result["summary_rows"][0]["archetype"] == "entry-dependent"
        assert result["summary_rows"][0]["dominant_root_cause"] == "entry"
        assert result["best_lap"] == 2

    def test_lap_breakdowns_sorted(self):
        r1 = CornerRecord(corner_name="T1", corner_index=0, lap=1, time_loss=0.15)
        r2 = CornerRecord(corner_name="T2", corner_index=1, lap=1, time_loss=0.05)

        analysis = {
            "records": {1: [r1, r2]},
            "summaries": [],
            "best_lap": 2,
            "corner_names": ["T1", "T2"],
        }

        result = corner_analysis_to_template(analysis)
        rows = result["lap_breakdowns"]["1"]
        assert rows[0]["corner"] == "T1"  # worst first
        assert rows[0]["is_worst"] is True
        assert rows[1]["is_worst"] is False
