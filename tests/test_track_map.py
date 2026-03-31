"""Tests for scripts/analysis/track_map.py."""

from scripts.analysis.track_map import create_speed_track_map, create_lateral_g_track_map


class TestSpeedMap:
    def test_returns_dict(self, telemetry_df):
        result = create_speed_track_map(telemetry_df, best_lap=1)
        assert isinstance(result, dict)
        assert "lat" in result
        assert "lon" in result
        assert "values" in result
        assert "colorscale" in result
        assert "colorbar" in result
        assert "title" in result

    def test_data_lengths_match(self, telemetry_df):
        result = create_speed_track_map(telemetry_df, best_lap=1)
        assert len(result["lat"]) == len(result["lon"]) == len(result["values"])

    def test_no_latlon(self, telemetry_df_minimal):
        result = create_speed_track_map(telemetry_df_minimal)
        assert result is None


class TestLateralGMap:
    def test_returns_dict(self, telemetry_df):
        result = create_lateral_g_track_map(telemetry_df, best_lap=1)
        assert isinstance(result, dict)
        assert "lat" in result
        assert "lon" in result
        assert "values" in result

    def test_colorscale(self, telemetry_df):
        result = create_lateral_g_track_map(telemetry_df, best_lap=1)
        assert result["colorscale"] == "RdYlBu"
