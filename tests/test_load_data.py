"""Tests for scripts/load_data.py."""

import numpy as np
import pandas as pd
import yaml

from scripts.load_data import parse_laptime, load_laptimes, load_race_metadata, extract_laptimes_from_telemetry


class TestParseLaptime:
    def test_seconds(self):
        assert parse_laptime("69.742") == 69.742

    def test_mmss(self):
        assert parse_laptime("1:09.742") == 69.742

    def test_float_passthrough(self):
        assert parse_laptime(70.5) == 70.5


class TestLoadLaptimes:
    def test_load(self, tmp_path):
        csv = tmp_path / "laptimes.csv"
        csv.write_text("lap,time\n1,1:09.742\n2,70.5\n3,71.2\n")
        df = load_laptimes(str(csv))
        assert "seconds" in df.columns
        assert len(df) == 3
        assert abs(df["seconds"].iloc[0] - 69.742) < 0.001
        assert df["seconds"].dtype == float


class TestLoadRaceMetadata:
    def test_load(self, tmp_path):
        meta = tmp_path / "race.yaml"
        meta.write_text(yaml.dump({"track": "Kiltorcan", "date": "2026-03-08"}))
        result = load_race_metadata(str(tmp_path))
        assert result["track"] == "Kiltorcan"
        assert "date" in result

    def test_missing_file(self, tmp_path):
        result = load_race_metadata(str(tmp_path))
        assert result == {}


class TestExtractLaptimesFromTelemetry:
    def test_basic(self):
        """Derives lap times from elapsed_time per lap_number."""
        df = pd.DataFrame({
            "lap_number": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "elapsed_time": [0.0, 30.0, 70.0, 70.0, 100.0, 140.5, 140.5, 170.0, 211.0],
        })
        result = extract_laptimes_from_telemetry(df)
        assert len(result) == 3
        assert list(result.columns) == ["lap", "time", "seconds"]
        assert result.iloc[0]["lap"] == 1
        assert result.iloc[0]["seconds"] == 70.0
        assert result.iloc[1]["seconds"] == 70.5
        assert result.iloc[2]["seconds"] == 70.5

    def test_drops_nan_laps(self):
        df = pd.DataFrame({
            "lap_number": [np.nan, np.nan, 1, 1, 1],
            "elapsed_time": [0.0, 5.0, 10.0, 40.0, 80.0],
        })
        result = extract_laptimes_from_telemetry(df)
        assert len(result) == 1
        assert result.iloc[0]["lap"] == 1

    def test_empty_telemetry(self):
        result = extract_laptimes_from_telemetry(pd.DataFrame())
        assert result.empty
        assert "seconds" in result.columns

    def test_none_telemetry(self):
        result = extract_laptimes_from_telemetry(None)
        assert result.empty

    def test_multi_lap_fixture(self, telemetry_df_multi_lap):
        result = extract_laptimes_from_telemetry(telemetry_df_multi_lap)
        assert len(result) == 3
        # Each lap should be ~60s based on the fixture
        for _, row in result.iterrows():
            assert 55.0 < row["seconds"] < 65.0
