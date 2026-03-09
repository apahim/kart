"""Helpers for loading and parsing kart racing data."""

import os
import pandas as pd
import yaml


def parse_laptime(time_val):
    """Parse a lap time value (MM:SS.mmm string or float) into total seconds."""
    if isinstance(time_val, (int, float)):
        return float(time_val)
    time_str = str(time_val).strip()
    if ":" in time_str:
        parts = time_str.split(":")
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    return float(time_str)


def load_laptimes(csv_path):
    """Load official lap times from a CSV file.

    Expects at minimum columns: lap, time
    Returns a DataFrame with an added 'seconds' column.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    if "time" in df.columns:
        df["seconds"] = df["time"].apply(parse_laptime)
    return df


def _dedup_columns(header_row_values, source_row_values):
    """Deduplicate column names using the source row (sensor info).

    RaceChrono v3 exports can have duplicate column names like 'speed' (gps)
    and 'speed' (calc). This function appends sensor suffixes to make them unique.
    """
    sensor_map = {
        "100: gps": "gps",
        "calc": "calc",
        "101: acc": "acc",
        "102: gyro": "gyro",
    }

    seen = {}
    new_names = []
    for i, col in enumerate(header_row_values):
        col = col.strip()
        source = source_row_values[i].strip() if i < len(source_row_values) else ""
        suffix = sensor_map.get(source, "")

        if col in seen:
            if suffix:
                new_col = f"{col}_{suffix}"
            else:
                seen[col] += 1
                new_col = f"{col}_{seen[col]}"
        else:
            seen[col] = 0
            # Check if this column name will appear again
            future_dupes = [h.strip() for h in header_row_values[i + 1:]]
            if col in future_dupes and suffix:
                new_col = f"{col}_{suffix}"
            else:
                new_col = col
        new_names.append(new_col)

    return new_names


def load_racechrono_session(csv_path):
    """Load a single RaceChrono Pro CSV export (v3 format).

    Skips the metadata header and units/source rows, keeping only the
    column names row and the actual data. Deduplicates column names
    using sensor source information.
    """
    header_row = 0
    with open(csv_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.startswith("timestamp,"):
            header_row = i
            break

    header_values = lines[header_row].strip().split(",")
    source_row_idx = header_row + 2
    if source_row_idx < len(lines):
        source_values = lines[source_row_idx].strip().split(",")
    else:
        source_values = []

    new_columns = _dedup_columns(header_values, source_values)

    df = pd.read_csv(
        csv_path,
        header=header_row,
        skiprows=[header_row + 1, header_row + 2],
    )
    df.columns = new_columns

    return df


def load_telemetry(race_dir):
    """Load the telemetry CSV from a race directory.

    Looks for telemetry.csv in the race directory.
    Returns a DataFrame, or None if no telemetry file exists.
    """
    path = os.path.join(race_dir, "telemetry.csv")
    if not os.path.exists(path):
        return None
    return load_racechrono_session(path)


def extract_laptimes_from_telemetry(telemetry_df):
    """Derive lap times from telemetry elapsed_time and lap_number columns.

    Returns a DataFrame with columns: lap, time, seconds.
    """
    if telemetry_df is None or telemetry_df.empty:
        return pd.DataFrame(columns=["lap", "time", "seconds"])

    df = telemetry_df.dropna(subset=["lap_number"])
    if df.empty or "elapsed_time" not in df.columns:
        return pd.DataFrame(columns=["lap", "time", "seconds"])

    grouped = df.groupby("lap_number")["elapsed_time"].agg(["min", "max"])
    grouped["duration"] = grouped["max"] - grouped["min"]

    rows = []
    for lap_num, row in grouped.iterrows():
        rows.append({
            "lap": int(lap_num),
            "time": round(row["duration"], 3),
            "seconds": round(row["duration"], 3),
        })

    return pd.DataFrame(rows)


def load_race_metadata(race_dir):
    """Load race metadata from race.yaml in the race directory.

    Returns a dict with metadata, or an empty dict if no file exists.
    """
    yaml_path = os.path.join(race_dir, "race.yaml")
    if not os.path.exists(yaml_path):
        return {}
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f) or {}
