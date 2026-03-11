"""Cross-race evolution analysis — tracking driver improvement across sessions."""

import os
import glob

import numpy as np
import pandas as pd
import yaml

from scripts.load_data import extract_laptimes_from_telemetry, load_telemetry
from scripts.analysis.outliers import filter_non_race_laps, detect_outliers
from scripts.analysis.utils import project_to_meters, format_laptime


def load_all_races(data_dir="data/races"):
    """Scan all race directories and build a consolidated DataFrame.

    Each row represents one race, with summary metrics + metadata.
    """
    rows = []
    for race_dir in sorted(glob.glob(os.path.join(data_dir, "*"))):
        if not os.path.isdir(race_dir):
            continue

        summary_path = os.path.join(race_dir, "summary_generated.yaml")
        meta_path = os.path.join(race_dir, "race.yaml")

        if not os.path.exists(summary_path):
            continue

        with open(summary_path, "r") as f:
            summary = yaml.safe_load(f) or {}

        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = yaml.safe_load(f) or {}

        row = {
            "race_dir": os.path.basename(race_dir),
            "date": meta.get("date", os.path.basename(race_dir)[:10]),
            "track": meta.get("track", "Unknown"),
            "kart_number": meta.get("kart_number"),
            "driver_weight_kg": meta.get("driver_weight_kg"),
            "session_type": meta.get("session_type"),
            "weather_condition": summary.get("weather", {}).get("condition") if isinstance(summary.get("weather"), dict) else None,
            "weather_temp_c": summary.get("weather", {}).get("temp_c") if isinstance(summary.get("weather"), dict) else None,
            "weather_wind_kmh": summary.get("weather", {}).get("wind_kmh") if isinstance(summary.get("weather"), dict) else None,
            "weather_wind_direction": summary.get("weather", {}).get("wind_direction") if isinstance(summary.get("weather"), dict) else None,
            "best_lap_time": summary.get("best_lap", {}).get("time"),
            "average": summary.get("average"),
            "median": summary.get("median"),
            "std_dev": summary.get("std_dev"),
            "consistency_pct": summary.get("consistency_pct"),
            "total_laps": summary.get("total_laps"),
            "clean_laps": summary.get("clean_laps"),
            "top_speed_kmh": summary.get("top_speed_kmh"),
            "max_lateral_g": summary.get("max_lateral_g"),
            "max_braking_g": summary.get("max_braking_g"),
            "max_acceleration_g": summary.get("max_acceleration_g"),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_all_laptimes(data_dir="data/races"):
    """Load individual lap times from all race directories.

    Returns a combined DataFrame with columns:
        race_dir, date, track, lap, seconds, is_outlier
    """
    frames = []
    for race_dir in sorted(glob.glob(os.path.join(data_dir, "*"))):
        if not os.path.isdir(race_dir):
            continue

        meta_path = os.path.join(race_dir, "race.yaml")

        if not os.path.exists(meta_path):
            continue

        with open(meta_path, "r") as f:
            meta = yaml.safe_load(f) or {}

        telemetry_df = load_telemetry(race_dir)
        if telemetry_df is None:
            continue

        laps_df = extract_laptimes_from_telemetry(telemetry_df)
        laps_df = filter_non_race_laps(laps_df)
        if laps_df.empty or "seconds" not in laps_df.columns:
            continue

        clean_df, _ = detect_outliers(laps_df)
        clean_indices = set(clean_df.index)

        race_name = os.path.basename(race_dir)
        date_str = meta.get("date", race_name[:10])

        for _, row in laps_df.iterrows():
            frames.append({
                "race_dir": race_name,
                "date": date_str,
                "track": meta.get("track", "Unknown"),
                "session_type": meta.get("session_type"),
                "lap": int(row["lap"]),
                "seconds": float(row["seconds"]),
                "is_outlier": row.name not in clean_indices,
            })

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def prepare_raceline_data(track_name, current_race_dir, data_dir="data/races"):
    """Prepare racing line data for cross-session comparison in the dashboard.

    Returns a JSON-serializable dict with GPS coordinates for each valid lap
    across all sessions at the same track.
    """
    current_race_name = os.path.basename(current_race_dir)

    # Find all race dirs for this track
    matching_dirs = []
    for race_dir in sorted(glob.glob(os.path.join(data_dir, "*"))):
        if not os.path.isdir(race_dir):
            continue
        meta_path = os.path.join(race_dir, "race.yaml")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path, "r") as f:
            meta = yaml.safe_load(f) or {}
        if meta.get("track", "") == track_name:
            matching_dirs.append((race_dir, meta))

    if not matching_dirs:
        return None

    # Compute shared projection center from the current race's best lap
    center = None
    for race_dir, meta in matching_dirs:
        if os.path.basename(race_dir) == current_race_name:
            tel = load_telemetry(race_dir)
            if tel is not None:
                lat_col, lon_col = _find_gps_cols(tel)
                if lat_col and lon_col:
                    laps = extract_laptimes_from_telemetry(tel)
                    laps = filter_non_race_laps(laps)
                    if not laps.empty and "seconds" in laps.columns:
                        clean, _ = detect_outliers(laps)
                        if not clean.empty:
                            best_lap = int(clean.loc[clean["seconds"].idxmin(), "lap"])
                            lap_data = tel[tel["lap_number"] == best_lap].dropna(subset=[lat_col, lon_col])
                            if len(lap_data) >= 20:
                                center = (
                                    float(np.mean(lap_data[lat_col].values)),
                                    float(np.mean(lap_data[lon_col].values)),
                                )
            break

    sessions = []
    for race_dir, meta in matching_dirs:
        race_name = os.path.basename(race_dir)
        tel = load_telemetry(race_dir)
        if tel is None:
            continue

        lat_col, lon_col = _find_gps_cols(tel)
        if not lat_col or not lon_col:
            continue

        laps_df = extract_laptimes_from_telemetry(tel)
        laps_df = filter_non_race_laps(laps_df)
        if laps_df.empty or "seconds" not in laps_df.columns:
            continue

        clean_df, _ = detect_outliers(laps_df)
        clean_indices = set(clean_df.index)
        best_idx = clean_df["seconds"].idxmin() if not clean_df.empty else None
        best_lap_num = int(clean_df.loc[best_idx, "lap"]) if best_idx is not None else None

        date_str = str(meta.get("date", race_name[:10]))
        session_type = meta.get("session_type", "")

        session_laps = []
        for _, row in laps_df.iterrows():
            lap_num = int(row["lap"])
            lap_data = tel[tel["lap_number"] == lap_num].copy()
            lap_data = lap_data.dropna(subset=[lat_col, lon_col])
            if len(lap_data) < 20:
                continue

            lat_vals = lap_data[lat_col].values
            lon_vals = lap_data[lon_col].values
            x_m, y_m = project_to_meters(lat_vals, lon_vals, center=center)

            # Downsample to ~200 points
            if len(x_m) > 200:
                indices = np.linspace(0, len(x_m) - 1, 200, dtype=int)
                x_m = x_m[indices]
                y_m = y_m[indices]

            is_outlier = row.name not in clean_indices
            seconds = float(row["seconds"])

            session_laps.append({
                "lap": lap_num,
                "time_fmt": format_laptime(seconds),
                "seconds": round(seconds, 3),
                "is_best": lap_num == best_lap_num,
                "is_outlier": is_outlier,
                "x": [round(float(v), 2) for v in x_m],
                "y": [round(float(v), 2) for v in y_m],
            })

        if session_laps:
            sessions.append({
                "date": date_str,
                "session_type": session_type,
                "is_current": race_name == current_race_name,
                "laps": session_laps,
            })

    if not sessions:
        return None

    return {"sessions": sessions}


def _find_gps_cols(df):
    """Find latitude and longitude column names in a DataFrame."""
    lat_col = lon_col = None
    for col in df.columns:
        cl = col.lower()
        if "latitude" in cl:
            lat_col = col
        if "longitude" in cl:
            lon_col = col
    return lat_col, lon_col
