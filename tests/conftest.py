"""Shared fixtures for kart racing analysis tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def laptimes_df():
    """10 laps with realistic kart lap times (69-73s range)."""
    times = [71.2, 70.5, 69.8, 70.1, 72.3, 69.742, 70.9, 71.5, 70.3, 71.0]
    return pd.DataFrame({"lap": list(range(1, 11)), "seconds": times})


@pytest.fixture
def laptimes_df_single():
    """Single lap — edge case."""
    return pd.DataFrame({"lap": [1], "seconds": [70.0]})


@pytest.fixture
def telemetry_df():
    """~200 rows simulating 1 lap around a circular track."""
    n = 200
    t = np.linspace(0, 60, n)
    angle = np.linspace(0, 2 * np.pi, n)

    # Circular track centered at (52.5, -7.5)
    lat = 52.5 + 0.001 * np.sin(angle)
    lon = -7.5 + 0.001 * np.cos(angle)

    # Speed varies sinusoidally (simulating corners)
    speed = 10 + 5 * np.sin(4 * angle)  # m/s, 4 corners

    # Accelerations
    lateral_acc = 0.8 * np.sin(4 * angle)
    longitudinal_acc = 0.5 * np.cos(4 * angle)

    # Distance traveled
    dt = np.diff(t, prepend=t[0])
    distance = np.cumsum(speed * dt)

    return pd.DataFrame({
        "timestamp": t,
        "elapsed_time": t,
        "latitude": lat,
        "longitude": lon,
        "speed": speed,
        "speed_gps": speed,
        "lateral_acc": lateral_acc,
        "longitudinal_acc": longitudinal_acc,
        "distance_traveled": distance,
        "lap_number": np.ones(n, dtype=int),
    })


@pytest.fixture
def telemetry_df_multi_lap(telemetry_df):
    """3 laps of telemetry data."""
    frames = []
    lap_duration = telemetry_df["elapsed_time"].max()
    for lap in range(1, 4):
        lap_df = telemetry_df.copy()
        lap_df["lap_number"] = lap
        lap_df["elapsed_time"] = lap_df["elapsed_time"] + (lap - 1) * lap_duration
        lap_df["distance_traveled"] = lap_df["distance_traveled"] + (lap - 1) * lap_df["distance_traveled"].iloc[-1]
        frames.append(lap_df)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def races_df():
    """3-row DataFrame simulating 3 races at the same track."""
    return pd.DataFrame({
        "race_dir": ["2026-02-27-Kiltorcan", "2026-03-08-Kiltorcan", "2026-03-15-Kiltorcan"],
        "date": pd.to_datetime(["2026-02-27", "2026-03-08", "2026-03-15"]),
        "track": ["Kiltorcan", "Kiltorcan", "Kiltorcan"],
        "kart_number": [11, 11, 11],
        "best_lap_time": [70.041, 69.742, 69.500],
        "average": [70.87, 70.36, 70.10],
        "median": [70.75, 70.25, 70.00],
        "std_dev": [0.52, 0.38, 0.35],
        "consistency_pct": [99.3, 99.5, 99.5],
        "total_laps": [24, 24, 24],
        "clean_laps": [23, 24, 24],
        "top_speed_kmh": [85.2, 86.1, 86.5],
        "max_lateral_g": [1.15, 1.18, 1.20],
        "max_braking_g": [0.95, 0.98, 1.00],
        "max_acceleration_g": [0.65, 0.68, 0.70],
        "driver_weight_kg": [75, 75, 75],
        "weather_condition": ["overcast", "sunny", "overcast"],
        "weather_temp_c": [8, 12, 10],
    })


@pytest.fixture
def all_laps_df():
    """~30 rows of laps across 3 races with is_outlier column."""
    frames = []
    race_configs = [
        ("2026-02-27-Kiltorcan", "2026-02-27", [70.5, 70.8, 71.2, 70.1, 70.9, 71.5, 70.3, 72.8, 70.6, 70.04]),
        ("2026-03-08-Kiltorcan", "2026-03-08", [70.2, 70.0, 69.8, 70.5, 70.1, 69.742, 70.3, 70.6, 70.4, 70.1]),
        ("2026-03-15-Kiltorcan", "2026-03-15", [69.9, 69.7, 70.1, 69.5, 70.3, 70.0, 69.8, 70.2, 69.6, 70.0]),
    ]
    for race_dir, date_str, times in race_configs:
        for i, t in enumerate(times):
            frames.append({
                "race_dir": race_dir,
                "date": date_str,
                "track": "Kiltorcan",
                "lap": i + 1,
                "seconds": t,
                "is_outlier": False,
            })
    # Mark the 72.8 lap as an outlier
    frames[7]["is_outlier"] = True

    df = pd.DataFrame(frames)
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture
def telemetry_df_minimal():
    """Minimal telemetry — only timestamp and lap_number."""
    n = 50
    return pd.DataFrame({
        "timestamp": np.linspace(0, 30, n),
        "lap_number": np.ones(n, dtype=int),
    })
