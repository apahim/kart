"""Plot telemetry data from RaceChrono Pro exports (RaceBox Mini)."""

import sys
import os
import matplotlib.pyplot as plt
from load_data import load_telemetry


def plot_speed(df, title="Speed"):
    """Plot speed over time from a RaceChrono session."""
    # RaceChrono typically exports 'Speed (km/h)' or similar
    speed_col = None
    for col in df.columns:
        if "speed" in col.lower():
            speed_col = col
            break

    if speed_col is None:
        print(f"  No speed column found. Available columns: {list(df.columns)}")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df[speed_col], linewidth=0.8)
    ax.set_ylabel(speed_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_track_map(df, title="Track Map"):
    """Plot GPS track from a RaceChrono session."""
    lat_col, lon_col = None, None
    for col in df.columns:
        col_lower = col.lower()
        if "lat" in col_lower:
            lat_col = col
        if "lon" in col_lower:
            lon_col = col

    if lat_col is None or lon_col is None:
        print(f"  No lat/lon columns found. Available columns: {list(df.columns)}")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(df[lon_col], df[lat_col], linewidth=0.8)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()


def plot_telemetry(race_dir):
    df = load_telemetry(race_dir)

    if df is None:
        print(f"No telemetry.csv found in {race_dir}")
        sys.exit(1)

    print(f"Plotting {os.path.basename(race_dir)} ({len(df)} rows, {len(df.columns)} columns)")
    plot_speed(df, title=f"Speed - {os.path.basename(race_dir)}")
    plot_track_map(df, title=f"Track - {os.path.basename(race_dir)}")

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_telemetry.py <race_directory>")
        sys.exit(1)
    plot_telemetry(sys.argv[1])
