"""Generate a per-race interactive dashboard (HTML) and summary (YAML).

Usage:
    python scripts/generate_dashboard.py data/races/2026-03-08-Kiltorcan/
"""

import os
import sys

from jinja2 import Environment, FileSystemLoader

# Allow running as a script from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.load_data import extract_laptimes_from_telemetry, load_telemetry, load_race_metadata
from scripts.analysis.outliers import filter_non_race_laps
from scripts.analysis.summary import generate_summary, write_summary
from scripts.analysis.tracks import load_track_coords, load_track_corners, auto_detect_corners
from scripts.analysis.weather import fetch_weather
from scripts.analysis.utils import format_laptime, safe_chart
from scripts.analysis.laptimes import (
    create_laptime_bar_chart,
    create_delta_to_best_chart,
)
from scripts.analysis.track_map import create_speed_track_map
from scripts.analysis.speed import (
    create_cumulative_time_delta,
    create_throttle_brake_phases,
)
from scripts.analysis.corners import (
    create_corner_time_loss_chart,
)
from scripts.analysis.braking import create_braking_track_map, create_braking_consistency_chart
from scripts.analysis.sectors import create_sector_times_table
from scripts.analysis.coaching import generate_coaching_summary


def main(race_dir):
    race_dir = race_dir.rstrip("/")
    race_name = os.path.basename(race_dir)

    # Load data
    metadata = load_race_metadata(race_dir)

    # Load telemetry (required)
    telemetry_df = load_telemetry(race_dir)
    if telemetry_df is None:
        print(f"Error: no telemetry.csv in {race_dir}")
        sys.exit(1)
    print(f"Loaded telemetry: {len(telemetry_df)} rows, columns: {list(telemetry_df.columns)}")

    # Derive lap times from telemetry
    laptimes_df = extract_laptimes_from_telemetry(telemetry_df)
    laptimes_df = filter_non_race_laps(laptimes_df)
    print(f"Extracted {len(laptimes_df)} race laps from telemetry")

    # Generate summary
    summary = generate_summary(laptimes_df, telemetry_df)

    # Fetch weather from Open-Meteo if we have the required metadata
    track_name = metadata.get("track", "")
    session_start = metadata.get("session_start")
    race_date = metadata.get("date")
    if track_name and session_start and race_date:
        coords = load_track_coords(track_name)
        if coords:
            lat, lon, tz = coords
            print(f"Fetching weather for {race_date} {session_start} at {track_name}...")
            weather = fetch_weather(str(race_date), str(session_start), lat, lon, tz)
            if weather:
                summary["weather"] = weather
                print(f"  Weather: {weather['condition']}, {weather['temp_c']}°C, wind {weather['wind_kmh']} km/h")
            else:
                print("  Weather fetch returned no data")

    summary_path = os.path.join(race_dir, "summary_generated.yaml")
    write_summary(summary, summary_path)
    print(f"Summary written to {summary_path}")

    # Find best lap for telemetry charts
    best_lap = summary["best_lap"]["lap"]

    # Generate charts — each wrapped in try/except so one failure doesn't kill the dashboard
    charts = {}
    charts["laptime_bar"] = safe_chart("laptime_bar", create_laptime_bar_chart, laptimes_df)
    charts["delta_to_best"] = safe_chart("delta_to_best", create_delta_to_best_chart, laptimes_df)

    weather = summary.get("weather")
    sector_data = None

    # Load track-defined corners for consistent detection
    track_corners = load_track_corners(track_name) if track_name else None
    if not track_corners and track_name and telemetry_df is not None:
        print(f"No corners defined for {track_name} — auto-detecting from telemetry...")
        track_corners = auto_detect_corners(track_name)
    if track_corners:
        print(f"Using {len(track_corners)} track-defined corners for {track_name}")

    if telemetry_df is not None:
        charts["speed_track_map"] = safe_chart("speed_track_map", create_speed_track_map, telemetry_df, best_lap=best_lap, weather=weather, track_corners=track_corners)
        charts["braking_map"] = safe_chart("braking_map", create_braking_track_map, telemetry_df, best_lap=best_lap, weather=weather, track_corners=track_corners)
        charts["braking_consistency"] = safe_chart("braking_consistency", create_braking_consistency_chart, telemetry_df, laptimes_df, track_corners=track_corners)
        charts["cumulative_delta"] = safe_chart("cumulative_delta", create_cumulative_time_delta, telemetry_df, laptimes_df, track_corners=track_corners)
        charts["corner_time_loss"] = safe_chart("corner_time_loss", create_corner_time_loss_chart, telemetry_df, laptimes_df, track_corners=track_corners)
        charts["throttle_brake_phases"] = safe_chart("throttle_brake_phases", create_throttle_brake_phases, telemetry_df, laptimes_df)
        try:
            sector_data = create_sector_times_table(telemetry_df, laptimes_df, metadata=metadata, track_corners=track_corners)
        except Exception as e:
            print(f"Warning: sector_times failed: {e}")
            sector_data = None

    # Coaching summary / action plan
    coaching = None
    try:
        coaching = generate_coaching_summary(
            telemetry_df, laptimes_df,
            sector_data=sector_data,
            track_corners=track_corners,
        )
    except Exception as e:
        print(f"Warning: coaching summary failed: {e}")

    # Render template
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("dashboard.html.j2")

    title = metadata.get("track", race_name)

    html = template.render(
        title=title,
        metadata=metadata,
        summary=summary,
        charts=charts,
        coaching=coaching,
        sector_data=sector_data,
    )

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(repo_root, "docs", "races", race_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dashboard.html")
    with open(output_path, "w") as f:
        f.write(html)

    print(f"Dashboard written to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/generate_dashboard.py <race_directory>")
        sys.exit(1)
    main(sys.argv[1])
