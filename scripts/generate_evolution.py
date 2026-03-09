"""Generate the cross-race evolution dashboard.

Usage:
    python scripts/generate_evolution.py
"""

import os
import sys

from jinja2 import Environment, FileSystemLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.analysis.utils import safe_chart
from scripts.analysis.evolution import (
    load_all_races,
    load_all_laptimes,
    enrich_races_with_quartiles,
    create_laptime_progression,
    create_consistency_trend,
    create_speed_gforce_trends,
    create_lap_distribution,
    create_session_overlay,
    create_improvement_summary,
    create_kart_comparison,
    create_weather_impact,
    create_weight_vs_performance,
    create_temp_vs_laptime,
)


def build_charts(races_df, all_laps_df):
    """Build the full set of charts from the given DataFrames."""
    return {
        "laptime_progression": safe_chart("laptime_progression", create_laptime_progression, races_df),
        "lap_distribution": safe_chart("lap_distribution", create_lap_distribution, all_laps_df),
        "session_overlay": safe_chart("session_overlay", create_session_overlay, all_laps_df),
        "improvement_summary": safe_chart("improvement_summary", create_improvement_summary, races_df),
        "consistency_trend": safe_chart("consistency_trend", create_consistency_trend, races_df),
        "speed_gforce_trends": safe_chart("speed_gforce_trends", create_speed_gforce_trends, races_df),
        "kart_comparison": safe_chart("kart_comparison", create_kart_comparison, races_df),
        "weather_impact": safe_chart("weather_impact", create_weather_impact, races_df),
        "weight_vs_performance": safe_chart("weight_vs_performance", create_weight_vs_performance, races_df),
        "temp_vs_laptime": safe_chart("temp_vs_laptime", create_temp_vs_laptime, races_df),
    }


def main():
    races_df = load_all_races()

    if races_df.empty:
        print("No race summaries found. Generate per-race dashboards first.")
        sys.exit(1)

    print(f"Found {len(races_df)} race(s)")

    all_laps_df = load_all_laptimes()
    if not all_laps_df.empty:
        races_df = enrich_races_with_quartiles(races_df, all_laps_df)
        print(f"Loaded {len(all_laps_df)} individual laps across all races")

    # Build charts per track
    tracks = sorted(races_df["track"].unique())
    per_track_charts = {}
    for track in tracks:
        track_races = races_df[races_df["track"] == track]
        track_laps = all_laps_df[all_laps_df["track"] == track] if not all_laps_df.empty else all_laps_df
        per_track_charts[track] = build_charts(track_races, track_laps)
        print(f"  Generated charts for {track} ({len(track_races)} session(s))")

    # Default to the most recent race's track
    default_track = races_df.sort_values("date").iloc[-1]["track"]

    races = races_df.to_dict("records")

    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("evolution.html.j2")

    html = template.render(
        race_count=len(races_df),
        races=races,
        per_track_charts=per_track_charts,
        tracks=tracks,
        default_track=default_track,
    )

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(repo_root, "docs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "evolution.html")
    with open(output_path, "w") as f:
        f.write(html)

    print(f"Evolution dashboard written to {output_path}")


if __name__ == "__main__":
    main()
