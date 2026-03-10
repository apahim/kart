"""Generate the cross-race evolution dashboard.

Usage:
    python scripts/generate_evolution.py
"""

import json
import os
import sys

import pandas as pd
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
    create_weather_impact,
    create_weight_vs_performance,
    create_temp_vs_laptime,
    create_raceline_evolution,
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
        "weather_impact": safe_chart("weather_impact", create_weather_impact, races_df),
        "weight_vs_performance": safe_chart("weight_vs_performance", create_weight_vs_performance, races_df),
        "temp_vs_laptime": safe_chart("temp_vs_laptime", create_temp_vs_laptime, races_df),
        "raceline_evolution": safe_chart("raceline_evolution", create_raceline_evolution, races_df),
    }


def _build_race_list(races_df):
    """Build a list of race info dicts for the JS race selector."""
    result = []
    for _, row in races_df.iterrows():
        date_str = row["date"].strftime("%Y-%m-%d") if pd.notna(row.get("date")) else ""
        label = date_str
        if row.get("session_type"):
            label += f" ({row['session_type']})"
        result.append({"date": date_str, "label": label})
    return result


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

    # Build charts per (track, session_type)
    tracks = sorted(races_df["track"].unique())
    per_track_type_charts = {}
    race_list = {}

    for track in tracks:
        track_races = races_df[races_df["track"] == track]
        track_laps = all_laps_df[all_laps_df["track"] == track] if not all_laps_df.empty else all_laps_df

        per_track_type_charts[track] = {}
        race_list[track] = {}

        # "All" session types
        per_track_type_charts[track]["All"] = build_charts(track_races, track_laps)
        race_list[track]["All"] = _build_race_list(track_races)
        print(f"  Generated charts for {track} / All ({len(track_races)} session(s))")

        # Per session type
        for stype in sorted(track_races["session_type"].dropna().unique()):
            filtered_races = track_races[track_races["session_type"] == stype]
            filtered_laps = track_laps[track_laps["session_type"] == stype] if not track_laps.empty else track_laps
            per_track_type_charts[track][stype] = build_charts(filtered_races, filtered_laps)
            race_list[track][stype] = _build_race_list(filtered_races)
            print(f"  Generated charts for {track} / {stype} ({len(filtered_races)} session(s))")

    # Default to the most recent race's track
    default_track = races_df.sort_values("date").iloc[-1]["track"]

    races = races_df.to_dict("records")

    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("evolution.html.j2")

    html = template.render(
        race_count=len(races_df),
        races=races,
        per_track_type_charts=per_track_type_charts,
        race_list_json=json.dumps(race_list),
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
