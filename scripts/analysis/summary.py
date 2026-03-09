"""Race summary generation with IQR-filtered statistics."""

from datetime import datetime

import yaml

from scripts.analysis.outliers import detect_outliers
from scripts.analysis.utils import format_laptime


def generate_summary(laptimes_df, telemetry_df=None, time_col="seconds"):
    """Generate a race summary dict from lap times and optional telemetry.

    Args:
        laptimes_df: DataFrame with lap and seconds columns.
        telemetry_df: Optional telemetry DataFrame with speed/g-force data.
        time_col: Column name for lap times.

    Returns:
        Dict suitable for writing as summary.yaml.
    """
    clean_df, excluded = detect_outliers(laptimes_df, time_col=time_col)
    clean_times = clean_df[time_col]

    best_idx = clean_df[time_col].idxmin()
    worst_idx = clean_df[time_col].idxmax()

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "total_laps": int(len(laptimes_df)),
        "clean_laps": int(len(clean_df)),
        "excluded_laps": excluded,
        "best_lap": {
            "lap": int(clean_df.loc[best_idx, "lap"]),
            "time": round(float(clean_df.loc[best_idx, time_col]), 3),
        },
        "worst_clean_lap": {
            "lap": int(clean_df.loc[worst_idx, "lap"]),
            "time": round(float(clean_df.loc[worst_idx, time_col]), 3),
        },
        "average": round(float(clean_times.mean()), 2),
        "median": round(float(clean_times.median()), 2),
        "std_dev": round(float(clean_times.std()), 2),
        "consistency_pct": round(
            float(100 * (1 - clean_times.std() / clean_times.mean())), 1
        ),
        "best_lap_time_fmt": format_laptime(float(clean_df.loc[best_idx, time_col])),
        "worst_clean_time_fmt": format_laptime(float(clean_df.loc[worst_idx, time_col])),
        "average_fmt": format_laptime(float(clean_times.mean())),
        "median_fmt": format_laptime(float(clean_times.median())),
    }

    if telemetry_df is not None:
        # Speed columns (in m/s from RaceChrono)
        if "speed_gps" in telemetry_df.columns:
            top_speed = telemetry_df["speed_gps"].max() * 3.6
            summary["top_speed_kmh"] = round(float(top_speed), 1)
        elif "speed" in telemetry_df.columns:
            top_speed = telemetry_df["speed"].max() * 3.6
            summary["top_speed_kmh"] = round(float(top_speed), 1)

        if "lateral_acc" in telemetry_df.columns:
            summary["max_lateral_g"] = round(
                float(telemetry_df["lateral_acc"].abs().max()), 2
            )
        if "longitudinal_acc" in telemetry_df.columns:
            long_acc = telemetry_df["longitudinal_acc"]
            summary["max_braking_g"] = round(float(long_acc.min() * -1), 2)
            summary["max_acceleration_g"] = round(float(long_acc.max()), 2)

    return summary


def write_summary(summary, output_path):
    """Write summary dict to a YAML file."""
    with open(output_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
