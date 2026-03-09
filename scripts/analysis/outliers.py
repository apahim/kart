"""IQR-based outlier detection and non-race lap filtering."""

import pandas as pd


def filter_non_race_laps(laptimes_df, time_col="seconds"):
    """Remove outlap, cooldown, and incomplete laps from telemetry-derived lap times.

    Drops the last lap (always cooldown/incomplete in RaceChrono exports),
    then removes laps outside median * [0.5, 1.5] range.

    Returns the filtered DataFrame.
    """
    if laptimes_df.empty or len(laptimes_df) < 2:
        return laptimes_df

    # Drop the last lap (cooldown/incomplete)
    max_lap = laptimes_df["lap"].max()
    df = laptimes_df[laptimes_df["lap"] != max_lap].copy()

    if df.empty:
        return df

    # Remove laps far from median (catches remaining partial/cooldown laps)
    median = df[time_col].median()
    df = df[(df[time_col] >= median * 0.5) & (df[time_col] <= median * 1.5)].copy()

    return df


def detect_outliers(laptimes_df, time_col="seconds", iqr_multiplier=1.5):
    """Flag outlier laps using the IQR method.

    Args:
        laptimes_df: DataFrame with lap times.
        time_col: Column name containing lap times in seconds.
        iqr_multiplier: Multiplier for IQR bounds (default 1.5).

    Returns:
        Tuple of (clean_df, excluded_list) where:
        - clean_df: DataFrame with outlier laps removed.
        - excluded_list: List of dicts with lap info and exclusion reason.
    """
    times = laptimes_df[time_col]
    q1 = times.quantile(0.25)
    q3 = times.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr

    mask = (times >= lower) & (times <= upper)
    clean_df = laptimes_df[mask].copy()

    excluded = []
    for _, row in laptimes_df[~mask].iterrows():
        reason = "IQR outlier"
        if row[time_col] > upper:
            reason += f" (>{upper:.3f}s)"
        else:
            reason += f" (<{lower:.3f}s)"
        excluded.append({
            "lap": int(row["lap"]),
            "time": round(float(row[time_col]), 3),
            "reason": reason,
        })

    return clean_df, excluded
