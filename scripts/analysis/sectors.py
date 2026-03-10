"""Sector time analysis."""

import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks

from scripts.analysis.outliers import detect_outliers


def detect_sectors(df, best_lap, n_sectors=3, manual_sectors=None, track_corners=None):
    """Detect sector boundaries from track corners or speed peaks.

    When track_corners are provided, creates one sector per corner with
    boundaries at midpoints between consecutive corners.

    Args:
        df: Telemetry DataFrame.
        best_lap: Lap number for analysis.
        n_sectors: Number of sectors to detect (used only for speed-based fallback).
        manual_sectors: Optional list of distance fractions from race.yaml.
        track_corners: Optional list of {name, lat, lon} dicts from tracks.yaml.

    Returns:
        List of boundary distance fractions (length n_sectors+1, starting with 0 and ending with 1).
    """
    if manual_sectors:
        boundaries = [0.0] + sorted(manual_sectors) + [1.0]
        return boundaries

    if "distance_traveled" not in df.columns or "lap_number" not in df.columns:
        return None

    lap_data = df[df["lap_number"] == best_lap].copy().reset_index(drop=True)
    if len(lap_data) < 50:
        return None

    dist = lap_data["distance_traveled"].values
    dist_norm = dist - dist[0]
    lap_length = dist_norm[-1]
    if lap_length <= 0:
        return None

    # GPS-based sectors from track corners
    if track_corners:
        lat_col = lon_col = None
        for col in lap_data.columns:
            cl = col.lower()
            if "latitude" in cl:
                lat_col = col
            if "longitude" in cl:
                lon_col = col

        if lat_col and lon_col:
            lat = lap_data[lat_col].values
            lon = lap_data[lon_col].values
            lat_mean = np.mean(lat)
            lon_mean = np.mean(lon)
            cos_lat = np.cos(np.radians(lat_mean))

            x_data = (lon - lon_mean) * 111320 * cos_lat
            y_data = (lat - lat_mean) * 110540

            corner_fracs = []
            for tc in track_corners:
                cx = (tc["lon"] - lon_mean) * 111320 * cos_lat
                cy = (tc["lat"] - lat_mean) * 110540
                dists = np.sqrt((x_data - cx) ** 2 + (y_data - cy) ** 2)
                idx = int(np.argmin(dists))
                corner_fracs.append(dist_norm[idx] / lap_length)

            corner_fracs.sort()
            # Boundaries at midpoints between consecutive corners
            boundaries = [0.0]
            for i in range(len(corner_fracs) - 1):
                boundaries.append((corner_fracs[i] + corner_fracs[i + 1]) / 2)
            boundaries.append(1.0)
            return boundaries

    # Fallback: speed-based detection
    speed_col = "speed_gps" if "speed_gps" in df.columns else "speed"
    if speed_col not in df.columns:
        return None

    speed = lap_data[speed_col].values
    kernel_size = min(15, len(speed) // 5)
    if kernel_size > 1:
        speed_smooth = np.convolve(speed, np.ones(kernel_size) / kernel_size, mode="same")
    else:
        speed_smooth = speed

    # Find speed peaks (top of straights)
    peaks, properties = find_peaks(speed_smooth, prominence=0.5, distance=len(speed) // (n_sectors + 2))
    if len(peaks) < n_sectors:
        # Fall back to equal spacing
        return [i / n_sectors for i in range(n_sectors + 1)]

    # Sort by prominence and pick top n_sectors
    prom_order = np.argsort(properties["prominences"])[::-1]
    top_peaks = sorted(peaks[prom_order[:n_sectors]])

    boundaries = [0.0]
    for p in top_peaks:
        boundaries.append(dist_norm[p] / lap_length)
    boundaries.append(1.0)

    return boundaries


def compute_sector_times(df, laptimes_df, sector_boundaries, time_col="seconds"):
    """Compute time per sector per lap.

    Returns dict with:
        sector_times: {lap: [s1_time, s2_time, ...]}
        best_sectors: [best_s1, best_s2, ...]
        theoretical_best: sum of best sectors
        clean_laps: list of clean lap numbers
        best_lap: overall best lap
    """
    speed_col = "speed_gps" if "speed_gps" in df.columns else "speed"
    if speed_col not in df.columns or "lap_number" not in df.columns:
        return None
    if "distance_traveled" not in df.columns:
        return None

    clean_df, _ = detect_outliers(laptimes_df, time_col=time_col)
    best_lap = int(clean_df.loc[clean_df[time_col].idxmin(), "lap"])
    all_laps = sorted(laptimes_df["lap"].astype(int))

    # Build lookup of actual lap times for proportional scaling
    lap_time_lookup = dict(zip(laptimes_df["lap"].astype(int), laptimes_df[time_col]))

    n_sectors = len(sector_boundaries) - 1
    sector_times = {}

    for lap in all_laps:
        lap_data = df[df["lap_number"] == lap].copy().reset_index(drop=True)
        if len(lap_data) < 20:
            continue

        dist = lap_data["distance_traveled"].values
        dist_norm = dist - dist[0]
        lap_length = dist_norm[-1]
        if lap_length <= 0:
            continue

        speed = lap_data[speed_col].values
        speed = np.maximum(speed, 0.1)

        frac = dist_norm / lap_length
        times = []
        for si in range(n_sectors):
            start_frac = sector_boundaries[si]
            end_frac = sector_boundaries[si + 1]
            mask = (frac >= start_frac) & (frac <= end_frac)
            indices = np.where(mask)[0]
            if len(indices) < 2:
                times.append(None)
                continue
            seg_dist = dist_norm[indices]
            seg_speed = speed[indices]
            dd = np.diff(seg_dist)
            dt = dd / seg_speed[:-1]
            times.append(float(np.sum(dt)))

        if None not in times:
            # Scale sector times so they sum to the actual lap time
            raw_total = sum(times)
            if raw_total > 0:
                actual_time = lap_time_lookup.get(lap, raw_total)
                factor = actual_time / raw_total
                times = [t * factor for t in times]
            sector_times[lap] = times

    if not sector_times:
        return None

    best_sectors = []
    best_sector_laps = []
    for si in range(n_sectors):
        best_time = min(times[si] for times in sector_times.values())
        best_sectors.append(best_time)
        best_sector_laps.append(
            min(lap for lap, times in sector_times.items() if times[si] == best_time)
        )

    theoretical_best = sum(best_sectors)

    return {
        "sector_times": sector_times,
        "best_sectors": best_sectors,
        "best_sector_laps": best_sector_laps,
        "theoretical_best": theoretical_best,
        "clean_laps": sorted(clean_df["lap"].astype(int)),
        "best_lap": best_lap,
        "sector_boundaries": sector_boundaries,
    }


def create_sector_times_table(df, laptimes_df=None, time_col="seconds", metadata=None, track_corners=None):
    """Build sector times data for HTML table rendering.

    Returns a dict with keys: headers, rows, theoretical_row, best_lap.
    Each row is a dict with 'lap', 'sectors' (list of {value, css_class}),
    'total', 'delta', 'is_best'.
    Returns None if insufficient data.
    """
    if laptimes_df is None or time_col not in laptimes_df.columns:
        return None

    speed_col = "speed_gps" if "speed_gps" in df.columns else "speed"
    if speed_col not in df.columns or "lap_number" not in df.columns:
        return None
    if "distance_traveled" not in df.columns:
        return None

    clean_df, _ = detect_outliers(laptimes_df, time_col=time_col)
    if len(clean_df) < 2:
        return None
    best_lap = int(clean_df.loc[clean_df[time_col].idxmin(), "lap"])

    manual_sectors = None
    if metadata and "sectors" in metadata:
        manual_sectors = metadata["sectors"]

    boundaries = detect_sectors(df, best_lap, manual_sectors=manual_sectors, track_corners=track_corners)
    if boundaries is None:
        return None

    result = compute_sector_times(df, laptimes_df, boundaries, time_col=time_col)
    if result is None:
        return None

    sector_times = result["sector_times"]
    best_sectors = result["best_sectors"]
    n_sectors = len(best_sectors)

    laps = sorted(sector_times.keys())
    if track_corners and len(track_corners) == n_sectors:
        headers = [tc["name"] for tc in track_corners]
    else:
        headers = [f"S{i+1}" for i in range(n_sectors)]
    worst_sectors = [max(sector_times[lap][si] for lap in laps) for si in range(n_sectors)]

    rows = []
    for lap in laps:
        times = sector_times[lap]
        total = sum(times)
        delta = total - result["theoretical_best"]
        sectors = []
        for si in range(n_sectors):
            if times[si] == best_sectors[si]:
                css = "sector-best"
            elif times[si] == worst_sectors[si]:
                css = "sector-slow"
            else:
                css = ""
            sectors.append({"value": f"{times[si]:.3f}", "css_class": css})
        rows.append({
            "lap": int(lap),
            "sectors": sectors,
            "total": f"{total:.3f}",
            "delta": f"{delta:+.3f}",
            "is_best": lap == best_lap,
        })

    theoretical_row = {
        "sectors": [{"value": f"{bs:.3f}", "css_class": "sector-best"} for bs in best_sectors],
        "total": f"{result['theoretical_best']:.3f}",
    }

    return {
        "headers": headers,
        "rows": rows,
        "theoretical_row": theoretical_row,
        "best_lap": best_lap,
        "sector_boundaries": boundaries,
        "best_sectors": best_sectors,
        "best_sector_laps": result["best_sector_laps"],
        "sector_times": sector_times,
    }
