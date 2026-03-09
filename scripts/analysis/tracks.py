"""Track configuration helpers: load coordinates, corners, and auto-detection."""

import os

import yaml


def _tracks_yaml_path():
    """Return the absolute path to data/tracks.yaml."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "tracks.yaml",
    )


def load_track_coords(track_name):
    """Load coordinates and timezone for a track from data/tracks.yaml.

    Args:
        track_name: Track name (e.g. "Kiltorcan Raceway").

    Returns:
        (lat, lon, timezone) tuple, or None if not found.
    """
    tracks_path = _tracks_yaml_path()
    if not os.path.exists(tracks_path):
        return None

    with open(tracks_path, "r") as f:
        tracks = yaml.safe_load(f) or {}

    # Normalize: lowercase, spaces to underscores
    key = track_name.lower().replace(" ", "_")
    entry = tracks.get(key)
    if entry:
        return entry["lat"], entry["lon"], entry.get("timezone", "UTC")

    return None


def load_track_corners(track_name):
    """Load corner definitions for a track from data/tracks.yaml.

    Args:
        track_name: Track name (e.g. "Kiltorcan Raceway").

    Returns:
        List of {name, lat, lon} dicts, or None if not defined.
    """
    tracks_path = _tracks_yaml_path()
    if not os.path.exists(tracks_path):
        return None

    with open(tracks_path, "r") as f:
        tracks = yaml.safe_load(f) or {}

    key = track_name.lower().replace(" ", "_")
    entry = tracks.get(key)
    if entry and "corners" in entry:
        return entry["corners"]

    return None


def auto_detect_corners(track_name):
    """Auto-detect corner positions by clustering speed minima across all sessions.

    Scans all race directories for the given track, runs speed-based corner
    detection on each session's best lap, clusters the resulting GPS positions,
    filters outliers, and writes the corners back to tracks.yaml.

    Returns the list of {name, lat, lon} dicts, or None if detection fails.
    """
    import glob
    import numpy as np
    from collections import Counter

    from scripts.load_data import load_race_metadata, load_telemetry, extract_laptimes_from_telemetry
    from scripts.analysis.corners import detect_corners
    from scripts.analysis.outliers import filter_non_race_laps, detect_outliers

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "races",
    )

    # Find all sessions for this track
    race_dirs = sorted(
        d for d in glob.glob(os.path.join(data_dir, "*"))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "race.yaml"))
    )

    # Collect corner GPS positions from each session's best lap
    all_corner_positions = []  # list of lists of (lat, lon)

    for race_dir in race_dirs:
        metadata = load_race_metadata(race_dir)
        if metadata.get("track", "").lower() != track_name.lower():
            continue

        telemetry_df = load_telemetry(race_dir)
        if telemetry_df is None:
            continue

        # Find lat/lon columns
        lat_col = lon_col = None
        for col in telemetry_df.columns:
            cl = col.lower()
            if "latitude" in cl:
                lat_col = col
            if "longitude" in cl:
                lon_col = col
        if not lat_col or not lon_col:
            continue

        # Get best lap
        laptimes_df = extract_laptimes_from_telemetry(telemetry_df)
        laptimes_df = filter_non_race_laps(laptimes_df)
        if laptimes_df.empty or "seconds" not in laptimes_df.columns:
            continue
        clean_df, _ = detect_outliers(laptimes_df, time_col="seconds")
        if clean_df.empty:
            continue
        best_lap = int(clean_df.loc[clean_df["seconds"].idxmin(), "lap"])

        # Run speed-based detection (no track_corners)
        corners, lap_data = detect_corners(telemetry_df, best_lap=best_lap)
        if corners is None or len(corners) == 0:
            continue

        positions = []
        for idx in corners:
            positions.append((
                float(lap_data[lat_col].iloc[idx]),
                float(lap_data[lon_col].iloc[idx]),
            ))
        all_corner_positions.append(positions)

    if not all_corner_positions:
        print(f"  Auto-detect: no telemetry sessions found for {track_name}")
        return None

    print(f"  Auto-detect: found {len(all_corner_positions)} session(s) with corners")

    # Find the most common corner count (mode) — sessions with different
    # counts are likely detection artifacts
    counts = Counter(len(p) for p in all_corner_positions)
    target_count = counts.most_common(1)[0][0]
    filtered = [p for p in all_corner_positions if len(p) == target_count]

    if len(filtered) < 1:
        print(f"  Auto-detect: no sessions with consistent corner count")
        return None

    print(f"  Auto-detect: {len(filtered)} session(s) with {target_count} corners (mode)")

    # Cluster: for each corner index, collect all positions and take the median
    # (robust to outliers from individual sessions)
    corners_result = []
    for ci in range(target_count):
        lats = np.array([p[ci][0] for p in filtered])
        lons = np.array([p[ci][1] for p in filtered])

        # Remove outliers: drop positions more than 2 * IQR from median
        if len(lats) >= 3:
            lat_med = np.median(lats)
            lon_med = np.median(lons)
            # Distance from median in meters
            lat_mean_rad = np.radians(lat_med)
            cos_lat = np.cos(lat_mean_rad)
            dx = (lons - lon_med) * 111320 * cos_lat
            dy = (lats - lat_med) * 110540
            dists = np.sqrt(dx**2 + dy**2)
            q75 = np.percentile(dists, 75)
            q25 = np.percentile(dists, 25)
            iqr = q75 - q25
            cutoff = q75 + 2 * max(iqr, 5.0)  # min 5m to avoid over-filtering
            mask = dists <= cutoff
            lats = lats[mask]
            lons = lons[mask]

        if len(lats) == 0:
            continue

        corners_result.append({
            "name": f"T{ci + 1}",
            "lat": round(float(np.median(lats)), 5),
            "lon": round(float(np.median(lons)), 5),
        })

    if not corners_result:
        print(f"  Auto-detect: clustering produced no valid corners")
        return None

    # Write back to tracks.yaml
    tracks_path = _tracks_yaml_path()
    if os.path.exists(tracks_path):
        with open(tracks_path, "r") as f:
            tracks = yaml.safe_load(f) or {}
    else:
        tracks = {}

    key = track_name.lower().replace(" ", "_")
    if key not in tracks:
        tracks[key] = {"name": track_name}
    tracks[key]["corners"] = corners_result

    with open(tracks_path, "w") as f:
        yaml.dump(tracks, f, default_flow_style=False, sort_keys=False)

    print(f"  Auto-detect: wrote {len(corners_result)} corners to tracks.yaml for {track_name}")
    return corners_result
