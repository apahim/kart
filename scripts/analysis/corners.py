"""Corner detection and per-corner statistics."""

import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks

from scripts.analysis.utils import format_laptime


def detect_corners(df, best_lap=None, track_corners=None, prominence=0.5, min_distance=50):
    """Detect corners by finding speed minima or matching to track-defined GPS positions.

    Args:
        df: Telemetry DataFrame.
        best_lap: If set, analyze only this lap.
        track_corners: List of {name, lat, lon} dicts from tracks.yaml.
            When provided, corners are matched by GPS proximity instead of speed.
        prominence: Minimum prominence for peak detection (m/s).
        min_distance: Minimum distance between peaks (samples).

    Returns:
        Tuple of (corner_indices, lap_data) or (None, None) if insufficient data.
    """
    speed_col = "speed_gps" if "speed_gps" in df.columns else "speed"
    if speed_col not in df.columns:
        return None, None

    if best_lap is not None and "lap_number" in df.columns:
        lap_data = df[df["lap_number"] == best_lap].copy().reset_index(drop=True)
    else:
        lap_data = df.copy().reset_index(drop=True)

    if len(lap_data) < min_distance * 2:
        return None, None

    # GPS-based matching when track corners are defined
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
            # Project to meters for distance calculation
            lat_mean = np.mean(lat)
            lon_mean = np.mean(lon)
            lat_mean_rad = np.radians(lat_mean)
            cos_lat = np.cos(lat_mean_rad)

            x_data = (lon - lon_mean) * 111320 * cos_lat
            y_data = (lat - lat_mean) * 110540

            peaks = []
            for tc in track_corners:
                cx = (tc["lon"] - lon_mean) * 111320 * cos_lat
                cy = (tc["lat"] - lat_mean) * 110540
                dists = np.sqrt((x_data - cx) ** 2 + (y_data - cy) ** 2)
                peaks.append(int(np.argmin(dists)))

            return np.array(peaks), lap_data

    # Fallback: speed-based detection
    speed = lap_data[speed_col].values
    # Smooth speed
    kernel_size = min(10, len(speed) // 5)
    if kernel_size > 1:
        speed_smooth = np.convolve(speed, np.ones(kernel_size) / kernel_size, mode="same")
    else:
        speed_smooth = speed

    inverted = -speed_smooth
    peaks, properties = find_peaks(inverted, prominence=prominence, distance=min_distance)

    return peaks, lap_data


def create_corner_analysis(df, laptimes_df=None, time_col="seconds", track_corners=None):
    """Create corner analysis chart showing speed minima on the best lap."""
    from scripts.analysis.outliers import detect_outliers

    best_lap = None
    if laptimes_df is not None and time_col in laptimes_df.columns:
        clean_df, _ = detect_outliers(laptimes_df, time_col=time_col)
        best_lap = int(clean_df.loc[clean_df[time_col].idxmin(), "lap"])

    corners, lap_data = detect_corners(df, best_lap=best_lap, track_corners=track_corners)
    if corners is None or len(corners) == 0:
        return None

    speed_col = "speed_gps" if "speed_gps" in df.columns else "speed"
    speed_kmh = lap_data[speed_col].values * 3.6

    # Use distance in meters for x-axis if available
    if "distance_traveled" in lap_data.columns:
        dist = lap_data["distance_traveled"].values
        x_vals = dist - dist[0]
        x_label = "Distance (m)"
        corner_x = x_vals[corners].tolist()
        hover_x = "Distance: %{x:.0f}m"
    else:
        x_vals = list(range(len(speed_kmh)))
        x_label = "Sample"
        corner_x = corners.tolist()
        hover_x = "Sample: %{x}"

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=speed_kmh,
        mode="lines",
        name="Speed",
        line=dict(color="#3498db", width=1.5),
    ))

    fig.add_trace(go.Scatter(
        x=corner_x,
        y=speed_kmh[corners].tolist(),
        mode="markers+text",
        name="Corners",
        marker=dict(size=10, color="#e74c3c", symbol="triangle-down"),
        text=[track_corners[i]["name"] if track_corners and i < len(track_corners) else f"T{i+1}" for i in range(len(corners))],
        textposition="bottom center",
        hovertemplate="Corner %{text}<br>Min Speed: %{y:.1f} km/h<extra></extra>",
    ))

    title = f"Corner Detection (Lap {best_lap})" if best_lap else "Corner Detection"
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="Speed (km/h)",
        template="plotly_white",
        height=400,
    )

    return fig


def create_corner_comparison_table(df, laptimes_df=None, time_col="seconds", track_corners=None):
    """Table showing per-corner stats across all clean laps."""
    from scripts.analysis.outliers import detect_outliers

    speed_col = "speed_gps" if "speed_gps" in df.columns else "speed"
    if speed_col not in df.columns or "lap_number" not in df.columns:
        return None
    if "distance_traveled" not in df.columns:
        return None

    if laptimes_df is None or time_col not in laptimes_df.columns:
        return None

    clean_df, _ = detect_outliers(laptimes_df, time_col=time_col)
    best_lap = int(clean_df.loc[clean_df[time_col].idxmin(), "lap"])
    clean_laps = set(clean_df["lap"].astype(int))

    corners, ref_lap_data = detect_corners(df, best_lap=best_lap, track_corners=track_corners)
    if corners is None or len(corners) == 0:
        return None

    corner_names = [track_corners[i]["name"] if track_corners and i < len(track_corners) else f"T{i + 1}" for i in range(len(corners))]

    # Reference distances for corners (as fraction of lap)
    ref_dist = ref_lap_data["distance_traveled"].values
    ref_dist_norm = ref_dist - ref_dist[0]
    lap_length = ref_dist_norm[-1]
    if lap_length <= 0:
        return None
    corner_fracs = [ref_dist_norm[c] / lap_length for c in corners]

    # Collect per-corner min speed for each lap
    n_corners = len(corners)
    sample_window = 20
    corner_data = {i: {"min_speeds": [], "entry_speeds": [], "exit_speeds": []} for i in range(n_corners)}

    laps = sorted(df["lap_number"].dropna().unique())
    laps = [l for l in laps if l > 0 and l in clean_laps]

    for lap in laps:
        lap_data = df[df["lap_number"] == lap].copy().reset_index(drop=True)
        if len(lap_data) < 50:
            continue
        dist = lap_data["distance_traveled"].values
        dist_norm = dist - dist[0]
        this_len = dist_norm[-1]
        if this_len <= 0:
            continue
        speed = lap_data[speed_col].values * 3.6

        for ci, cf in enumerate(corner_fracs):
            # Find the index closest to this corner fraction
            target_dist = cf * this_len
            idx = np.argmin(np.abs(dist_norm - target_dist))

            # Search window around the corner for min speed
            win_start = max(0, idx - sample_window)
            win_end = min(len(speed), idx + sample_window)
            window_speed = speed[win_start:win_end]
            if len(window_speed) == 0:
                continue

            min_idx = win_start + np.argmin(window_speed)
            corner_data[ci]["min_speeds"].append(speed[min_idx])

            # Entry speed (sample_window before min)
            entry_idx = max(0, min_idx - sample_window)
            corner_data[ci]["entry_speeds"].append(speed[entry_idx])

            # Exit speed (sample_window after min)
            exit_idx = min(len(speed) - 1, min_idx + sample_window)
            corner_data[ci]["exit_speeds"].append(speed[exit_idx])

    # Build table data
    headers = ["Corner", "Best Min Speed", "Avg Min Speed", "Std Dev", "Best Entry", "Best Exit"]
    corner_labels = []
    best_mins = []
    avg_mins = []
    std_devs = []
    best_entries = []
    best_exits = []

    for ci in range(n_corners):
        if not corner_data[ci]["min_speeds"]:
            continue
        mins = corner_data[ci]["min_speeds"]
        entries = corner_data[ci]["entry_speeds"]
        exits = corner_data[ci]["exit_speeds"]

        corner_labels.append(corner_names[ci])
        best_mins.append(f"{max(mins):.1f}")
        avg_mins.append(f"{np.mean(mins):.1f}")
        std_devs.append(f"{np.std(mins):.2f}")
        best_entries.append(f"{max(entries):.1f}")
        best_exits.append(f"{max(exits):.1f}")

    if not corner_labels:
        return None

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=headers,
            fill_color="#3498db",
            font=dict(color="white", size=12),
            align="center",
        ),
        cells=dict(
            values=[corner_labels, best_mins, avg_mins, std_devs, best_entries, best_exits],
            fill_color=[["#f0f8ff" if i % 2 == 0 else "white" for i in range(len(corner_labels))]] * 6,
            align="center",
            font=dict(size=11),
        ),
    )])

    fig.update_layout(
        title="Corner Comparison (Clean Laps)",
        template="plotly_white",
        height=max(200, 60 + 35 * len(corner_labels)),
    )

    return fig


def create_corner_time_loss_chart(df, laptimes_df=None, time_col="seconds", track_corners=None):
    """Horizontal bar chart ranking corners by time lost/gained vs best lap."""
    from scripts.analysis.speed import compute_time_delta

    result = compute_time_delta(df, laptimes_df, time_col=time_col, track_corners=track_corners)
    if result is None:
        return None

    dist_grid, cum_delta, best_lap, median_lap, corners_info = result
    if len(corners_info) < 2:
        return None

    # Get corner distances and sort by distance
    corner_dists = []
    for c in corners_info:
        if "distance" in c and c["distance"] <= dist_grid[-1]:
            corner_dists.append((c["label"], c["distance"]))
    if len(corner_dists) < 2:
        return None
    corner_dists.sort(key=lambda x: x[1])

    # Compute boundaries at midpoints between adjacent corners
    boundaries = [0]
    for i in range(len(corner_dists) - 1):
        boundaries.append((corner_dists[i][1] + corner_dists[i + 1][1]) / 2)
    boundaries.append(dist_grid[-1])

    # Extract delta between entry/exit of each corner zone
    labels = []
    deltas = []
    for i, (label, _) in enumerate(corner_dists):
        entry_dist = boundaries[i]
        exit_dist = boundaries[i + 1]
        entry_idx = np.argmin(np.abs(dist_grid - entry_dist))
        exit_idx = np.argmin(np.abs(dist_grid - exit_dist))
        delta = cum_delta[exit_idx] - cum_delta[entry_idx]
        labels.append(label)
        deltas.append(delta)

    # Sort by magnitude (biggest loss first)
    sorted_pairs = sorted(zip(labels, deltas), key=lambda x: x[1])
    labels = [p[0] for p in sorted_pairs]
    deltas = [p[1] for p in sorted_pairs]

    colors = ["rgba(46,204,113,0.8)" if d >= 0 else "rgba(231,76,60,0.8)" for d in deltas]

    fig = go.Figure(go.Bar(
        y=labels, x=deltas, orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:+.3f}s<extra></extra>",
    ))

    fig.update_layout(
        title=f"Per-Corner Time Loss: Best (L{best_lap}) vs Median (L{median_lap})",
        xaxis_title="Time Delta (s) — green=gained, red=lost",
        template="plotly_white",
        height=max(250, 50 + 40 * len(labels)),
    )

    return fig


def create_corner_min_speed_chart(df, laptimes_df=None, time_col="seconds", track_corners=None):
    """Grouped bar chart of min speed per corner per lap, best lap highlighted."""
    from scripts.analysis.outliers import detect_outliers

    speed_col = "speed_gps" if "speed_gps" in df.columns else "speed"
    if speed_col not in df.columns or "lap_number" not in df.columns:
        return None
    if "distance_traveled" not in df.columns:
        return None
    if laptimes_df is None or time_col not in laptimes_df.columns:
        return None

    clean_df, _ = detect_outliers(laptimes_df, time_col=time_col)
    best_lap = int(clean_df.loc[clean_df[time_col].idxmin(), "lap"])
    clean_laps = sorted(clean_df["lap"].astype(int))

    corners, ref_lap_data = detect_corners(df, best_lap=best_lap, track_corners=track_corners)
    if corners is None or len(corners) == 0:
        return None

    corner_names = [track_corners[i]["name"] if track_corners and i < len(track_corners) else f"T{i + 1}" for i in range(len(corners))]

    ref_dist = ref_lap_data["distance_traveled"].values
    ref_dist_norm = ref_dist - ref_dist[0]
    lap_length = ref_dist_norm[-1]
    if lap_length <= 0:
        return None
    corner_fracs = [ref_dist_norm[c] / lap_length for c in corners]
    n_corners = len(corners)

    fig = go.Figure()
    sample_window = 20

    for lap in clean_laps:
        lap_data = df[df["lap_number"] == lap].copy().reset_index(drop=True)
        if len(lap_data) < 50:
            continue
        dist = lap_data["distance_traveled"].values
        dist_norm = dist - dist[0]
        this_len = dist_norm[-1]
        if this_len <= 0:
            continue
        speed = lap_data[speed_col].values * 3.6

        min_speeds = []
        for cf in corner_fracs:
            target = cf * this_len
            idx = np.argmin(np.abs(dist_norm - target))
            win_start = max(0, idx - sample_window)
            win_end = min(len(speed), idx + sample_window)
            min_speeds.append(float(np.min(speed[win_start:win_end])))

        is_best = lap == best_lap
        fig.add_trace(go.Bar(
            x=corner_names,
            y=min_speeds,
            name=f"Lap {int(lap)}",
            marker_color="#e74c3c" if is_best else None,
            opacity=1.0 if is_best else 0.6,
        ))

    fig.update_layout(
        barmode="group",
        title="Min Speed per Corner (All Clean Laps)",
        xaxis_title="Corner",
        yaxis_title="Min Speed (km/h)",
        template="plotly_white",
        height=400,
    )

    return fig
