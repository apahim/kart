"""Braking and acceleration zone analysis."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scripts.analysis.utils import detect_corners_with_positions, wind_data
from scripts.analysis.track_map import _get_coords, _corners_list


def create_braking_track_map(df, best_lap=None, weather=None, track_corners=None):
    """Prepare braking track map data for MapKit JS rendering."""
    lat_col, lon_col = _get_coords(df)
    if not lat_col or not lon_col:
        return None
    if "longitudinal_acc" not in df.columns:
        return None

    if best_lap is not None and "lap_number" in df.columns:
        plot_df = df[df["lap_number"] == best_lap].copy()
    else:
        plot_df = df.copy()

    plot_df = plot_df.dropna(subset=[lat_col, lon_col, "longitudinal_acc"])
    long_g = plot_df["longitudinal_acc"].values

    title = f"Braking & Acceleration (Lap {best_lap})" if best_lap else "Braking & Acceleration"

    return {
        "title": title,
        "lat": [round(float(v), 6) for v in plot_df[lat_col].values],
        "lon": [round(float(v), 6) for v in plot_df[lon_col].values],
        "values": [round(float(v), 3) for v in long_g],
        "colorscale": "RdYlGn",
        "colorbar": {
            "title": "Long. G",
            "min": round(float(np.nanmin(long_g)), 2),
            "max": round(float(np.nanmax(long_g)), 2),
        },
        "cmid": 0.0,
        "corners": _corners_list(df, best_lap=best_lap, track_corners=track_corners),
        "wind": wind_data(weather),
    }


def create_all_laps_braking_map(df, clean_laps, weather=None, track_corners=None):
    """Generate braking track map data for each clean lap.

    Returns dict {lap_number: map_data_dict}.
    """
    result = {}
    for lap in clean_laps:
        data = create_braking_track_map(df, best_lap=lap, weather=weather, track_corners=track_corners)
        if data is not None:
            result[lap] = data
    return result


def create_braking_consistency_chart(df, laptimes_df=None, time_col="seconds", track_corners=None):
    """Box plot of braking point distance per corner across all laps.

    Large spread = inconsistent braking = easy time gain.
    """
    if "longitudinal_acc" not in df.columns or "lap_number" not in df.columns:
        return None
    if "distance_traveled" not in df.columns:
        return None

    from scripts.analysis.outliers import detect_outliers
    from scripts.analysis.corners import detect_corners

    # Find best lap for corner detection
    best_lap = None
    if laptimes_df is not None and time_col in laptimes_df.columns:
        clean_df, _ = detect_outliers(laptimes_df, time_col=time_col)
        best_lap = int(clean_df.loc[clean_df[time_col].idxmin(), "lap"])

    corners, ref_lap_data = detect_corners(df, best_lap=best_lap, track_corners=track_corners)
    if corners is None or len(corners) == 0:
        return None

    corner_names = [track_corners[i]["name"] if track_corners and i < len(track_corners) else f"T{i + 1}" for i in range(len(corners))]

    # Get reference distances for each corner
    if "distance_traveled" not in ref_lap_data.columns:
        return None
    ref_dist = ref_lap_data["distance_traveled"].values
    ref_dist_norm = ref_dist - ref_dist[0]
    lap_length = ref_dist_norm[-1]
    if lap_length <= 0:
        return None

    # Corner positions as fraction of lap
    corner_fracs = [ref_dist_norm[c] / lap_length for c in corners]

    clean_laps = set(clean_df["lap"].astype(int)) if laptimes_df is not None else None
    laps = sorted(df["lap_number"].dropna().unique())
    laps = [l for l in laps if l > 0 and (clean_laps is None or l in clean_laps)]

    # For each lap, find braking points near each corner
    braking_threshold = -0.15
    corner_braking_distances = {i: [] for i in range(len(corners))}

    for lap in laps:
        lap_data = df[df["lap_number"] == lap].copy()
        if len(lap_data) < 50:
            continue

        dist = lap_data["distance_traveled"].values
        dist_norm = dist - dist[0]
        this_lap_length = dist_norm[-1]
        if this_lap_length <= 0:
            continue

        long_acc = lap_data["longitudinal_acc"].values

        for ci, corner_frac in enumerate(corner_fracs):
            # Search window: 30% of lap before the corner to 5% after
            search_start = corner_frac - 0.30
            search_end = corner_frac + 0.05
            frac = dist_norm / this_lap_length

            mask = (frac >= search_start) & (frac <= search_end)
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue

            # Find where braking starts (first point below threshold going toward corner)
            braking_indices = indices[long_acc[indices] < braking_threshold]
            if len(braking_indices) > 0:
                brake_start = braking_indices[0]
                corner_braking_distances[ci].append(dist_norm[brake_start])

    # Create box plot
    fig = go.Figure()
    labels = corner_names

    for ci in range(len(corners)):
        if corner_braking_distances[ci]:
            fig.add_trace(go.Box(
                y=corner_braking_distances[ci],
                name=labels[ci],
                boxmean=True,
                marker_color="#e74c3c",
            ))

    fig.update_layout(
        title="Braking Consistency by Corner",
        xaxis_title="Corner",
        yaxis_title="Braking Point Distance (m)",
        template="plotly_white",
        height=400,
    )

    return fig


def create_brake_release_chart(df, laptimes_df=None, time_col="seconds", track_corners=None):
    """Box plots of brake application distance and trail braking depth per corner."""
    if "longitudinal_acc" not in df.columns or "lap_number" not in df.columns:
        return None
    if "distance_traveled" not in df.columns:
        return None

    from scripts.analysis.outliers import detect_outliers
    from scripts.analysis.corners import detect_corners

    best_lap = None
    clean_laps = None
    if laptimes_df is not None and time_col in laptimes_df.columns:
        clean_df, _ = detect_outliers(laptimes_df, time_col=time_col)
        best_lap = int(clean_df.loc[clean_df[time_col].idxmin(), "lap"])
        clean_laps = set(clean_df["lap"].astype(int))

    corners, ref_lap_data = detect_corners(df, best_lap=best_lap, track_corners=track_corners)
    if corners is None or len(corners) == 0:
        return None
    if "distance_traveled" not in ref_lap_data.columns:
        return None

    ref_dist = ref_lap_data["distance_traveled"].values
    ref_dist_norm = ref_dist - ref_dist[0]
    lap_length = ref_dist_norm[-1]
    if lap_length <= 0:
        return None
    corner_fracs = [ref_dist_norm[c] / lap_length for c in corners]
    n_corners = len(corners)

    laps = sorted(df["lap_number"].dropna().unique())
    laps = [l for l in laps if l > 0 and (clean_laps is None or l in clean_laps)]

    brake_threshold = -0.15
    release_threshold = -0.05
    brake_distances = {i: [] for i in range(n_corners)}
    trail_depths = {i: [] for i in range(n_corners)}

    for lap in laps:
        lap_data = df[df["lap_number"] == lap].copy().reset_index(drop=True)
        if len(lap_data) < 50:
            continue
        dist = lap_data["distance_traveled"].values
        dist_norm = dist - dist[0]
        this_len = dist_norm[-1]
        if this_len <= 0:
            continue
        long_acc = lap_data["longitudinal_acc"].values
        frac = dist_norm / this_len

        for ci, cf in enumerate(corner_fracs):
            # Search window around corner
            search_start = cf - 0.25
            search_end = cf + 0.10
            mask = (frac >= search_start) & (frac <= search_end)
            indices = np.where(mask)[0]
            if len(indices) < 5:
                continue

            # Find braking zone
            braking = long_acc[indices] < brake_threshold
            if not np.any(braking):
                continue

            brake_indices = indices[braking]
            brake_start_idx = brake_indices[0]
            brake_end_idx = brake_indices[-1]

            # Brake application distance
            b_dist = dist_norm[brake_end_idx] - dist_norm[brake_start_idx]
            brake_distances[ci].append(b_dist)

            # Trail braking: braking that continues past the corner apex
            corner_idx = np.argmin(np.abs(frac - cf))
            past_apex = brake_indices[brake_indices >= corner_idx]
            if len(past_apex) > 0:
                trail = dist_norm[past_apex[-1]] - dist_norm[corner_idx]
                trail_depths[ci].append(trail)
            else:
                trail_depths[ci].append(0.0)

    # Check we have data
    has_data = any(brake_distances[i] for i in range(n_corners))
    if not has_data:
        return None

    labels = [track_corners[i]["name"] if track_corners and i < len(track_corners) else f"T{i + 1}" for i in range(n_corners)]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Brake Application Distance", "Trail Braking Depth"],
                        vertical_spacing=0.12)

    for ci in range(n_corners):
        if brake_distances[ci]:
            fig.add_trace(go.Box(
                y=brake_distances[ci], name=labels[ci],
                boxmean=True, marker_color="#e74c3c",
                legendgroup=labels[ci], showlegend=True,
            ), row=1, col=1)
        if trail_depths[ci]:
            fig.add_trace(go.Box(
                y=trail_depths[ci], name=labels[ci],
                boxmean=True, marker_color="#3498db",
                legendgroup=labels[ci], showlegend=False,
            ), row=2, col=1)

    fig.update_yaxes(title_text="Distance (m)", row=1, col=1)
    fig.update_yaxes(title_text="Depth (m)", row=2, col=1)
    fig.update_layout(
        title="Brake Release Timing Analysis",
        template="plotly_white",
        height=600,
    )

    return fig
