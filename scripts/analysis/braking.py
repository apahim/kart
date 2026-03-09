"""Braking and acceleration zone analysis."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scripts.analysis.utils import project_to_meters, detect_corners_with_positions, add_wind_arrow


def create_braking_track_map(df, best_lap=None, weather=None, track_corners=None):
    """Create a track map colored by longitudinal G (braking/acceleration)."""
    lat_col = lon_col = None
    for col in df.columns:
        cl = col.lower()
        if "latitude" in cl:
            lat_col = col
        if "longitude" in cl:
            lon_col = col

    if not lat_col or not lon_col:
        return None
    if "longitudinal_acc" not in df.columns:
        return None

    if best_lap is not None and "lap_number" in df.columns:
        plot_df = df[df["lap_number"] == best_lap].copy()
    else:
        plot_df = df.copy()

    plot_df = plot_df.dropna(subset=[lat_col, lon_col, "longitudinal_acc"])

    x_m, y_m = project_to_meters(plot_df[lat_col].values, plot_df[lon_col].values)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_m,
        y=y_m,
        mode="markers",
        marker=dict(
            size=5,
            color=plot_df["longitudinal_acc"],
            colorscale="RdYlGn",
            cmid=0,
            colorbar=dict(title="Long. G"),
            showscale=True,
        ),
        hovertemplate="Long G: %{marker.color:.2f}<extra></extra>",
    ))

    corners_info = detect_corners_with_positions(df, best_lap=best_lap, track_corners=track_corners)
    if corners_info:
        lat_mean = np.mean(plot_df[lat_col].values)
        lon_mean = np.mean(plot_df[lon_col].values)
        lat_mean_rad = np.radians(lat_mean)
        for corner in corners_info:
            if "lat" in corner and "lon" in corner:
                cx = (corner["lon"] - lon_mean) * 111320 * np.cos(lat_mean_rad)
                cy = (corner["lat"] - lat_mean) * 110540
                fig.add_annotation(
                    x=cx, y=cy,
                    text=f"<b>{corner['label']}</b>",
                    showarrow=False,
                    font=dict(size=11, color="black"),
                    bgcolor="rgba(255,255,255,0.7)",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=2,
                )

    title = f"Braking & Acceleration (Lap {best_lap})" if best_lap else "Braking & Acceleration"
    axis_opts = dict(showticklabels=False, showgrid=False, zeroline=False, title="")
    fig.update_layout(
        title=title,
        xaxis=dict(scaleanchor="y", scaleratio=1, **axis_opts),
        yaxis=axis_opts,
        template="plotly_white",
        height=500,
    )

    add_wind_arrow(fig, weather)

    return fig


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
