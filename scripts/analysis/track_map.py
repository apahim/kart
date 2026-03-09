"""Track map visualizations using Plotly."""

import numpy as np
import plotly.graph_objects as go

from scripts.analysis.utils import project_to_meters, detect_corners_with_positions, add_wind_arrow


def _get_coords(df):
    """Extract latitude and longitude columns from telemetry."""
    lat_col = lon_col = None
    for col in df.columns:
        cl = col.lower()
        if "latitude" in cl or col == "latitude":
            lat_col = col
        if "longitude" in cl or col == "longitude":
            lon_col = col
    return lat_col, lon_col


def _add_corner_annotations(fig, corners_info, lat, lon):
    """Add corner label annotations to a track map figure."""
    if not corners_info:
        return

    x_m, y_m = project_to_meters(lat, lon)

    for corner in corners_info:
        if "lat" in corner and "lon" in corner:
            cx, cy = project_to_meters(
                np.array([corner["lat"]]), np.array([corner["lon"]])
            )
            # Use same projection center as the main data
            lat_mean = np.mean(lat)
            lon_mean = np.mean(lon)
            lat_mean_rad = np.radians(lat_mean)
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


def _hide_axes(fig):
    """Hide axis labels, ticks, and gridlines for clean track maps."""
    axis_opts = dict(
        showticklabels=False, showgrid=False, zeroline=False,
        title="",
    )
    fig.update_layout(xaxis=axis_opts, yaxis=axis_opts)


def create_speed_track_map(df, best_lap=None, weather=None, track_corners=None):
    """Create a track map colored by speed using meters projection."""
    lat_col, lon_col = _get_coords(df)
    if not lat_col or not lon_col:
        return None

    speed_col = "speed_gps" if "speed_gps" in df.columns else "speed"
    if speed_col not in df.columns:
        return None

    if best_lap is not None and "lap_number" in df.columns:
        plot_df = df[df["lap_number"] == best_lap].copy()
    else:
        plot_df = df.copy()

    plot_df = plot_df.dropna(subset=[lat_col, lon_col, speed_col])
    speed_kmh = plot_df[speed_col] * 3.6

    x_m, y_m = project_to_meters(plot_df[lat_col].values, plot_df[lon_col].values)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_m,
        y=y_m,
        mode="markers",
        marker=dict(
            size=5,
            color=speed_kmh,
            colorscale="RdYlGn",
            colorbar=dict(title="km/h"),
            showscale=True,
        ),
        hovertemplate="Speed: %{marker.color:.1f} km/h<extra></extra>",
    ))

    corners_info = detect_corners_with_positions(df, best_lap=best_lap, track_corners=track_corners)
    _add_corner_annotations(fig, corners_info, plot_df[lat_col].values, plot_df[lon_col].values)

    fig.update_layout(
        title="Track Map - Speed",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(),
        template="plotly_white",
        height=500,
    )
    _hide_axes(fig)
    add_wind_arrow(fig, weather)

    return fig


def create_lateral_g_track_map(df, best_lap=None, weather=None, track_corners=None):
    """Create a track map colored by lateral G-force using meters projection."""
    lat_col, lon_col = _get_coords(df)
    if not lat_col or not lon_col:
        return None

    if "lateral_acc" not in df.columns:
        return None

    if best_lap is not None and "lap_number" in df.columns:
        plot_df = df[df["lap_number"] == best_lap].copy()
    else:
        plot_df = df.copy()

    plot_df = plot_df.dropna(subset=[lat_col, lon_col, "lateral_acc"])

    x_m, y_m = project_to_meters(plot_df[lat_col].values, plot_df[lon_col].values)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_m,
        y=y_m,
        mode="markers",
        marker=dict(
            size=5,
            color=plot_df["lateral_acc"],
            colorscale="RdYlBu",
            cmid=0,
            colorbar=dict(title="Lateral G"),
            showscale=True,
        ),
        hovertemplate="Lateral G: %{marker.color:.2f}<extra></extra>",
    ))

    corners_info = detect_corners_with_positions(df, best_lap=best_lap, track_corners=track_corners)
    _add_corner_annotations(fig, corners_info, plot_df[lat_col].values, plot_df[lon_col].values)

    fig.update_layout(
        title="Track Map - Lateral G",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(),
        template="plotly_white",
        height=500,
    )
    _hide_axes(fig)
    add_wind_arrow(fig, weather)

    return fig
