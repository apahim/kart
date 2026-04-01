"""Shared utilities for kart racing analysis."""

import base64
import json
import re
import struct
import traceback

import numpy as np
import plotly.io as pio

# Matches plotly's binary array format: {"dtype":"f8","bdata":"base64..."}
_BDATA_RE = re.compile(r'\{"dtype":"([^"]+)","bdata":"([^"]+)"\}')

_DTYPE_FMT = {"f8": "d", "f4": "f", "i4": "i", "i2": "h", "i1": "b",
              "u4": "I", "u2": "H", "u1": "B"}


def _decode_bdata(match):
    """Decode a plotly bdata object back into a JSON array."""
    try:
        dtype, bdata = match.group(1), match.group(2)
        fmt = _DTYPE_FMT.get(dtype)
        if fmt is None:
            return match.group(0)
        # Plotly HTML-escapes / as \u002f in base64 strings — undo that
        bdata = bdata.replace("\\u002f", "/")
        raw = base64.b64decode(bdata)
        count = len(raw) // struct.calcsize(fmt)
        values = list(struct.unpack(f"<{count}{fmt}", raw))
        return json.dumps(values)
    except Exception:
        return match.group(0)


def apply_mobile_layout(fig):
    """Apply mobile-friendly defaults to a Plotly figure.

    Tightens margins, shrinks fonts, and enables autosize so charts
    render well on narrow smartphone screens.
    """
    fig.update_layout(
        autosize=True,
        margin=dict(l=40, r=30, t=40, b=40),
        title_font_size=14,
        font_size=11,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font_size=10,
        ),
    )
    return fig


def fig_to_html(fig):
    """Convert a Plotly figure to an embeddable HTML div."""
    if fig is None:
        return None
    apply_mobile_layout(fig)
    html = pio.to_html(
        fig, full_html=False, include_plotlyjs=False,
        config={"responsive": True},
    )
    # Decode any binary bdata arrays back to plain JSON arrays so the
    # chart renders correctly with CDN versions of plotly.js.
    return _BDATA_RE.sub(_decode_bdata, html)


def fig_to_json(fig):
    """Convert a Plotly figure to a JSON-serialisable dict (data + layout).

    Used for charts that are rendered client-side via Plotly.react().
    """
    if fig is None:
        return None
    apply_mobile_layout(fig)
    return json.loads(fig.to_json())


def safe_chart(name, func, *args, **kwargs):
    """Call a chart function, returning None on error instead of crashing."""
    try:
        return fig_to_html(func(*args, **kwargs))
    except Exception as e:
        print(f"Warning: chart '{name}' failed: {e}")
        traceback.print_exc()
        return None


def safe_map_data(name, func, *args, **kwargs):
    """Call a map data function, returning None on error instead of crashing."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"Warning: map data '{name}' failed: {e}")
        traceback.print_exc()
        return None


def wind_data(weather):
    """Extract wind info dict from weather data for map overlays.

    Returns dict with arrow_deg, cardinal, speed_kmh or None.
    """
    if not weather:
        return None
    speed = weather.get("wind_kmh")
    deg = weather.get("wind_direction_deg")
    cardinal = weather.get("wind_direction", "")
    if speed is None or deg is None:
        return None
    return {"arrow_deg": deg, "cardinal": cardinal, "speed_kmh": speed}


def format_laptime(seconds):
    """Format lap time from seconds to M:SS.mmm (e.g., 69.742 -> 1:09.742)."""
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    if minutes > 0:
        return f"{minutes}:{remainder:06.3f}"
    return f"{remainder:.3f}"


def add_wind_arrow(fig, weather):
    """Add a wind direction arrow annotation to a track map figure.

    Args:
        fig: Plotly figure to annotate.
        weather: Dict with wind_kmh, wind_direction_deg, wind_direction keys.
                 No-op if None or missing keys.
    """
    if not weather or not fig:
        return
    speed = weather.get("wind_kmh")
    deg = weather.get("wind_direction_deg")
    cardinal = weather.get("wind_direction", "")
    if speed is None or deg is None:
        return

    # Meteorological convention: deg is direction wind comes FROM (0=N, clockwise).
    # Arrow shows where wind comes FROM: N wind → arrow points up.
    # Plotly textangle rotates text clockwise; ↑ at textangle=0 points up (north).
    plotly_angle = deg

    fig.add_annotation(
        text="<b>↑</b>",
        x=0.02, y=0.08,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=18, color="#333"),
        textangle=plotly_angle,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#999",
        borderwidth=1,
        borderpad=4,
    )
    fig.add_annotation(
        text=f"{cardinal} {speed} km/h",
        x=0.02, y=0.02,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=10, color="#666"),
        bgcolor="rgba(255,255,255,0.8)",
        borderpad=2,
    )


def project_to_meters(lat, lon, center=None):
    """Project lat/lon to local XY meters using equirectangular projection.

    Accurate enough for a ~400m kart track.

    Args:
        lat: Latitude values.
        lon: Longitude values.
        center: Optional (lat_mean, lon_mean) tuple to use as projection origin.
                If None, auto-centers on the mean of the input data.

    Returns:
        Tuple of (x_meters, y_meters) as numpy arrays.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    if center is not None:
        lat_mean, lon_mean = center
    else:
        lat_mean = np.mean(lat)
        lon_mean = np.mean(lon)
    lat_mean_rad = np.radians(lat_mean)

    x = (lon - lon_mean) * 111320 * np.cos(lat_mean_rad)
    y = (lat - lat_mean) * 110540

    return x, y


def detect_corners_with_positions(df, best_lap=None, track_corners=None):
    """Detect corners and enrich with position data.

    Returns a list of dicts with keys:
        label, index, lat, lon, x, y, distance, min_speed_kmh
    Returns empty list if detection fails.
    """
    from scripts.analysis.corners import detect_corners

    corners, lap_data = detect_corners(df, best_lap=best_lap, track_corners=track_corners)
    if corners is None or len(corners) == 0:
        return []

    speed_col = "speed_gps" if "speed_gps" in df.columns else "speed"
    lat_col = lon_col = None
    for col in df.columns:
        cl = col.lower()
        if "latitude" in cl:
            lat_col = col
        if "longitude" in cl:
            lon_col = col

    result = []
    for i, idx in enumerate(corners):
        label = track_corners[i]["name"] if track_corners and i < len(track_corners) else f"T{i + 1}"
        corner = {
            "label": label,
            "index": int(idx),
            "min_speed_kmh": float(lap_data[speed_col].iloc[idx] * 3.6),
        }
        if lat_col and lon_col:
            corner["lat"] = float(lap_data[lat_col].iloc[idx])
            corner["lon"] = float(lap_data[lon_col].iloc[idx])
        if "distance_traveled" in lap_data.columns:
            dist = lap_data["distance_traveled"].values
            corner["distance"] = float(dist[idx] - dist[0])
        result.append(corner)

    return result
