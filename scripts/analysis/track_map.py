"""Track map data preparation for MapKit JS rendering."""

import numpy as np

from scripts.analysis.utils import detect_corners_with_positions, wind_data


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


def _corners_list(df, best_lap=None, track_corners=None):
    """Get corner label data as a list of dicts with label, lat, lon."""
    corners_info = detect_corners_with_positions(df, best_lap=best_lap, track_corners=track_corners)
    return [
        {"label": c["label"], "lat": c["lat"], "lon": c["lon"]}
        for c in corners_info
        if "lat" in c and "lon" in c
    ]


def create_speed_track_map(df, best_lap=None, weather=None, track_corners=None):
    """Prepare speed track map data for MapKit JS rendering."""
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
    speed_kmh = (plot_df[speed_col] * 3.6).values

    return {
        "title": "Track Map - Speed",
        "lat": [round(float(v), 6) for v in plot_df[lat_col].values],
        "lon": [round(float(v), 6) for v in plot_df[lon_col].values],
        "values": [round(float(v), 1) for v in speed_kmh],
        "colorscale": "RdYlGn",
        "colorbar": {
            "title": "km/h",
            "min": round(float(np.nanmin(speed_kmh)), 1),
            "max": round(float(np.nanmax(speed_kmh)), 1),
        },
        "cmid": None,
        "corners": _corners_list(df, best_lap=best_lap, track_corners=track_corners),
        "wind": wind_data(weather),
    }


def create_sector_delta_map(df, best_lap, sector_data, weather=None, track_corners=None):
    """Prepare sector delta track map data for MapKit JS rendering."""
    if sector_data is None:
        return None

    sector_boundaries = sector_data.get("sector_boundaries")
    best_sectors = sector_data.get("best_sectors")
    sector_times_dict = sector_data.get("sector_times")
    if not sector_boundaries or not best_sectors or not sector_times_dict:
        return None
    if best_lap not in sector_times_dict:
        return None

    lat_col, lon_col = _get_coords(df)
    if not lat_col or not lon_col:
        return None

    if "lap_number" not in df.columns or "distance_traveled" not in df.columns:
        return None

    lap_data = df[df["lap_number"] == best_lap].copy().reset_index(drop=True)
    lap_data = lap_data.dropna(subset=[lat_col, lon_col])
    if len(lap_data) < 20:
        return None

    dist = lap_data["distance_traveled"].values
    dist_norm = dist - dist[0]
    lap_length = dist_norm[-1]
    if lap_length <= 0:
        return None
    frac = dist_norm / lap_length

    best_lap_sectors = sector_times_dict[best_lap]
    n_sectors = len(best_sectors)

    deltas = np.zeros(len(frac))
    for i in range(len(frac)):
        for si in range(n_sectors):
            if frac[i] >= sector_boundaries[si] and frac[i] <= sector_boundaries[si + 1]:
                deltas[i] = best_lap_sectors[si] - best_sectors[si]
                break

    delta_max = max(abs(float(deltas.max())), 0.01)

    return {
        "title": "Track Map - Sector Delta (Best Lap vs Best Sectors)",
        "lat": [round(float(v), 6) for v in lap_data[lat_col].values],
        "lon": [round(float(v), 6) for v in lap_data[lon_col].values],
        "values": [round(float(v), 3) for v in deltas],
        "colorscale": "RdYlGn_r",
        "colorbar": {
            "title": "Delta (s)",
            "min": 0.0,
            "max": round(delta_max, 3),
        },
        "cmid": None,
        "corners": _corners_list(df, best_lap=best_lap, track_corners=track_corners),
        "wind": wind_data(weather),
    }


def create_all_laps_speed_map(df, clean_laps, weather=None, track_corners=None):
    """Generate speed track map data for each clean lap.

    Returns dict {lap_number: map_data_dict}.
    """
    result = {}
    for lap in clean_laps:
        data = create_speed_track_map(df, best_lap=lap, weather=weather, track_corners=track_corners)
        if data is not None:
            result[lap] = data
    return result


def create_all_laps_sector_delta_map(df, clean_laps, sector_data, weather=None, track_corners=None):
    """Generate sector delta map data for each clean lap.

    Returns dict {lap_number: map_data_dict}.
    """
    result = {}
    for lap in clean_laps:
        data = create_sector_delta_map(df, lap, sector_data, weather=weather, track_corners=track_corners)
        if data is not None:
            result[lap] = data
    return result


def create_lateral_g_track_map(df, best_lap=None, weather=None, track_corners=None):
    """Prepare lateral G track map data for MapKit JS rendering."""
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
    lat_g = plot_df["lateral_acc"].values

    return {
        "title": "Track Map - Lateral G",
        "lat": [round(float(v), 6) for v in plot_df[lat_col].values],
        "lon": [round(float(v), 6) for v in plot_df[lon_col].values],
        "values": [round(float(v), 3) for v in lat_g],
        "colorscale": "RdYlBu",
        "colorbar": {
            "title": "Lateral G",
            "min": round(float(np.nanmin(lat_g)), 2),
            "max": round(float(np.nanmax(lat_g)), 2),
        },
        "cmid": 0.0,
        "corners": _corners_list(df, best_lap=best_lap, track_corners=track_corners),
        "wind": wind_data(weather),
    }
