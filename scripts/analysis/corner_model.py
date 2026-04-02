"""Unified Corner Object model — per-corner, per-lap metrics with classification."""

from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np

from scripts.analysis.corners import detect_corners
from scripts.analysis.outliers import detect_outliers


@dataclass
class CornerRecord:
    """Per-corner, per-lap metrics."""

    corner_name: str
    corner_index: int
    lap: int
    # Position
    lat: Optional[float] = None
    lon: Optional[float] = None
    distance: float = 0.0
    corner_frac: float = 0.0
    # Speed (km/h)
    entry_speed: float = 0.0
    min_speed: float = 0.0
    exit_speed: float = 0.0
    # Time
    time_loss: float = 0.0  # seconds vs best lap (positive = lost time)
    # Braking
    braking_point: Optional[float] = None  # distance from lap start where braking begins
    braking_distance: Optional[float] = None  # meters of braking before apex
    trail_braking_depth: Optional[float] = None  # meters past apex still under braking
    # Classification
    root_cause: Optional[str] = None  # "entry" | "mid" | "exit" | None

    def to_dict(self):
        return asdict(self)


@dataclass
class CornerSummary:
    """Cross-lap aggregation for one corner."""

    corner_name: str
    corner_index: int
    lat: Optional[float] = None
    lon: Optional[float] = None
    distance: float = 0.0
    archetype: str = "flow"  # "entry-dependent" | "exit-dependent" | "flow"
    # Aggregated metrics
    avg_time_loss: float = 0.0
    total_time_loss: float = 0.0
    best_min_speed: float = 0.0
    avg_min_speed: float = 0.0
    std_min_speed: float = 0.0
    best_entry_speed: float = 0.0
    best_exit_speed: float = 0.0
    braking_spread: Optional[float] = None  # IQR of braking points (meters)
    dominant_root_cause: Optional[str] = None
    records: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d["records"] = [r.to_dict() if isinstance(r, CornerRecord) else r for r in self.records]
        return d


def classify_root_cause(record, best_record):
    """Classify why a corner cost time: entry, mid, or exit problem.

    Compares speed deltas at each phase against the best lap's corner record.
    Returns None if time loss is negligible (< 0.03s).
    """
    if record.time_loss < 0.03:
        return None

    entry_delta = record.entry_speed - best_record.entry_speed
    mid_delta = record.min_speed - best_record.min_speed
    exit_delta = record.exit_speed - best_record.exit_speed

    # The most negative delta (biggest speed deficit) is the root cause.
    # Use absolute values — we want the phase where speed was most below best.
    deficits = {
        "entry": -entry_delta if entry_delta < 0 else 0,
        "mid": -mid_delta if mid_delta < 0 else 0,
        "exit": -exit_delta if exit_delta < 0 else 0,
    }

    if all(v == 0 for v in deficits.values()):
        return None

    return max(deficits, key=deficits.get)


def classify_archetype(best_record, next_corner_distance=None, median_inter_corner=None):
    """Classify a corner's type from the best-lap speed ratios.

    - entry-dependent: High entry/min ratio (braking technique matters)
    - exit-dependent: High exit/min ratio AND followed by a long straight
    - flow: Neither dominant (momentum/line consistency)
    """
    if best_record.min_speed <= 0:
        return "flow"

    entry_ratio = best_record.entry_speed / best_record.min_speed
    exit_ratio = best_record.exit_speed / best_record.min_speed

    # Exit-dependent: high exit ratio and followed by a long straight
    if exit_ratio > 1.3:
        if next_corner_distance is not None and median_inter_corner is not None:
            if next_corner_distance > median_inter_corner:
                return "exit-dependent"
        # If we don't have distance info, still classify if ratio is very high
        if exit_ratio > 1.5:
            return "exit-dependent"

    if entry_ratio > 1.4:
        return "entry-dependent"

    return "flow"


def build_corner_analysis(df, laptimes_df, time_col="seconds", track_corners=None):
    """Build unified corner objects for all clean laps.

    Detects corners once, iterates laps once, computes all metrics in a single pass.

    Returns dict with:
        records: dict[int, list[CornerRecord]]  — lap number -> corner records
        summaries: list[CornerSummary]  — one per corner, sorted by avg_time_loss
        best_lap: int
        median_lap: int
    Returns None if analysis cannot be performed.
    """
    speed_col = "speed_gps" if "speed_gps" in df.columns else "speed"
    if speed_col not in df.columns or "lap_number" not in df.columns:
        return None
    if "distance_traveled" not in df.columns:
        return None
    if laptimes_df is None or time_col not in laptimes_df.columns:
        return None

    clean_df, _ = detect_outliers(laptimes_df, time_col=time_col)
    if len(clean_df) < 2:
        return None

    best_lap = int(clean_df.loc[clean_df[time_col].idxmin(), "lap"])
    clean_laps = sorted(clean_df["lap"].astype(int))

    # Detect corners on the best lap (single detection for entire analysis)
    corners, ref_lap_data = detect_corners(df, best_lap=best_lap, track_corners=track_corners)
    if corners is None or len(corners) == 0:
        return None

    n_corners = len(corners)
    corner_names = [
        track_corners[i]["name"] if track_corners and i < len(track_corners) else f"T{i + 1}"
        for i in range(n_corners)
    ]

    # Reference distances (corner fractions on best lap)
    ref_dist = ref_lap_data["distance_traveled"].values
    ref_dist_norm = ref_dist - ref_dist[0]
    lap_length = ref_dist_norm[-1]
    if lap_length <= 0:
        return None
    corner_fracs = [ref_dist_norm[c] / lap_length for c in corners]

    # Get corner GPS positions from ref lap
    lat_col = lon_col = None
    for col in df.columns:
        cl = col.lower()
        if "latitude" in cl:
            lat_col = col
        if "longitude" in cl:
            lon_col = col

    corner_positions = []
    for i, idx in enumerate(corners):
        pos = {"lat": None, "lon": None}
        if lat_col and lon_col:
            pos["lat"] = float(ref_lap_data[lat_col].iloc[idx])
            pos["lon"] = float(ref_lap_data[lon_col].iloc[idx])
        corner_positions.append(pos)

    # Compute per-lap time loss (each lap vs best lap, direct integration)
    per_lap_time_loss = _compute_per_lap_time_loss_vs_best(
        df, speed_col, corner_fracs, clean_laps, best_lap
    )

    # Check for braking data
    has_braking = "longitudinal_acc" in df.columns
    braking_threshold = -0.15
    sample_window = 20

    # Build records for all clean laps
    all_records = {}  # lap -> [CornerRecord]

    for lap in clean_laps:
        lap_data = df[df["lap_number"] == lap].copy().reset_index(drop=True)
        if len(lap_data) < 50:
            continue

        dist = lap_data["distance_traveled"].values
        dist_norm = dist - dist[0]
        this_len = dist_norm[-1]
        if this_len <= 0:
            continue

        speed = lap_data[speed_col].values * 3.6  # km/h

        lap_records = []
        for ci, cf in enumerate(corner_fracs):
            target_dist = cf * this_len
            idx = np.argmin(np.abs(dist_norm - target_dist))

            # Speed metrics in ±sample_window around corner
            win_start = max(0, idx - sample_window)
            win_end = min(len(speed), idx + sample_window)
            window_speed = speed[win_start:win_end]
            if len(window_speed) == 0:
                continue

            min_local_idx = np.argmin(window_speed)
            min_idx = win_start + min_local_idx
            min_speed = float(speed[min_idx])

            entry_idx = max(0, min_idx - sample_window)
            exit_idx = min(len(speed) - 1, min_idx + sample_window)
            entry_speed = float(speed[entry_idx])
            exit_speed = float(speed[exit_idx])

            # Braking metrics
            braking_point = None
            braking_distance = None
            trail_depth = None

            if has_braking:
                long_acc = lap_data["longitudinal_acc"].values
                # Search window: 30% before corner to 5% after
                frac = dist_norm / this_len
                search_mask = (frac >= cf - 0.30) & (frac <= cf + 0.05)
                search_indices = np.where(search_mask)[0]

                if len(search_indices) > 0:
                    braking_indices = search_indices[long_acc[search_indices] < braking_threshold]
                    if len(braking_indices) > 0:
                        brake_start = braking_indices[0]
                        braking_point = float(dist_norm[brake_start])
                        braking_distance = float(dist_norm[min_idx] - dist_norm[brake_start])
                        if braking_distance < 0:
                            braking_distance = 0.0

                # Trail braking: braking past the apex
                post_apex_mask = (frac >= cf) & (frac <= cf + 0.10)
                post_indices = np.where(post_apex_mask)[0]
                if len(post_indices) > 0:
                    trail_indices = post_indices[long_acc[post_indices] < braking_threshold]
                    if len(trail_indices) > 0:
                        trail_end = trail_indices[-1]
                        trail_depth = float(dist_norm[trail_end] - dist_norm[min_idx])
                        if trail_depth < 0:
                            trail_depth = 0.0

            # Time loss for this corner in this lap
            lap_tl = 0.0
            if per_lap_time_loss and lap in per_lap_time_loss:
                lap_corner_losses = per_lap_time_loss[lap]
                if ci < len(lap_corner_losses):
                    lap_tl = lap_corner_losses[ci]

            record = CornerRecord(
                corner_name=corner_names[ci],
                corner_index=ci,
                lap=lap,
                lat=corner_positions[ci]["lat"],
                lon=corner_positions[ci]["lon"],
                distance=float(ref_dist_norm[corners[ci]]),
                corner_frac=cf,
                entry_speed=entry_speed,
                min_speed=min_speed,
                exit_speed=exit_speed,
                time_loss=lap_tl,
                braking_point=braking_point,
                braking_distance=braking_distance,
                trail_braking_depth=trail_depth,
            )
            lap_records.append(record)

        all_records[lap] = lap_records

    if not all_records:
        return None

    # Classify root causes (compare each lap's records against best lap)
    best_records = all_records.get(best_lap, [])
    if best_records:
        best_by_corner = {r.corner_index: r for r in best_records}
        for lap, records in all_records.items():
            if lap == best_lap:
                continue
            for record in records:
                if record.corner_index in best_by_corner:
                    record.root_cause = classify_root_cause(record, best_by_corner[record.corner_index])

    # Compute inter-corner distances for archetype classification
    corner_distances = [float(ref_dist_norm[c]) for c in corners]
    inter_corner_dists = []
    for i in range(len(corner_distances) - 1):
        inter_corner_dists.append(corner_distances[i + 1] - corner_distances[i])
    # Wrap-around distance for last corner to first
    if len(corner_distances) >= 2:
        inter_corner_dists.append(lap_length - corner_distances[-1] + corner_distances[0])
    median_inter_corner = float(np.median(inter_corner_dists)) if inter_corner_dists else None

    # Build summaries
    summaries = []
    for ci in range(n_corners):
        corner_records = []
        for lap_records in all_records.values():
            for r in lap_records:
                if r.corner_index == ci:
                    corner_records.append(r)

        if not corner_records:
            continue

        min_speeds = [r.min_speed for r in corner_records]
        entry_speeds = [r.entry_speed for r in corner_records]
        exit_speeds = [r.exit_speed for r in corner_records]
        time_losses = [r.time_loss for r in corner_records]

        # Braking spread (IQR of braking points)
        braking_points = [r.braking_point for r in corner_records if r.braking_point is not None]
        braking_spread = None
        if len(braking_points) >= 4:
            q75, q25 = np.percentile(braking_points, [75, 25])
            braking_spread = round(float(q75 - q25), 1)

        # Dominant root cause
        causes = [r.root_cause for r in corner_records if r.root_cause is not None]
        dominant_cause = None
        if causes:
            from collections import Counter
            dominant_cause = Counter(causes).most_common(1)[0][0]

        # Archetype
        next_dist = inter_corner_dists[ci] if ci < len(inter_corner_dists) else None
        best_rec = best_by_corner.get(ci) if best_records else None
        archetype = "flow"
        if best_rec:
            archetype = classify_archetype(best_rec, next_dist, median_inter_corner)

        summary = CornerSummary(
            corner_name=corner_names[ci],
            corner_index=ci,
            lat=corner_positions[ci]["lat"],
            lon=corner_positions[ci]["lon"],
            distance=float(ref_dist_norm[corners[ci]]),
            archetype=archetype,
            avg_time_loss=float(np.mean(time_losses)),
            total_time_loss=float(np.sum(time_losses)),
            best_min_speed=float(np.max(min_speeds)),
            avg_min_speed=float(np.mean(min_speeds)),
            std_min_speed=float(np.std(min_speeds)),
            best_entry_speed=float(np.max(entry_speeds)),
            best_exit_speed=float(np.max(exit_speeds)),
            braking_spread=braking_spread,
            dominant_root_cause=dominant_cause,
            records=corner_records,
        )
        summaries.append(summary)

    # Sort summaries by avg_time_loss (worst = most positive first)
    summaries.sort(key=lambda s: s.avg_time_loss, reverse=True)

    return {
        "records": all_records,
        "summaries": summaries,
        "best_lap": best_lap,
        "corner_names": corner_names,
    }


def _compute_per_lap_time_loss_vs_best(df, speed_col, corner_fracs, clean_laps, best_lap, n_points=500):
    """Compute per-corner time loss for each lap vs the best lap.

    Uses direct distance-based time integration — compares each lap's speed
    profile against the best lap at each corner zone.

    Returns dict {lap: [delta_corner_0, delta_corner_1, ...]} or None.
    Positive delta = this lap lost time vs best. Negative = gained time.
    """
    n_corners = len(corner_fracs)
    if n_corners < 2:
        return None

    # Get best lap speed profile
    best_data = df[df["lap_number"] == best_lap].copy().reset_index(drop=True)
    if len(best_data) < 50:
        return None
    best_dist = best_data["distance_traveled"].values
    best_dist_norm = best_dist - best_dist[0]
    best_len = best_dist_norm[-1]
    if best_len <= 0:
        return None
    best_speed = np.maximum(best_data[speed_col].values, 0.003)  # clamp ~0.01 km/h

    # Corner zone boundaries (midpoints between adjacent corners, as fractions)
    boundaries_frac = [0.0]
    for i in range(n_corners - 1):
        boundaries_frac.append((corner_fracs[i] + corner_fracs[i + 1]) / 2)
    boundaries_frac.append(1.0)

    # Interpolate best lap to common grid
    dist_grid = np.linspace(0, best_len, n_points)
    best_interp = np.interp(dist_grid, best_dist_norm, best_speed)

    # Pre-compute best lap time per zone
    dd = np.diff(dist_grid)
    best_cum_time = np.insert(np.cumsum(dd / best_interp[:-1]), 0, 0.0)

    best_zone_times = []
    for i in range(n_corners):
        entry_idx = int(boundaries_frac[i] * (n_points - 1))
        exit_idx = int(boundaries_frac[i + 1] * (n_points - 1))
        best_zone_times.append(best_cum_time[exit_idx] - best_cum_time[entry_idx])

    per_lap = {}
    for lap in clean_laps:
        if lap == best_lap:
            per_lap[lap] = [0.0] * n_corners
            continue

        lap_data = df[df["lap_number"] == lap].copy().reset_index(drop=True)
        if len(lap_data) < 50:
            continue
        lap_dist = lap_data["distance_traveled"].values
        lap_dist_norm = lap_dist - lap_dist[0]
        lap_len = lap_dist_norm[-1]
        if lap_len <= 0:
            continue
        lap_speed = np.maximum(lap_data[speed_col].values, 0.003)

        # Interpolate this lap to the same distance grid (using best lap's length)
        max_dist = min(best_len, lap_len)
        lap_grid = np.linspace(0, max_dist, n_points)
        lap_interp = np.interp(lap_grid, lap_dist_norm, lap_speed)

        lap_dd = np.diff(lap_grid)
        lap_cum_time = np.insert(np.cumsum(lap_dd / lap_interp[:-1]), 0, 0.0)

        deltas = []
        for i in range(n_corners):
            entry_idx = int(boundaries_frac[i] * (n_points - 1))
            exit_idx = int(boundaries_frac[i + 1] * (n_points - 1))
            lap_zone_time = lap_cum_time[exit_idx] - lap_cum_time[entry_idx]
            # Positive = this lap spent more time (lost time vs best)
            delta = lap_zone_time - best_zone_times[i]
            deltas.append(float(delta))

        per_lap[lap] = deltas

    return per_lap if per_lap else None


def build_corner_map_data(df, corner_analysis):
    """Build track map data for the Corner Analysis tab.

    Shows the best lap's track outline colored by speed, with corner labels
    annotated with avg time loss.

    Returns a map data dict compatible with setupTrackMap(), or None.
    """
    if corner_analysis is None:
        return None

    speed_col = "speed_gps" if "speed_gps" in df.columns else "speed"
    if speed_col not in df.columns or "lap_number" not in df.columns:
        return None

    lat_col = lon_col = None
    for col in df.columns:
        cl = col.lower()
        if "latitude" in cl:
            lat_col = col
        if "longitude" in cl:
            lon_col = col
    if not lat_col or not lon_col:
        return None

    best_lap = corner_analysis["best_lap"]
    plot_df = df[df["lap_number"] == best_lap].copy()
    plot_df = plot_df.dropna(subset=[lat_col, lon_col, speed_col])
    if len(plot_df) < 10:
        return None

    speed_kmh = (plot_df[speed_col] * 3.6).values

    # Build corner labels with time loss annotation
    corners = []
    for s in corner_analysis["summaries"]:
        if s.lat is not None and s.lon is not None:
            loss_str = f"+{s.avg_time_loss:.2f}s" if s.avg_time_loss > 0 else f"{s.avg_time_loss:.2f}s"
            corners.append({
                "label": f"{s.corner_name} ({loss_str})",
                "lat": s.lat,
                "lon": s.lon,
            })

    return {
        "title": "Corner Map — Avg Time Loss",
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
        "corners": corners,
        "wind": None,
    }


def corner_analysis_to_template(corner_analysis, laptimes_df=None, time_col="seconds"):
    """Convert corner analysis to template-friendly dicts.

    Returns a dict suitable for passing to the Jinja2 template.
    """
    if corner_analysis is None:
        return None

    from scripts.analysis.utils import format_laptime

    summaries = corner_analysis["summaries"]
    records = corner_analysis["records"]
    best_lap = corner_analysis["best_lap"]

    # Build lap time lookup
    lap_times = {}
    if laptimes_df is not None and time_col in laptimes_df.columns:
        for _, row in laptimes_df.iterrows():
            lap_times[int(row["lap"])] = float(row[time_col])

    # Summary table data (sorted by worst time loss)
    summary_rows = []
    for s in summaries:
        summary_rows.append({
            "corner": s.corner_name,
            "archetype": s.archetype,
            "avg_time_loss": round(s.avg_time_loss, 3),
            "avg_time_loss_fmt": f"+{s.avg_time_loss:.3f}s" if s.avg_time_loss > 0 else f"{s.avg_time_loss:.3f}s",
            "best_min_speed": round(s.best_min_speed, 1),
            "avg_min_speed": round(s.avg_min_speed, 1),
            "std_min_speed": round(s.std_min_speed, 2),
            "best_entry_speed": round(s.best_entry_speed, 1),
            "best_exit_speed": round(s.best_exit_speed, 1),
            "braking_spread": s.braking_spread,
            "braking_spread_fmt": f"{s.braking_spread:.1f}m" if s.braking_spread is not None else "-",
            "dominant_root_cause": s.dominant_root_cause or "-",
        })

    # Per-lap breakdown
    lap_breakdowns = {}
    for lap, lap_records in sorted(records.items()):
        rows = []
        for r in sorted(lap_records, key=lambda x: x.time_loss, reverse=True):
            rows.append({
                "corner": r.corner_name,
                "time_loss": round(r.time_loss, 3),
                "time_loss_fmt": f"+{r.time_loss:.3f}s" if r.time_loss > 0 else f"{r.time_loss:.3f}s",
                "entry_speed": round(r.entry_speed, 1),
                "min_speed": round(r.min_speed, 1),
                "exit_speed": round(r.exit_speed, 1),
                "braking_distance": round(r.braking_distance, 1) if r.braking_distance is not None else None,
                "root_cause": r.root_cause or "-",
                "is_worst": False,
            })
        # Mark the worst corner
        if rows:
            rows[0]["is_worst"] = True
        lap_breakdowns[str(lap)] = rows

    # Build laps list with times for the selector dropdown
    laps_list = []
    for lap in sorted(records.keys()):
        lap_info = {
            "lap": lap,
            "is_best": lap == best_lap,
        }
        if lap in lap_times:
            lap_info["time_fmt"] = format_laptime(lap_times[lap])
        else:
            lap_info["time_fmt"] = None
        laps_list.append(lap_info)

    return {
        "summary_rows": summary_rows,
        "lap_breakdowns": lap_breakdowns,
        "laps": laps_list,
        "best_lap": best_lap,
        "corner_names": corner_analysis["corner_names"],
    }
