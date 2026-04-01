"""Speed trace analysis using Plotly."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scripts.analysis.outliers import detect_outliers
from scripts.analysis.utils import detect_corners_with_positions


def create_speed_traces(df, laptimes_df=None, time_col="seconds", track_corners=None):
    """Create speed envelope view: best/worst/median laps + gray min/max envelope."""
    speed_col = "speed_gps" if "speed_gps" in df.columns else "speed"
    if speed_col not in df.columns or "lap_number" not in df.columns:
        return None
    if "distance_traveled" not in df.columns:
        return None

    best_lap = worst_lap = median_lap = None
    if laptimes_df is not None and time_col in laptimes_df.columns:
        clean_df, _ = detect_outliers(laptimes_df, time_col=time_col)
        best_lap = int(clean_df.loc[clean_df[time_col].idxmin(), "lap"])
        worst_lap = int(clean_df.loc[clean_df[time_col].idxmax(), "lap"])
        median_time = clean_df[time_col].median()
        median_lap = int(clean_df.iloc[(clean_df[time_col] - median_time).abs().argsort().iloc[0]]["lap"])

    laps = sorted(df["lap_number"].dropna().unique())
    laps = [l for l in laps if l > 0]

    # Interpolate all laps to a common distance grid
    n_points = 500
    lap_speeds = {}
    max_dist = 0

    for lap in laps:
        lap_data = df[df["lap_number"] == lap].copy()
        if len(lap_data) < 10:
            continue
        dist = lap_data["distance_traveled"].values
        dist_norm = dist - dist[0]
        lap_total = dist_norm[-1]
        if lap_total <= 0:
            continue
        max_dist = max(max_dist, lap_total)
        speed_kmh = lap_data[speed_col].values * 3.6
        lap_speeds[lap] = (dist_norm, speed_kmh, lap_total)

    if not lap_speeds:
        return None

    # Use median lap length for the grid
    lap_lengths = [v[2] for v in lap_speeds.values()]
    grid_length = np.median(lap_lengths)
    dist_grid = np.linspace(0, grid_length, n_points)

    # Discard laps that are too short (< 50% of median) — incomplete laps
    # cause np.interp to flat-line, distorting the envelope
    min_length = grid_length * 0.5
    lap_speeds = {k: v for k, v in lap_speeds.items() if v[2] >= min_length}
    if not lap_speeds:
        return None

    # Interpolate all laps to common grid
    interp_speeds = {}
    for lap, (dist_norm, speed_kmh, lap_total) in lap_speeds.items():
        interp_speeds[lap] = np.interp(dist_grid, dist_norm, speed_kmh)

    # Compute envelope (min/max across all laps)
    all_speeds = np.array(list(interp_speeds.values()))
    env_min = np.min(all_speeds, axis=0)
    env_max = np.max(all_speeds, axis=0)

    fig = go.Figure()

    # Gray envelope
    fig.add_trace(go.Scatter(
        x=dist_grid, y=env_max, mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dist_grid, y=env_min, mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(180,180,180,0.25)",
        name="All Laps Range",
        hoverinfo="skip",
    ))

    # Featured laps
    featured = []
    if median_lap and median_lap in interp_speeds:
        featured.append((median_lap, "Median Lap", "#3498db", 2))
    if worst_lap and worst_lap in interp_speeds:
        featured.append((worst_lap, "Worst Clean", "#f39c12", 2))
    if best_lap and best_lap in interp_speeds:
        featured.append((best_lap, "Best Lap", "#e74c3c", 3))

    for lap, name, color, width in featured:
        fig.add_trace(go.Scatter(
            x=dist_grid,
            y=interp_speeds[lap],
            mode="lines",
            name=f"{name} (L{int(lap)})",
            line=dict(width=width, color=color),
            hovertemplate=f"{name}<br>Distance: %{{x:.0f}}m<br>Speed: %{{y:.1f}} km/h<extra></extra>",
        ))

    # Corner annotations
    corners_info = detect_corners_with_positions(df, best_lap=best_lap, track_corners=track_corners)
    for corner in corners_info:
        if "distance" in corner:
            fig.add_vline(
                x=corner["distance"], line_dash="dash",
                line_color="rgba(0,0,0,0.3)", line_width=1,
            )
            fig.add_annotation(
                x=corner["distance"], y=1.02, yref="paper",
                text=corner["label"], showarrow=False,
                font=dict(size=10, color="#555"),
            )

    fig.update_layout(
        title="Speed Traces (Envelope View)",
        xaxis_title="Distance (m)",
        yaxis_title="Speed (km/h)",
        template="plotly_white",
        height=400,
        showlegend=True,
    )

    return fig


def compute_time_delta(df, laptimes_df, time_col="seconds", n_points=500, track_corners=None, target_lap=None):
    """Compute cumulative time delta between a target lap and the median lap.

    Args:
        target_lap: Lap number to compare against the median. If None, uses
                    the best (fastest) clean lap.

    Returns (dist_grid, cum_delta, target_lap, median_lap, corners_info) or None.
    cum_delta is positive where median is slower (target gained time).
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

    if target_lap is None:
        target_lap = int(clean_df.loc[clean_df[time_col].idxmin(), "lap"])
    best_lap = target_lap
    median_time = clean_df[time_col].median()
    median_lap = int(clean_df.iloc[(clean_df[time_col] - median_time).abs().argsort().iloc[0]]["lap"])

    if best_lap == median_lap:
        return None

    def get_lap_data(lap_num):
        lap_data = df[df["lap_number"] == lap_num].copy()
        if len(lap_data) < 10:
            return None, None
        dist = lap_data["distance_traveled"].values
        dist_norm = dist - dist[0]
        if dist_norm[-1] <= 0:
            return None, None
        speed = lap_data[speed_col].values
        speed = np.maximum(speed, 0.1)  # clamp to avoid div by zero
        return dist_norm, speed

    best_dist, best_speed = get_lap_data(best_lap)
    med_dist, med_speed = get_lap_data(median_lap)
    if best_dist is None or med_dist is None:
        return None

    max_dist = min(best_dist[-1], med_dist[-1])
    dist_grid = np.linspace(0, max_dist, n_points)

    # Interpolate speed (m/s) to common grid
    best_interp = np.interp(dist_grid, best_dist, best_speed)
    med_interp = np.interp(dist_grid, med_dist, med_speed)

    # Integrate 1/speed over distance to get cumulative time
    dd = np.diff(dist_grid)
    best_dt = np.cumsum(dd / best_interp[:-1])
    med_dt = np.cumsum(dd / med_interp[:-1])

    best_dt = np.insert(best_dt, 0, 0.0)
    med_dt = np.insert(med_dt, 0, 0.0)

    # Positive = median slower = best gained time
    cum_delta = med_dt - best_dt

    corners_info = detect_corners_with_positions(df, best_lap=best_lap, track_corners=track_corners)

    return dist_grid, cum_delta, best_lap, median_lap, corners_info


def create_cumulative_time_delta(df, laptimes_df=None, time_col="seconds", track_corners=None, target_lap=None):
    """Cumulative time delta chart: target lap vs median lap."""
    result = compute_time_delta(df, laptimes_df, time_col=time_col, track_corners=track_corners, target_lap=target_lap)
    if result is None:
        return None

    dist_grid, cum_delta, best_lap, median_lap, corners_info = result

    fig = go.Figure()

    # Green fill where best is faster (positive delta), red where slower
    pos_delta = np.where(cum_delta >= 0, cum_delta, 0)
    neg_delta = np.where(cum_delta < 0, cum_delta, 0)

    fig.add_trace(go.Scatter(
        x=dist_grid, y=pos_delta, mode="lines", fill="tozeroy",
        fillcolor="rgba(46,204,113,0.4)", line=dict(width=0),
        name="Best faster", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dist_grid, y=neg_delta, mode="lines", fill="tozeroy",
        fillcolor="rgba(231,76,60,0.4)", line=dict(width=0),
        name="Median faster", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dist_grid, y=cum_delta, mode="lines",
        line=dict(color="#2c3e50", width=2),
        name="Time Delta",
        hovertemplate="Distance: %{x:.0f}m<br>Delta: %{y:.3f}s<extra></extra>",
    ))

    # Corner annotations
    for corner in corners_info:
        if "distance" in corner and corner["distance"] <= dist_grid[-1]:
            fig.add_vline(
                x=corner["distance"], line_dash="dash",
                line_color="rgba(0,0,0,0.2)", line_width=1,
            )
            fig.add_annotation(
                x=corner["distance"], y=1.02, yref="paper",
                text=corner["label"], showarrow=False,
                font=dict(size=9, color="#555"),
            )

    fig.update_layout(
        title=f"Cumulative Time Delta: Best (L{best_lap}) vs Median (L{median_lap})",
        xaxis_title="Distance (m)",
        yaxis_title="Time Delta (s)",
        template="plotly_white",
        height=350,
        showlegend=True,
    )

    return fig


def create_throttle_brake_phases(df, laptimes_df=None, time_col="seconds", target_lap=None):
    """Speed trace with throttle/brake/coast phase coloring from longitudinal_acc.

    Args:
        target_lap: Lap number to display. If None, uses the best (fastest) clean lap.
    """
    if "longitudinal_acc" not in df.columns:
        return None

    speed_col = "speed_gps" if "speed_gps" in df.columns else "speed"
    if speed_col not in df.columns or "lap_number" not in df.columns:
        return None
    if "distance_traveled" not in df.columns:
        return None
    if laptimes_df is None or time_col not in laptimes_df.columns:
        return None

    clean_df, _ = detect_outliers(laptimes_df, time_col=time_col)
    if len(clean_df) < 1:
        return None

    best_lap = target_lap if target_lap is not None else int(clean_df.loc[clean_df[time_col].idxmin(), "lap"])
    lap_data = df[df["lap_number"] == best_lap].copy().reset_index(drop=True)
    if len(lap_data) < 20:
        return None

    dist = lap_data["distance_traveled"].values
    dist_norm = dist - dist[0]
    if dist_norm[-1] <= 0:
        return None

    speed_kmh = lap_data[speed_col].values * 3.6
    long_acc = lap_data["longitudinal_acc"].values

    # Phase thresholds (in G)
    BRAKE_THRESH = -0.15
    ACCEL_THRESH = 0.05

    fig = go.Figure()

    # Draw colored segments for each phase
    # Classify each point
    phases = np.where(long_acc < BRAKE_THRESH, "brake",
             np.where(long_acc > ACCEL_THRESH, "accel", "coast"))

    phase_colors = {"brake": "rgba(231,76,60,0.25)", "coast": "rgba(241,196,15,0.25)",
                    "accel": "rgba(46,204,113,0.25)"}

    # Draw phase bands as filled regions at y=0 to speed
    current_phase = phases[0]
    seg_start = 0
    for i in range(1, len(phases)):
        if phases[i] != current_phase or i == len(phases) - 1:
            end = i if i == len(phases) - 1 else i
            seg_dist = dist_norm[seg_start:end+1]
            seg_speed = speed_kmh[seg_start:end+1]
            fig.add_trace(go.Scatter(
                x=np.concatenate([seg_dist, seg_dist[::-1]]),
                y=np.concatenate([seg_speed, np.zeros(len(seg_speed))]),
                fill="toself",
                fillcolor=phase_colors[current_phase],
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))
            seg_start = i
            current_phase = phases[i]

    # Speed line on top
    fig.add_trace(go.Scatter(
        x=dist_norm, y=speed_kmh, mode="lines",
        line=dict(color="#2c3e50", width=2),
        name=f"Speed (L{best_lap})",
        hovertemplate="Distance: %{x:.0f}m<br>Speed: %{y:.1f} km/h<extra></extra>",
    ))

    # Detect hesitation zones (coast between brake and accel mid-corner)
    hesitations = []
    for i in range(1, len(phases) - 1):
        if phases[i] == "coast":
            # Look for brake->coast->accel pattern
            # Search backward for brake, forward for accel
            has_brake_before = any(phases[j] == "brake" for j in range(max(0, i-15), i))
            has_accel_after = any(phases[j] == "accel" for j in range(i+1, min(len(phases), i+15)))
            if has_brake_before and has_accel_after:
                # Check if this is mid-corner (speed below median)
                if speed_kmh[i] < np.median(speed_kmh):
                    hesitations.append(i)

    # Annotate distinct hesitation zones (deduplicate nearby)
    if hesitations:
        deduped = [hesitations[0]]
        for h in hesitations[1:]:
            if h - deduped[-1] > 20:
                deduped.append(h)
        for h in deduped[:5]:  # max 5 annotations
            fig.add_annotation(
                x=dist_norm[h], y=speed_kmh[h],
                text="hesitation", showarrow=True,
                arrowhead=2, arrowsize=0.8,
                font=dict(size=9, color="#e67e22"),
                bgcolor="rgba(255,255,255,0.8)",
            )

    # Add legend entries for phases
    for phase, color, label in [("brake", "#e74c3c", "Braking"),
                                 ("coast", "#f1c40f", "Coast"),
                                 ("accel", "#2ecc71", "Acceleration")]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=color),
            name=label,
        ))

    fig.update_layout(
        title=f"Throttle/Brake Phases (Lap {best_lap})",
        xaxis_title="Distance (m)",
        yaxis_title="Speed (km/h)",
        template="plotly_white",
        height=400,
        showlegend=True,
    )

    return fig


def create_best_vs_comparison_speed(df, laptimes_df=None, time_col="seconds", track_corners=None):
    """Two-row subplot: best vs median speed, and delta between them."""
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
    median_time = clean_df[time_col].median()
    median_lap = int(clean_df.iloc[(clean_df[time_col] - median_time).abs().argsort().iloc[0]]["lap"])

    if best_lap == median_lap:
        return None

    def get_lap_speed(lap_num):
        lap_data = df[df["lap_number"] == lap_num].copy()
        if len(lap_data) < 10:
            return None, None
        dist = lap_data["distance_traveled"].values
        dist_norm = dist - dist[0]
        if dist_norm[-1] <= 0:
            return None, None
        return dist_norm, lap_data[speed_col].values * 3.6

    best_dist, best_speed = get_lap_speed(best_lap)
    comp_dist, comp_speed = get_lap_speed(median_lap)
    if best_dist is None or comp_dist is None:
        return None

    # Interpolate to common grid
    n_points = 500
    max_dist = min(best_dist[-1], comp_dist[-1])
    dist_grid = np.linspace(0, max_dist, n_points)

    best_interp = np.interp(dist_grid, best_dist, best_speed)
    comp_interp = np.interp(dist_grid, comp_dist, comp_speed)
    delta = best_interp - comp_interp

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
                        vertical_spacing=0.08,
                        subplot_titles=[f"Best (L{best_lap}) vs Median (L{median_lap})", "Speed Delta"])

    fig.add_trace(go.Scatter(
        x=dist_grid, y=best_interp, mode="lines",
        name=f"Best (L{best_lap})", line=dict(color="#e74c3c", width=2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dist_grid, y=comp_interp, mode="lines",
        name=f"Median (L{median_lap})", line=dict(color="#3498db", width=2),
    ), row=1, col=1)

    # Delta: green where best is faster, red where slower
    pos_delta = np.where(delta >= 0, delta, 0)
    neg_delta = np.where(delta < 0, delta, 0)

    fig.add_trace(go.Scatter(
        x=dist_grid, y=pos_delta, mode="lines", fill="tozeroy",
        fillcolor="rgba(46,204,113,0.4)", line=dict(width=0),
        name="Best faster", hoverinfo="skip",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=dist_grid, y=neg_delta, mode="lines", fill="tozeroy",
        fillcolor="rgba(231,76,60,0.4)", line=dict(width=0),
        name="Median faster", hoverinfo="skip",
    ), row=2, col=1)

    # Corner annotations
    corners_info = detect_corners_with_positions(df, best_lap=best_lap, track_corners=track_corners)
    for corner in corners_info:
        if "distance" in corner and corner["distance"] <= max_dist:
            for row in [1, 2]:
                fig.add_vline(
                    x=corner["distance"], line_dash="dash",
                    line_color="rgba(0,0,0,0.2)", line_width=1,
                    row=row, col=1,
                )
            fig.add_annotation(
                x=corner["distance"], y=1.02, yref="y domain",
                text=corner["label"], showarrow=False,
                font=dict(size=9, color="#555"),
                row=1, col=1,
            )

    fig.update_layout(
        template="plotly_white", height=500, showlegend=True,
    )
    fig.update_xaxes(title_text="Distance (m)", row=2, col=1)
    fig.update_yaxes(title_text="Speed (km/h)", row=1, col=1)
    fig.update_yaxes(title_text="Delta (km/h)", row=2, col=1)

    return fig


def create_all_laps_cumulative_delta(df, laptimes_df, clean_laps, time_col="seconds", track_corners=None):
    """Generate cumulative time delta Plotly JSON for each clean lap.

    Returns dict {lap_number: plotly_json_dict} for use with Plotly.react().
    """
    from scripts.analysis.utils import fig_to_json

    result = {}
    for lap in clean_laps:
        fig = create_cumulative_time_delta(df, laptimes_df, time_col=time_col,
                                           track_corners=track_corners, target_lap=lap)
        data = fig_to_json(fig)
        if data is not None:
            result[lap] = data
    return result


def create_all_laps_throttle_brake(df, laptimes_df, clean_laps, time_col="seconds"):
    """Generate throttle/brake phases Plotly JSON for each clean lap.

    Returns dict {lap_number: plotly_json_dict} for use with Plotly.react().
    """
    from scripts.analysis.utils import fig_to_json

    result = {}
    for lap in clean_laps:
        fig = create_throttle_brake_phases(df, laptimes_df, time_col=time_col, target_lap=lap)
        data = fig_to_json(fig)
        if data is not None:
            result[lap] = data
    return result
