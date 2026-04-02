"""Coaching summary and structured action plan."""

import numpy as np

from scripts.analysis.outliers import detect_outliers
from scripts.analysis.utils import format_laptime


def compute_corner_time_loss(df, laptimes_df, time_col="seconds", track_corners=None):
    """Compute per-corner time loss vs best lap (data only, no chart).

    Returns list of dicts with 'label' and 'delta' (seconds, negative = time lost),
    sorted by delta ascending (worst first), or None.
    """
    from scripts.analysis.speed import compute_time_delta

    result = compute_time_delta(df, laptimes_df, time_col=time_col, track_corners=track_corners)
    if result is None:
        return None

    dist_grid, cum_delta, best_lap, median_lap, corners_info = result
    if len(corners_info) < 2:
        return None

    corner_dists = []
    for c in corners_info:
        if "distance" in c and c["distance"] <= dist_grid[-1]:
            corner_dists.append((c["label"], c["distance"]))
    if len(corner_dists) < 2:
        return None
    corner_dists.sort(key=lambda x: x[1])

    boundaries = [0]
    for i in range(len(corner_dists) - 1):
        boundaries.append((corner_dists[i][1] + corner_dists[i + 1][1]) / 2)
    boundaries.append(dist_grid[-1])

    results = []
    for i, (label, _) in enumerate(corner_dists):
        entry_dist = boundaries[i]
        exit_dist = boundaries[i + 1]
        entry_idx = np.argmin(np.abs(dist_grid - entry_dist))
        exit_idx = np.argmin(np.abs(dist_grid - exit_dist))
        delta = cum_delta[exit_idx] - cum_delta[entry_idx]
        results.append({"label": label, "delta": delta})

    results.sort(key=lambda x: x["delta"])
    return results


def compute_braking_spread(df, laptimes_df, time_col="seconds", track_corners=None):
    """Compute braking point spread (IQR in meters) per corner.

    Returns list of dicts with 'label' and 'spread_m', sorted by spread descending,
    or None.
    """
    if "longitudinal_acc" not in df.columns or "lap_number" not in df.columns:
        return None
    if "distance_traveled" not in df.columns:
        return None

    from scripts.analysis.corners import detect_corners

    clean_df, _ = detect_outliers(laptimes_df, time_col=time_col)
    best_lap = int(clean_df.loc[clean_df[time_col].idxmin(), "lap"])
    clean_laps = set(clean_df["lap"].astype(int))

    corners, ref_lap_data = detect_corners(df, best_lap=best_lap, track_corners=track_corners)
    if corners is None or len(corners) == 0:
        return None
    if "distance_traveled" not in ref_lap_data.columns:
        return None

    corner_names = [
        track_corners[i]["name"] if track_corners and i < len(track_corners) else f"T{i + 1}"
        for i in range(len(corners))
    ]

    ref_dist = ref_lap_data["distance_traveled"].values
    ref_dist_norm = ref_dist - ref_dist[0]
    lap_length = ref_dist_norm[-1]
    if lap_length <= 0:
        return None
    corner_fracs = [ref_dist_norm[c] / lap_length for c in corners]

    laps = sorted(df["lap_number"].dropna().unique())
    laps = [l for l in laps if l > 0 and l in clean_laps]

    braking_threshold = -0.15
    corner_braking_distances = {i: [] for i in range(len(corners))}

    for lap in laps:
        lap_data = df[df["lap_number"] == lap].copy()
        if len(lap_data) < 50:
            continue
        dist = lap_data["distance_traveled"].values
        dist_norm = dist - dist[0]
        this_len = dist_norm[-1]
        if this_len <= 0:
            continue
        long_acc = lap_data["longitudinal_acc"].values

        for ci, corner_frac in enumerate(corner_fracs):
            search_start = corner_frac - 0.30
            search_end = corner_frac + 0.05
            frac = dist_norm / this_len
            mask = (frac >= search_start) & (frac <= search_end)
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue
            braking_indices = indices[long_acc[indices] < braking_threshold]
            if len(braking_indices) > 0:
                brake_start = braking_indices[0]
                corner_braking_distances[ci].append(dist_norm[brake_start])

    results = []
    for ci in range(len(corners)):
        pts = corner_braking_distances[ci]
        if len(pts) >= 4:
            q75, q25 = np.percentile(pts, [75, 25])
            spread = q75 - q25
            results.append({"label": corner_names[ci], "spread_m": round(spread, 1)})

    results.sort(key=lambda x: x["spread_m"], reverse=True)
    return results if results else None


def generate_coaching_summary(df, laptimes_df, corner_scores=None, time_col="seconds",
                              sector_data=None, track_corners=None, corner_analysis=None):
    """Generate structured action plan with corner-specific advice.

    Args:
        corner_analysis: Optional pre-built corner analysis dict from
            build_corner_analysis(). When provided, skips redundant
            corner time loss and braking spread computation.

    Returns dict with:
        - action_items: list of dicts with 'icon', 'title', 'detail'
        - coaching_text: list of strings (legacy, kept for compatibility)
    """
    action_items = []
    tips = []

    clean_df = None
    times = None
    if laptimes_df is not None and time_col in laptimes_df.columns:
        clean_df, _ = detect_outliers(laptimes_df, time_col=time_col)
        if len(clean_df) >= 2:
            times = clean_df[time_col].values

    # --- Top priority corner (from time loss data) ---
    corner_time_loss = None
    braking_spread = None

    if corner_analysis is not None:
        # Extract from pre-built Corner Object
        corner_time_loss = [
            {"label": s.corner_name, "delta": -s.avg_time_loss}
            for s in corner_analysis["summaries"]
            if s.avg_time_loss != 0
        ]
        corner_time_loss.sort(key=lambda x: x["delta"])

        braking_spread = [
            {"label": s.corner_name, "spread_m": s.braking_spread}
            for s in corner_analysis["summaries"]
            if s.braking_spread is not None
        ]
        braking_spread.sort(key=lambda x: x["spread_m"], reverse=True)
        if not braking_spread:
            braking_spread = None
    else:
        if df is not None and laptimes_df is not None:
            try:
                corner_time_loss = compute_corner_time_loss(df, laptimes_df, time_col=time_col, track_corners=track_corners)
            except Exception:
                pass

        if df is not None and laptimes_df is not None:
            try:
                braking_spread = compute_braking_spread(df, laptimes_df, time_col=time_col, track_corners=track_corners)
            except Exception:
                pass

    if corner_time_loss:
        # Worst corner (most negative delta = most time lost)
        worst = corner_time_loss[0]
        if worst["delta"] < -0.05:
            detail = f"You lose {abs(worst['delta']):.2f}s here vs your best lap."
            # Add braking info if available for this corner
            if braking_spread:
                spread_for_corner = next((b for b in braking_spread if b["label"] == worst["label"]), None)
                if spread_for_corner and spread_for_corner["spread_m"] > 2:
                    detail += f" Your braking point varies by {spread_for_corner['spread_m']:.0f}m."
            action_items.append({
                "icon": "target",
                "title": f"Focus on {worst['label']}",
                "detail": detail,
            })
            tips.append(f"Focus on {worst['label']} — {detail}")

    # --- Consistency target ---
    if times is not None and len(times) >= 5:
        sorted_times = np.sort(times)
        best_5 = sorted_times[:5]
        best_5_avg = np.mean(best_5)
        slow_laps = times[times > best_5_avg + 1.0]
        if len(slow_laps) >= 2:
            best_5_fmt = format_laptime(best_5_avg)
            detail = f"Your best 5 laps average {best_5_fmt}. {len(slow_laps)} laps were >1s off that pace."
            if braking_spread:
                inconsistent = [b for b in braking_spread if b["spread_m"] > 3]
                if inconsistent:
                    corner_names = ", ".join(b["label"] for b in inconsistent[:2])
                    detail += f" Match your braking at {corner_names}."
            action_items.append({
                "icon": "consistency",
                "title": "Improve consistency",
                "detail": detail,
            })
            tips.append(detail)
        elif len(times) >= 4:
            std = np.std(times)
            if std < 0.3:
                action_items.append({
                    "icon": "check",
                    "title": "Excellent consistency",
                    "detail": f"Lap time spread is only {std:.2f}s — very tight.",
                })
                tips.append("Excellent consistency — your lap times are very tight.")

    # --- Pace trend ---
    if times is not None and len(times) >= 4:
        first_half = np.mean(times[:len(times) // 2])
        second_half = np.mean(times[len(times) // 2:])
        diff = first_half - second_half
        if diff > 0.2:
            detail = f"You were {diff:.1f}s faster in the second half — keep doing what changed."
            action_items.append({
                "icon": "trending-up",
                "title": "Pace improved through session",
                "detail": detail,
            })
            tips.append(detail)
        elif diff < -0.2:
            detail = f"Your pace dropped {abs(diff):.1f}s in the second half — consider shorter stints or hydration."
            action_items.append({
                "icon": "trending-down",
                "title": "Pace faded late in session",
                "detail": detail,
            })
            tips.append(detail)

    # --- Theoretical best from sector data ---
    if sector_data and times is not None:
        try:
            theoretical_best = float(sector_data["theoretical_row"]["total"])
            actual_best = float(np.min(times))
            gap = actual_best - theoretical_best
            if gap > 0.05:
                theo_fmt = format_laptime(theoretical_best)
                actual_fmt = format_laptime(actual_best)
                detail = f"Your ideal lap (best sectors combined) is {theo_fmt} — {gap:.2f}s faster than your actual best of {actual_fmt}."
                action_items.append({
                    "icon": "zap",
                    "title": f"Theoretical best: {theo_fmt}",
                    "detail": detail,
                })
                tips.append(detail)
        except (ValueError, KeyError):
            pass

    # --- Braking insight ---
    if braking_spread and not any("braking" in item.get("detail", "").lower() for item in action_items):
        worst_brake = braking_spread[0]
        if worst_brake["spread_m"] > 3:
            detail = f"Braking point at {worst_brake['label']} varies by {worst_brake['spread_m']:.0f}m — most inconsistent corner."
            action_items.append({
                "icon": "brake",
                "title": f"Tighten braking at {worst_brake['label']}",
                "detail": detail,
            })
            tips.append(detail)

    if not action_items:
        action_items.append({
            "icon": "check",
            "title": "Solid session",
            "detail": "Keep up the consistent driving — review corner speeds for specific improvement areas.",
        })
        tips.append("Keep up the consistent driving — review corner speeds for specific improvement areas.")

    return {
        "action_items": action_items[:5],
        "coaching_text": tips[:5],
    }
