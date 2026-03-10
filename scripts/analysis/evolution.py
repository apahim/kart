"""Cross-race evolution analysis — tracking driver improvement across sessions."""

import os
import glob

import numpy as np
import pandas as pd
import yaml
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scripts.load_data import extract_laptimes_from_telemetry, load_telemetry
from scripts.analysis.outliers import filter_non_race_laps, detect_outliers
from scripts.analysis.utils import project_to_meters, format_laptime


def load_all_races(data_dir="data/races"):
    """Scan all race directories and build a consolidated DataFrame.

    Each row represents one race, with summary metrics + metadata.
    """
    rows = []
    for race_dir in sorted(glob.glob(os.path.join(data_dir, "*"))):
        if not os.path.isdir(race_dir):
            continue

        summary_path = os.path.join(race_dir, "summary_generated.yaml")
        meta_path = os.path.join(race_dir, "race.yaml")

        if not os.path.exists(summary_path):
            continue

        with open(summary_path, "r") as f:
            summary = yaml.safe_load(f) or {}

        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = yaml.safe_load(f) or {}

        row = {
            "race_dir": os.path.basename(race_dir),
            "date": meta.get("date", os.path.basename(race_dir)[:10]),
            "track": meta.get("track", "Unknown"),
            "kart_number": meta.get("kart_number"),
            "driver_weight_kg": meta.get("driver_weight_kg"),
            "session_type": meta.get("session_type"),
            "weather_condition": summary.get("weather", {}).get("condition") if isinstance(summary.get("weather"), dict) else None,
            "weather_temp_c": summary.get("weather", {}).get("temp_c") if isinstance(summary.get("weather"), dict) else None,
            "weather_wind_kmh": summary.get("weather", {}).get("wind_kmh") if isinstance(summary.get("weather"), dict) else None,
            "weather_wind_direction": summary.get("weather", {}).get("wind_direction") if isinstance(summary.get("weather"), dict) else None,
            "best_lap_time": summary.get("best_lap", {}).get("time"),
            "average": summary.get("average"),
            "median": summary.get("median"),
            "std_dev": summary.get("std_dev"),
            "consistency_pct": summary.get("consistency_pct"),
            "total_laps": summary.get("total_laps"),
            "clean_laps": summary.get("clean_laps"),
            "top_speed_kmh": summary.get("top_speed_kmh"),
            "max_lateral_g": summary.get("max_lateral_g"),
            "max_braking_g": summary.get("max_braking_g"),
            "max_acceleration_g": summary.get("max_acceleration_g"),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_all_laptimes(data_dir="data/races"):
    """Load individual lap times from all race directories.

    Returns a combined DataFrame with columns:
        race_dir, date, track, lap, seconds, is_outlier
    """
    frames = []
    for race_dir in sorted(glob.glob(os.path.join(data_dir, "*"))):
        if not os.path.isdir(race_dir):
            continue

        meta_path = os.path.join(race_dir, "race.yaml")

        if not os.path.exists(meta_path):
            continue

        with open(meta_path, "r") as f:
            meta = yaml.safe_load(f) or {}

        telemetry_df = load_telemetry(race_dir)
        if telemetry_df is None:
            continue

        laps_df = extract_laptimes_from_telemetry(telemetry_df)
        laps_df = filter_non_race_laps(laps_df)
        if laps_df.empty or "seconds" not in laps_df.columns:
            continue

        clean_df, _ = detect_outliers(laps_df)
        clean_indices = set(clean_df.index)

        race_name = os.path.basename(race_dir)
        date_str = meta.get("date", race_name[:10])

        for _, row in laps_df.iterrows():
            frames.append({
                "race_dir": race_name,
                "date": date_str,
                "track": meta.get("track", "Unknown"),
                "session_type": meta.get("session_type"),
                "lap": int(row["lap"]),
                "seconds": float(row["seconds"]),
                "is_outlier": row.name not in clean_indices,
            })

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def enrich_races_with_quartiles(races_df, all_laps_df):
    """Add Q1 and Q3 columns to races_df computed from per-lap data."""
    if all_laps_df.empty or races_df.empty:
        return races_df

    clean = all_laps_df[~all_laps_df["is_outlier"]]
    quartiles = clean.groupby("race_dir")["seconds"].quantile([0.25, 0.75]).unstack()
    quartiles.columns = ["q1", "q3"]
    quartiles = quartiles.reset_index()

    return races_df.merge(quartiles, on="race_dir", how="left")


def _context_hover(row):
    """Build a context string for hover tooltips from race metadata."""
    parts = []
    if pd.notna(row.get("kart_number")):
        parts.append(f"Kart {int(row['kart_number'])}")
    if pd.notna(row.get("driver_weight_kg")):
        parts.append(f"{row['driver_weight_kg']}kg")
    if row.get("session_type"):
        parts.append(row["session_type"])
    if row.get("weather_condition"):
        parts.append(row["weather_condition"])
    if pd.notna(row.get("weather_temp_c")):
        parts.append(f"{row['weather_temp_c']}°C")
    return " | ".join(parts) if parts else ""


def _weather_symbol(condition):
    """Return a simple text marker for weather conditions."""
    if not condition:
        return ""
    c = condition.lower()
    if "rain" in c or "shower" in c or "wet" in c:
        return "Rain"
    if "cloud" in c or "overcast" in c:
        return "Overcast"
    if "sun" in c or "clear" in c or "dry" in c:
        return "Dry"
    return condition.capitalize()


def create_laptime_progression(races_df):
    """Best lap, average lap time, and IQR band over time with contextual annotations."""
    if races_df.empty:
        return None

    fig = go.Figure()
    context_texts = [_context_hover(row) for _, row in races_df.iterrows()]

    # IQR band if q1/q3 columns exist
    if "q1" in races_df.columns and "q3" in races_df.columns:
        fig.add_trace(go.Scatter(
            x=races_df["date"], y=races_df["q3"],
            mode="lines", name="Q3",
            line=dict(width=0), showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=races_df["date"], y=races_df["q1"],
            mode="lines", name="IQR Range",
            fill="tonexty", fillcolor="rgba(52, 152, 219, 0.15)",
            line=dict(width=0),
            hovertemplate="%{x|%Y-%m-%d}<br>Q1: %{y:.3f}s<extra></extra>",
        ))

    # Median trace
    if "median" in races_df.columns and races_df["median"].notna().any():
        fig.add_trace(go.Scatter(
            x=races_df["date"], y=races_df["median"],
            mode="markers+lines", name="Median",
            marker=dict(size=8, color="#9b59b6", symbol="diamond"),
            line=dict(color="#9b59b6", width=2, dash="dash"),
            customdata=context_texts,
            hovertemplate="%{x|%Y-%m-%d}<br>Median: %{y:.3f}s<br>%{customdata}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=races_df["date"], y=races_df["best_lap_time"],
        mode="markers+lines", name="Best Lap",
        marker=dict(size=10, color="#2ecc71"),
        line=dict(color="#2ecc71", width=2),
        customdata=context_texts,
        hovertemplate="%{x|%Y-%m-%d}<br>Best: %{y:.3f}s<br>%{customdata}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=races_df["date"], y=races_df["average"],
        mode="markers+lines", name="Clean Average",
        marker=dict(size=10, color="#3498db"),
        line=dict(color="#3498db", width=2),
        customdata=context_texts,
        hovertemplate="%{x|%Y-%m-%d}<br>Avg: %{y:.2f}s<br>%{customdata}<extra></extra>",
    ))

    # Annotate weather/kart changes on the chart
    for i, (_, row) in enumerate(races_df.iterrows()):
        annotations = []
        weather_sym = _weather_symbol(row.get("weather_condition"))
        if weather_sym:
            annotations.append(weather_sym)
        if pd.notna(row.get("kart_number")):
            annotations.append(f"K{int(row['kart_number'])}")
        if annotations:
            fig.add_annotation(
                x=row["date"], y=row["best_lap_time"],
                text=" ".join(annotations),
                showarrow=True, arrowhead=0, arrowcolor="#999",
                ax=0, ay=-30,
                font=dict(size=10, color="#555"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#ccc", borderwidth=1, borderpad=3,
            )

    fig.update_layout(
        title="Lap Time Progression",
        xaxis_title="Date", yaxis_title="Time (seconds)",
        template="plotly_white", height=400,
    )
    return fig


def create_consistency_trend(races_df):
    """Consistency % and std dev over time, with narrowed y-axis."""
    if races_df.empty:
        return None

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=races_df["date"], y=races_df["consistency_pct"],
        mode="markers+lines", name="Consistency %",
        marker=dict(size=10, color="#2ecc71"),
        hovertemplate="%{x|%Y-%m-%d}<br>Consistency: %{y:.1f}%<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=races_df["date"], y=races_df["std_dev"],
        mode="markers+lines", name="Std Dev",
        marker=dict(size=10, color="#e74c3c"),
        hovertemplate="%{x|%Y-%m-%d}<br>Std Dev: %{y:.2f}s<extra></extra>",
    ), secondary_y=True)

    # Narrow the y-axis so small differences are visible
    min_consistency = races_df["consistency_pct"].min()
    y_min = max(0, min_consistency - 0.5)
    fig.update_yaxes(title_text="Consistency %", range=[y_min, 100], secondary_y=False)
    fig.update_yaxes(title_text="Std Dev (s)", secondary_y=True)

    fig.update_layout(
        title="Consistency Trend",
        xaxis_title="Date",
        template="plotly_white", height=400,
    )
    return fig


def create_speed_gforce_trends(races_df):
    """Top speed, max lateral G, max braking G over time."""
    if races_df.empty:
        return None

    fig = make_subplots(rows=1, cols=3, subplot_titles=["Top Speed", "Max Lateral G", "Max Braking G"])

    if races_df["top_speed_kmh"].notna().any():
        fig.add_trace(go.Scatter(
            x=races_df["date"], y=races_df["top_speed_kmh"],
            mode="markers+lines", name="Top Speed",
            marker=dict(size=8, color="#3498db"),
        ), row=1, col=1)

    if races_df["max_lateral_g"].notna().any():
        fig.add_trace(go.Scatter(
            x=races_df["date"], y=races_df["max_lateral_g"],
            mode="markers+lines", name="Max Lat G",
            marker=dict(size=8, color="#e74c3c"),
        ), row=1, col=2)

    if races_df["max_braking_g"].notna().any():
        fig.add_trace(go.Scatter(
            x=races_df["date"], y=races_df["max_braking_g"],
            mode="markers+lines", name="Max Brake G",
            marker=dict(size=8, color="#f39c12"),
        ), row=1, col=3)

    fig.update_layout(
        title="Speed & G-Force Trends",
        template="plotly_white", height=350,
        showlegend=False,
    )
    return fig


def create_lap_distribution(all_laps_df):
    """Box plots per race date showing full spread of lap times."""
    if all_laps_df is None or all_laps_df.empty:
        return None

    clean = all_laps_df[~all_laps_df["is_outlier"]]
    if clean.empty:
        return None

    fig = go.Figure()
    for race_dir in clean["race_dir"].unique():
        race_data = clean[clean["race_dir"] == race_dir]
        date_str = race_data["date"].iloc[0].strftime("%Y-%m-%d")
        fig.add_trace(go.Box(
            y=race_data["seconds"],
            name=date_str,
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.8,
            hovertemplate="Lap %{text}<br>%{y:.3f}s<extra></extra>",
            text=race_data["lap"].astype(str),
        ))

    fig.update_layout(
        title="Lap Time Distribution by Session",
        yaxis_title="Lap Time (seconds)",
        template="plotly_white", height=450,
    )
    return fig


def create_session_overlay(all_laps_df):
    """All laps from all races overlaid on same axes, one line per session."""
    if all_laps_df is None or all_laps_df.empty:
        return None

    clean = all_laps_df[~all_laps_df["is_outlier"]]
    if clean.empty:
        return None

    colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"]
    fig = go.Figure()

    for i, race_dir in enumerate(clean["race_dir"].unique()):
        race_data = clean[clean["race_dir"] == race_dir].sort_values("lap")
        date_str = race_data["date"].iloc[0].strftime("%Y-%m-%d")
        color = colors[i % len(colors)]

        # Median reference line
        median_val = race_data["seconds"].median()
        fig.add_hline(
            y=median_val, line_dash="dot", line_color=color,
            annotation_text=f"{date_str} median: {median_val:.3f}s",
            annotation_position="top right",
            opacity=0.5,
        )

        fig.add_trace(go.Scatter(
            x=race_data["lap"], y=race_data["seconds"],
            mode="markers+lines", name=date_str,
            marker=dict(size=6, color=color),
            line=dict(color=color, width=2),
            hovertemplate="Lap %{x}<br>%{y:.3f}s<extra></extra>",
        ))

    fig.update_layout(
        title="Session Overlay — Lap Times Compared",
        xaxis_title="Lap Number", yaxis_title="Lap Time (seconds)",
        template="plotly_white", height=450,
    )
    return fig


def create_improvement_summary(races_df):
    """Grouped bar chart of deltas between consecutive races at the same track."""
    if races_df.empty or len(races_df) < 2:
        return None

    fig = go.Figure()
    metrics = [
        ("best_lap_time", "Best Lap"),
        ("average", "Average"),
        ("median", "Median"),
    ]

    for metric_col, label in metrics:
        if metric_col not in races_df.columns or races_df[metric_col].isna().all():
            continue

        deltas = []
        labels = []
        colors = []

        sorted_df = races_df.sort_values("date")
        prev = None
        for _, row in sorted_df.iterrows():
            if prev is not None and row["track"] == prev["track"]:
                delta = row[metric_col] - prev[metric_col]
                deltas.append(delta)
                date_str = row["date"].strftime("%Y-%m-%d")
                labels.append(date_str)
                # Negative delta = improvement (faster), green
                colors.append("#2ecc71" if delta < 0 else "#e74c3c")
            prev = row

        if deltas:
            fig.add_trace(go.Bar(
                x=labels, y=deltas, name=label,
                marker_color=colors,
                hovertemplate="%{x}<br>" + label + ": %{y:+.3f}s<extra></extra>",
            ))

    if not fig.data:
        return None

    fig.update_layout(
        title="Improvement Between Sessions (negative = faster)",
        xaxis_title="Date", yaxis_title="Delta (seconds)",
        template="plotly_white", height=400,
        barmode="group",
    )
    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)
    return fig


def create_kart_comparison(races_df):
    """Box plots of lap times grouped by kart number."""
    df = races_df.dropna(subset=["kart_number"])
    if df.empty:
        return None

    fig = go.Figure()
    for kart in sorted(df["kart_number"].unique()):
        kart_data = df[df["kart_number"] == kart]
        date_labels = [d.strftime("%Y-%m-%d") if pd.notna(d) else "" for d in kart_data["date"]]
        fig.add_trace(go.Box(
            y=kart_data["best_lap_time"],
            name=f"Kart {int(kart)}",
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.8,
            text=date_labels,
            hovertemplate="Kart %{x}<br>Best Lap: %{y:.3f}s<br>%{text}<extra></extra>",
        ))

    subtitle = ""
    if df["kart_number"].nunique() == 1:
        subtitle = " (single kart — more data needed for comparison)"

    fig.update_layout(
        title=f"Kart Comparison (Best Lap Times){subtitle}",
        yaxis_title="Best Lap Time (s)",
        template="plotly_white", height=400,
    )
    return fig


def create_weather_impact(races_df):
    """Performance metrics split by weather condition with temperature on secondary axis."""
    df = races_df.dropna(subset=["weather_condition"])
    if df.empty:
        return None

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for condition in sorted(df["weather_condition"].unique()):
        cond_data = df[df["weather_condition"] == condition]
        context = [_context_hover(row) for _, row in cond_data.iterrows()]
        fig.add_trace(go.Scatter(
            x=cond_data["date"],
            y=cond_data["best_lap_time"],
            mode="markers",
            name=condition.capitalize(),
            marker=dict(size=12),
            customdata=context,
            hovertemplate="%{x|%Y-%m-%d}<br>Best: %{y:.3f}s<br>%{customdata}<extra></extra>",
        ), secondary_y=False)

    # Temperature line if available
    temp_df = df.dropna(subset=["weather_temp_c"])
    if not temp_df.empty:
        fig.add_trace(go.Scatter(
            x=temp_df["date"], y=temp_df["weather_temp_c"],
            mode="markers+lines", name="Temp (°C)",
            marker=dict(size=8, symbol="triangle-up", color="#f39c12"),
            line=dict(color="#f39c12", width=2, dash="dot"),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y}°C<extra></extra>",
        ), secondary_y=True)

    fig.update_yaxes(title_text="Best Lap Time (s)", secondary_y=False)
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=True)

    fig.update_layout(
        title="Weather Impact on Lap Times",
        xaxis_title="Date",
        template="plotly_white", height=400,
    )
    return fig


def create_weight_vs_performance(races_df):
    """Scatter plot of driver weight vs best/average lap time."""
    df = races_df.dropna(subset=["driver_weight_kg", "best_lap_time"])
    if df.empty:
        return None

    fig = go.Figure()
    context = [_context_hover(row) for _, row in df.iterrows()]
    date_labels = [d.strftime("%Y-%m-%d") if pd.notna(d) else "" for d in df["date"]]

    fig.add_trace(go.Scatter(
        x=df["driver_weight_kg"], y=df["best_lap_time"],
        mode="markers+text", name="Best Lap",
        marker=dict(size=12, color="#2ecc71"),
        text=date_labels, textposition="top center",
        customdata=context,
        hovertemplate="Weight: %{x}kg<br>Best: %{y:.3f}s<br>%{customdata}<extra></extra>",
    ))

    avg_df = df.dropna(subset=["average"])
    if not avg_df.empty:
        avg_context = [_context_hover(row) for _, row in avg_df.iterrows()]
        fig.add_trace(go.Scatter(
            x=avg_df["driver_weight_kg"], y=avg_df["average"],
            mode="markers", name="Average",
            marker=dict(size=10, color="#3498db", symbol="diamond"),
            customdata=avg_context,
            hovertemplate="Weight: %{x}kg<br>Avg: %{y:.3f}s<br>%{customdata}<extra></extra>",
        ))

    fig.update_layout(
        title="Driver Weight vs. Performance",
        xaxis_title="Driver Weight (kg)", yaxis_title="Lap Time (s)",
        template="plotly_white", height=400,
    )
    return fig


def create_temp_vs_laptime(races_df):
    """Scatter plot of ambient temperature vs lap times."""
    df = races_df.dropna(subset=["weather_temp_c", "best_lap_time"])
    if df.empty:
        return None

    fig = go.Figure()
    context = [_context_hover(row) for _, row in df.iterrows()]
    date_labels = [d.strftime("%Y-%m-%d") if pd.notna(d) else "" for d in df["date"]]

    fig.add_trace(go.Scatter(
        x=df["weather_temp_c"], y=df["best_lap_time"],
        mode="markers+text", name="Best Lap",
        marker=dict(size=12, color="#e74c3c"),
        text=date_labels, textposition="top center",
        customdata=context,
        hovertemplate="Temp: %{x}°C<br>Best: %{y:.3f}s<br>%{customdata}<extra></extra>",
    ))

    avg_df = df.dropna(subset=["average"])
    if not avg_df.empty:
        avg_context = [_context_hover(row) for _, row in avg_df.iterrows()]
        fig.add_trace(go.Scatter(
            x=avg_df["weather_temp_c"], y=avg_df["average"],
            mode="markers", name="Average",
            marker=dict(size=10, color="#3498db", symbol="diamond"),
            customdata=avg_context,
            hovertemplate="Temp: %{x}°C<br>Avg: %{y:.3f}s<br>%{customdata}<extra></extra>",
        ))

    fig.update_layout(
        title="Temperature vs. Lap Time",
        xaxis_title="Ambient Temperature (°C)", yaxis_title="Lap Time (s)",
        template="plotly_white", height=400,
    )
    return fig


def create_raceline_evolution(races_df):
    """Overlay best lap racing lines from each session on a single track map.

    Shows how the driver's line evolves across sessions.
    """
    if races_df.empty:
        return None

    colors = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
    ]

    fig = go.Figure()
    trace_count = 0

    for i, (_, row) in enumerate(races_df.iterrows()):
        race_dir_name = row["race_dir"]
        race_dir = os.path.join("data/races", race_dir_name)

        telemetry_df = load_telemetry(race_dir)
        if telemetry_df is None:
            continue

        laps_df = extract_laptimes_from_telemetry(telemetry_df)
        laps_df = filter_non_race_laps(laps_df)
        if laps_df.empty or "seconds" not in laps_df.columns:
            continue

        clean_df, _ = detect_outliers(laps_df)
        if clean_df.empty:
            continue
        best_idx = clean_df["seconds"].idxmin()
        best_lap = int(clean_df.loc[best_idx, "lap"])
        best_time = float(clean_df.loc[best_idx, "seconds"])

        # Find GPS columns
        lat_col = lon_col = None
        for col in telemetry_df.columns:
            cl = col.lower()
            if "latitude" in cl:
                lat_col = col
            if "longitude" in cl:
                lon_col = col
        if not lat_col or not lon_col:
            continue

        lap_data = telemetry_df[telemetry_df["lap_number"] == best_lap].copy()
        lap_data = lap_data.dropna(subset=[lat_col, lon_col])
        if len(lap_data) < 20:
            continue

        x_m, y_m = project_to_meters(lap_data[lat_col].values, lap_data[lon_col].values)

        date_str = row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else race_dir_name[:10]
        label = date_str
        if row.get("session_type"):
            label += f" ({row['session_type']})"
        label += f" - {format_laptime(best_time)}"

        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=x_m, y=y_m,
            mode="lines",
            name=label,
            line=dict(color=color, width=2),
            hovertemplate=f"{label}<extra></extra>",
        ))
        trace_count += 1

    if trace_count == 0:
        return None

    fig.update_layout(
        title="Racing Line Evolution",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(),
        template="plotly_white",
        height=500,
        showlegend=True,
    )

    # Hide axes for clean map look
    axis_opts = dict(
        showticklabels=False, showgrid=False, zeroline=False, title="",
    )
    fig.update_layout(xaxis=axis_opts, yaxis=axis_opts)

    return fig
