"""Lap time analysis charts using Plotly."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scripts.analysis.outliers import detect_outliers
from scripts.analysis.utils import format_laptime


def create_laptime_bar_chart(laptimes_df, time_col="seconds"):
    """Create a bar chart of lap times with best lap highlighted and outliers marked."""
    clean_df, excluded = detect_outliers(laptimes_df, time_col=time_col)
    excluded_laps = {e["lap"] for e in excluded}

    best_idx = clean_df[time_col].idxmin()
    best_lap = int(clean_df.loc[best_idx, "lap"])

    colors = []
    for _, row in laptimes_df.iterrows():
        lap = int(row["lap"])
        if lap == best_lap:
            colors.append("#2ecc71")
        elif lap in excluded_laps:
            colors.append("#e74c3c")
        else:
            colors.append("#3498db")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=laptimes_df["lap"],
        y=laptimes_df[time_col],
        marker_color=colors,
        text=[format_laptime(t) for t in laptimes_df[time_col]],
        textposition="outside",
        hovertemplate="Lap %{x}<br>Time: %{text}<extra></extra>",
    ))

    best_time = clean_df.loc[best_idx, time_col]
    fig.add_hline(y=best_time, line_dash="dash", line_color="#2ecc71",
                  annotation_text=f"Best: {format_laptime(best_time)}",
                  annotation_position="bottom left",
                  annotation_font=dict(color="#2ecc71", size=11))

    avg_time = clean_df[time_col].mean()
    fig.add_hline(y=avg_time, line_dash="dot", line_color="#f39c12",
                  annotation_text=f"Avg: {format_laptime(avg_time)}",
                  annotation_position="bottom left",
                  annotation_font=dict(color="#f39c12", size=11))

    # Narrow y-axis range so differences are visible
    worst_time = laptimes_df[time_col].max()
    fig.update_layout(
        title="Lap Times",
        xaxis_title="Lap",
        yaxis_title="Time",
        yaxis_range=[best_time - 0.5, worst_time + 0.5],
        template="plotly_white",
        height=400,
    )

    return fig


def create_delta_to_best_chart(laptimes_df, time_col="seconds"):
    """Create a delta-to-best lap chart showing all laps."""
    clean_df, _ = detect_outliers(laptimes_df, time_col=time_col)
    best_time = clean_df[time_col].min()

    deltas = laptimes_df[time_col] - best_time
    colors = ["#2ecc71" if d <= 0.5 else "#f39c12" if d <= 1.0 else "#e74c3c"
              for d in deltas]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=laptimes_df["lap"],
        y=deltas,
        marker_color=colors,
        text=[f"+{d:.3f}s" for d in deltas.values],
        textposition="outside",
        hovertemplate="Lap %{x}<br>Delta: +%{y:.3f}s<extra></extra>",
    ))

    fig.update_layout(
        title="Delta to Best Lap",
        xaxis_title="Lap",
        yaxis_title="Delta (seconds)",
        template="plotly_white",
        height=350,
    )

    return fig


def create_rolling_average_chart(laptimes_df, time_col="seconds", window=3):
    """Create a rolling average lap time chart with session-segment shading."""
    clean_df, _ = detect_outliers(laptimes_df, time_col=time_col)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=clean_df["lap"],
        y=clean_df[time_col],
        mode="markers+lines",
        name="Lap Time",
        line=dict(color="#3498db", width=1),
        marker=dict(size=6),
        hovertemplate="Lap %{x}<br>Time: %{customdata}<extra></extra>",
        customdata=[format_laptime(t) for t in clean_df[time_col]],
    ))

    rolling = clean_df[time_col].rolling(window=window, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=clean_df["lap"],
        y=rolling,
        mode="lines",
        name=f"{window}-Lap Rolling Avg",
        line=dict(color="#e74c3c", width=2, dash="dash"),
    ))

    # Session segment shading (thirds)
    n_laps = len(clean_df)
    if n_laps >= 6:
        third = n_laps // 3
        segments = [
            (0, third, "Early", "rgba(46,204,113,0.08)"),
            (third, 2 * third, "Mid", "rgba(52,152,219,0.08)"),
            (2 * third, n_laps, "Late", "rgba(231,76,60,0.08)"),
        ]
        laps = clean_df["lap"].values
        times = clean_df[time_col].values
        for start, end, label, color in segments:
            seg_laps = laps[start:end]
            seg_avg = float(np.mean(times[start:end]))
            fig.add_vrect(
                x0=seg_laps[0] - 0.5, x1=seg_laps[-1] + 0.5,
                fillcolor=color, line_width=0,
            )
            fig.add_annotation(
                x=(seg_laps[0] + seg_laps[-1]) / 2,
                y=seg_avg,
                text=f"{label}: {format_laptime(seg_avg)}",
                showarrow=False,
                font=dict(size=10, color="#555"),
                bgcolor="rgba(255,255,255,0.7)",
            )

    fig.update_layout(
        title="Lap Time Trend",
        xaxis_title="Lap",
        yaxis_title="Time",
        template="plotly_white",
        height=350,
    )

    return fig


def create_laptime_histogram(laptimes_df, time_col="seconds"):
    """Histogram of lap time distribution with best and mean lines."""
    clean_df, _ = detect_outliers(laptimes_df, time_col=time_col)
    times = clean_df[time_col]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=times,
        nbinsx=min(10, max(5, len(times) // 2)),
        marker_color="#3498db",
        opacity=0.8,
        hovertemplate="Time: %{x:.3f}s<br>Count: %{y}<extra></extra>",
    ))

    best = times.min()
    mean = times.mean()

    fig.add_vline(x=best, line_dash="dash", line_color="#2ecc71", line_width=2,
                  annotation_text=f"Best: {format_laptime(best)}")
    fig.add_vline(x=mean, line_dash="dot", line_color="#f39c12", line_width=2,
                  annotation_text=f"Mean: {format_laptime(mean)}")

    fig.update_layout(
        title="Lap Time Distribution",
        xaxis_title="Time (seconds)",
        yaxis_title="Count",
        template="plotly_white",
        height=350,
    )

    return fig
