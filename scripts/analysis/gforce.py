"""G-force / friction circle analysis using Plotly."""

import plotly.graph_objects as go
import numpy as np


def create_gg_diagram(df, best_lap=None):
    """Create a GG diagram (friction circle) with quadrant coloring."""
    if "lateral_acc" not in df.columns or "longitudinal_acc" not in df.columns:
        return None

    if best_lap is not None and "lap_number" in df.columns:
        plot_df = df[df["lap_number"] == best_lap].copy()
        title = f"GG Diagram (Lap {best_lap})"
    else:
        plot_df = df.copy()
        title = "GG Diagram (All Laps)"

    plot_df = plot_df.dropna(subset=["lateral_acc", "longitudinal_acc"])

    lat_g = plot_df["lateral_acc"].values
    lon_g = plot_df["longitudinal_acc"].values

    fig = go.Figure()

    # Color-code by quadrant
    quadrants = [
        (lat_g >= 0, lon_g >= 0, "Accel+Right", "rgba(46,204,113,0.6)"),
        (lat_g < 0, lon_g >= 0, "Accel+Left", "rgba(52,152,219,0.6)"),
        (lat_g >= 0, lon_g < 0, "Brake+Right", "rgba(231,76,60,0.6)"),
        (lat_g < 0, lon_g < 0, "Brake+Left", "rgba(243,156,18,0.6)"),
    ]

    for mask_lat, mask_lon, name, color in quadrants:
        mask = mask_lat & mask_lon
        if mask.sum() == 0:
            continue
        fig.add_trace(go.Scatter(
            x=lat_g[mask],
            y=lon_g[mask],
            mode="markers",
            marker=dict(size=4, color=color),
            name=name,
            hovertemplate="Lateral: %{x:.2f}G<br>Longitudinal: %{y:.2f}G<extra></extra>",
        ))

    # Reference circles
    max_g = max(
        np.abs(lat_g).max() if len(lat_g) > 0 else 0.5,
        np.abs(lon_g).max() if len(lon_g) > 0 else 0.5,
    ) * 0.95
    theta = np.linspace(0, 2 * np.pi, 100)
    for r in [0.5, 1.0, max_g]:
        if r > 0:
            fig.add_trace(go.Scatter(
                x=r * np.cos(theta),
                y=r * np.sin(theta),
                mode="lines",
                line=dict(color="gray", width=1, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            ))

    # Quadrant labels
    label_offset = max_g * 0.75
    annotations = [
        (label_offset * 0.7, label_offset * 0.7, "Accel+Right"),
        (-label_offset * 0.7, label_offset * 0.7, "Accel+Left"),
        (label_offset * 0.7, -label_offset * 0.7, "Brake+Right"),
        (-label_offset * 0.7, -label_offset * 0.7, "Brake+Left"),
    ]
    for ax, ay, text in annotations:
        fig.add_annotation(
            x=ax, y=ay, text=text, showarrow=False,
            font=dict(size=10, color="#555"),
            bgcolor="rgba(255,255,255,0.6)",
        )

    # Axis labels as annotations
    fig.add_annotation(
        x=0, y=-max_g * 1.1, text="<b>Braking</b>", showarrow=False,
        font=dict(size=11, color="#888"),
    )
    fig.add_annotation(
        x=0, y=max_g * 1.1, text="<b>Acceleration</b>", showarrow=False,
        font=dict(size=11, color="#888"),
    )
    fig.add_annotation(
        x=-max_g * 1.1, y=0, text="<b>Left</b>", showarrow=False,
        font=dict(size=11, color="#888"),
    )
    fig.add_annotation(
        x=max_g * 1.1, y=0, text="<b>Right</b>", showarrow=False,
        font=dict(size=11, color="#888"),
    )

    fig.update_layout(
        title=title,
        xaxis=dict(title="Lateral G", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Longitudinal G"),
        template="plotly_white",
        height=500,
    )

    return fig
