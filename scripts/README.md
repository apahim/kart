# Scripts

Python scripts for generating kart racing dashboards and analyzing telemetry data.

## Generator Scripts

| Script | Description |
|--------|-------------|
| `generate_all.py` | Orchestrates dashboard, evolution, and index generation |
| `generate_dashboard.py` | Generates a single-race interactive HTML dashboard |
| `generate_evolution.py` | Generates cross-session progression dashboard |
| `generate_index.py` | Generates the index page linking all race dashboards |

## Analysis Modules (`analysis/`)

| Module | Description |
|--------|-------------|
| `laptimes.py` | Lap time charts with best lap highlighting and outlier marking |
| `track_map.py` | Track map visualizations with corner annotations and weather indicators |
| `speed.py` | Speed trace envelope showing best/worst/median laps |
| `gforce.py` | Friction circle (GG diagram) with quadrant coloring |
| `corners.py` | Corner detection and per-corner time loss statistics |
| `braking.py` | Braking/acceleration zone analysis with track map coloring |
| `sectors.py` | Sector time analysis using track corner boundaries |
| `coaching.py` | Coaching summary and structured action plan generation |
| `evolution.py` | Cross-race improvement tracking across sessions |
| `summary.py` | Race summary statistics with IQR-filtered lap times |
| `outliers.py` | IQR-based outlier detection and non-race lap filtering |
| `weather.py` | Weather data fetching from Open-Meteo archive API |
| `tracks.py` | Track configuration loading from `data/tracks.yaml` |
| `utils.py` | Shared Plotly utilities, chart encoding, and formatting helpers |

## Data Loading

`load_data.py` provides functions for loading race data:

- `load_telemetry(race_dir)` — loads and parses `telemetry.csv` (RaceChrono v3 format)
- `extract_laptimes_from_telemetry()` — derives lap times from elapsed time and lap number columns
- `load_race_yaml(race_dir)` — loads session metadata from `race.yaml`

## Templates (`templates/`)

| Template | Description |
|----------|-------------|
| `dashboard.html.j2` | Single-race dashboard with all chart sections |
| `evolution.html.j2` | Cross-session evolution page |
| `index.html.j2` | Index page listing all race sessions |

## Legacy Scripts

| Script | Description |
|--------|-------------|
| `plot_laptimes.py` | Matplotlib bar chart of lap times (superseded by dashboard) |
| `plot_telemetry.py` | Matplotlib telemetry plots (superseded by dashboard) |
