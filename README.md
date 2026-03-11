# Kart Racing Data Analysis

Telemetry-driven kart racing analysis using data from a RaceBox Mini GPS logger collected via RaceChrono Pro. Generates interactive HTML dashboards with Plotly charts, coaching insights, and cross-session progression tracking.

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Generate all dashboards
python scripts/generate_all.py --all
```

## Dashboards

Generated HTML is written to `docs/` (committed for GitHub Pages).

### Session Dashboard

Each race session gets an interactive dashboard with three sections:

**Session Overview** — weather conditions, lap time statistics, action plan summary, and a lap times chart showing per-lap performance with best lap highlighting and outlier marking.

**Best Lap Deep Dive** — track map with corner annotations, cumulative delta chart comparing laps, speed trace envelope (best/worst/median), throttle/brake zone map, and a friction circle (GG diagram).

**Where to Improve** — per-corner time loss analysis, braking consistency comparison, and sector times table with delta highlighting.

### Racing Line Comparison

Each session dashboard includes a racing line comparison panel at the bottom. It overlays GPS traces from any lap in the current session or any other session at the same track, allowing visual comparison of lines across sessions.

### Generate commands

```bash
# Single race + index
python scripts/generate_all.py data/races/2026-03-08-Kiltorcan/

# All races + index
python scripts/generate_all.py --all

# Individual generators
python scripts/generate_dashboard.py data/races/2026-03-08-Kiltorcan/
python scripts/generate_index.py
```

Output structure:
```
docs/
├── index.html
└── races/
    └── 2026-03-08-Kiltorcan/
        └── dashboard.html
```

## Adding Race Data

1. Create a directory under `data/races/` named `YYYY-MM-DD-TrackName`
2. Export the RaceChrono Pro session CSV as `telemetry.csv` in the race directory
3. Create a `race.yaml` with session metadata:

```yaml
track: Track Name
date: 2026-03-08
driver_weight_kg: 80
kart_number: 7
weather:                       # optional — auto-fetched from Open-Meteo if omitted
  condition: overcast          # sunny / overcast / rainy / wet
  temp_c: 12
  wind: light                  # calm / light / moderate / strong
notes: "Session notes"
```

Weather data is automatically fetched from the Open-Meteo archive API when not provided in `race.yaml`.

## Track Configuration

Tracks are defined in `data/tracks.yaml` with GPS coordinates and corner definitions used for sector analysis and corner annotations on track maps.

```yaml
tracks:
  Track Name:
    lat: 52.123
    lon: -7.456
    timezone: Europe/Dublin
    corners:
      - { name: "Turn 1", lat: 52.123, lon: -7.456 }
```

Corner definitions are used to split laps into sectors and to annotate track map visualizations.

## Project Structure

```
data/
  races/YYYY-MM-DD-TrackName/   # race.yaml + telemetry.csv + summary_generated.yaml
  tracks.yaml                   # track corner definitions
scripts/
  generate_all.py               # orchestrates all generators
  generate_dashboard.py         # single-race dashboard
  generate_index.py             # index page
  load_data.py                  # telemetry & metadata loading
  analysis/                     # 14 analysis modules (see scripts/README.md)
  templates/                    # Jinja2 HTML templates
tests/                          # pytest test suite
docs/                           # generated HTML output (GitHub Pages)
```

## Running Tests

```bash
python -m pytest tests/ -x -q
```

## Legacy Scripts

`plot_laptimes.py` and `plot_telemetry.py` are matplotlib-based scripts superseded by the dashboard system.
