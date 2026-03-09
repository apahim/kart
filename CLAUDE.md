# CLAUDE.md

## Environment

- Always use the project venv: `venv/bin/python` and `venv/bin/pip`
- Never use system python

## Key Commands

```bash
venv/bin/python -m pytest tests/ -x -q          # run tests
venv/bin/python scripts/generate_all.py --all    # regenerate all dashboards
```

## Project Layout

- `data/races/YYYY-MM-DD-TrackName/` — race.yaml + telemetry.csv + summary_generated.yaml
- `data/tracks.yaml` — track corner definitions and coordinates
- `scripts/` — generator scripts (generate_all, generate_dashboard, generate_evolution, generate_index)
- `scripts/analysis/` — 14 analysis modules (laptimes, track_map, speed, gforce, corners, braking, sectors, coaching, evolution, summary, outliers, weather, tracks, utils)
- `scripts/templates/` — Jinja2 HTML templates (dashboard.html.j2, evolution.html.j2, index.html.j2)
- `scripts/load_data.py` — telemetry and metadata loading functions
- `tests/` — pytest test suite
- `docs/` — generated HTML output (committed for GitHub Pages)

## Data Conventions

- RaceChrono v3 CSV: header row, units row, source row; duplicate columns deduplicated by sensor suffix
- Speed in telemetry is m/s (multiply by 3.6 for km/h)
- Lap times derived from telemetry via `extract_laptimes_from_telemetry()`
- Non-race laps filtered by `filter_non_race_laps()` in `scripts/analysis/outliers.py`
- Weather auto-fetched from Open-Meteo API if not provided in race.yaml

## Code Conventions

- Plotly for all charts (no matplotlib in dashboards)
- Jinja2 templates for HTML generation
- IQR-based outlier filtering for lap time statistics
- Track corners loaded from `data/tracks.yaml` for sector boundaries and map annotations

## Workflow

After making changes:
1. Run tests: `venv/bin/python -m pytest tests/ -x -q`
2. Regenerate dashboards: `venv/bin/python scripts/generate_all.py --all`
