# Race Data

Each race is stored in its own directory named `YYYY-MM-DD-TrackName`.

## Directory Structure

```
YYYY-MM-DD-TrackName/
├── race.yaml                # Session metadata
├── telemetry.csv            # RaceChrono Pro CSV export (RaceBox Mini telemetry)
└── summary_generated.yaml   # Auto-generated race summary statistics
```

## Telemetry (`telemetry.csv`)

Export sessions from RaceChrono Pro as CSV. The app exports files with columns for timestamp, GPS coordinates, speed, acceleration, and other telemetry channels from the RaceBox Mini.

Lap times are derived automatically from the `elapsed_time` and `lap_number` columns in the telemetry data.

## Weather

Weather data is auto-fetched from the Open-Meteo archive API if not provided in `race.yaml`.
