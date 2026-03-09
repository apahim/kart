"""Fetch historical weather data from the Open-Meteo archive API."""

import json
import urllib.request
import urllib.error


# WMO weather interpretation codes
_WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def weathercode_to_condition(code):
    """Map a WMO weather code to a readable condition string."""
    return _WMO_CODES.get(code, f"Unknown ({code})")


def _degrees_to_cardinal(deg):
    """Convert wind direction in degrees to a cardinal/intercardinal string."""
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = round(deg / 45) % 8
    return directions[idx]


def fetch_weather(date, session_start, lat, lon, timezone="UTC"):
    """Fetch weather for a specific date/time/location from Open-Meteo archive.

    Args:
        date: Date string "YYYY-MM-DD".
        session_start: Time string "HH:MM".
        lat: Latitude.
        lon: Longitude.
        timezone: IANA timezone string.

    Returns:
        Dict with condition, temp_c, wind_kmh, weathercode, or None on failure.
    """
    try:
        hour = int(session_start.split(":")[0])

        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={date}&end_date={date}"
            f"&hourly=temperature_2m,weathercode,windspeed_10m,winddirection_10m"
            f"&timezone={timezone}"
        )

        req = urllib.request.Request(url, headers={"User-Agent": "kart-dashboard/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        codes = hourly.get("weathercode", [])
        winds = hourly.get("windspeed_10m", [])
        wind_dirs = hourly.get("winddirection_10m", [])

        if not times or hour >= len(times):
            return None

        code = codes[hour]
        wind_dir_deg = wind_dirs[hour] if hour < len(wind_dirs) else None
        return {
            "condition": weathercode_to_condition(code),
            "temp_c": round(temps[hour], 1) if temps[hour] is not None else None,
            "wind_kmh": round(winds[hour], 1) if winds[hour] is not None else None,
            "wind_direction_deg": round(wind_dir_deg) if wind_dir_deg is not None else None,
            "wind_direction": _degrees_to_cardinal(wind_dir_deg) if wind_dir_deg is not None else None,
            "weathercode": code,
        }

    except (urllib.error.URLError, OSError, KeyError, IndexError, ValueError) as e:
        print(f"  Weather fetch failed: {e}")
        return None
