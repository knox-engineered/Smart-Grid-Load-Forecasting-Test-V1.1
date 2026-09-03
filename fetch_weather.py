"""
Fetch historical weather data from Open-Meteo for India's 5 grid regions and
combine into a demand-share-WEIGHTED national proxy (better than a simple
average, since regions don't contribute equally to All-India demand).

No API key required. Docs: https://open-meteo.com/en/docs/historical-weather-api

Usage:
    python3 fetch_weather.py
"""
import requests
import pandas as pd

# ---- Config: edit to match your data range ----
START_DATE = "2026-08-01"
END_DATE = "2026-08-31"

# One representative city per NLDC grid region, weighted by each region's
# ACTUAL month-averaged share of All-India "Energy Met (MU)" - computed
# directly from your own August 2026 NLDC PSP files (MOP_E sheet) via
# extract_regional_weights.py. See regional_weights.csv for the source values.
REGIONS = {
    "Northern_Delhi":    {"lat": 28.6139, "lon": 77.2090, "weight": 0.333631},
    "Western_Mumbai":    {"lat": 19.0760, "lon": 72.8777, "weight": 0.278947},
    "Southern_Bengaluru":{"lat": 12.9716, "lon": 77.5946, "weight": 0.248591},
    "Eastern_Kolkata":   {"lat": 22.5726, "lon": 88.3639, "weight": 0.124711},
    "NorthEastern_Guwahati": {"lat": 26.1445, "lon": 91.7362, "weight": 0.014120},
}

HOURLY_VARS = "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,wind_speed_10m"
COMFORT_BASELINE_C = 22  # for Cooling Degree Day calculation


def fetch_city_weather(name, lat, lon):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": HOURLY_VARS,
        "timezone": "Asia/Kolkata",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["hourly"]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    return df


def main():
    weighted_vars = {v: None for v in
                      ["temperature_2m", "relative_humidity_2m", "apparent_temperature",
                       "precipitation", "wind_speed_10m"]}

    total_weight = sum(r["weight"] for r in REGIONS.values())
    assert abs(total_weight - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total_weight}"

    per_region = {}
    for name, cfg in REGIONS.items():
        print(f"Fetching weather for {name} (weight={cfg['weight']})...")
        df = fetch_city_weather(name, cfg["lat"], cfg["lon"])
        per_region[name] = df

        for var in weighted_vars:
            contribution = df[var] * cfg["weight"]
            weighted_vars[var] = contribution if weighted_vars[var] is None else weighted_vars[var] + contribution

    weighted = pd.DataFrame(weighted_vars)
    weighted.columns = [f"weighted_{c}" for c in weighted.columns]

    # Cooling Degree Day feature: only counts heat ABOVE the comfort baseline
    weighted["weighted_CDD"] = (weighted["weighted_temperature_2m"] - COMFORT_BASELINE_C).clip(lower=0)

    weighted.to_csv("weather_hourly.csv")
    print(f"\nSaved weather_hourly.csv: {weighted.shape[0]} hourly rows")
    print("Columns:", weighted.columns.tolist())
    print("Date range:", weighted.index.min(), "to", weighted.index.max())


if __name__ == "__main__":
    main()
