# Data directory

The original soil-moisture observations and Sentinel-2 imagery are not included because they are private and/or too large for GitHub.

By default, the notebooks expect this layout:

```text
data/
├── weather/
│   ├── Dair_Alla.csv
│   └── Dair_Alla_2015_2025.csv
├── sentinel_downloads_1km/
├── sentinel_downloads_2km/
├── sentinel_downloads_3km/
└── testing/
    ├── classification/
    ├── classification_2km/
    ├── classification_3km/
    └── regression/
```

Each Sentinel sample is a GeoTIFF containing or derived from Sentinel-2 B8A and B11 bands. The weather CSV must include `date` and `soil_moisture_0_to_7cm` columns.

To keep data elsewhere, set `SOIL_DATA_DIR` in `.env` to an absolute or repository-relative path. See `.env.example`.

Public source services used by the project:

- [Sentinel Hub](https://www.sentinel-hub.com/) for Sentinel-2 imagery
- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) for weather and soil-moisture observations
