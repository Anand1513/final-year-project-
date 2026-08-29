# Climate Feature Decision

## Available Data
We audited the datasets in `data/raw/` to locate climate data (Temperature, Humidity, Rainfall).
- **Rainfall**: Abundant district-level and regional-level rainfall data exists (e.g., `district wise rainfall normal.csv` and `Agroclimatic_regions.shp` contains `avgann_rf`).
- **Temperature & Humidity**: The historical temperature and humidity at the District + Year level are missing from the raw CSV datasets. The only temperature available is the static regional-level average January and July temperatures in the `Agroclimatic_regions.shp` file.

## Decision
As per the strict requirements:
1. We **will not** invent values or use random data.
2. We **will not** use current weather APIs for historical years.

**Strategy**: The regional model is explicitly **climate-data-limited** in terms of time-variant Temperature and Humidity. 
We will train the regional model using the available static `avgtmp_jan` and `avgtmp_jul` from the Agroclimatic Region mappings. Humidity will be omitted from the regional model entirely. Rainfall will be incorporated using the robust normal rainfall datasets.

This limitation is documented here and will be reflected in the model metadata.
