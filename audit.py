import pandas as pd
import geopandas as gpd
import os

os.makedirs('reports', exist_ok=True)

with open('reports/regional_data_audit.md', 'w') as f:
    f.write("# Regional Data Audit\n\n")

    datasets = [
        ('soil-nutrient-analysis.csv', pd.read_csv),
        ('district wise rainfall normal.csv', pd.read_csv),
        ('crop_production.csv', pd.read_csv),
        ('ICRISAT-District Level Data.csv', lambda x: pd.read_csv(x, encoding='latin1')),
        ('Agroclimatic_regions/Agroclimatic_regions.shp', gpd.read_file)
    ]

    for filename, reader in datasets:
        path = os.path.join('data/raw', filename)
        f.write(f"## {filename}\n")
        try:
            df = reader(path)
            f.write(f"- **Rows:** {len(df)}\n")
            f.write(f"- **Columns:** {len(df.columns)}\n")
            f.write(f"- **Schema:**\n")
            for col in df.columns:
                f.write(f"  - `{col}` ({df[col].dtype})\n")
            f.write(f"- **Missing Values:**\n")
            missing = df.isnull().sum()
            for col, count in missing.items():
                if count > 0:
                    f.write(f"  - `{col}`: {count}\n")
            f.write(f"- **Duplicates:** {df.duplicated().sum()}\n")
            f.write("\n")
        except Exception as e:
            f.write(f"**Error loading dataset:** {e}\n\n")
