import os
import pandas as pd
import geopandas as gpd
import numpy as np

def clean_text(series):
    return series.astype(str).str.strip().str.upper()

def preprocess():
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('scripts', exist_ok=True)

    print("Loading datasets...")
    # 1. Base table: crop_production.csv
    base_df = pd.read_csv('data/raw/crop_production.csv')
    base_df.columns = ['State', 'District', 'Year', 'Season', 'Crop', 'Area', 'Production']
    base_df['State'] = clean_text(base_df['State'])
    base_df['District'] = clean_text(base_df['District'])
    base_df['Crop'] = clean_text(base_df['Crop'])
    base_df['Season'] = clean_text(base_df['Season'])
    
    # Drop rows with NaN production or area == 0
    base_df = base_df.dropna(subset=['Production', 'Area'])
    base_df = base_df[base_df['Area'] > 0]
    base_df['Yield'] = base_df['Production'] / base_df['Area']

    # 2. Process soil nutrient data
    print("Processing soil data...")
    soil_df = pd.read_csv('data/raw/soil-nutrient-analysis.csv')
    soil_df['state_name'] = clean_text(soil_df['state_name'])
    soil_df['district_name'] = clean_text(soil_df['district_name'])
    soil_df['nutrient_name'] = clean_text(soil_df['nutrient_name'])
    
    # Pivot soil data at district level
    # We want median value per district for N, P, K, pH (if available)
    valid_nutrients = ['NITROGEN', 'PHOSPHORUS', 'POTASSIUM', 'SOIL PH']
    soil_filtered = soil_df[soil_df['nutrient_name'].isin(valid_nutrients)]
    
    soil_agg = soil_filtered.groupby(['state_name', 'district_name', 'nutrient_name'])['value'].median().reset_index()
    soil_pivot = soil_agg.pivot(index=['state_name', 'district_name'], columns='nutrient_name', values='value').reset_index()
    soil_pivot = soil_pivot.rename(columns={'state_name': 'State', 'district_name': 'District', 'PHOSPHORUS': 'PHOSPHOROUS', 'SOIL PH': 'PH'})
    
    # 3. Process Agroclimatic regions
    print("Processing agroclimatic data...")
    try:
        agro_gdf = gpd.read_file('data/raw/Agroclimatic_regions/Agroclimatic_regions.shp')
        agro_gdf['state'] = clean_text(agro_gdf['state'])
        # Simplified: map states to their primary agroclimatic region for simplicity in this join
        state_agro = agro_gdf.groupby('state')[['avgtmp_jan', 'avgtmp_jul', 'avgann_rf']].first().reset_index()
        state_agro = state_agro.rename(columns={'state': 'State'})
    except Exception as e:
        print(f"Warning: Could not process Agroclimatic regions properly: {e}")
        state_agro = pd.DataFrame(columns=['State', 'avgtmp_jan', 'avgtmp_jul', 'avgann_rf'])

    # 4. Process Rainfall
    print("Processing rainfall data...")
    rain_df = pd.read_csv('data/raw/district wise rainfall normal.csv')
    rain_df['STATE_UT_NAME'] = clean_text(rain_df['STATE_UT_NAME'])
    rain_df['DISTRICT'] = clean_text(rain_df['DISTRICT'])
    rain_df = rain_df.rename(columns={'STATE_UT_NAME': 'State', 'DISTRICT': 'District'})
    
    # 5. Merge all together
    print("Merging features...")
    df = pd.merge(base_df, soil_pivot, on=['State', 'District'], how='left')
    df = pd.merge(df, state_agro, on='State', how='left')
    df = pd.merge(df, rain_df[['State', 'District', 'ANNUAL', 'Jun-Sep']], on=['State', 'District'], how='left')
    
    # Save the processed base table
    df.to_csv('data/processed/merged_regional_data.csv', index=False)
    print(f"Saved merged_regional_data.csv with shape {df.shape}")

if __name__ == "__main__":
    preprocess()
