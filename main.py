from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import date
from utils import pred_crop, pred_rainfall, pred_temp_hum
import pickle
import pandas as pd
import numpy as np
import geopandas as gpd
import os
from dotenv import load_dotenv
import google.generativeai as genai
from utils.explainer import generate_shap_plot_base64
import pickle

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

with open("outputmodel1/crop_recommendation_rf_model.pkl", "rb") as file:
    rf_model_for_shap = pickle.load(file)
with open("outputmodel1/crop_recommendation_scaler.pkl", "rb") as file:
    scaler_for_shap = pickle.load(file)

# Load the exported regional models
try:
    with open('model2/models/regional_rf_model.pkl', 'rb') as f:
        regional_rf_model = pickle.load(f)
    with open('model2/models/regional_label_encoders.pkl', 'rb') as f:
        regional_label_encoders = pickle.load(f)
    with open('model2/models/regional_mlb.pkl', 'rb') as f:
        regional_mlb = pickle.load(f)
except Exception as e:
    print(f"Warning: Could not load regional models: {e}")
    regional_rf_model = None

# Pre-load data for automated lookup
try:
    agro_gdf = gpd.read_file('model2/data/raw/Agroclimatic_regions/Agroclimatic_regions.shp')
    agro_gdf['state'] = agro_gdf['state'].astype(str).str.strip().str.upper()
    state_agro = agro_gdf.groupby('state')[['avgtmp_jan', 'avgtmp_jul', 'avgann_rf']].first()
except:
    state_agro = pd.DataFrame(columns=['avgtmp_jan', 'avgtmp_jul', 'avgann_rf'])

try:
    rain_df = pd.read_csv('model2/data/raw/district wise rainfall normal.csv')
    rain_df['STATE_UT_NAME'] = rain_df['STATE_UT_NAME'].astype(str).str.strip().str.upper()
    rain_df['DISTRICT'] = rain_df['DISTRICT'].astype(str).str.strip().str.upper()
    rain_df = rain_df.set_index(['STATE_UT_NAME', 'DISTRICT'])
except:
    rain_df = pd.DataFrame()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


class Inputs(BaseModel):
    nitrogen: float
    phosphorous: float
    potassium: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


@app.post("/predict/")
async def predict(inputs: Inputs):
    try:
        prediction = pred_crop.predict_crop(
            inputs.nitrogen, inputs.phosphorous, inputs.potassium,
            inputs.temperature, inputs.humidity, inputs.ph, inputs.rainfall)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"result": prediction}

class RegionalInputs(BaseModel):
    state: str
    district: str
    season: str
    nitrogen: float
    phosphorous: float
    potassium: float
    ph: float
    rainfall: float 

@app.post("/predict_regional/")
async def predict_regional(inputs: RegionalInputs):
    if not regional_rf_model:
        raise HTTPException(status_code=500, detail="Regional models not loaded.")
        
    state = inputs.state.strip().upper()
    district = inputs.district.strip().upper()
    season = inputs.season.strip().upper()
    
    # 1. Look up missing agroclimatic data
    if state in state_agro.index:
        temp_jan = float(state_agro.loc[state, 'avgtmp_jan'])
        temp_jul = float(state_agro.loc[state, 'avgtmp_jul'])
        avgann_rf = float(state_agro.loc[state, 'avgann_rf'])
    else:
        temp_jan, temp_jul, avgann_rf = -1.0, -1.0, -1.0
        
    # 2. Look up missing rainfall data
    if (state, district) in rain_df.index:
        jun_sep_rf = float(rain_df.loc[(state, district), 'Jun-Sep'])
        cur_ann_rf = inputs.rainfall if inputs.rainfall > 0 else float(rain_df.loc[(state, district), 'ANNUAL'])
    else:
        jun_sep_rf = -1.0
        cur_ann_rf = inputs.rainfall if inputs.rainfall > 0 else -1.0

    # Handle NaNs from lookup
    if pd.isna(temp_jan): temp_jan = -1.0
    if pd.isna(temp_jul): temp_jul = -1.0
    if pd.isna(avgann_rf): avgann_rf = -1.0
    if pd.isna(jun_sep_rf): jun_sep_rf = -1.0
    if pd.isna(cur_ann_rf): cur_ann_rf = -1.0
        
    try:
        # Safe transform with fallback for unknown classes if needed, but LabelEncoder throws ValueError for unknown
        state_enc = regional_label_encoders['State'].transform([state])[0]
        district_enc = regional_label_encoders['District'].transform([district])[0]
        season_enc = regional_label_encoders['Season'].transform([season])[0]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid Region/Season combination: {e}")
        
    input_features = np.array([[state_enc, district_enc, season_enc, inputs.nitrogen, inputs.phosphorous, inputs.potassium, inputs.ph, temp_jan, temp_jul, avgann_rf, cur_ann_rf, jun_sep_rf]])
    
    prediction = regional_rf_model.predict(input_features)
    predicted_crops = regional_mlb.inverse_transform(prediction)[0]
    
    if len(predicted_crops) == 0:
        return {"results": []}
    
    return {"results": [{"crop": c.title(), "confidence": 100} for c in predicted_crops]}


import requests

class HybridInputs(BaseModel):
    state: str
    district: str

# Pre-load features for hybrid model
try:
    feature_df = pd.read_csv('model2/data/processed/feature_table.csv', low_memory=False)
    cols = ['NITROGEN', 'PHOSPHOROUS', 'POTASSIUM', 'PH', 'avgann_rf']
    for col in cols:
        feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
    district_features = feature_df.groupby(['State', 'District'])[cols].median().reset_index()
    district_features.set_index(['State', 'District'], inplace=True)
    
    # Pre-load historical crops for validation
    historical_crops = feature_df.groupby(['State', 'District'])['Crop'].unique().to_dict()
except Exception as e:
    print(f"Warning: Could not load feature_table.csv for hybrid model: {e}")
    district_features = pd.DataFrame()
    historical_crops = {}

def fetch_weather(district_name: str):
    try:
        # 1. Geocode
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={district_name}&count=1&language=en&format=json"
        geo_resp = requests.get(geo_url).json()
        if not geo_resp.get("results"):
            return None, None
            
        lat = geo_resp["results"][0]["latitude"]
        lon = geo_resp["results"][0]["longitude"]
        
        # 2. Weather
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m"
        weather_resp = requests.get(weather_url).json()
        
        temp = weather_resp["current"]["temperature_2m"]
        hum = weather_resp["current"]["relative_humidity_2m"]
        return temp, hum
    except Exception as e:
        print(f"Weather API Error: {e}")
        return None, None

@app.post("/predict_hybrid/")
async def predict_hybrid(inputs: HybridInputs):
    state = inputs.state.strip().upper()
    district = inputs.district.strip().upper()
    
    # 1. Fetch live weather
    temp, hum = fetch_weather(district)
    if temp is None:
        temp, hum = 25.0, 70.0 # Fallback
        
    # 2. Fetch historical soil data
    if (state, district) in district_features.index:
        n_raw = district_features.loc[(state, district), 'NITROGEN']
        p_raw = district_features.loc[(state, district), 'PHOSPHOROUS']
        k_raw = district_features.loc[(state, district), 'POTASSIUM']
        ph_raw = district_features.loc[(state, district), 'PH']
        rf_raw = district_features.loc[(state, district), 'avgann_rf']
    else:
        raise HTTPException(status_code=400, detail=f"Invalid Region: '{district}' does not exist in '{state}'. Please verify spelling.")
        
    # Heuristic scaling for 7-parameter model
    n_val = 90.0 if n_raw > 1.5 else (50.0 if n_raw > 0.5 else 20.0)
    p_val = 80.0 if p_raw > 1.5 else (40.0 if p_raw > 0.5 else 15.0)
    k_val = 80.0 if k_raw > 1.5 else (40.0 if k_raw > 0.5 else 15.0)
    ph_val = 7.5 if ph_raw > 1.5 else (6.5 if ph_raw > 0.5 else 5.5)
    rf_val = rf_raw if rf_raw > 0 else 150.0
    
    # 3. Predict using high-accuracy Model 1
    try:
        recommended_crops = pred_crop.predict_crop(n_val, p_val, k_val, temp, hum, ph_val, rf_val, top_k=3)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model 1 Prediction Failed: {e}")
        
    # 4. Regional Validation
    valid_crops = historical_crops.get((state, district), [])
    valid_crops_upper = [str(c).upper() for c in valid_crops if str(c).lower() != 'nan']
    
    # Check if the #1 recommended crop is historically validated
    is_validated = recommended_crops[0]["crop"].upper() in valid_crops_upper
    
    return {
        "crop": recommended_crops,
        "validated": is_validated,
        "live_temp": temp,
        "live_humidity": hum,
        "soil_data": {"N": n_val, "P": p_val, "K": k_val, "pH": ph_val, "Rainfall": rf_val}
    }

class ExplainInputs(BaseModel):
    crop: str
    nitrogen: float
    phosphorous: float
    potassium: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

@app.post("/explain/")
async def explain_recommendation(inputs: ExplainInputs):
    if not os.getenv("GEMINI_API_KEY"):
        return {"explanation": "Gemini API key is not configured. Unable to provide AI explanation."}
        
    try:
        model = genai.GenerativeModel('gemini-3.6-flash')
        prompt = (
            f"You are an AI Agricultural Assistant. A Random Forest model has recommended the crop '{inputs.crop}' "
            f"based on soil (N={inputs.nitrogen}, P={inputs.phosphorous}, K={inputs.potassium}, pH={inputs.ph}) "
            f"and weather (Temp={inputs.temperature}C, Humidity={inputs.humidity}%, Rain={inputs.rainfall}mm). "
            f"Explain to a farmer in simple terms why this crop is suitable for these conditions, "
            f"and give 2 brief tips for farming it. Keep it under 4 sentences."
        )
        response = model.generate_content(prompt)
        return {"explanation": response.text}
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return {"explanation": "AI explanation is temporarily unavailable."}

@app.post("/explain_plots/")
async def explain_plots(inputs: ExplainInputs):
    try:
        feature_values = [
            inputs.nitrogen, inputs.phosphorous, inputs.potassium, 
            inputs.temperature, inputs.humidity, inputs.ph, inputs.rainfall
        ]
        feature_names = ['Nitrogen', 'Phosphorous', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall']
        
        shap_b64 = generate_shap_plot_base64(rf_model_for_shap, scaler_for_shap, feature_values, feature_names)
        
        return {
            "shap_base64": shap_b64,
            "lime_base64": None
        }
    except Exception as e:
        print(f"Explain Plots Error: {e}")
        return {"shap_base64": None, "lime_base64": None}
