import joblib
import json
import numpy as np
import pandas as pd
import os

# Define absolute paths assuming execution from project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'regional')

def load_regional_pipeline():
    model = joblib.load(os.path.join(MODEL_DIR, 'regional_model.pkl'))
    preprocessor = joblib.load(os.path.join(MODEL_DIR, 'regional_preprocessor.pkl'))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, 'regional_label_encoder.pkl'))
    with open(os.path.join(MODEL_DIR, 'regional_feature_names.json'), 'r') as f:
        feature_names = json.load(f)
    return model, preprocessor, label_encoder, feature_names

def predict_regional_top3(features_dict):
    """
    Predict Top 3 crops based on regional features.
    features_dict must contain keys matching regional_feature_names.json.
    """
    model, preprocessor, label_encoder, feature_names = load_regional_pipeline()
    
    # Create DataFrame from input
    df = pd.DataFrame([features_dict])
    
    # Ensure all required features exist
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = -1 # default missing value
            
    # Apply label encoders for categorical features
    for col in ['State', 'District', 'Season']:
        if col in df.columns and col in preprocessor:
            le = preprocessor[col]
            # Handle unseen categories by using the first class as a fallback
            df[col] = df[col].astype(str).apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
            
    # Order features correctly
    X = df[feature_names]
    
    # Predict probabilities
    probs = model.predict_proba(X)[0]
    
    # Get top 3 indices
    top3_idx = np.argsort(probs)[-3:][::-1]
    
    # Format output
    top3_crops = label_encoder.inverse_transform(top3_idx)
    top3_probs = probs[top3_idx]
    
    result = []
    for rank, (crop, prob) in enumerate(zip(top3_crops, top3_probs)):
        result.append({
            "rank": rank + 1,
            "crop": crop,
            "score": round(float(prob), 4)
        })
        
    return result
