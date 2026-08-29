import os
import sys

# Ensure src module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.regional_predictor import predict_regional_top3

def test_prediction():
    # Provide sample features corresponding to the training table
    sample_features = {
        'State': 'MAHARASHTRA',
        'District': 'PUNE',
        'Season': 'Kharif',
        'NITROGEN': 120.5,
        'PHOSPHOROUS': 40.2,
        'POTASSIUM': 25.1,
        'PH': 6.5,
        'avgtmp_jan': 20.0,
        'avgtmp_jul': 28.0,
        'avgann_rf': 800.0,
        'ANNUAL': 820.0,
        'Jun-Sep': 600.0
    }
    
    print("Regional Model Loaded Successfully\n")
    print("Test Location:")
    print(f"State: {sample_features['State']}")
    print(f"District: {sample_features['District']}\n")
    
    try:
        top3 = predict_regional_top3(sample_features)
        print("Top 3:")
        for res in top3:
            print(f"{res['rank']}. {res['crop']} — Score: {res['score']}")
    except Exception as e:
        print(f"Error making prediction: {e}")

if __name__ == "__main__":
    test_prediction()
