import pickle
import numpy as np

def predict_crop(nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall, top_k=3):
    return get_prediction((nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall), top_k)

def get_prediction(x, top_k=3):
    # Load Random Forest Model
    with open("outputmodel1/crop_recommendation_rf_model.pkl", "rb") as file:
        model = pickle.load(file)
        
    # Load Scaler
    with open("outputmodel1/crop_recommendation_scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
        
    # The Crop Dictionary mapping
    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    # Input should be a 2D array: shape (1, 7)
    input_vector = np.array([x])
    
    # Scale input
    scaled_vector = scaler.transform(input_vector)
    
    # Get probabilities
    probs = model.predict_proba(scaled_vector)[0]
    
    # Get top K indices (sorted descending)
    topk_indices = np.argsort(probs)[-top_k:][::-1]
    
    results = []
    for idx in topk_indices:
        class_id = model.classes_[idx]
        prob = probs[idx]
        confidence = round(prob * 100, 1)
        crop_name = crop_dict.get(class_id, "Unknown")
        results.append({"crop": crop_name, "confidence": confidence})

    return results
