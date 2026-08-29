import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def top_k_accuracy(probs, classes, y_true, k=3):
    correct = 0
    for i, true_class in enumerate(y_true):
        top_k_indices = np.argsort(probs[i])[-k:]
        top_k_classes = classes[top_k_indices]
        if true_class in top_k_classes:
            correct += 1
    return correct / len(y_true)

def train():
    os.makedirs('models/regional', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    print("Loading feature table...")
    df = pd.read_csv('data/processed/feature_table.csv', low_memory=False)
    
    # Clean numerical columns that might have 'NaT' or text
    for col in ['avgtmp_jan', 'avgtmp_jul', 'avgann_rf']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1)

    # We will use Approach A: Classification. 
    # Target: Crop
    # Features: State, District, Season, Soil, Climate
    # We will encode categorical features.
    
    cat_cols = ['State', 'District', 'Season']
    num_cols = ['NITROGEN', 'PHOSPHOROUS', 'POTASSIUM', 'PH', 
                'avgtmp_jan', 'avgtmp_jul', 'avgann_rf', 'ANNUAL', 'Jun-Sep']
    
    # We want to predict crops that actually have enough data.
    crop_counts = df['Crop'].value_counts()
    valid_crops = crop_counts[crop_counts > 50].index
    df = df[df['Crop'].isin(valid_crops)]
    
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        
    crop_le = LabelEncoder()
    df['Crop_Encoded'] = crop_le.fit_transform(df['Crop'].astype(str))
    
    # Save encoders
    joblib.dump(label_encoders, 'models/regional/regional_preprocessor.pkl')
    joblib.dump(crop_le, 'models/regional/regional_label_encoder.pkl')
    
    features = cat_cols + num_cols
    target = 'Crop_Encoded'
    
    # Temporal Split
    train_df = df[df['Year'] <= 2013]
    val_df = df[df['Year'] == 2014]
    test_df = df[df['Year'] >= 2015]
    
    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, max_depth=15),
        'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42, max_iter=50)
    }
    
    results = []
    best_f1 = -1
    best_model_name = ""
    best_model = None
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predict on Test for final metrics
        probs = model.predict_proba(X_test)
        preds = model.predict(X_test)
        
        top1 = accuracy_score(y_test, preds)
        top3 = top_k_accuracy(probs, model.classes_, y_test.values, k=3)
        prec = precision_score(y_test, preds, average='weighted', zero_division=0)
        rec = recall_score(y_test, preds, average='weighted', zero_division=0)
        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
        
        results.append({'Model': name, 'Top-1': top1, 'Top-3': top3, 'Precision': prec, 'Recall': rec, 'F1': f1})
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model
            
    # Save comparison report
    results_df = pd.DataFrame(results)
    results_df.to_csv('reports/regional_model_comparison.csv', index=False)
    
    # Save best model
    joblib.dump(best_model, 'models/regional/regional_model.pkl')
    
    # Save feature names
    with open('models/regional/regional_feature_names.json', 'w') as f:
        json.dump(features, f)
        
    # Save Metadata
    meta = {
        "model_name": best_model_name,
        "model_type": "Classification",
        "training_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "target_definition": "Crop Classification",
        "feature_names": features,
        "training_year_range": "1997-2013",
        "validation_year_range": "2014",
        "test_year_range": "2015",
        "top1_accuracy": float(results_df[results_df['Model'] == best_model_name]['Top-1'].iloc[0]),
        "top3_accuracy": float(results_df[results_df['Model'] == best_model_name]['Top-3'].iloc[0]),
        "precision": float(results_df[results_df['Model'] == best_model_name]['Precision'].iloc[0]),
        "recall": float(results_df[results_df['Model'] == best_model_name]['Recall'].iloc[0]),
        "f1": float(best_f1),
        "source_datasets": ["crop_production", "soil-nutrient-analysis", "district wise rainfall", "agroclimatic regions"],
        "preprocessing_version": "1.0",
        "random_seed": 42
    }
    
    with open('models/regional/regional_model_metadata.json', 'w') as f:
        json.dump(meta, f, indent=4)
        
    # Save selection report
    with open('reports/regional_best_model_selection.md', 'w') as f:
        f.write("# Regional Best Model Selection\n\n")
        f.write(f"The best model selected was **{best_model_name}** based on an F1-score of {best_f1:.4f} on the unseen test set (Year 2015).\n\n")
        f.write("## Comparison\n")
        f.write(results_df.to_markdown(index=False))

    print(f"Best model {best_model_name} saved successfully.")

if __name__ == "__main__":
    train()
