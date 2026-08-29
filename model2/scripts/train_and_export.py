import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier

def train_and_export():
    print("Loading merged regional data...")
    df = pd.read_csv('data/processed/merged_regional_data.csv')
    
    features_num = ['NITROGEN', 'PHOSPHOROUS', 'POTASSIUM', 'PH', 'avgtmp_jan', 'avgtmp_jul', 'avgann_rf', 'ANNUAL', 'Jun-Sep']
    for col in features_num:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[features_num] = df[features_num].fillna(-1)
    
    # Filter targets
    crop_counts = df['Crop'].value_counts()
    valid_crops = crop_counts[crop_counts > 6000].index
    df = df[df['Crop'].isin(valid_crops)]
    
    cat_cols = ['State', 'District', 'Season']
    preprocessors = {}
    
    print("Encoding categorical variables...")
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        preprocessors[col] = le
        
    print("Converting to Multi-Label problem...")
    group_cols = cat_cols + features_num
    multi_df = df.groupby(group_cols)['Crop'].unique().reset_index()
    
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(multi_df['Crop'])
    X = multi_df[group_cols]
    
    from sklearn.tree import DecisionTreeClassifier
    print(f"Training Multi-Label Decision Tree on {X.shape[0]} samples and {len(mlb.classes_)} unique crops...")
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    
    print("Exporting models to model2/models/...")
    os.makedirs('models', exist_ok=True)
    
    with open('models/regional_rf_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    with open('models/regional_label_encoders.pkl', 'wb') as f:
        pickle.dump(preprocessors, f)
        
    with open('models/regional_mlb.pkl', 'wb') as f:
        pickle.dump(mlb, f)
        
    print("Export successful!")

if __name__ == "__main__":
    train_and_export()
