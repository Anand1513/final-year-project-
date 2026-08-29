import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Set up directories
os.makedirs('../output image 2', exist_ok=True)
os.makedirs('../output model 2', exist_ok=True)

print("1. Loading Data...")
base_df = pd.read_csv('../data/raw/crop_production.csv')
base_df.columns = ['State', 'District', 'Year', 'Season', 'Crop', 'Area', 'Production']
for col in ['State', 'District', 'Crop', 'Season']:
    base_df[col] = base_df[col].astype(str).str.strip().str.upper()

base_df = base_df.dropna(subset=['Production', 'Area'])
base_df = base_df[base_df['Area'] > 0]
base_df['Yield'] = base_df['Production'] / base_df['Area']

soil_df = pd.read_csv('../data/raw/soil-nutrient-analysis.csv')
for col in ['state_name', 'district_name', 'nutrient_name']:
    soil_df[col] = soil_df[col].astype(str).str.strip().str.upper()

valid_nutrients = ['NITROGEN', 'PHOSPHORUS', 'POTASSIUM', 'SOIL PH']
soil_filtered = soil_df[soil_df['nutrient_name'].isin(valid_nutrients)]
soil_agg = soil_filtered.groupby(['state_name', 'district_name', 'nutrient_name'])['value'].median().reset_index()
soil_pivot = soil_agg.pivot(index=['state_name', 'district_name'], columns='nutrient_name', values='value').reset_index()
soil_pivot = soil_pivot.rename(columns={'state_name': 'State', 'district_name': 'District', 'PHOSPHORUS': 'PHOSPHOROUS', 'SOIL PH': 'PH'})

rain_df = pd.read_csv('../data/raw/district wise rainfall normal.csv')
rain_df['STATE_UT_NAME'] = rain_df['STATE_UT_NAME'].astype(str).str.strip().str.upper()
rain_df['DISTRICT'] = rain_df['DISTRICT'].astype(str).str.strip().str.upper()
rain_df = rain_df.rename(columns={'STATE_UT_NAME': 'State', 'DISTRICT': 'District'})

try:
    agro_gdf = gpd.read_file('../data/raw/Agroclimatic_regions/Agroclimatic_regions.shp')
    agro_gdf['state'] = agro_gdf['state'].astype(str).str.strip().str.upper()
    state_agro = agro_gdf.groupby('state')[['avgtmp_jan', 'avgtmp_jul', 'avgann_rf']].first().reset_index()
    state_agro = state_agro.rename(columns={'state': 'State'})
except:
    state_agro = pd.DataFrame(columns=['State', 'avgtmp_jan', 'avgtmp_jul', 'avgann_rf'])

df = pd.merge(base_df, soil_pivot, on=['State', 'District'], how='left')
df = pd.merge(df, state_agro, on='State', how='left')
df = pd.merge(df, rain_df[['State', 'District', 'ANNUAL', 'Jun-Sep']], on=['State', 'District'], how='left')

for col in ['avgtmp_jan', 'avgtmp_jul', 'avgann_rf']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("2. Feature Engineering...")
df = df.sort_values(by=['State', 'District', 'Season', 'Crop', 'Year']).reset_index(drop=True)
group_cols = ['State', 'District', 'Season', 'Crop']
df['historical_mean_yield'] = df.groupby(group_cols)['Yield'].apply(lambda x: x.shift(1).expanding().mean()).reset_index(level=group_cols, drop=True)
df['historical_mean_yield'] = df['historical_mean_yield'].fillna(df.groupby(['State', 'Season', 'Crop'])['Yield'].transform('mean'))
df = df.fillna(-1)

crop_counts = df['Crop'].value_counts()
valid_crops = crop_counts[crop_counts > 6000].index
df = df[df['Crop'].isin(valid_crops)]

print("3. Preprocessing for Multi-Label...")
cat_cols = ['State', 'District', 'Season']
num_cols = ['NITROGEN', 'PHOSPHOROUS', 'POTASSIUM', 'PH', 'avgtmp_jan', 'avgtmp_jul', 'avgann_rf', 'ANNUAL', 'Jun-Sep', 'historical_mean_yield']

preprocessors = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    preprocessors[col] = le

group_cols_multi = ['State', 'District', 'Season', 'NITROGEN', 'PHOSPHOROUS', 'POTASSIUM', 'PH', 'avgtmp_jan', 'avgtmp_jul', 'avgann_rf', 'ANNUAL', 'Jun-Sep']
multi_df = df.groupby(group_cols_multi)['Crop'].unique().reset_index()

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(multi_df['Crop'])
X = multi_df[group_cols_multi]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print("4. Training Random Forest Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
preds = model.predict(X_test)

print("5. Evaluating and Generating Metrics...")
def jaccard_score_sample(y_true, y_pred):
    scores = []
    for t, p in zip(y_true, y_pred):
        intersection = np.logical_and(t, p).sum()
        union = np.logical_or(t, p).sum()
        if union == 0: scores.append(1.0)
        else: scores.append(intersection / union)
    return np.mean(scores)

acc = accuracy_score(y_test, preds)
jaccard = jaccard_score_sample(y_test, preds)
prec = precision_score(y_test, preds, average='micro', zero_division=0)
rec = recall_score(y_test, preds, average='micro', zero_division=0)
f1 = f1_score(y_test, preds, average='micro', zero_division=0)

metrics_text = f"Accuracy: {acc:.4f}\nJaccard: {jaccard:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1 Score: {f1:.4f}"
print(metrics_text)

# Save metrics to a text file in output image 2
with open('../output image 2/evaluation_metrics.txt', 'w') as f:
    f.write("Model 2 Evaluation Metrics\n")
    f.write("==========================\n")
    f.write(metrics_text)

print("6. Plotting Feature Importance...")
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(10, 6))
plt.title("Feature Importances for Regional Multi-Label Crop Prediction")
plt.bar(range(X.shape[1]), importances[indices], align="center", color='#4ade80')
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45, ha='right')
plt.tight_layout()
plt.savefig('../output image 2/feature_importance.png', dpi=300)
plt.close()

print("7. Plotting Performance Graph...")
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Jaccard']
metrics_values = [acc, prec, rec, f1, jaccard]

plt.figure(figsize=(8, 5))
plt.bar(metrics_names, metrics_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
plt.title('Regional Model Performance Metrics')
plt.ylim(0, 1)
for i, v in enumerate(metrics_values):
    plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('../output image 2/performance_metrics.png', dpi=300)
plt.close()

print("8. Saving Models to /output model 2/...")
with open('../output model 2/regional_rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
with open('../output model 2/regional_label_encoders.pkl', 'wb') as f:
    pickle.dump(preprocessors, f)
    
with open('../output model 2/regional_mlb.pkl', 'wb') as f:
    pickle.dump(mlb, f)

print("Done! Check 'model2/output image 2/' and 'model2/output model 2/'.")
