import pandas as pd
df = pd.read_csv('../data/raw/crop_production.csv')
print(df['Crop'].value_counts().head(20))
