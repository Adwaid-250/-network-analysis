import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from joblib import dump 

# 1. Load the dataset
df = pd.read_csv("mesh_idps_dataset.csv")  # Replace with your actual filename

# 2. Drop or encode non-numeric columns
drop_cols = ['ID', 'Timestamp', 'Protocol', 'Type', 'SourceIP', 'DestIP']
df = df.drop(columns=drop_cols, errors='ignore')

# Encode categorical columns
if 'Protocol' in df.columns:
    df['Protocol'] = LabelEncoder().fit_transform(df['Protocol'])
# if df['Label'].dtype == 'object':
#     df['Label'] = LabelEncoder().fit_transform(df['Label'])

# 3. Sample exactly 200 entries per class (if available)
sampled_df = pd.concat([
    group.sample(n=300, random_state=42)
    for _, group in df.groupby('Label')
    if len(group) >= 300
])

X = sampled_df.drop(columns=['Label'])
y = sampled_df['Label']

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Normalize feature values (fit only on training data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
dump(scaler, 'scaler.joblib')

# === Save to CSVs ===
# Convert scaled arrays back to DataFrames before saving
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv("X_trainscaled.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv("X_testscaled.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print(f" Done! Training samples: {X_train.shape}, Classes: {y_train.nunique()}")
print("Final shape for QML:", X_train_scaled.shape, y_train.shape)
