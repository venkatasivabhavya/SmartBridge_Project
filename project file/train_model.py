import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Sample dataset creation
data = {
    'Age': [45, 34, 50, 40, 60, 55, 48, 33, 58, 62],
    'Total_Bilirubin': [1.0, 0.9, 2.5, 1.1, 3.2, 2.8, 1.4, 0.8, 3.5, 4.0],
    'Direct_Bilirubin': [0.3, 0.2, 1.0, 0.4, 1.2, 1.0, 0.5, 0.2, 1.5, 1.8],
    'Alkaline_Phosphotase': [210, 180, 300, 240, 320, 310, 230, 190, 340, 360],
    'Alamine_Aminotransferase': [35, 30, 45, 38, 50, 48, 37, 32, 52, 60],
    'Aspartate_Aminotransferase': [50, 40, 65, 55, 70, 68, 52, 42, 75, 80],
    'Total_Protiens': [6.5, 6.8, 5.5, 6.2, 5.0, 5.2, 6.1, 6.9, 5.3, 5.1],
    'Albumin': [3.0, 3.2, 2.8, 3.1, 2.5, 2.6, 3.0, 3.4, 2.7, 2.4],
    'Albumin_and_Globulin_Ratio': [1.0, 1.1, 0.8, 1.0, 0.7, 0.8, 1.0, 1.2, 0.75, 0.65],
    'Liver_Cirrhosis': [1, 0, 1, 0, 1, 1, 0, 0, 1, 1]
}

df = pd.DataFrame(data)

# Features and target
X = df.drop('Liver_Cirrhosis', axis=1)
y = df['Liver_Cirrhosis']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
with open('rf_acc_68.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the scaler
with open('normalizer.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully!")