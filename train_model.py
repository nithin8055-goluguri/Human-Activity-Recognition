import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

csv_file = os.path.join("data", "webcam_har_data.csv")
if not os.path.exists(csv_file):
    print("[✗] Error: CSV data file not found!")
    exit()

print("⏳ Loading dataset and preprocessing columns...")
df = pd.read_csv(csv_file)

X = df.drop('label', axis=1).values
y = df['label'].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("⏳ Training Random Forest Classifier model...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"[✓] Training complete! Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")

pickle.dump(model,  open("model.pkl", "wb"))
pickle.dump(le,     open("label_encoder.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
print("[✓] Successfully exported model.pkl, label_encoder.pkl, and scaler.pkl!")
