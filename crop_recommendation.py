# Basic libraries
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")
print("Dataset loaded. Shape:", df.shape)
print(df.head())

# Dataset info
print("\nDataset Info:")
df.info()

print("\nDataset Description:")
print(df.describe())

# Feature & Target Split
X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
y = df["label"]

# Label Encoding
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
print("\nEncoded Labels Sample:", y_encoded[:10])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Model Training (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("\nModel trained successfully.")

# Prediction & Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)
print("Train Score:", model.score(X_train, y_train))
print("Test Score:", model.score(X_test, y_test))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# Single Input Prediction Example
sample_input = np.array([[75, 40, 40, 24, 70, 6.8, 160]])
prediction = model.predict(sample_input)
predicted_crop = encoder.inverse_transform(prediction)
print("\nSample Prediction for input", sample_input[0], ":", predicted_crop[0])

# Save Model & Encoder
with open("crop_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("\n✅ Model and Label Encoder saved as 'crop_model.pkl' and 'label_encoder.pkl'")

