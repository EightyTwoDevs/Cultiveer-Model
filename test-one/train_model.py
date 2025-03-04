import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. LOAD THE CLEANED DATA
df = pd.read_csv('cleaned_analytics_data.csv')
print("Loaded cleaned data:", df.shape)

# 2. PREPARE FEATURES & TARGETS
feature_cols = [
    'airTemp', 'humidity', 'lightLux', 'soilMoisture',
    'soilTemp1', 'soilTemp2', 'ph', 'ec',
    'NPK_temperature', 'NPK_humidity',
    'hour', 'day', 'weekday'
]

target_cols = ['nitrogen', 'phosphorus', 'potassium']

X = df[feature_cols]
y = df[target_cols]

# (Optional) If you want scaling for later methods:
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X[feature_cols] = scaler.fit_transform(X[feature_cols])

# 3. SPLIT INTO TRAIN/TEST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. TRAIN RANDOM FOREST
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 5. EVALUATE
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred, multioutput='uniform_average')

print("Random Forest MSE:", mse)
print("Random Forest R^2:", r2)

# Evaluate each target individually
import numpy as np
y_pred_N = y_pred[:, 0]
y_pred_P = y_pred[:, 1]
y_pred_K = y_pred[:, 2]

r2_n = r2_score(y_test['nitrogen'], y_pred_N)
r2_p = r2_score(y_test['phosphorus'], y_pred_P)
r2_k = r2_score(y_test['potassium'], y_pred_K)

print(f"R^2 (N): {r2_n:.3f}")
print(f"R^2 (P): {r2_p:.3f}")
print(f"R^2 (K): {r2_k:.3f}")

# 6. SAVE THE MODEL
# change the modle name to npk_rf_model.pkl
joblib.dump(rf, 'npk_rf_model.pkl')
print("Model saved to npk_rf_model.pkl")
