import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------
# 1. LOAD YOUR CLEANED DATA
# -------------------------
# Make sure your CSV has valid sensor columns and NPK columns.
# This might be the "cleaned_analytics_data.csv" from your previous steps.
df = pd.read_csv('cleaned_analytics_data.csv')
print("Loaded data shape:", df.shape)

# -------------------------
# 2. DEFINE FEATURES & TARGETS
# -------------------------
# "Inputs" (features) come from pot sensors only:
feature_cols = [
    'airTemp',
    'humidity',
    'lightLux',
    'soilMoisture',
    'soilTemp1',
    'soilTemp2'
    # Optionally add hour, day, weekday if you created time features
]

# "Targets" (outputs) are all the readings from the NPK sensor:
target_cols = [
    'nitrogen',
    'phosphorus',
    'potassium',
    'ph',
    'ec',
    'NPK_temperature',
    'NPK_humidity'
]

X = df[feature_cols]
y = df[target_cols]

# -------------------------
# 3. SPLIT INTO TRAIN/TEST
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42
)

# -------------------------
# 4. TRAIN A RANDOM FOREST
# -------------------------
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)

# -------------------------
# 5. EVALUATE THE MODEL
# -------------------------
y_pred = rf.predict(X_test)  # shape: (num_samples, 7)

# Overall MSE and R^2 (averaged across all 7 targets)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred, multioutput='uniform_average')

print("Random Forest MSE:", mse)
print("Random Forest R^2:", r2)

# Evaluate each target individually
print("\nPer-target R^2 scores:")
for i, col in enumerate(target_cols):
    r2_individual = r2_score(y_test[col], y_pred[:, i])
    print(f"{col}: R^2 = {r2_individual:.3f}")

# -------------------------
# 6. SAVE THE MODEL
# -------------------------
joblib.dump(rf, 'npk_multioutput_model.pkl')
print("Model saved to npk_multioutput_model.pkl")
