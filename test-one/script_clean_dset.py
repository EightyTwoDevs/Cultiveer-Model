import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 1) LOAD CSV
df = pd.read_csv('analytics_data_test.csv')

print("Initial shape:", df.shape)
print(df.info())

# ----------------------------------------------------------------------------
# 2) CLEAN DATA

# (a) Remove rows where N, P, K are 0 (invalid readings)
#     Adjust these column names to match your actual CSV
df = df[(df['nitrogen'] != 0) &
        (df['phosphorus'] != 0) &
        (df['potassium'] != 0)]

# (b) Remove negative values in other relevant columns
#     Example: you might want to ensure no negative soilMoisture, lightLux, etc.
#     Create a list of columns we expect to be >= 0
non_negative_cols = ['airTemp', 'humidity', 'lightLux', 'soilMoisture',
                     'soilTemp1', 'soilTemp2', 'ph', 'ec',
                     'NPK_temperature', 'NPK_humidity']
for col in non_negative_cols:
    df = df[df[col] >= 0]

print("After removing zeros in N, P, K and negatives in others:", df.shape)

# (c) (Optional) Remove or cap extreme outliers
#     Example: remove rows beyond 3 standard deviations for each column
#     (especially if you have some physically impossible spikes)
zscore_threshold = 3
for col in non_negative_cols + ['nitrogen', 'phosphorus', 'potassium']:
    # Compute mean & std
    col_mean = df[col].mean()
    col_std = df[col].std()
    # Filter out outliers beyond threshold
    df = df[(df[col] > col_mean - zscore_threshold * col_std) &
            (df[col] < col_mean + zscore_threshold * col_std)]

print("After optional outlier removal:", df.shape)

# (d) Convert timestamp if needed (example format: '2025-02-14 14:02:09')
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
# Drop any rows where timestamp could not be parsed
df.dropna(subset=['created_at'], inplace=True)

# (e) (Optional) Extract useful time features (hour, day, etc.)
df['hour'] = df['created_at'].dt.hour
df['day'] = df['created_at'].dt.day
df['weekday'] = df['created_at'].dt.weekday  # Monday=0, Sunday=6

# ----------------------------------------------------------------------------
# 3) PREPARE FEATURES & TARGETS

# Choose your feature columns (adjust to your dataset)
feature_cols = [
    'airTemp', 'humidity', 'lightLux', 'soilMoisture',
    'soilTemp1', 'soilTemp2', 'ph', 'ec', 'NPK_temperature',
    'NPK_humidity',
    # Possibly the derived time features
    'hour', 'day', 'weekday'
]

# Multi‐output: you want to predict N, P, K
target_cols = ['nitrogen', 'phosphorus', 'potassium']

X = df[feature_cols]
y = df[target_cols]

# (Optional) Scale or normalize features
#   Random Forest usually doesn't need it, but if you plan to try linear models
#   or neural networks later, scaling can help.
#   Uncomment if you want to scale:

# scaler = StandardScaler()
# X[feature_cols] = scaler.fit_transform(X[feature_cols])

# ----------------------------------------------------------------------------
# 4) SPLIT INTO TRAIN/TEST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------------------------------------
# 5) TRAIN A RANDOM FOREST MODEL
rf = RandomForestRegressor(
    n_estimators=100,  # number of trees
    random_state=42,
    # max_depth=..., min_samples_split=..., etc., if you want to tune
)
rf.fit(X_train, y_train)

# ----------------------------------------------------------------------------
# 6) EVALUATE THE MODEL
y_pred = rf.predict(X_test)

# Evaluate with common regression metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred, multioutput='uniform_average')

print("Random Forest - MSE:", mse)
print("Random Forest - R^2:", r2)

# (Optional) Evaluate each target separately
# if you want separate R2 for N, P, K:
y_pred_N = y_pred[:, 0]  # first column is nitrogen
y_pred_P = y_pred[:, 1]  # second column is phosphorus
y_pred_K = y_pred[:, 2]  # third column is potassium

r2_n = r2_score(y_test['nitrogen'], y_pred_N)
r2_p = r2_score(y_test['phosphorus'], y_pred_P)
r2_k = r2_score(y_test['potassium'], y_pred_K)

print("R^2 (N):", r2_n)
print("R^2 (P):", r2_p)
print("R^2 (K):", r2_k)
