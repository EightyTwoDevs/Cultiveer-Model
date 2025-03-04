import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. LOAD CLEANED DATA & MODEL
df = pd.read_csv('cleaned_analytics_data.csv')
rf = joblib.load('npk_rf_model.pkl')

feature_cols = [
    'airTemp', 'humidity', 'lightLux', 'soilMoisture',
    'soilTemp1', 'soilTemp2', 'ph', 'ec',
    'NPK_temperature', 'NPK_humidity',
    'hour', 'day', 'weekday'
]
target_cols = ['nitrogen', 'phosphorus', 'potassium']

X = df[feature_cols]
y = df[target_cols]

# Split same way as in training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Predict on test set
y_pred = rf.predict(X_test)

# Combine predictions with true values for convenient plotting
results_df = X_test.copy()
results_df['true_n'] = y_test['nitrogen']
results_df['true_p'] = y_test['phosphorus']
results_df['true_k'] = y_test['potassium']
results_df['pred_n'] = y_pred[:, 0]
results_df['pred_p'] = y_pred[:, 1]
results_df['pred_k'] = y_pred[:, 2]

# ------------------------------------------------------------------------------
# 2. PREDICTED vs ACTUAL PLOT FOR EACH N, P, K

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
targets = ['n', 'p', 'k']

for i, t in enumerate(targets):
    true_col = f'true_{t}'
    pred_col = f'pred_{t}'

    axes[i].scatter(
        results_df[true_col],
        results_df[pred_col],
        alpha=0.3
    )
    # Diagonal line for reference
    min_val = min(results_df[true_col].min(), results_df[pred_col].min())
    max_val = max(results_df[true_col].max(), results_df[pred_col].max())
    axes[i].plot([min_val, max_val], [min_val, max_val], 'r--')
    axes[i].set_xlabel(f'Actual {t.upper()}')
    axes[i].set_ylabel(f'Predicted {t.upper()}')
    axes[i].set_title(f'{t.upper()}: Predicted vs. Actual')

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 3. RESIDUAL PLOTS (Actual - Predicted)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, t in enumerate(targets):
    true_col = f'true_{t}'
    pred_col = f'pred_{t}'
    residuals = results_df[true_col] - results_df[pred_col]

    axes[i].scatter(results_df[pred_col], residuals, alpha=0.3)
    axes[i].axhline(0, color='red', linestyle='--')
    axes[i].set_xlabel(f'Predicted {t.upper()}')
    axes[i].set_ylabel('Residuals')
    axes[i].set_title(f'{t.upper()}: Residual Plot')

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 4. DISTRIBUTION OF ERRORS

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, t in enumerate(targets):
    true_col = f'true_{t}'
    pred_col = f'pred_{t}'
    residuals = results_df[true_col] - results_df[pred_col]

    axes[i].hist(residuals, bins=30)
    axes[i].axvline(0, color='red', linestyle='--')
    axes[i].set_xlabel('Error (Actual - Predicted)')
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'{t.upper()}: Error Distribution')

plt.tight_layout()
plt.show()
