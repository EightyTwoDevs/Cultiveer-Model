import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split

# 1) LOAD CLEANED DATA AND MODEL
df = pd.read_csv('cleaned_analytics_data.csv')
rf = joblib.load('npk_multioutput_model.pkl')  # The multi‐output RF from your training script

# Make sure you use the SAME features and random_state as in training
feature_cols = [
    'airTemp',
    'humidity',
    'lightLux',
    'soilMoisture',
    'soilTemp1',
    'soilTemp2'
]
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 2) MAKE PREDICTIONS ON THE TEST SET
y_pred = rf.predict(X_test)  # shape: (num_samples, 7)

# Combine true values and predictions into one DataFrame for easier plotting
results_df = X_test.copy()
for i, col in enumerate(target_cols):
    results_df[f'true_{col}'] = y_test[col].values
    results_df[f'pred_{col}'] = y_pred[:, i]

# ------------------------------------------------------------------------------
# 3) PREDICTED VS. ACTUAL SCATTER PLOTS
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()  # Flatten (2x4) into a list for easy iteration

for i, col in enumerate(target_cols):
    ax = axes[i]
    true_col = f'true_{col}'
    pred_col = f'pred_{col}'

    ax.scatter(results_df[true_col], results_df[pred_col], alpha=0.3)
    # Diagonal line
    min_val = min(results_df[true_col].min(), results_df[pred_col].min())
    max_val = max(results_df[true_col].max(), results_df[pred_col].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')

    ax.set_xlabel(f'Actual {col}')
    ax.set_ylabel(f'Predicted {col}')
    ax.set_title(f'{col}: Pred vs Actual')

# If there are unused subplots (because we have only 7 columns but 8 subplots),
# hide the extra axis if needed
if len(target_cols) < len(axes):
    for j in range(len(target_cols), len(axes)):
        axes[j].set_visible(False)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 4) RESIDUAL PLOTS
#   Residual = actual - predicted
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

for i, col in enumerate(target_cols):
    ax = axes[i]
    true_col = f'true_{col}'
    pred_col = f'pred_{col}'

    residuals = results_df[true_col] - results_df[pred_col]
    ax.scatter(results_df[pred_col], residuals, alpha=0.3)
    ax.axhline(0, color='red', linestyle='--')

    ax.set_xlabel(f'Predicted {col}')
    ax.set_ylabel('Residuals')
    ax.set_title(f'{col}: Residual Plot')

# Hide any unused subplot
if len(target_cols) < len(axes):
    for j in range(len(target_cols), len(axes)):
        axes[j].set_visible(False)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 5) ERROR DISTRIBUTIONS
#   Another way to look at residuals is via a histogram
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

for i, col in enumerate(target_cols):
    ax = axes[i]
    true_col = f'true_{col}'
    pred_col = f'pred_{col}'

    residuals = results_df[true_col] - results_df[pred_col]
    sns.histplot(residuals, bins=30, kde=True, ax=ax)
    ax.axvline(0, color='red', linestyle='--')
    ax.set_xlabel(f'Residual (Actual - Predicted)')
    ax.set_ylabel('Count')
    ax.set_title(f'{col}: Error Distribution')

# Hide any unused subplot
if len(target_cols) < len(axes):
    for j in range(len(target_cols), len(axes)):
        axes[j].set_visible(False)

plt.tight_layout()
plt.show()
