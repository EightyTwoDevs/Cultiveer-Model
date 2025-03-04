import pandas as pd
import numpy as np

# 1. LOAD RAW DATA
df = pd.read_csv('analytics_data_test.csv')  # <-- Adjust filename as needed
print("Initial shape:", df.shape)
print(df.info())

# 2. REMOVE INVALID NPK VALUES (zeros)
df = df[(df['nitrogen'] != 0) &
        (df['phosphorus'] != 0) &
        (df['potassium'] != 0)]
print("After removing zero N/P/K:", df.shape)

# 3. REMOVE NEGATIVE VALUES FOR CERTAIN COLUMNS
#    List columns that should be >= 0 in your domain
non_negative_cols = [
    'airTemp', 'humidity', 'lightLux', 'soilMoisture',
    'soilTemp1', 'soilTemp2', 'ph', 'ec',
    'NPK_temperature', 'NPK_humidity'
]
for col in non_negative_cols:
    df = df[df[col] >= 0]

print("After removing negatives:", df.shape)

# 4. OUTLIER REMOVAL (Optional) USING Z-SCORE
zscore_threshold = 3
for col in non_negative_cols + ['nitrogen', 'phosphorus', 'potassium']:
    mean_val = df[col].mean()
    std_val = df[col].std()
    df = df[(df[col] > mean_val - zscore_threshold * std_val) &
            (df[col] < mean_val + zscore_threshold * std_val)]

print("After removing outliers (3 SD):", df.shape)

# 5. CONVERT TIMESTAMP COLUMN (IF ANY)
#    Adjust the datetime format if needed
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

# Drop rows where timestamp couldn't be parsed (if that occurs)
df.dropna(subset=['created_at'], inplace=True)

# 6. (Optional) CREATE TIME FEATURES
df['hour'] = df['created_at'].dt.hour
df['day'] = df['created_at'].dt.day
df['weekday'] = df['created_at'].dt.weekday

print("Final cleaned shape:", df.shape)

# 7. SAVE THE CLEANED DATA
df.to_csv('cleaned_analytics_data.csv', index=False)
print("Cleaned data saved to cleaned_analytics_data.csv")
