import joblib
import pandas as pd

# Load the trained multi-output model
model = joblib.load('npk_multioutput_model.pkl')

# Suppose you have new sensor readings in a DataFrame or 2D array with columns:
# ['airTemp', 'humidity', 'lightLux', 'soilMoisture', 'soilTemp1', 'soilTemp2']
new_data = pd.DataFrame({
    #tesing  line number 24202
    'airTemp': [28],
    'humidity': [17],
    'lightLux': [43.33],
    'soilMoisture': [77],
    'soilTemp1': [29.31],
    'soilTemp2': [29.37]
})

# Make the prediction
predictions = model.predict(new_data)  # shape (1, 7)

print("Predicted [N, P, K, pH, EC, NPK_temp, NPK_humidity]:")
print(predictions[0])
