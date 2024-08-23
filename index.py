# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

# Load datasets for the years 2020 and 2021
df1 = pd.read_csv("anantapur 2020-01-01 to 2020-12-31.csv")
df2 = pd.read_csv("anantapur 2021-01-01 to 2021-12-31.csv")

# Combine datasets and save to a new CSV file
df_combined = pd.concat([df1, df2], ignore_index=True)
df_combined.to_csv('df_combined(2020+2021)_without_FeatureScaling.csv', index=False)

# Convert 'datetime' column to datetime format
df_combined['datetime'] = pd.to_datetime(df_combined['datetime'])

# Forward fill missing values
df_combined.ffill(inplace=True)

# Calculate mean temperature
df_combined['mean_temp'] = (df_combined['tempmax'] + df_combined['tempmin']) / 2

# Convert sunrise and sunset times to datetime format and calculate daylight hours
df_combined['sunrise'] = pd.to_datetime(df_combined['sunrise'])
df_combined['sunset'] = pd.to_datetime(df_combined['sunset'])
df_combined['daylight_hours'] = (df_combined['sunset'] - df_combined['sunrise']).dt.total_seconds() / 3600

# Define conditions for planting
df_combined['is_planting_time'] = (
    (df_combined['mean_temp'] >= 20) & (df_combined['mean_temp'] <= 42) &  # Ideal temperature for planting
    (df_combined['humidity'] >= 70) & (df_combined['humidity'] <= 90) &  # Sufficient humidity for paddy planting
    (df_combined['daylight_hours'] >= 10)  # Sufficient daylight
).astype(int)

# Define conditions for harvesting
df_combined['is_harvest_time'] = (
    (df_combined['mean_temp'] >= 20) & (df_combined['mean_temp'] <= 25) &  # Ideal temperature for harvesting
    (df_combined['humidity'] <= 70)  # Low humidity to prevent spoilage
).astype(int)

# Plot temperature trends over time
plt.figure(figsize=(12, 6))
plt.plot(df_combined['datetime'], df_combined['mean_temp'], label='Mean Temperature')
plt.axhline(42, color='r', linestyle='--', label='Planting Upper Temp Limit')
plt.axhline(20, color='g', linestyle='--', label='Planting Lower and Harvesting Lower Temp Limit')
plt.axhline(25, color='b', linestyle='--', label='Harvesting Upper Temp Limit')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Trend Over Time')
plt.legend()
plt.show()

# Plot humidity trends over time
plt.figure(figsize=(12, 6))
plt.plot(df_combined['datetime'], df_combined['humidity'], label='Humidity')
plt.axhline(70, color='r', linestyle='--', label='Harvesting Humidity Limit')
plt.axhline(90, color='b', linestyle='--', label='Planting Humidity Limit')
plt.xlabel('Date')
plt.ylabel('Humidity (%)')
plt.title('Humidity Trend Over Time')
plt.legend()
plt.show()

# Split the data into training and testing sets based on the year
train_data = df_combined[df_combined['datetime'].dt.year == 2020].copy()
test_data = df_combined[df_combined['datetime'].dt.year == 2021].copy()

# Define features and labels for training
X_train = train_data[['tempmax', 'tempmin', 'humidity',  'daylight_hours']]
y_train_planting = train_data['is_planting_time']
y_train_harvesting = train_data['is_harvest_time']

X_test = test_data[['tempmax', 'tempmin', 'humidity', 'daylight_hours']]
y_test_planting = test_data['is_planting_time']
y_test_harvesting = test_data['is_harvest_time']

# Train XGBoost model for planting prediction
model_xgb_planting = xgb.XGBClassifier(scale_pos_weight=10, random_state=42)
model_xgb_planting.fit(X_train, y_train_planting)

# Predict and evaluate planting model
y_pred_planting = model_xgb_planting.predict(X_test)
print("Planting Time Prediction:")
print(classification_report(y_test_planting, y_pred_planting))
print(f"Overall Accuracy (Planting): {accuracy_score(y_test_planting, y_pred_planting):.6f}")

# Train XGBoost model for harvesting prediction
model_xgb_harvesting = xgb.XGBClassifier(scale_pos_weight=10, random_state=42)
model_xgb_harvesting.fit(X_train, y_train_harvesting)

# Predict and evaluate harvesting model
y_pred_harvesting = model_xgb_harvesting.predict(X_test)
print("Harvesting Time Prediction:")
print(classification_report(y_test_harvesting, y_pred_harvesting))
print(f"Overall Accuracy (Harvesting): {accuracy_score(y_test_harvesting, y_pred_harvesting):.6f}")

# Plot predicted planting months for 2021
test_data['predicted_planting'] = y_pred_planting
planting_months = test_data[test_data['predicted_planting'] == 1].groupby(test_data['datetime'].dt.to_period('M')).size()

plt.figure(figsize=(12, 6))
planting_months.plot(kind='bar', color='skyblue')
plt.title('Predicted Planting Months for 2021')
plt.xlabel('Month')
plt.ylabel('Number of Days')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot predicted harvesting months for 2021
test_data['predicted_harvesting'] = y_pred_harvesting
harvesting_months = test_data[test_data['predicted_harvesting'] == 1].groupby(test_data['datetime'].dt.to_period('M')).size()

plt.figure(figsize=(12, 6))
harvesting_months.plot(kind='bar', color='salmon')
plt.title('Predicted Harvesting Months for 2021')
plt.xlabel('Month')
plt.ylabel('Number of Days')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
