#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# Read the Sunspots dataset
data = pd.read_csv("C:\\Users\\admin\\Downloads\\archive (4).zip")

# Convert Date column to datetime and set as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Focus on the 'Monthly Mean Total Sunspot Number' column
sunspot_data = data[['Monthly Mean Total Sunspot Number']]

# Display the shape and first 10 rows
print("Shape of the dataset:", sunspot_data.shape)
print("First 10 rows of the dataset:")
print(sunspot_data.head(10))

# Plot original data
plt.figure(figsize=(12, 6))
plt.plot(sunspot_data, label='Original Sunspot Data', color='blue')
plt.title('Original Monthly Mean Total Sunspot Data')
plt.xlabel('Year')
plt.ylabel('Sunspot Count')
plt.legend()
plt.grid()
plt.show()

# Moving averages (window = 5 and 10)
rolling_mean_5 = sunspot_data['Monthly Mean Total Sunspot Number'].rolling(window=5).mean()
rolling_mean_10 = sunspot_data['Monthly Mean Total Sunspot Number'].rolling(window=10).mean()

print("\nFirst 10 values of rolling mean (window=5):")
print(rolling_mean_5.head(10))
print("\nFirst 20 values of rolling mean (window=10):")
print(rolling_mean_10.head(20))

# Plot moving averages
plt.figure(figsize=(12, 6))
plt.plot(sunspot_data, label='Original Data', color='blue')
plt.plot(rolling_mean_5, label='Moving Average (window=5)', linestyle='--')
plt.plot(rolling_mean_10, label='Moving Average (window=10)', linestyle='-.')
plt.title('Moving Average of Sunspot Data')
plt.xlabel('Year')
plt.ylabel('Sunspot Count')
plt.legend()
plt.grid()
plt.show()

# Data transformation - scaling
scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(
        sunspot_data.values.reshape(-1, 1)
    ).flatten(),
    index=sunspot_data.index
)

# Add +1 to avoid zero/negative values (for multiplicative models)
scaled_data = scaled_data + 1

# Train-test split (80%-20%)
x = int(len(scaled_data) * 0.8)
train_data = scaled_data[:x]
test_data = scaled_data[x:]

# Exponential Smoothing model (additive trend, multiplicative seasonality)
model_add = ExponentialSmoothing(
    train_data, trend='add', seasonal='mul', seasonal_periods=12
).fit()

# Forecast test data
test_predictions_add = model_add.forecast(steps=len(test_data))

# Plot actual vs predicted
ax = train_data.plot(label='Train Data', figsize=(12,6))
test_predictions_add.plot(ax=ax, label='Predictions')
test_data.plot(ax=ax, label='Test Data')
ax.set_title('Holt-Winters Forecast on Sunspot Data')
ax.legend()
plt.grid()
plt.show()

# RMSE
rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print("\nRoot Mean Squared Error (RMSE):", rmse)

# Mean and variance
print("Variance:", scaled_data.var())
print("Mean:", scaled_data.mean())

# Future Forecast (one-fourth of dataset length)
model_final = ExponentialSmoothing(
    scaled_data, trend='add', seasonal='mul', seasonal_periods=12
).fit()

future_predictions = model_final.forecast(steps=int(len(scaled_data) / 4))

# Plot future forecast
ax = scaled_data.plot(label='Original Data', figsize=(12,6))
future_predictions.plot(ax=ax, label='Future Predictions', color='red')
ax.set_xlabel('Year')
ax.set_ylabel('Scaled Sunspot Count')
ax.set_title('Future Forecast of Monthly Mean Sunspot Number')
ax.legend()
plt.grid()
plt.show()


# In[ ]:




