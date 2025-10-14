# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 14-10-2025


### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# Read the Sunspots dataset
data = pd.read_csv("Sunspots.csv")

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
```

### OUTPUT:

#### Moving Average:
```
First 10 values of rolling mean (window=5):
Date
1749-01-31       NaN
1749-02-28       NaN
1749-03-31       NaN
1749-04-30       NaN
1749-05-31    110.44
1749-06-30    118.94
1749-07-31    129.68
1749-08-31    128.44
1749-09-30    135.18
1749-10-31    132.00
Name: Monthly Mean Total Sunspot Number, dtype: float64

First 20 values of rolling mean (window=10):
Date
1749-01-31       NaN
1749-02-28       NaN
1749-03-31       NaN
1749-04-30       NaN
1749-05-31       NaN
1749-06-30       NaN
1749-07-31       NaN
1749-08-31       NaN
1749-09-30       NaN
1749-10-31    121.22
1749-11-30    137.98
1749-12-31    141.75
1750-01-31    142.30
1750-02-28    145.67
1750-03-31    146.37
1750-04-30    147.17
1750-05-31    146.37
1750-06-30    151.99
1750-07-31    153.57
1750-08-31    158.16
Name: Monthly Mean Total Sunspot Number, dtype: float64
```
#### Plot Transform Dataset:
<img width="1005" height="545" alt="image" src="https://github.com/user-attachments/assets/2b7cca88-27f0-43f2-9b5d-8c827b0d0277" />
<img width="992" height="545" alt="image" src="https://github.com/user-attachments/assets/5a92df90-f686-4af0-adf7-a4c382f2dfd8" />


```
Root Mean Squared Error (RMSE): 0.17906611954618418
Variance: 0.02906697583970718
Mean: 1.2053711071952424
```



#### Exponential Smoothing:
<img width="1001" height="545" alt="image" src="https://github.com/user-attachments/assets/654c491f-bfca-4cc4-80b3-1cc57a34606a" />



### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
