#Libraries
import numpy as np
# Constants
total_days = 365
average_demand = 28500

# Generate time series data with seasonal pattern and noise
time = np.arange(total_days)
##Scenario1
seasonality_pattern1 = (20000 * np.sin (2 * np.pi * time / total_days - np.pi/2) +
    10000 * np.sin (4 * np.pi * time / total_days - np.pi/4) +
    5000 * np.sin (6 * np.pi * time / total_days + np.pi/4))
noise_std1 = 4000
noise1 = np.random.normal (scale=noise_std1, size=total_days)
daily_demand1 = average_demand + seasonality_pattern1 + noise1
daily_demand1 = np.maximum(daily_demand1, 5000)
##Scenario2
seasonality_pattern2 = (15000 * np.sin(2 * np.pi * time / total_days) +
    5000 * np.cos(4 * np.pi * time / total_days) +
    1000 * time / total_days)
noise_std2 = 4000
noise2 = np.random.normal (scale=noise_std2, size=total_days)
daily_demand2 = average_demand + seasonality_pattern2 + noise2
daily_demand2 = np.maximum(daily_demand2, 5000)
##Scenario 3
seasonality_pattern3 = (25000 * np.sin(2 * np.pi * time / total_days) +
    15000 * np.sin(4 * np.pi * time / total_days) +
    10000 * np.sin(6 * np.pi * time / total_days))
noise_std3 = 3000 * np.abs(np.sin(2 * np.pi * time / total_days))
noise3 = np.random.normal (scale=noise_std3, size=total_days)
daily_demand3 = average_demand + seasonality_pattern3 + noise3
daily_demand3 = np.maximum(daily_demand3, 5000)
