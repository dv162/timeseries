📈 Time Series Forecasting Strategies – Direct vs Recursive
This project explores two classic time series forecasting strategies – Direct Forecasting and Recursive Forecasting – using a univariate CO₂ concentration dataset. Both approaches are implemented using Linear Regression, with visualization and evaluation included.

🔁 Recursive Forecasting
Window Size: 5
Forecast Horizon: 1 step
Approach: Predict one value, feed it back recursively for next step
Model: LinearRegression

⏳ Direct Forecasting
Window Size: 5
Forecast Horizon: 3 steps
Approach: Train 3 separate models, one for each step ahead
Model: LinearRegression (one per target)


📊 Project Overview
We explore and compare two modeling strategies:

Strategy | Direct Forecasting | Recursive Forecasting
Description | Train one model per future step | Train one model for next step and reuse it
Use Case | Better for multi-step precision | Simpler, efficient for shorter horizons
Model | LinearRegression × 3 (one per step) | LinearRegression (single model)
Both use a sliding window of 5 time steps as input to predict future CO₂ levels.
