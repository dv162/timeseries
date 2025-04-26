# 📈 Time Series Forecasting Strategies – Direct vs Recursive
This project explores two classic time series forecasting strategies – Direct Forecasting and Recursive Forecasting – using a univariate CO₂ concentration dataset. Both approaches are implemented using Linear Regression, with visualization and evaluation included.

# 🔁 Recursive Forecasting
Window Size: 5
Forecast Horizon: 1 step
Approach: Predict one value, feed it back recursively for next step
Model: LinearRegression
![Image](https://github.com/user-attachments/assets/3dcb9baa-a1fb-46ca-8bd2-2a88b9a22519)



# ⏳ Direct Forecasting
Window Size: 5
Forecast Horizon: 3 steps
Approach: Train 3 separate models, one for each step ahead
Model: LinearRegression (one per target)
![Image](https://github.com/user-attachments/assets/7b060761-c76f-40c6-b56b-bcab984f0960)
