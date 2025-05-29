"""
Sweden Temperature Prediction
Dataset source: https://www.kaggle.com/datasets/adamwurdits/finland-norway-and-sweden-weather-data-20152019
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# 1. Load data
df = pd.read_csv("nordics_weather.csv")
df.columns = df.columns.str.strip()

# 2. Filter for Sweden
sweden = df[df["country"] == "Sweden"].copy()

# 3. Check for missing values
print("Missing values:")
for col, val in sweden.isnull().sum().items():
    print(f"{col}: {val}")

# Remove rows with missing temperature or precipitation
sweden = sweden.dropna(subset=["tavg", "precipitation"])

# 4. Check for extreme values (optional)
print("Temperature range:", sweden["tavg"].min(), "to", sweden["tavg"].max())
print("Precipitation range:", sweden["precipitation"].min(), "to", sweden["precipitation"].max())

# 5. Convert date to datetime and extract the day of the year
sweden["date"] = pd.to_datetime(sweden["date"])
sweden["day_of_year"] = sweden["date"].dt.dayofyear

# 6. Normalize precipitation (optional, for linear regression)
sweden["precip_norm"] = sweden["precipitation"] / sweden["precipitation"].max()

# 7. Feature selection
X = sweden[["day_of_year", "precip_norm"]]
y = sweden["tavg"]

# 8. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. Train linear regression
model = LinearRegression()
model.fit(X_train, y_train)

# 10. Predict on test data
y_pred = model.predict(X_test)

# 11. Results for Linear regression
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Linear regression - Mean Squared Error: {mse:.2f}")
print(f"Linear regression - R^2 Score: {r2:.2f}")

# 12. Simple plot: Actual vs Predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Temperature (°C)")
plt.ylabel("Predicted Temperature (°C)")
plt.title("Actual vs Predicted Temperature (Sweden)")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')
plt.show()

# 13. Train and evaluate Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_rf_pred = rf_model.predict(X_test)

# 14. Results for Random Forest
rf_mse = mean_squared_error(y_test, y_rf_pred)
rf_r2 = r2_score(y_test, y_rf_pred)
print(f"Random Forest - Mean Squared Error: {rf_mse:.2f}")
print(f"Random Forest - R^2 Score: {rf_r2:.2f}")
