
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load cleaned orders
data = pd.read_csv("output/cleaned_orders.csv")

data.columns = data.columns.str.strip().str.lower()

# 2. Convert order timestamp to date
data["order_datetime"] = pd.to_datetime(
    data["order_datetime"], errors="coerce"
)

data = data.dropna(subset=["order_datetime"])

# Use all orders, including cancelled orders, to forecast demand
data["order_date"] = data["order_datetime"].dt.floor("D")

# 3. Aggregate order volume by day
daily = (
    data.groupby("order_date")
    .size()
    .reset_index(name="order_count")
    .sort_values("order_date")
)

daily["order_date"] = pd.to_datetime(daily["order_date"])

# Include dates with zero orders, if any
daily = daily.set_index("order_date").asfreq("D", fill_value=0)
daily.index.name = "order_date"
daily = daily.reset_index()

# 4. Create calendar and historical features
daily["day_of_week"] = daily["order_date"].dt.dayofweek
daily["day_of_month"] = daily["order_date"].dt.day
daily["month"] = daily["order_date"].dt.month
daily["year"] = daily["order_date"].dt.year
daily["is_weekend"] = (
    daily["day_of_week"] >= 5
).astype(int)

# Historical order counts: use only past observations
daily["lag_1"] = daily["order_count"].shift(1)
daily["lag_7"] = daily["order_count"].shift(7)
daily["rolling_mean_7"] = (
    daily["order_count"].shift(1).rolling(7).mean()
)

# Remove rows without enough history
model_data = daily.dropna().copy()

features = [
    "day_of_week",
    "day_of_month",
    "month",
    "year",
    "is_weekend",
    "lag_1",
    "lag_7",
    "rolling_mean_7"
]

X = model_data[features]
y = model_data["order_count"]

if len(model_data) < 30:
    raise ValueError(
        "Not enough daily history to train the demand model."
    )

# 5. Chronological split: earliest 80% for training
split = int(len(model_data) * 0.80)

X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

# 6. Train the model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    min_samples_leaf=3
)

model.fit(X_train, y_train)

# 7. Predict the held-out period
predictions = np.maximum(0, model.predict(X_test))

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

# Baseline: predict each test day using the previous week's count
baseline = X_test["lag_7"].to_numpy()

baseline_mae = mean_absolute_error(y_test, baseline)

# 8. Save forecast and evaluation results
out = Path("output/data_mining")
out.mkdir(parents=True, exist_ok=True)

forecast = pd.DataFrame({
    "date": model_data["order_date"].iloc[split:].values,
    "actual_orders": y_test.values,
    "predicted_orders": np.round(predictions, 1),
    "previous_week_orders": baseline
})

forecast.to_csv(
    out / "daily_demand_forecast.csv",
    index=False
)

with open(
    out / "demand_prediction_metrics.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write("DAILY DEMAND PREDICTION RESULTS\n")
    file.write(f"Training days: {len(X_train)}\n")
    file.write(f"Testing days: {len(X_test)}\n")
    file.write(f"Model MAE: {mae:.2f} orders/day\n")
    file.write(f"Model RMSE: {rmse:.2f} orders/day\n")
    file.write(f"Weekly baseline MAE: {baseline_mae:.2f}\n")

# 9. Display results
print("\n--- DAILY DEMAND PREDICTION ---")
print("Training days:", len(X_train))
print("Testing days:", len(X_test))
print(f"Model MAE: {mae:.2f} orders/day")
print(f"Model RMSE: {rmse:.2f} orders/day")
print(f"Weekly baseline MAE: {baseline_mae:.2f}")

print("\nSample forecasts:")
print(forecast.head(10).to_string(index=False))

print("\nSaved files:")
print(out / "daily_demand_forecast.csv")
print(out / "demand_prediction_metrics.txt")
