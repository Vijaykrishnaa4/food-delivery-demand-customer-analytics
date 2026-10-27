
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 1. Load orders
data = pd.read_csv("output/cleaned_orders.csv")
data.columns = data.columns.str.strip().str.lower()

data["order_datetime"] = pd.to_datetime(
    data["order_datetime"], errors="coerce"
)
data = data.dropna(subset=["order_datetime"])

# 2. Count orders per day
data["order_date"] = data["order_datetime"].dt.floor("D")

daily = (
    data.groupby("order_date")
    .size()
    .rename("order_count")
    .to_frame()
    .sort_index()
)

daily = daily.asfreq("D", fill_value=0)
daily.index.name = "order_date"
daily = daily.reset_index()

# 3. Create calendar and lag features
daily["day_of_week"] = daily["order_date"].dt.dayofweek
daily["day_of_month"] = daily["order_date"].dt.day
daily["month"] = daily["order_date"].dt.month
daily["year"] = daily["order_date"].dt.year
daily["is_weekend"] = (
    daily["day_of_week"] >= 5
).astype(int)

daily["lag_1"] = daily["order_count"].shift(1)
daily["lag_7"] = daily["order_count"].shift(7)
daily["lag_14"] = daily["order_count"].shift(14)
daily["rolling_mean_7"] = (
    daily["order_count"].shift(1).rolling(7).mean()
)
daily["rolling_mean_14"] = (
    daily["order_count"].shift(1).rolling(14).mean()
)

daily = daily.dropna().reset_index(drop=True)

features = [
    "day_of_week",
    "day_of_month",
    "month",
    "year",
    "is_weekend",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_14"
]

X = daily[features]
y = daily["order_count"]

# 4. Chronological train/test split
split = int(len(daily) * 0.80)

X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

# 5. Train several models
models = {
    "Ridge Regression": make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0)
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
}

results = []
predictions_by_model = {}

# Previous week's demand baseline
baseline = X_test["lag_7"].to_numpy()
results.append({
    "model": "Previous Week Baseline",
    "mae": mean_absolute_error(y_test, baseline),
    "rmse": np.sqrt(mean_squared_error(y_test, baseline))
})

predictions_by_model["Previous Week Baseline"] = baseline

for name, model in models.items():
    model.fit(X_train, y_train)

    pred = np.maximum(0, model.predict(X_test))

    results.append({
        "model": name,
        "mae": mean_absolute_error(y_test, pred),
        "rmse": np.sqrt(mean_squared_error(y_test, pred))
    })

    predictions_by_model[name] = pred

# 6. Compare models
comparison = pd.DataFrame(results).sort_values("mae")

print("\n--- MODEL COMPARISON ---")
print(comparison.round(2).to_string(index=False))

best_name = comparison.iloc[0]["model"]
best_mae = comparison.iloc[0]["mae"]
baseline_mae = comparison.loc[
    comparison["model"] == "Previous Week Baseline", "mae"
].iloc[0]

print("\nBest model:", best_name)
print(f"Best MAE: {best_mae:.2f} orders/day")
print(f"Baseline MAE: {baseline_mae:.2f} orders/day")

# 7. Save results
out = Path("output/data_mining")
out.mkdir(parents=True, exist_ok=True)

comparison.to_csv(out / "demand_model_comparison.csv", index=False)

forecast = pd.DataFrame({
    "date": daily["order_date"].iloc[split:].values,
    "actual_orders": y_test.values,
    "previous_week_baseline": baseline,
    "best_model": predictions_by_model[best_name].round(1)
})

forecast.to_csv(
    out / "improved_demand_forecast.csv",
    index=False
)

if best_name != "Previous Week Baseline" and best_mae < baseline_mae:
    print("\nThe best tested model beats the baseline on MAE.")
else:
    print("\nNo tested model beats the baseline on MAE.")
    print("Keep the baseline unless further validation supports a better model.")

print("\nSaved:")
print(out / "demand_model_comparison.csv")
print(out / "improved_demand_forecast.csv")
