
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Set project paths
DATA_DIR = Path("output/data_mining")
CHART_DIR = DATA_DIR / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

# 2. Customer segmentation chart
segments = pd.read_csv(DATA_DIR / "customer_segments.csv")

cluster_counts = (
    segments["customer_cluster"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(9, 5))
cluster_counts.plot(kind="bar")
plt.title("Customer Segmentation Using K-Means")
plt.xlabel("Customer Cluster")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(CHART_DIR / "customer_segments.png", dpi=300)
plt.close()

# 3. Order classification confusion matrix
matrix = pd.read_csv(
    DATA_DIR / "order_confusion_matrix.csv",
    index_col=0
)

plt.figure(figsize=(7, 5))
plt.imshow(matrix.values, interpolation="nearest")
plt.title("Order Classification Confusion Matrix")
plt.xticks(
    range(len(matrix.columns)),
    matrix.columns,
    rotation=15,
    ha="right"
)
plt.yticks(range(len(matrix.index)), matrix.index)

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        plt.text(
            j, i, str(matrix.iloc[i, j]),
            ha="center", va="center", fontsize=12
        )

plt.colorbar(label="Number of Orders")
plt.tight_layout()
plt.savefig(CHART_DIR / "order_confusion_matrix.png", dpi=300)
plt.close()

# 4. Actual versus predicted demand
forecast = pd.read_csv(
    DATA_DIR / "improved_demand_forecast.csv",
    parse_dates=["date"]
)

plt.figure(figsize=(13, 6))
plt.plot(
    forecast["date"],
    forecast["actual_orders"],
    label="Actual Orders",
    linewidth=2
)
plt.plot(
    forecast["date"],
    forecast["best_model"],
    label="Ridge Regression Prediction",
    linewidth=1.5
)
plt.plot(
    forecast["date"],
    forecast["previous_week_baseline"],
    label="Previous Week Baseline",
    linewidth=1.2,
    linestyle="--"
)

plt.title("Daily Food Delivery Demand: Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Number of Orders")
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(CHART_DIR / "demand_forecast_comparison.png", dpi=300)
plt.close()

# 5. Model comparison chart
comparison = pd.read_csv(
    DATA_DIR / "demand_model_comparison.csv"
).sort_values("mae")

plt.figure(figsize=(9, 5))
plt.bar(comparison["model"], comparison["mae"])
plt.title("Demand Forecasting Model Comparison")
plt.xlabel("Model")
plt.ylabel("Mean Absolute Error (lower is better)")
plt.xticks(rotation=15, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(CHART_DIR / "demand_model_comparison.png", dpi=300)
plt.close()

print("Data-mining visualizations created successfully!")
print("\nCharts saved in:", CHART_DIR)

for file in sorted(CHART_DIR.glob("*.png")):
    print("-", file.name)
