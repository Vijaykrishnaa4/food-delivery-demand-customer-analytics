
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Load warehouse tables
fact = pd.read_csv("output/facts/fact_orders.csv")
customers = pd.read_csv("output/dimensions/DIM_CUSTOMER.csv")

# Standardize column names
fact.columns = fact.columns.str.strip().str.lower()
customers.columns = customers.columns.str.strip().str.lower()

# 2. Calculate customer behaviour features
fact["total_amount"] = pd.to_numeric(
    fact["total_amount"], errors="coerce"
)

if "order_status" in fact.columns:
    status = fact["order_status"].astype(str).str.lower()
    delivered = fact[status.str.contains(
        "delivered|completed", na=False
    )].copy()
else:
    delivered = fact.copy()

# Count all orders; calculate spending from completed orders
order_count = fact.groupby("customer_key").size().rename("order_count")

spending = delivered.groupby("customer_key").agg(
    total_spend=("total_amount", "sum"),
    avg_order_value=("total_amount", "mean")
)

features = order_count.to_frame().join(spending, how="left")

# Add delivery time and rating when available
for col, output_col in [
    ("delivery_time_mins", "avg_delivery_time"),
    ("customer_rating", "avg_rating")
]:
    if col in delivered.columns:
        delivered[col] = pd.to_numeric(delivered[col], errors="coerce")
        values = delivered.groupby("customer_key")[col].mean()
        features = features.join(values.rename(output_col), how="left")

# Include customers with no orders
customers["customer_key"] = pd.to_numeric(
    customers["customer_key"], errors="coerce"
)
features = customers[["customer_key"]].merge(
    features.reset_index(), on="customer_key", how="left"
).set_index("customer_key")

features = features.fillna(0)

# 3. Prepare features for clustering
X = features.select_dtypes(include="number")
X = X.loc[:, X.nunique() > 1]

if len(X) < 3 or X.shape[1] == 0:
    raise ValueError("Not enough customer data for 3 clusters.")

X_scaled = StandardScaler().fit_transform(X)

# 4. Apply K-Means
model = KMeans(n_clusters=3, random_state=42, n_init=10)
features["customer_cluster"] = model.fit_predict(X_scaled)

# 5. Save results
out = Path("output/data_mining")
out.mkdir(parents=True, exist_ok=True)

features.to_csv(out / "customer_segments.csv")

print("Customer segmentation completed!")
print("\nCustomers per cluster:")
print(features["customer_cluster"].value_counts().sort_index())

print("\nCluster averages:")
print(features.groupby("customer_cluster").mean().round(2))

print("\nSaved: output/data_mining/customer_segments.csv")
