
import pandas as pd

data = pd.read_csv("output/cleaned_orders.csv")
data.columns = data.columns.str.strip().str.lower()

data["target"] = data["order_status"].apply(
    lambda s: (
        "Delivered"
        if str(s).strip().lower() == "delivered"
        else "Cancelled"
    )
)

features = [
    "order_value",
    "discount_amount",
    "delivery_fee",
    "taxes",
    "order_year",
    "order_month",
    "order_day",
    "order_hour",
    "is_weekend",
    "payment_method"
]

features = [c for c in features if c in data.columns]

print("\n--- FEATURE COMPARISON ---")
print(data.groupby("target")[features].mean(numeric_only=True).round(2).T)

for col in features:
    if data[col].nunique(dropna=False) <= 30:
        print(f"\n--- {col}: target distribution ---")
        print(pd.crosstab(data[col], data["target"]))

print("\n--- MISSING VALUES ---")
print(data[features].isna().sum())

print("\n--- DUPLICATE INPUT PATTERNS ---")
print(
    "Duplicate feature rows:",
    data.duplicated(subset=features).sum()
)
