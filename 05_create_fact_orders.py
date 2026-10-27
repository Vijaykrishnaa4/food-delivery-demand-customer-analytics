import pandas as pd
import os


# ============================================================
# 1. LOAD CLEANED ORDERS AND DIMENSION TABLES
# ============================================================

orders = pd.read_csv(
    "output/cleaned_orders.csv"
)

dim_customer = pd.read_csv(
    "output/dimensions/dim_customer.csv"
)

dim_restaurant = pd.read_csv(
    "output/dimensions/dim_restaurant.csv"
)

dim_date = pd.read_csv(
    "output/dimensions/dim_date.csv"
)


print("\n========== DATA LOADED ==========")

print("Orders:", orders.shape)
print("Customers:", dim_customer.shape)
print("Restaurants:", dim_restaurant.shape)
print("Dates:", dim_date.shape)


# ============================================================
# 2. CONVERT ORDER DATETIME
# ============================================================

orders["order_datetime"] = pd.to_datetime(
    orders["order_datetime"],
    errors="coerce"
)


# Create order date
orders["order_date"] = (
    orders["order_datetime"]
    .dt.normalize()
)


# ============================================================
# 3. MAP CUSTOMER ID TO CUSTOMER KEY
# ============================================================

print("\n========== MAPPING CUSTOMER KEYS ==========")

customer_mapping = dim_customer[
    [
        "customer_id",
        "customer_key"
    ]
]

orders = orders.merge(
    customer_mapping,
    on="customer_id",
    how="left"
)


# Check unmatched customers
missing_customer_keys = orders["customer_key"].isnull().sum()

print(
    "Orders with missing customer keys:",
    missing_customer_keys
)


# ============================================================
# 4. MAP RESTAURANT ID TO RESTAURANT KEY
# ============================================================

print("\n========== MAPPING RESTAURANT KEYS ==========")

restaurant_mapping = dim_restaurant[
    [
        "restaurant_id",
        "restaurant_key"
    ]
]

orders = orders.merge(
    restaurant_mapping,
    on="restaurant_id",
    how="left"
)


# Check unmatched restaurants
missing_restaurant_keys = orders["restaurant_key"].isnull().sum()

print(
    "Orders with missing restaurant keys:",
    missing_restaurant_keys
)


# ============================================================
# 5. MAP ORDER DATE TO DATE KEY
# ============================================================

print("\n========== MAPPING DATE KEYS ==========")

date_mapping = dim_date[
    [
        "full_date",
        "date_key"
    ]
].copy()


# Convert dimension date to datetime
date_mapping["full_date"] = pd.to_datetime(
    date_mapping["full_date"],
    errors="coerce"
)


orders = orders.merge(
    date_mapping,
    left_on="order_date",
    right_on="full_date",
    how="left"
)


# Check unmatched dates
missing_date_keys = orders["date_key"].isnull().sum()

print(
    "Orders with missing date keys:",
    missing_date_keys
)


# ============================================================
# 6. CREATE FACT_ORDERS
# ============================================================

print("\n========== CREATING FACT_ORDERS ==========")


fact_orders = orders[
    [
        "order_id",
        "customer_key",
        "restaurant_key",
        "date_key",

        "order_value",
        "discount_amount",
        "delivery_fee",
        "taxes",
        "total_amount",

        "payment_method",
        "delivery_time_mins",
        "customer_rating",
        "order_status"
    ]
].copy()


# ============================================================
# 7. CONVERT KEYS TO INTEGER
# ============================================================

fact_orders["customer_key"] = (
    fact_orders["customer_key"]
    .astype("Int64")
)

fact_orders["restaurant_key"] = (
    fact_orders["restaurant_key"]
    .astype("Int64")
)

fact_orders["date_key"] = (
    fact_orders["date_key"]
    .astype("Int64")
)


# ============================================================
# 8. FACT TABLE VALIDATION
# ============================================================

print("\n========== FACT TABLE VALIDATION ==========")

print(
    "Number of fact rows:",
    len(fact_orders)
)

print(
    "Unique order IDs:",
    fact_orders["order_id"].nunique()
)

print(
    "Duplicate order IDs:",
    fact_orders["order_id"].duplicated().sum()
)

print(
    "Missing customer keys:",
    fact_orders["customer_key"].isnull().sum()
)

print(
    "Missing restaurant keys:",
    fact_orders["restaurant_key"].isnull().sum()
)

print(
    "Missing date keys:",
    fact_orders["date_key"].isnull().sum()
)


# ============================================================
# 9. CHECK FACT TABLE GRAIN
# ============================================================

print("\n========== FACT TABLE GRAIN CHECK ==========")

print(
    "Expected rows:",
    len(orders)
)

print(
    "Actual fact rows:",
    len(fact_orders)
)

if len(fact_orders) == len(orders):
    print("Grain check: PASSED")
else:
    print("Grain check: FAILED")


# ============================================================
# 10. DISPLAY FACT TABLE
# ============================================================

print("\n========== FACT_ORDERS SAMPLE ==========")

print(
    fact_orders.head(10)
)


# ============================================================
# 11. CREATE FACT TABLE DIRECTORY
# ============================================================

os.makedirs(
    "output/facts",
    exist_ok=True
)


# ============================================================
# 12. SAVE FACT TABLE
# ============================================================

fact_orders.to_csv(
    "output/facts/fact_orders.csv",
    index=False
)


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n========== FACT_ORDERS CREATED ==========")

print(
    "Saved to: output/facts/fact_orders.csv"
)

print(
    "Rows:",
    len(fact_orders)
)

print(
    "Columns:",
    len(fact_orders.columns)
)


print("\n========== STAGE 2.4 COMPLETED ==========")