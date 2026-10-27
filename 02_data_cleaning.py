import pandas as pd


# ============================================================
# 1. LOAD RAW DATA
# ============================================================

customers = pd.read_csv("C:/Users/Vijay Krishnaa S/Downloads/zomato_customers.csv")
orders = pd.read_csv("C:/Users/Vijay Krishnaa S/Downloads/zomato_orders.csv")
restaurants = pd.read_csv("C:/Users/Vijay Krishnaa S/Downloads/zomato_restaurants.csv")


print("========== RAW DATA LOADED ==========")
print("Customers:", customers.shape)
print("Orders:", orders.shape)
print("Restaurants:", restaurants.shape)


# ============================================================
# 2. STANDARDIZE COLUMN NAMES
# ============================================================

customers.columns = customers.columns.str.strip().str.lower()
orders.columns = orders.columns.str.strip().str.lower()
restaurants.columns = restaurants.columns.str.strip().str.lower()


print("\n========== COLUMN NAMES ==========")

print("Customers:")
print(customers.columns.tolist())

print("\nOrders:")
print(orders.columns.tolist())

print("\nRestaurants:")
print(restaurants.columns.tolist())


# ============================================================
# 3. CONVERT DATE COLUMNS
# ============================================================

customers["signup_date"] = pd.to_datetime(
    customers["signup_date"],
    errors="coerce"
)

orders["order_datetime"] = pd.to_datetime(
    orders["order_datetime"],
    errors="coerce"
)


# ============================================================
# 4. CHECK DATE CONVERSION
# ============================================================

print("\n========== DATE CHECK ==========")

print(
    "Invalid customer signup dates:",
    customers["signup_date"].isnull().sum()
)

print(
    "Invalid order dates:",
    orders["order_datetime"].isnull().sum()
)


# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================

print("\n========== MISSING VALUES BEFORE CLEANING ==========")

print("\nOrders:")
print(orders.isnull().sum())


# Delivery time is missing for orders that were not delivered.
# We keep these values as missing because there was no completed
# delivery time to record.

# Customer rating is optional, so missing ratings are preserved.

print("\nMissing delivery times:")
print(orders["delivery_time_mins"].isnull().sum())

print("Missing customer ratings:")
print(orders["customer_rating"].isnull().sum())


# ============================================================
# 6. CHECK DUPLICATES
# ============================================================

print("\n========== DUPLICATE CHECK ==========")

print("Duplicate customers:", customers.duplicated().sum())
print("Duplicate orders:", orders.duplicated().sum())
print("Duplicate restaurants:", restaurants.duplicated().sum())


# ============================================================
# 7. CHECK INVALID NUMERIC VALUES
# ============================================================

print("\n========== BUSINESS RULE CHECK ==========")

print(
    "Orders with negative order value:",
    (orders["order_value"] < 0).sum()
)

print(
    "Orders with negative discount:",
    (orders["discount_amount"] < 0).sum()
)

print(
    "Orders with negative delivery fee:",
    (orders["delivery_fee"] < 0).sum()
)

print(
    "Orders with negative taxes:",
    (orders["taxes"] < 0).sum()
)

print(
    "Orders with negative total amount:",
    (orders["total_amount"] < 0).sum()
)

print(
    "Orders with invalid delivery time:",
    (orders["delivery_time_mins"] < 0).sum()
)


# ============================================================
# 8. CREATE USEFUL DATE FEATURES
# ============================================================

orders["order_date"] = orders["order_datetime"].dt.date
orders["order_year"] = orders["order_datetime"].dt.year
orders["order_month"] = orders["order_datetime"].dt.month
orders["order_day"] = orders["order_datetime"].dt.day
orders["order_hour"] = orders["order_datetime"].dt.hour
orders["day_of_week"] = orders["order_datetime"].dt.day_name()


# Weekend indicator
orders["is_weekend"] = (
    orders["order_datetime"].dt.dayofweek >= 5
)


# ============================================================
# 9. DISPLAY TRANSFORMED DATA
# ============================================================

print("\n========== TRANSFORMED ORDERS ==========")

print(
    orders[
        [
            "order_id",
            "order_datetime",
            "order_date",
            "order_year",
            "order_month",
            "order_hour",
            "day_of_week",
            "is_weekend"
        ]
    ].head()
)


# ============================================================
# 10. FINAL DATASET SUMMARY
# ============================================================

print("\n========== CLEANED DATA SUMMARY ==========")

print("Customers:", customers.shape)
print("Orders:", orders.shape)
print("Restaurants:", restaurants.shape)

print("\n========== STAGE 2.1 COMPLETED ==========")

# ============================================================
# 11. CREATE OUTPUT FOLDER
# ============================================================

import os

os.makedirs("output", exist_ok=True)


# ============================================================
# 12. SAVE CLEANED DATASETS
# ============================================================

customers.to_csv(
    "output/cleaned_customers.csv",
    index=False
)

orders.to_csv(
    "output/cleaned_orders.csv",
    index=False
)

restaurants.to_csv(
    "output/cleaned_restaurants.csv",
    index=False
)


# ============================================================
# 13. VERIFY SAVED FILES
# ============================================================

print("\n========== CLEANED FILES CREATED ==========")

print("output/cleaned_customers.csv")
print("output/cleaned_orders.csv")
print("output/cleaned_restaurants.csv")

print("\n========== FINAL ROW COUNTS ==========")

print("Customers:", len(customers))
print("Orders:", len(orders))
print("Restaurants:", len(restaurants))


print("\n========== STAGE 2.2 COMPLETED ==========")