import pandas as pd
import os


# ============================================================
# 1. CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs("output/dimensions", exist_ok=True)


# ============================================================
# 2. LOAD CLEANED DATA
# ============================================================

customers = pd.read_csv("C:/Users/Vijay Krishnaa S/Downloads/zomato_customers.csv")
orders = pd.read_csv("C:/Users/Vijay Krishnaa S/Downloads/zomato_orders.csv")
restaurants = pd.read_csv("C:/Users/Vijay Krishnaa S/Downloads/zomato_restaurants.csv")


print("\n========== CLEANED DATA LOADED ==========")

print("Customers:", customers.shape)
print("Orders:", orders.shape)
print("Restaurants:", restaurants.shape)


# ============================================================
# 3. DIM_CUSTOMER
# ============================================================

print("\n========== CREATING DIM_CUSTOMER ==========")


# Create surrogate key
customers.insert(
    0,
    "customer_key",
    range(1, len(customers) + 1)
)


# Select required columns
dim_customer = customers[
    [
        "customer_key",
        "customer_id",
        "customer_name",
        "city",
        "signup_date",
        "is_premium_member"
    ]
].copy()


# Convert date
dim_customer["signup_date"] = pd.to_datetime(
    dim_customer["signup_date"],
    errors="coerce"
)


# Validate
print("Rows:", len(dim_customer))
print(
    "Unique customer keys:",
    dim_customer["customer_key"].nunique()
)
print(
    "Unique customer IDs:",
    dim_customer["customer_id"].nunique()
)
print(
    "Duplicate customer keys:",
    dim_customer["customer_key"].duplicated().sum()
)
print(
    "Duplicate customer IDs:",
    dim_customer["customer_id"].duplicated().sum()
)


# Save
dim_customer.to_csv(
    "output/dimensions/dim_customer.csv",
    index=False
)


print("DIM_CUSTOMER saved successfully.")


# ============================================================
# 4. DIM_RESTAURANT
# ============================================================

print("\n========== CREATING DIM_RESTAURANT ==========")


# Create surrogate key
restaurants.insert(
    0,
    "restaurant_key",
    range(1, len(restaurants) + 1)
)


# Select required columns
dim_restaurant = restaurants[
    [
        "restaurant_key",
        "restaurant_id",
        "restaurant_name",
        "city",
        "cuisine",
        "cost_for_two",
        "is_veg_only",
        "avg_rating",
        "onboarded_year"
    ]
].copy()


# Validate
print("Rows:", len(dim_restaurant))
print(
    "Unique restaurant keys:",
    dim_restaurant["restaurant_key"].nunique()
)
print(
    "Unique restaurant IDs:",
    dim_restaurant["restaurant_id"].nunique()
)
print(
    "Duplicate restaurant keys:",
    dim_restaurant["restaurant_key"].duplicated().sum()
)
print(
    "Duplicate restaurant IDs:",
    dim_restaurant["restaurant_id"].duplicated().sum()
)


# Save
dim_restaurant.to_csv(
    "output/dimensions/dim_restaurant.csv",
    index=False
)


print("DIM_RESTAURANT saved successfully.")


# ============================================================
# 5. PREPARE ORDER DATE
# ============================================================

print("\n========== PREPARING DIM_DATE ==========")


# Convert order_datetime to datetime
orders["order_datetime"] = pd.to_datetime(
    orders["order_datetime"],
    errors="coerce"
)


# Extract unique dates
date_values = (
    orders["order_datetime"]
    .dt.normalize()
    .dropna()
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)


# ============================================================
# 6. CREATE DIM_DATE
# ============================================================

dim_date = pd.DataFrame()

# Date key in YYYYMMDD format
dim_date["date_key"] = (
    date_values.dt.strftime("%Y%m%d").astype(int)
)

# Actual date
dim_date["full_date"] = date_values

# Year
dim_date["year"] = date_values.dt.year

# Quarter number
dim_date["quarter"] = date_values.dt.quarter

# Quarter label
dim_date["quarter_name"] = (
    "Q" + date_values.dt.quarter.astype(str)
)

# Month number
dim_date["month"] = date_values.dt.month

# Month name
dim_date["month_name"] = date_values.dt.month_name()

# Day number
dim_date["day"] = date_values.dt.day

# Day name
dim_date["day_name"] = date_values.dt.day_name()

# Day of week number
dim_date["day_of_week"] = (
    date_values.dt.dayofweek + 1
)

# Weekend indicator
dim_date["is_weekend"] = (
    date_values.dt.dayofweek >= 5
)


# ============================================================
# 7. VALIDATE DIM_DATE
# ============================================================

print("\n========== DIM_DATE ==========")

print(dim_date.head())

print("\nRows:", len(dim_date))

print(
    "Unique date keys:",
    dim_date["date_key"].nunique()
)

print(
    "Duplicate date keys:",
    dim_date["date_key"].duplicated().sum()
)

print(
    "Missing date keys:",
    dim_date["date_key"].isnull().sum()
)


# ============================================================
# 8. SAVE DIM_DATE
# ============================================================

dim_date.to_csv(
    "output/dimensions/dim_date.csv",
    index=False
)


print("\nDIM_DATE saved successfully.")


# ============================================================
# 9. FINAL DIMENSION SUMMARY
# ============================================================

print("\n========== DIMENSION SUMMARY ==========")

print(
    "DIM_CUSTOMER:",
    dim_customer.shape
)

print(
    "DIM_RESTAURANT:",
    dim_restaurant.shape
)

print(
    "DIM_DATE:",
    dim_date.shape
)


print("\n========== ALL DIMENSIONS CREATED ==========")

print("1. output/dimensions/dim_customer.csv")
print("2. output/dimensions/dim_restaurant.csv")
print("3. output/dimensions/dim_date.csv")


print("\n========== STAGE 2.3 COMPLETED ==========")