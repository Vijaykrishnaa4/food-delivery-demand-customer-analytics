import pandas as pd
import os


# ============================================================
# 1. LOAD CLEANED CUSTOMER DATA
# ============================================================

customers = pd.read_csv("C:/Users/Vijay Krishnaa S/Downloads/zomato_customers.csv")

print("\n========== CLEANED CUSTOMER DATA ==========")
print(customers.head())


# ============================================================
# 2. CREATE CUSTOMER SURROGATE KEY
# ============================================================

customers.insert(
    0,
    "customer_key",
    range(1, len(customers) + 1)
)


# ============================================================
# 3. SELECT DIMENSION COLUMNS
# ============================================================

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


# ============================================================
# 4. CONVERT SIGNUP DATE
# ============================================================

dim_customer["signup_date"] = pd.to_datetime(
    dim_customer["signup_date"],
    errors="coerce"
)


# ============================================================
# 5. CHECK DIMENSION
# ============================================================

print("\n========== DIM_CUSTOMER ==========")

print(dim_customer.head())

print("\nNumber of rows:", len(dim_customer))

print(
    "Unique customer keys:",
    dim_customer["customer_key"].nunique()
)

print(
    "Unique customer IDs:",
    dim_customer["customer_id"].nunique()
)


# ============================================================
# 6. VALIDATE CUSTOMER KEYS
# ============================================================

print("\n========== CUSTOMER KEY VALIDATION ==========")

print(
    "Duplicate customer keys:",
    dim_customer["customer_key"].duplicated().sum()
)

print(
    "Duplicate customer IDs:",
    dim_customer["customer_id"].duplicated().sum()
)

print(
    "Missing customer keys:",
    dim_customer["customer_key"].isnull().sum()
)


# ============================================================
# 7. SAVE DIM_CUSTOMER
# ============================================================

os.makedirs("output/dimensions", exist_ok=True)

dim_customer.to_csv(
    "output/dimensions/dim_customer.csv",
    index=False
)


# ============================================================
# 8. COMPLETION
# ============================================================

print("\n========== DIM_CUSTOMER CREATED ==========")

print(
    "Saved to: output/dimensions/dim_customer.csv"
)

print("Rows:", len(dim_customer))
print("Columns:", len(dim_customer.columns))