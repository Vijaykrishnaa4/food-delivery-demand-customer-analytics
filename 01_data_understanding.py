import pandas as pd


# ============================================================
# 1. LOAD DATASETS
# ============================================================

customers = pd.read_csv("C:/Users/Vijay Krishnaa S/Downloads/zomato_customers.csv")
orders = pd.read_csv("C:/Users/Vijay Krishnaa S/Downloads/zomato_orders.csv")
restaurants = pd.read_csv("C:/Users/Vijay Krishnaa S/Downloads/zomato_restaurants.csv")


# ============================================================
# 2. DISPLAY FIRST 5 ROWS
# ============================================================

print("\n========== CUSTOMERS ==========")
print(customers.head())

print("\n========== ORDERS ==========")
print(orders.head())

print("\n========== RESTAURANTS ==========")
print(restaurants.head())


# ============================================================
# 3. DATASET INFORMATION
# ============================================================

print("\n========== DATASET INFORMATION ==========")

print("\n--- CUSTOMERS ---")
customers.info()

print("\n--- ORDERS ---")
orders.info()

print("\n--- RESTAURANTS ---")
restaurants.info()


# ============================================================
# 4. DATASET SIZE
# ============================================================

print("\n========== DATASET SIZE ==========")

print("Customers:", customers.shape)
print("Orders:", orders.shape)
print("Restaurants:", restaurants.shape)


# ============================================================
# 5. MISSING VALUES
# ============================================================

print("\n========== MISSING VALUES ==========")

print("\n--- CUSTOMERS ---")
print(customers.isnull().sum())

print("\n--- ORDERS ---")
print(orders.isnull().sum())

print("\n--- RESTAURANTS ---")
print(restaurants.isnull().sum())


# ============================================================
# 6. DUPLICATE RECORDS
# ============================================================

print("\n========== DUPLICATE RECORDS ==========")

print("Customer duplicates:", customers.duplicated().sum())
print("Order duplicates:", orders.duplicated().sum())
print("Restaurant duplicates:", restaurants.duplicated().sum())


# ============================================================
# 7. UNIQUE ID CHECK
# ============================================================

print("\n========== UNIQUE ID CHECK ==========")

print(
    "Unique customer IDs:",
    customers["customer_id"].nunique()
)

print(
    "Unique order IDs:",
    orders["order_id"].nunique()
)

print(
    "Unique restaurant IDs:",
    restaurants["restaurant_id"].nunique()
)


# ============================================================
# 8. REFERENTIAL INTEGRITY CHECK
# ============================================================

print("\n========== REFERENTIAL INTEGRITY CHECK ==========")

invalid_customers = orders[
    ~orders["customer_id"].isin(customers["customer_id"])
]

invalid_restaurants = orders[
    ~orders["restaurant_id"].isin(restaurants["restaurant_id"])
]

print(
    "Orders with invalid customer IDs:",
    len(invalid_customers)
)

print(
    "Orders with invalid restaurant IDs:",
    len(invalid_restaurants)
)


# ============================================================
# 9. ORDER STATISTICS
# ============================================================

print("\n========== ORDER STATISTICS ==========")

print(orders.describe())


# ============================================================
# 10. ORDER STATUS DISTRIBUTION
# ============================================================

print("\n========== ORDER STATUS DISTRIBUTION ==========")

print(orders["order_status"].value_counts())


# ============================================================
# END
# ============================================================

print("\n========== DATA UNDERSTANDING COMPLETED ==========")