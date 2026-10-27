import pandas as pd


# ============================================================
# 1. LOAD DATA
# ============================================================

fact_orders = pd.read_csv(
    "output/facts/fact_orders.csv"
)

dim_date = pd.read_csv(
    "output/dimensions/dim_date.csv"
)

dim_customer = pd.read_csv(
    "output/dimensions/dim_customer.csv"
)

print("\n========== OLAP ANALYSIS ==========")

print("\nData Loaded:")
print("Fact Orders:", fact_orders.shape)
print("Date Dim:", dim_date.shape)
print("Customer Dim:", dim_customer.shape)


# ============================================================
# 2. MERGE FACT TABLE WITH DATE DIMENSION
# ============================================================

olap_data = fact_orders.merge(
    dim_date,
    on="date_key",
    how="left"
)

print("\nMerged Data Shape:", olap_data.shape)


# ============================================================
# 3. ROLL-UP
# ============================================================
# Hierarchy:
#
# Day → Month → Quarter → Year
#
# Roll-up summarizes detailed data into higher levels.
# ============================================================


# ---------------- YEAR LEVEL ----------------

print("\n========== ROLL-UP: YEAR ==========")

yearly = olap_data.groupby(
    ["year"]
).agg(
    total_orders=("order_id", "count"),
    total_revenue=("total_amount", "sum"),
    avg_order_value=("total_amount", "mean")
).reset_index()

print(yearly)


# ---------------- QUARTER LEVEL ----------------

print("\n========== ROLL-UP: QUARTER ==========")

quarterly = olap_data.groupby(
    ["year", "quarter", "quarter_name"]
).agg(
    total_orders=("order_id", "count"),
    total_revenue=("total_amount", "sum"),
    avg_order_value=("total_amount", "mean")
).reset_index()

print(quarterly)


# ---------------- MONTH LEVEL ----------------

print("\n========== ROLL-UP: MONTH LEVEL ==========")

monthly = olap_data.groupby(
    ["year", "month", "month_name"]
).agg(
    total_orders=("order_id", "count"),
    total_revenue=("total_amount", "sum"),
    avg_order_value=("total_amount", "mean")
).reset_index()

print(monthly)


# ---------------- DAY LEVEL ----------------

print("\n========== DAY LEVEL (DETAIL) ==========")

daily = olap_data.groupby(
    ["full_date"]
).agg(
    total_orders=("order_id", "count"),
    total_revenue=("total_amount", "sum"),
    avg_order_value=("total_amount", "mean")
).reset_index()

print(daily.head(20))


# ============================================================
# 4. DRILL-DOWN
# ============================================================
# Drill-down:
#
# Year → Quarter → Month → Day
#
# Example:
# 2024 → Q1 → January → Individual Days
# ============================================================


# ---------------- YEAR → QUARTER ----------------

print("\n========== DRILL-DOWN: YEAR → QUARTER ==========")

year_selected = 2024

drill_quarter = olap_data[
    olap_data["year"] == year_selected
].groupby(
    ["year", "quarter", "quarter_name"]
).agg(
    total_orders=("order_id", "count"),
    total_revenue=("total_amount", "sum"),
    avg_order_value=("total_amount", "mean")
).reset_index()

print(drill_quarter)


# ---------------- QUARTER → MONTH ----------------

print("\n========== DRILL-DOWN: QUARTER → MONTH ==========")

quarter_selected = 1

drill_month = olap_data[
    (olap_data["year"] == year_selected) &
    (olap_data["quarter"] == quarter_selected)
].groupby(
    ["year", "quarter", "month", "month_name"]
).agg(
    total_orders=("order_id", "count"),
    total_revenue=("total_amount", "sum"),
    avg_order_value=("total_amount", "mean")
).reset_index()

print(drill_month)


# ---------------- MONTH → DAY ----------------

print("\n========== DRILL-DOWN: MONTH → DAY ==========")

month_selected = 1

drill_day = olap_data[
    (olap_data["year"] == year_selected) &
    (olap_data["quarter"] == quarter_selected) &
    (olap_data["month"] == month_selected)
].groupby(
    ["full_date"]
).agg(
    total_orders=("order_id", "count"),
    total_revenue=("total_amount", "sum"),
    avg_order_value=("total_amount", "mean")
).reset_index()

print(drill_day)


# ============================================================
# 5. SLICE
# ============================================================
# Slice selects ONE value from ONE dimension.
#
# Customer Dimension → City
# Selected City      → Mumbai
#
# ALL ORDERS
#      ↓
# City = Mumbai
#      ↓
# Mumbai Orders
# ============================================================

print("\n========== SLICE: CITY = MUMBAI ==========")


# ------------------------------------------------------------
# MERGE FACT ORDERS WITH CUSTOMER DIMENSION
# ------------------------------------------------------------

slice_data = fact_orders.merge(
    dim_customer[
        [
            "customer_key",
            "customer_name",
            "city"
        ]
    ],
    on="customer_key",
    how="left"
)


# ------------------------------------------------------------
# MERGE WITH DATE DIMENSION
# ------------------------------------------------------------
# This adds:
# year, quarter, month, month_name, full_date
# to the slice data.
# ------------------------------------------------------------

slice_data = slice_data.merge(
    dim_date[
        [
            "date_key",
            "full_date",
            "year",
            "quarter",
            "quarter_name",
            "month",
            "month_name"
        ]
    ],
    on="date_key",
    how="left"
)

print("\nSlice Data Shape:", slice_data.shape)


# ============================================================
# 6. SELECT CITY
# ============================================================

selected_city = "Mumbai"

mumbai_data = slice_data[
    slice_data["city"].str.lower() == selected_city.lower()
].copy()


# ============================================================
# 7. VALIDATE SLICE
# ============================================================

print("\nSelected City:", selected_city)

print(
    "Orders in selected city:",
    len(mumbai_data)
)

print(
    "Unique customers:",
    mumbai_data["customer_key"].nunique()
)


# ============================================================
# 8. MUMBAI SLICE SUMMARY
# ============================================================

slice_summary = pd.DataFrame({
    "city": [selected_city],

    "total_orders": [
        mumbai_data["order_id"].count()
    ],

    "total_revenue": [
        mumbai_data["total_amount"].sum()
    ],

    "avg_order_value": [
        mumbai_data["total_amount"].mean()
    ],

    "avg_delivery_time": [
        mumbai_data["delivery_time_mins"].mean()
    ],

    "avg_customer_rating": [
        mumbai_data["customer_rating"].mean()
    ]
})


print("\n========== MUMBAI SLICE SUMMARY ==========")

print(slice_summary)


# ============================================================
# 9. MUMBAI ORDERS BY MONTH
# ============================================================

print("\n========== MUMBAI ORDERS BY MONTH ==========")

mumbai_monthly = mumbai_data.groupby(
    ["year", "month", "month_name"]
).agg(
    total_orders=("order_id", "count"),
    total_revenue=("total_amount", "sum"),
    avg_order_value=("total_amount", "mean")
).reset_index()

print(mumbai_monthly)


# ============================================================
# 10. SAVE ROLL-UP RESULTS
# ============================================================

yearly.to_csv(
    "output/olap_yearly.csv",
    index=False
)

quarterly.to_csv(
    "output/olap_quarterly.csv",
    index=False
)

monthly.to_csv(
    "output/olap_monthly.csv",
    index=False
)

daily.to_csv(
    "output/olap_daily.csv",
    index=False
)


# ============================================================
# 11. SAVE DRILL-DOWN RESULTS
# ============================================================

drill_quarter.to_csv(
    "output/olap_drill_quarter.csv",
    index=False
)

drill_month.to_csv(
    "output/olap_drill_month.csv",
    index=False
)

drill_day.to_csv(
    "output/olap_drill_day.csv",
    index=False
)


# ============================================================
# 12. SAVE SLICE RESULTS
# ============================================================

slice_summary.to_csv(
    "output/olap_slice_mumbai.csv",
    index=False
)

mumbai_monthly.to_csv(
    "output/olap_slice_mumbai_monthly.csv",
    index=False
)


# ============================================================
# 13. COMPLETION
# ============================================================

print("\n========== OLAP ANALYSIS COMPLETED ==========")

print("\nCompleted Operations:")
print("1. Roll-up      ✓")
print("2. Drill-down   ✓")
print("3. Slice        ✓")

print("\nOutput files created:")
print("output/olap_yearly.csv")
print("output/olap_quarterly.csv")
print("output/olap_monthly.csv")
print("output/olap_daily.csv")
print("output/olap_drill_quarter.csv")
print("output/olap_drill_month.csv")
print("output/olap_drill_day.csv")
print("output/olap_slice_mumbai.csv")
print("output/olap_slice_mumbai_monthly.csv")

# ============================================================
# 13. DICE
# ============================================================
# Dice selects data using conditions from TWO OR MORE
# dimensions simultaneously.
#
# Dimension 1:
# Customer Dimension -> City
#
# Selected cities:
# Mumbai, Bengaluru
#
# Dimension 2:
# Date Dimension -> Year
#
# Selected years:
# 2023, 2024
#
# Therefore:
#
# City IN (Mumbai, Bengaluru)
# AND
# Year IN (2023, 2024)
#
# This creates a smaller OLAP sub-cube.
# ============================================================

print("\n========== DICE OPERATION ==========")


# ------------------------------------------------------------
# MERGE FACT ORDERS WITH CUSTOMER DIMENSION
# ------------------------------------------------------------

dice_data = fact_orders.merge(
    dim_customer[
        [
            "customer_key",
            "customer_name",
            "city"
        ]
    ],
    on="customer_key",
    how="left"
)


# ------------------------------------------------------------
# MERGE WITH DATE DIMENSION
# ------------------------------------------------------------

dice_data = dice_data.merge(
    dim_date[
        [
            "date_key",
            "full_date",
            "year",
            "quarter",
            "quarter_name",
            "month",
            "month_name"
        ]
    ],
    on="date_key",
    how="left"
)


print("\nOriginal Data:", len(dice_data))


# ============================================================
# DICE CONDITIONS
# ============================================================

selected_cities = [
    "Mumbai",
    "Bengaluru"
]

selected_years = [
    2023,
    2024
]


# ============================================================
# APPLY DICE
# ============================================================

dice_result = dice_data[
    dice_data["city"].isin(selected_cities)
    &
    dice_data["year"].isin(selected_years)
].copy()


# ============================================================
# DICE VALIDATION
# ============================================================

print("\nSelected Cities:", selected_cities)

print("Selected Years:", selected_years)

print(
    "Orders after Dice:",
    len(dice_result)
)

print(
    "Unique Customers:",
    dice_result["customer_key"].nunique()
)


# ============================================================
# DICE SUMMARY BY CITY AND YEAR
# ============================================================

print("\n========== DICE SUMMARY: CITY + YEAR ==========")

dice_summary = dice_result.groupby(
    [
        "city",
        "year"
    ]
).agg(
    total_orders=("order_id", "count"),
    total_revenue=("total_amount", "sum"),
    avg_order_value=("total_amount", "mean"),
    avg_delivery_time=("delivery_time_mins", "mean"),
    avg_customer_rating=("customer_rating", "mean")
).reset_index()


print(dice_summary)


# ============================================================
# DICE SUMMARY BY CITY + YEAR + QUARTER
# ============================================================

print("\n========== DICE: CITY + YEAR + QUARTER ==========")

dice_quarterly = dice_result.groupby(
    [
        "city",
        "year",
        "quarter",
        "quarter_name"
    ]
).agg(
    total_orders=("order_id", "count"),
    total_revenue=("total_amount", "sum"),
    avg_order_value=("total_amount", "mean")
).reset_index()


print(dice_quarterly)


# ============================================================
# DICE SUMMARY BY CITY + YEAR + MONTH
# ============================================================

print("\n========== DICE: CITY + YEAR + MONTH ==========")

dice_monthly = dice_result.groupby(
    [
        "city",
        "year",
        "month",
        "month_name"
    ]
).agg(
    total_orders=("order_id", "count"),
    total_revenue=("total_amount", "sum"),
    avg_order_value=("total_amount", "mean")
).reset_index()


print(dice_monthly)


# ============================================================
# SAVE DICE RESULTS
# ============================================================

dice_summary.to_csv(
    "output/olap_dice_city_year.csv",
    index=False
)

dice_quarterly.to_csv(
    "output/olap_dice_quarterly.csv",
    index=False
)

dice_monthly.to_csv(
    "output/olap_dice_monthly.csv",
    index=False
)


# ============================================================
# FINAL OLAP STATUS
# ============================================================

print("\n========== OLAP OPERATIONS COMPLETED ==========")

print("1. Roll-up      ✓")
print("2. Drill-down   ✓")
print("3. Slice        ✓")
print("4. Dice         ✓")

print("\nDice output files:")
print("1. output/olap_dice_city_year.csv")
print("2. output/olap_dice_quarterly.csv")
print("3. output/olap_dice_monthly.csv")

# ============================================================
# 14. PIVOT
# ============================================================
# Pivot changes the orientation of the OLAP view.
#
# Rows    -> City
# Columns -> Year
# Values  -> Total Orders
#
# This allows us to compare the number of orders
# for each city across different years.
# ============================================================

print("\n========== PIVOT: CITY × YEAR ==========")


# ------------------------------------------------------------
# CREATE CITY-YEAR SUMMARY
# ------------------------------------------------------------

pivot_data = dice_result.groupby(
    ["city", "year"]
).agg(
    total_orders=("order_id", "count"),
    total_revenue=("total_amount", "sum")
).reset_index()


print("\nCity-Year Data:")
print(pivot_data)


# ============================================================
# PIVOT 1: TOTAL ORDERS
# ============================================================

print("\n========== PIVOT: TOTAL ORDERS ==========")

pivot_orders = pivot_data.pivot(
    index="city",
    columns="year",
    values="total_orders"
).reset_index()


# Remove column name created by pivot
pivot_orders.columns.name = None


print(pivot_orders)


# ============================================================
# PIVOT 2: TOTAL REVENUE
# ============================================================

print("\n========== PIVOT: TOTAL REVENUE ==========")

pivot_revenue = pivot_data.pivot(
    index="city",
    columns="year",
    values="total_revenue"
).reset_index()


# Remove column name created by pivot
pivot_revenue.columns.name = None


print(pivot_revenue)


# ============================================================
# PIVOT 3: CITY × YEAR WITH BOTH MEASURES
# ============================================================

print("\n========== PIVOT: CITY × YEAR SUMMARY ==========")

pivot_summary = pivot_data.pivot(
    index="city",
    columns="year",
    values=[
        "total_orders",
        "total_revenue"
    ]
)


print(pivot_summary)


# ============================================================
# SAVE PIVOT RESULTS
# ============================================================

pivot_orders.to_csv(
    "output/olap_pivot_orders.csv",
    index=False
)

pivot_revenue.to_csv(
    "output/olap_pivot_revenue.csv",
    index=False
)

pivot_summary.to_csv(
    "output/olap_pivot_summary.csv"
)


# ============================================================
# FINAL OLAP STATUS
# ============================================================

print("\n========== OLAP OPERATIONS COMPLETED ==========")

print("1. Roll-up      ✓")
print("2. Drill-down   ✓")
print("3. Slice        ✓")
print("4. Dice         ✓")
print("5. Pivot        ✓")


print("\nOLAP output files:")

print("1. output/olap_yearly.csv")
print("2. output/olap_quarterly.csv")
print("3. output/olap_monthly.csv")
print("4. output/olap_daily.csv")

print("5. output/olap_drill_quarter.csv")
print("6. output/olap_drill_month.csv")
print("7. output/olap_drill_day.csv")

print("8. output/olap_slice_mumbai.csv")
print("9. output/olap_slice_mumbai_monthly.csv")

print("10. output/olap_dice_city_year.csv")
print("11. output/olap_dice_quarterly.csv")
print("12. output/olap_dice_monthly.csv")

print("13. output/olap_pivot_orders.csv")
print("14. output/olap_pivot_revenue.csv")
print("15. output/olap_pivot_summary.csv")

print("\n========== STAGE 3: OLAP COMPLETED ==========")