
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# 1. CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
MINING = OUTPUT / "data_mining"

st.set_page_config(
    page_title="FoodPulse | Analytics Intelligence",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# 2. PROFESSIONAL DESIGN SYSTEM
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #08111f;
    --panel: #101d30;
    --panel2: #14263d;
    --border: rgba(148, 163, 184, 0.18);
    --text: #edf4ff;
    --muted: #9fb0c8;
    --teal: #48e0c0;
    --blue: #6aa9ff;
}

.stApp {
    background:
        radial-gradient(ellipse at 5% 0%, rgba(41,105,148,.16), transparent 35%),
        radial-gradient(ellipse at 95% 8%, rgba(46,174,157,.10), transparent 30%),
        var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

/* Use the available browser width */
.block-container {
    max-width: 100% !important;
    padding: 2rem 2.5rem 3rem !important;
}

h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    letter-spacing: -.5px;
}

h1 { font-size: clamp(28px, 3vw, 42px) !important; }
h2 { font-size: 25px !important; }
h3 { font-size: 19px !important; }

[data-testid="stSidebar"] {
    background: #0b1728;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text);
}

[data-testid="stSidebar"] .stRadio label {
    font-size: 14px;
}

/* Hero banner */
.hero {
    padding: 32px;
    border: 1px solid rgba(106,169,255,.25);
    border-radius: 22px;
    background: linear-gradient(120deg, #153650, #102b3d 70%, #10343c);
    margin-bottom: 24px;
    box-shadow: 0 16px 50px rgba(0,0,0,.15);
    animation: enter .55s ease-out;
}

.hero-tag {
    display: inline-block;
    color: var(--teal);
    border: 1px solid rgba(72,224,192,.3);
    background: rgba(72,224,192,.08);
    border-radius: 30px;
    padding: 6px 12px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
}

.hero h1 {
    color: #f3f7ff;
    margin: 14px 0 10px;
    line-height: 1.15;
}

.hero p {
    color: #b8c9df;
    max-width: 1000px;
    line-height: 1.8;
    margin-bottom: 0;
    font-size: 15px;
}

/* KPI cards */
.kpi {
    background: linear-gradient(145deg, #152940, #101b2d);
    border: 1px solid var(--border);
    border-radius: 17px;
    padding: 23px;
    min-height: 145px;
    transition: transform .2s ease, border-color .2s ease;
    box-shadow: 0 9px 24px rgba(0,0,0,.12);
}

.kpi:hover {
    transform: translateY(-4px);
    border-color: rgba(72,224,192,.55);
}

.kpi-label {
    color: #a9bad0;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: .8px;
    text-transform: uppercase;
}

.kpi-value {
    color: #f2f7ff;
    font-size: clamp(24px, 2.3vw, 34px);
    font-weight: 800;
    font-family: 'Manrope', sans-serif;
    margin: 12px 0 6px;
}

.kpi-note {
    color: #8ea6c0;
    font-size: 12px;
}

/* Sections and information cards */
.section-label {
    color: var(--teal);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 5px;
}

.info-card {
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 19px 21px;
    background: rgba(16,29,48,.85);
    margin-bottom: 12px;
}

.info-card h4 {
    color: #eaf2ff;
    margin: 0 0 9px;
    font-size: 16px;
}

.info-card p {
    color: #aebed2;
    line-height: 1.75;
    font-size: 14px;
    margin: 0;
}

div[data-testid="stPlotlyChart"] {
    border: 1px solid rgba(148,163,184,.13);
    border-radius: 16px;
    padding: 8px;
    background: rgba(13,25,42,.5);
}

[data-testid="stMetric"] {
    background: rgba(20,38,61,.75);
    border: 1px solid var(--border);
    padding: 16px;
    border-radius: 13px;
}

.stButton > button, .stDownloadButton > button {
    border-radius: 10px;
    transition: all .2s ease;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: var(--teal);
    transform: translateY(-1px);
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

@keyframes enter {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

footer {visibility: hidden;}
#MainMenu {visibility: hidden;}

@media (max-width: 900px) {
    .block-container {
        padding: 1rem !important;
    }
    .hero {
        padding: 22px;
    }
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 3. DATA LOADING
# =========================================================

@st.cache_data
def load_csv(path_string):
    path = Path(path_string)

    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        return df
    except Exception:
        return pd.DataFrame()


orders = load_csv(str(OUTPUT / "cleaned_orders.csv"))
customers = load_csv(str(OUTPUT / "cleaned_customers.csv"))
restaurants = load_csv(str(OUTPUT / "cleaned_restaurants.csv"))

segments = load_csv(str(MINING / "customer_segments.csv"))
predictions = load_csv(
    str(MINING / "order_classification_predictions.csv")
)
confusion = load_csv(
    str(MINING / "order_confusion_matrix.csv")
)
model_comparison = load_csv(
    str(MINING / "demand_model_comparison.csv")
)
forecast = load_csv(
    str(MINING / "improved_demand_forecast.csv")
)

if orders.empty:
    orders = load_csv(str(OUTPUT / "facts" / "fact_orders.csv"))

if not orders.empty:
    if "order_datetime" in orders.columns:
        orders["order_datetime"] = pd.to_datetime(
            orders["order_datetime"], errors="coerce"
        )

    for col in [
        "total_amount", "order_value", "discount_amount",
        "delivery_fee", "taxes", "delivery_time_mins",
        "customer_rating"
    ]:
        if col in orders.columns:
            orders[col] = pd.to_numeric(orders[col], errors="coerce")

    if "order_status" in orders.columns:
        orders["order_status"] = (
            orders["order_status"].astype(str).str.strip()
        )

# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================

COLORS = {
    "teal": "#48e0c0",
    "blue": "#6aa9ff",
    "purple": "#a78bfa",
    "orange": "#f6bd60",
    "red": "#ff7373",
}

PLOT_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"family": "DM Sans", "color": "#dce8f7", "size": 12},
    "margin": {"l": 22, "r": 22, "t": 65, "b": 25},
    "legend": {"orientation": "h", "y": -0.18},
}

def style_fig(fig, height=390):
    fig.update_layout(**PLOT_LAYOUT, height=height)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(
        gridcolor="rgba(148,163,184,.12)",
        zeroline=False
    )
    return fig

def money(value):
    return f"₹{value:,.0f}" if pd.notna(value) else "—"

def number(value):
    return f"{value:,.0f}" if pd.notna(value) else "—"

def hero(title, subtitle, tag="FOOD DELIVERY INTELLIGENCE"):
    st.markdown(
        f"""
        <div class="hero">
            <span class="hero-tag">{tag}</span>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def kpi(label, value, note):
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def section(label, title):
    st.markdown(
        f'<div class="section-label">{label}</div>',
        unsafe_allow_html=True
    )
    st.subheader(title)

def info(title, body):
    st.markdown(
        f"""
        <div class="info-card">
            <h4>{title}</h4>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def download_df(df, label, filename):
    if not df.empty:
        st.download_button(
            label,
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=filename,
            mime="text/csv"
        )

def order_revenue_column(df):
    if "total_amount" in df.columns:
        return "total_amount"
    if "order_value" in df.columns:
        return "order_value"
    return None

# =========================================================
# 5. SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## 🍽️ FoodPulse")
    st.caption("DATA WAREHOUSING + DATA MINING")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Executive Overview",
            "Demand & OLAP",
            "Customer Intelligence",
            "Restaurant Analytics",
            "Machine Learning Lab",
            "Project Methodology",
        ]
    )

    st.divider()
    st.markdown("### PROJECT DATA")
    st.caption("Synthetic food-delivery dataset")
    st.caption("January 2023 – December 2024")

    if not orders.empty and "order_datetime" in orders.columns:
        valid_dates = orders["order_datetime"].dropna()

        if not valid_dates.empty:
            start_default = valid_dates.min().date()
            end_default = valid_dates.max().date()

            st.markdown("### FILTERS")
            selected_dates = st.date_input(
                "Order date range",
                value=(start_default, end_default),
                min_value=start_default,
                max_value=end_default
            )
        else:
            selected_dates = None
    else:
        selected_dates = None

    cities = []
    if not orders.empty and "city" in orders.columns:
        cities = sorted(
            orders["city"].dropna().astype(str).unique().tolist()
        )

    selected_cities = st.multiselect(
        "City",
        options=cities,
        default=cities
    ) if cities else []

    st.divider()
    st.caption("Python · Pandas · Streamlit · Plotly")

# =========================================================
# 6. APPLY FILTERS
# =========================================================

filtered = orders.copy()

if (
    not filtered.empty
    and selected_dates
    and isinstance(selected_dates, (tuple, list))
    and len(selected_dates) == 2
    and "order_datetime" in filtered.columns
):
    start_date, end_date = selected_dates
    filtered = filtered[
        filtered["order_datetime"].dt.date.between(
            start_date, end_date
        )
    ]

if not filtered.empty and "city" in filtered.columns and cities:
    if selected_cities:
        filtered = filtered[
            filtered["city"].astype(str).isin(selected_cities)
        ]
    else:
        filtered = filtered.iloc[0:0]

# =========================================================
# 7. EXECUTIVE OVERVIEW
# =========================================================

if page == "Executive Overview":

    hero(
        "Food Delivery Demand & Customer Analytics",
        "An integrated analytics framework that transforms raw food-delivery "
        "records into business insights using ETL, data warehousing, OLAP "
        "and machine-learning experiments."
    )

    if filtered.empty:
        st.warning("No order records are available for the selected filters.")

    else:
        revenue_col = order_revenue_column(filtered)

        revenue = (
            filtered[revenue_col].sum()
            if revenue_col else np.nan
        )

        total_orders = len(filtered)

        unique_customers = (
            filtered["customer_id"].nunique()
            if "customer_id" in filtered.columns else np.nan
        )

        unique_restaurants = (
            filtered["restaurant_id"].nunique()
            if "restaurant_id" in filtered.columns else np.nan
        )

        cols = st.columns(4, gap="medium")

        with cols[0]:
            kpi("TOTAL ORDERS", number(total_orders), "Selected records")

        with cols[1]:
            kpi("RECORDED REVENUE", money(revenue), "Sum of recorded order totals")

        with cols[2]:
            kpi("UNIQUE CUSTOMERS", number(unique_customers), "Customers represented")

        with cols[3]:
            kpi("RESTAURANTS", number(unique_restaurants), "Restaurants represented")

        st.write("")
        section("BUSINESS PERFORMANCE", "Platform performance at a glance")

        left, right = st.columns([1.45, 1], gap="large")

        if "order_datetime" in filtered.columns:
            temp = filtered.dropna(subset=["order_datetime"]).copy()
            temp["month"] = (
                temp["order_datetime"].dt.to_period("M").astype(str)
            )

            monthly = (
                temp.groupby("month")
                .size()
                .reset_index(name="orders")
            )

            with left:
                fig = px.area(
                    monthly,
                    x="month",
                    y="orders",
                    title="Monthly Order Volume",
                    markers=True,
                )
                fig.update_traces(
                    line_color=COLORS["teal"],
                    fillcolor="rgba(72,224,192,.13)"
                )
                st.plotly_chart(
                    style_fig(fig, 420),
                    use_container_width=True
                )

        if "order_status" in filtered.columns:
            status = (
                filtered["order_status"]
                .value_counts()
                .rename_axis("status")
                .reset_index(name="orders")
            )

            with right:
                fig = px.pie(
                    status,
                    names="status",
                    values="orders",
                    title="Order Status Distribution",
                    hole=0.64,
                    color_discrete_sequence=[
                        COLORS["teal"], COLORS["blue"],
                        COLORS["purple"], COLORS["orange"]
                    ],
                )
                fig.update_traces(
                    textposition="inside",
                    textinfo="percent"
                )
                st.plotly_chart(
                    style_fig(fig, 420),
                    use_container_width=True
                )

        left, right = st.columns(2, gap="large")

        if "city" in filtered.columns:
            city_orders = (
                filtered.groupby("city")
                .size()
                .reset_index(name="orders")
                .sort_values("orders", ascending=False)
                .head(10)
            )

            with left:
                fig = px.bar(
                    city_orders,
                    x="city",
                    y="orders",
                    title="Orders by City",
                    text_auto=True,
                )
                fig.update_traces(marker_color=COLORS["blue"])
                st.plotly_chart(
                    style_fig(fig, 360),
                    use_container_width=True
                )

        if revenue_col and "order_status" in filtered.columns:
            status_revenue = (
                filtered.groupby("order_status")[revenue_col]
                .sum()
                .reset_index()
            )

            with right:
                fig = px.bar(
                    status_revenue,
                    x="order_status",
                    y=revenue_col,
                    title="Recorded Order Value by Status",
                )
                fig.update_traces(marker_color=COLORS["teal"])
                st.plotly_chart(
                    style_fig(fig, 360),
                    use_container_width=True
                )

        info(
            "How to interpret this page",
            "Use the sidebar to filter by date and city. KPI cards summarize the "
            "selected order records, while the charts reveal monthly demand and "
            "differences between locations and order statuses. Recorded totals "
            "should not automatically be interpreted as realized revenue for cancelled orders."
        )

# =========================================================
# 8. DEMAND & OLAP
# =========================================================

elif page == "Demand & OLAP":

    hero(
        "Demand & OLAP Analytics",
        "Explore order patterns across dates, cities and time periods. "
        "This section combines interactive analysis with the OLAP summaries "
        "generated by your Python and Pandas implementation.",
        "MULTIDIMENSIONAL ANALYTICS"
    )

    if filtered.empty:
        st.warning("No order data is available for the selected filters.")

    elif "order_datetime" in filtered.columns:

        temp = filtered.dropna(subset=["order_datetime"]).copy()
        temp["date"] = temp["order_datetime"].dt.date
        temp["month"] = temp["order_datetime"].dt.to_period("M").astype(str)
        temp["hour"] = temp["order_datetime"].dt.hour
        temp["day_name"] = temp["order_datetime"].dt.day_name()

        left, right = st.columns(2, gap="large")

        daily = temp.groupby("date").size().reset_index(name="orders")

        with left:
            fig = px.line(
                daily,
                x="date",
                y="orders",
                title="Daily Order Demand",
            )
            fig.update_traces(
                line_color=COLORS["teal"],
                line_width=2
            )
            st.plotly_chart(
                style_fig(fig),
                use_container_width=True
            )

        monthly = temp.groupby("month").size().reset_index(name="orders")

        with right:
            fig = px.bar(
                monthly,
                x="month",
                y="orders",
                title="Monthly Demand",
            )
            fig.update_traces(marker_color=COLORS["blue"])
            st.plotly_chart(
                style_fig(fig),
                use_container_width=True
            )

        left, right = st.columns(2, gap="large")

        hourly = temp.groupby("hour").size().reset_index(name="orders")

        with left:
            fig = px.bar(
                hourly,
                x="hour",
                y="orders",
                title="Demand by Hour",
            )
            fig.update_traces(marker_color=COLORS["teal"])
            st.plotly_chart(
                style_fig(fig),
                use_container_width=True
            )

        day_order = [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"
        ]

        weekday = temp.groupby("day_name").size().reset_index(name="orders")
        weekday["day_name"] = pd.Categorical(
            weekday["day_name"],
            categories=day_order,
            ordered=True
        )
        weekday = weekday.sort_values("day_name")

        with right:
            fig = px.bar(
                weekday,
                x="day_name",
                y="orders",
                title="Demand by Day of Week",
            )
            fig.update_traces(marker_color=COLORS["purple"])
            st.plotly_chart(
                style_fig(fig),
                use_container_width=True
            )

    section("WAREHOUSE ANALYSIS", "Explore existing OLAP results")

    olap_files = {
        "Daily roll-up": "olap_daily.csv",
        "Monthly roll-up": "olap_monthly.csv",
        "Quarterly roll-up": "olap_quarterly.csv",
        "Yearly roll-up": "olap_yearly.csv",
        "Mumbai slice": "olap_slice_mumbai.csv",
        "City-year dice": "olap_dice_city_year.csv",
        "Monthly dice": "olap_dice_monthly.csv",
        "Order pivot": "olap_pivot_orders.csv",
        "Revenue pivot": "olap_pivot_revenue.csv",
    }

    chosen = st.selectbox(
        "Select an OLAP output",
        list(olap_files.keys())
    )

    olap_df = load_csv(str(OUTPUT / olap_files[chosen]))

    if olap_df.empty:
        st.info("The selected OLAP file was not found.")
    else:
        st.dataframe(
            olap_df,
            use_container_width=True,
            hide_index=True
        )
        download_df(
            olap_df,
            "Download OLAP result",
            "olap_result.csv"
        )

    with st.expander("Learn about the OLAP operations"):
        info("Roll-up", "Aggregates detailed values into summaries such as monthly, quarterly or yearly results.")
        info("Drill-down", "Moves from a summary level into more detailed periods.")
        info("Slice", "Selects one value from a dimension, such as one city.")
        info("Dice", "Filters multiple dimensions at the same time, such as selected cities and years.")
        info("Pivot", "Rearranges dimensions to compare values in a different layout.")

# =========================================================
# 9. CUSTOMER INTELLIGENCE
# =========================================================

elif page == "Customer Intelligence":

    hero(
        "Customer Intelligence",
        "Explore customer behaviour using the K-Means clustering experiment. "
        "Compare the sizes and average characteristics of the discovered groups.",
        "CUSTOMER SEGMENTATION"
    )

    if segments.empty:
        st.warning(
            "Customer segmentation results were not found. "
            "Run src/07_customer_segmentation.py first."
        )

    elif "customer_cluster" not in segments.columns:
        st.error("The customer segment file has no customer_cluster column.")

    else:
        counts = (
            segments["customer_cluster"]
            .value_counts()
            .sort_index()
            .rename_axis("cluster")
            .reset_index(name="customers")
        )

        total = len(segments)
        cols = st.columns(len(counts))

        for i, row in counts.iterrows():
            with cols[i]:
                kpi(
                    f"CLUSTER {row['cluster']}",
                    number(row["customers"]),
                    f"{row['customers'] / total:.1%} of segment records"
                )

        left, right = st.columns(2, gap="large")

        with left:
            fig = px.pie(
                counts,
                names="cluster",
                values="customers",
                title="Customer Distribution",
                hole=0.62,
                color_discrete_sequence=[
                    COLORS["teal"], COLORS["blue"], COLORS["purple"]
                ],
            )
            st.plotly_chart(
                style_fig(fig),
                use_container_width=True
            )

        with right:
            fig = px.bar(
                counts,
                x="cluster",
                y="customers",
                title="Customers per Cluster",
                text_auto=True,
            )
            fig.update_traces(marker_color=COLORS["teal"])
            st.plotly_chart(
                style_fig(fig),
                use_container_width=True
            )

        numeric_cols = [
            col for col in [
                "order_count", "total_spend", "avg_order_value",
                "avg_delivery_time", "avg_rating"
            ]
            if col in segments.columns
        ]

        if numeric_cols:
            section("SEGMENT PROFILES", "Average customer behaviour")

            summary = (
                segments.groupby("customer_cluster")[numeric_cols]
                .mean()
                .round(2)
            )

            st.dataframe(summary, use_container_width=True)

            if "total_spend" in segments.columns and "order_count" in segments.columns:
                fig = px.scatter(
                    segments,
                    x="order_count",
                    y="total_spend",
                    color="customer_cluster",
                    title="Customer Spending vs Order Frequency",
                    labels={
                        "order_count": "Number of Orders",
                        "total_spend": "Total Recorded Spend (₹)",
                        "customer_cluster": "Cluster"
                    },
                    hover_data=[
                        col for col in ["customer_key"]
                        if col in segments.columns
                    ]
                )
                st.plotly_chart(
                    style_fig(fig, 460),
                    use_container_width=True
                )

        info(
            "Interpretation note",
            "K-Means identifies groups from the selected numerical features. Cluster numbers "
            "are arbitrary labels, not rankings. The current segmentation fills missing "
            "features with zero, which can distort average delivery times and ratings; "
            "interpret those measures carefully."
        )

        st.dataframe(
            segments,
            use_container_width=True,
            hide_index=True
        )

        download_df(
            segments,
            "Download customer segments",
            "customer_segments.csv"
        )

# =========================================================
# 10. RESTAURANT ANALYTICS
# =========================================================

elif page == "Restaurant Analytics":

    hero(
        "Restaurant Analytics",
        "Compare recorded order activity and available restaurant attributes "
        "to identify patterns across restaurants and cuisine categories.",
        "RESTAURANT PERFORMANCE"
    )

    if filtered.empty:
        st.warning("No order data is available for the selected filters.")

    elif "restaurant_id" not in filtered.columns:
        st.info("Restaurant-level analysis requires a restaurant_id column.")

    else:
        summary = (
            filtered.groupby("restaurant_id")
            .size()
            .reset_index(name="orders")
        )

        if "total_amount" in filtered.columns:
            revenue = (
                filtered.groupby("restaurant_id")["total_amount"]
                .sum()
                .reset_index(name="recorded_value")
            )
            summary = summary.merge(
                revenue,
                on="restaurant_id",
                how="left"
            )

        if (
            not restaurants.empty
            and "restaurant_id" in restaurants.columns
        ):
            lookup_cols = [
                col for col in [
                    "restaurant_id", "restaurant_name", "city",
                    "cuisine", "avg_rating", "cost_for_two"
                ]
                if col in restaurants.columns
            ]

            lookup = restaurants[lookup_cols].drop_duplicates(
                "restaurant_id"
            )

            summary = summary.merge(
                lookup,
                on="restaurant_id",
                how="left"
            )

        metric = (
            "recorded_value"
            if "recorded_value" in summary.columns
            else "orders"
        )

        label = (
            "restaurant_name"
            if "restaurant_name" in summary.columns
            else "restaurant_id"
        )

        top = (
            summary.sort_values(metric, ascending=False)
            .head(15)
        )

        fig = px.bar(
            top.sort_values(metric),
            x=metric,
            y=label,
            orientation="h",
            title="Top 15 Restaurants",
            hover_data=[
                col for col in ["city", "cuisine", "orders"]
                if col in top.columns
            ],
        )
        fig.update_traces(marker_color=COLORS["teal"])

        st.plotly_chart(
            style_fig(fig, 520),
            use_container_width=True
        )

        if "cuisine" in summary.columns:
            cuisine = (
                summary.groupby("cuisine")["orders"]
                .sum()
                .reset_index()
                .sort_values("orders", ascending=False)
            )

            fig = px.bar(
                cuisine,
                x="cuisine",
                y="orders",
                title="Orders by Cuisine",
            )
            fig.update_traces(marker_color=COLORS["blue"])
            st.plotly_chart(
                style_fig(fig),
                use_container_width=True
            )

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        download_df(
            summary,
            "Download restaurant analytics",
            "restaurant_analytics.csv"
        )

# =========================================================
# 11. MACHINE LEARNING LAB
# =========================================================

elif page == "Machine Learning Lab":

    hero(
        "Machine Learning Lab",
        "Review the experiments used for order classification and demand prediction. "
        "Evaluation metrics and limitations are shown alongside the results.",
        "MODEL EVALUATION"
    )

    section("CLASSIFICATION", "Delivered vs cancelled orders")

    if not confusion.empty:
        matrix = confusion.copy()

        first_col = matrix.columns[0]

        if first_col.startswith("unnamed"):
            matrix = matrix.set_index(first_col)

        numeric_matrix = matrix.apply(
            pd.to_numeric,
            errors="coerce"
        )

        fig = go.Figure(
            data=go.Heatmap(
                z=numeric_matrix.values,
                x=numeric_matrix.columns.tolist(),
                y=numeric_matrix.index.tolist(),
                text=numeric_matrix.values,
                texttemplate="%{text}",
                colorscale=[
                    [0, "#172b46"],
                    [0.5, "#286c7c"],
                    [1, "#48e0c0"]
                ],
                showscale=True
            )
        )

        fig.update_layout(
            title="Order Classification Confusion Matrix",
            xaxis_title="Predicted Outcome",
            yaxis_title="Actual Outcome"
        )

        st.plotly_chart(
            style_fig(fig, 420),
            use_container_width=True
        )

    else:
        st.info("The classification confusion matrix was not found.")

    if not predictions.empty:
        with st.expander("Inspect sample predictions"):
            st.dataframe(
                predictions.head(50),
                use_container_width=True,
                hide_index=True
            )

    info(
        "Classification limitation",
        "The current classifier detects very few cancelled orders. Its overall accuracy "
        "is misleading because delivered orders dominate the dataset. This experiment "
        "requires further improvement before it can be treated as a reliable cancellation predictor."
    )

    section("DEMAND FORECASTING", "Compare forecasting models")

    if (
        not model_comparison.empty
        and {"model", "mae"}.issubset(model_comparison.columns)
    ):
        comparison = model_comparison.sort_values("mae")

        fig = px.bar(
            comparison,
            x="model",
            y="mae",
            color="model",
            title="Mean Absolute Error — Lower Is Better",
            text_auto=".2f",
        )

        st.plotly_chart(
            style_fig(fig, 400),
            use_container_width=True
        )

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("The demand model comparison file was not found.")

    if (
        not forecast.empty
        and "date" in forecast.columns
    ):
        forecast = forecast.copy()
        forecast["date"] = pd.to_datetime(
            forecast["date"],
            errors="coerce"
        )

        fig = go.Figure()

        if "actual_orders" in forecast.columns:
            fig.add_trace(
                go.Scatter(
                    x=forecast["date"],
                    y=forecast["actual_orders"],
                    name="Actual Orders",
                    line=dict(color=COLORS["teal"], width=2.5)
                )
            )

        if "best_model" in forecast.columns:
            fig.add_trace(
                go.Scatter(
                    x=forecast["date"],
                    y=forecast["best_model"],
                    name="Ridge Prediction",
                    line=dict(color=COLORS["blue"], width=2)
                )
            )

        if "previous_week_baseline" in forecast.columns:
            fig.add_trace(
                go.Scatter(
                    x=forecast["date"],
                    y=forecast["previous_week_baseline"],
                    name="Weekly Baseline",
                    line=dict(
                        color=COLORS["orange"],
                        width=1.7,
                        dash="dash"
                    )
                )
            )

        fig.update_layout(
            title="Actual vs Predicted Daily Demand",
            xaxis_title="Date",
            yaxis_title="Order Count"
        )

        st.plotly_chart(
            style_fig(fig, 460),
            use_container_width=True
        )

        download_df(
            forecast,
            "Download demand forecasts",
            "improved_demand_forecast.csv"
        )

    info(
        "Forecasting interpretation",
        "Ridge Regression achieved the lowest MAE among the tested models on the current "
        "chronological split. However, its RMSE was higher than the weekly baseline's RMSE. "
        "Further validation is needed before claiming a reliable forecasting improvement."
    )

# =========================================================
# 12. PROJECT METHODOLOGY
# =========================================================

elif page == "Project Methodology":

    hero(
        "Inside the Analytics Framework",
        "A guided explanation of how this project moves from raw data to a warehouse, "
        "multidimensional analysis and machine-learning experiments.",
        "PROJECT WALKTHROUGH"
    )

    stages = [
        (
            "01",
            "Dataset Understanding",
            "Inspect source CSVs, column types, missing values, duplicates and key relationships."
        ),
        (
            "02",
            "ETL & Data Warehouse",
            "Clean and transform source data, derive date attributes, and organize the information "
            "into dimension tables and a fact table."
        ),
        (
            "03",
            "OLAP Analysis",
            "Use roll-up, drill-down, slice, dice and pivot to examine measures across "
            "different dimensions."
        ),
        (
            "04",
            "Data Mining",
            "Apply K-Means for customer segmentation, Random Forest for order classification, "
            "and regression models for daily demand forecasting."
        ),
        (
            "05",
            "Dashboard & Interpretation",
            "Combine business KPIs, OLAP summaries, customer segments and model evaluation "
            "in one interactive analytics interface."
        ),
    ]

    for num, title, description in stages:
        info(
            f"STAGE {num} — {title}",
            description
        )

    section("DATA WAREHOUSE", "Star schema architecture")

    st.graphviz_chart("""
    digraph Warehouse {
        graph [
            bgcolor="transparent",
            rankdir=TB,
            nodesep=0.65,
            ranksep=0.8
        ]

        node [
            shape=box,
            style="rounded,filled",
            fontname="Arial",
            color="#47728c",
            fillcolor="#142b40",
            fontcolor="#eaf4ff",
            margin="0.22,0.16"
        ]

        edge [
            color="#48e0c0",
            arrowsize=0.8
        ]

        C [label="DIM_CUSTOMER"]
        R [label="DIM_RESTAURANT"]
        D [label="DIM_DATE"]
        F [
            label="FACT_ORDERS",
            fillcolor="#164b50",
            color="#48e0c0",
            penwidth=2
        ]

        {rank=same; C; R; D}

        C -> F
        R -> F
        D -> F
    }
    """)

    info(
        "Why use a data warehouse?",
        "The fact table stores order events and measurable values. Dimension tables describe "
        "customers, restaurants and dates. This structure supports consistent aggregation "
        "and multidimensional analysis."
    )

    info(
        "Dataset scope",
        "The project uses a synthetic Zomato-style dataset containing approximately "
        "9,000 orders, 2,500 customers and 60 restaurants over January 2023–December 2024. "
        "It is an academic dataset, not official Zomato data."
    )

    info(
        "Association rules",
        "The current source data does not provide item-level transaction lists. Market-basket "
        "association rules should not be fabricated; they require valid order-item data."
    )

    info(
        "Technology stack",
        "Python and Pandas handle ETL and OLAP operations. Scikit-learn provides the "
        "machine-learning experiments. Streamlit builds the interface, while Plotly "
        "provides interactive charts."
    )

# =========================================================
# 13. FOOTER
# =========================================================

st.divider()

st.markdown("""
<div style="text-align:center; padding:10px 0 20px; color:#8297b2; font-size:12px;">
    <b style="color:#48e0c0;">FOODPULSE</b>
    &nbsp; · &nbsp; Food Delivery Demand & Customer Analytics
    <br><br>
    Academic Data Warehousing & Data Mining Project
    <br>
    Insights are based on the supplied synthetic dataset.
</div>
""", unsafe_allow_html=True)
