# Food Delivery Demand and Customer Analytics

## Project Overview

This project, titled **“An Integrated Data Warehousing and Data Mining Framework for Food Delivery Demand and Customer Analytics,”** analyzes food-delivery data using data warehousing concepts, OLAP operations, machine learning algorithms, and interactive visualization.

The project transforms raw customer, order, and restaurant datasets into meaningful analytical insights to understand customer behaviour, historical demand, order status, and restaurant performance.

## Objectives

- Understand and preprocess food-delivery datasets.
- Perform ETL using Python and Pandas.
- Organize data using a star-schema warehouse design.
- Perform OLAP operations for multidimensional analysis.
- Segment customers using K-Means clustering.
- Classify order status using a Random Forest Classifier.
- Compare regression models for demand prediction.
- Visualize analytical results through an interactive dashboard.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Plotly
- CSV datasets

## Project Modules

1. Dataset Understanding
2. Data Cleaning and ETL
3. Dimension and Fact Table Creation
4. OLAP Analysis
5. Customer Segmentation
6. Order Classification
7. Demand Prediction
8. Data Mining Visualization
9. Interactive Analytics Dashboard

## Dashboard

The Streamlit dashboard provides an interface for exploring executive KPIs, demand and OLAP analysis, customer intelligence, restaurant analytics, machine learning results, and project methodology.

## How to Run the Dashboard

Install the required dependencies, then run the application from the project directory:

```bash
python -m pip install streamlit pandas plotly scikit-learn
python -m streamlit run app.py
```

Ensure the required project data and output files are available in their expected locations before running the application.

## Academic Domain

Data Warehousing and Data Mining (DWDM)

## Note

This project uses a Zomato-style food-delivery dataset sourced from a public GitHub repository. It is an academic analytics prototype and does not represent an official Zomato system.
