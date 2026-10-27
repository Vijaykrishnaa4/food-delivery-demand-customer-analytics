
import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load cleaned order data
data = pd.read_csv("output/cleaned_orders.csv")

data.columns = data.columns.str.strip().str.lower()

print("Dataset loaded:", data.shape)

# 2. Create the target variable
# Delivered = Delivered
# Both cancellation types = Cancelled

data = data[data["order_status"].notna()].copy()

data["target"] = data["order_status"].apply(
    lambda status: (
        "Delivered"
        if str(status).strip().lower() == "delivered"
        else "Cancelled"
    )
)

print("\nTarget distribution:")
print(data["target"].value_counts())

# 3. Select features known before the outcome
numeric_features = [
    "order_value",
    "discount_amount",
    "order_year",
    "order_month",
    "order_day",
    "order_hour",
    "is_weekend"
]

categorical_features = ["payment_method"]

# Keep only columns available in the dataset
numeric_features = [
    col for col in numeric_features if col in data.columns
]
categorical_features = [
    col for col in categorical_features if col in data.columns
]

features = numeric_features + categorical_features

if not features:
    raise ValueError("No usable feature columns were found.")

X = data[features].copy()
y = data["target"]

# Convert numeric columns safely
for col in numeric_features:
    X[col] = pd.to_numeric(X[col], errors="coerce")

for col in categorical_features:
    X[col] = X[col].fillna("Unknown").astype(str)

# 4. Prepare preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

transformers = []

if numeric_features:
    transformers.append(
        ("numeric", numeric_pipeline, numeric_features)
    )

if categorical_features:
    transformers.append(
        ("categorical", categorical_pipeline, categorical_features)
    )

preprocessor = ColumnTransformer(transformers)

# 5. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 6. Build and train the model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)

# 7. Make predictions
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
report = classification_report(
    y_test,
    predictions,
    zero_division=0
)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=["Cancelled", "Delivered"]
)

# 8. Create output folder
out = Path("output/data_mining")
out.mkdir(parents=True, exist_ok=True)

# Save predictions
results = pd.DataFrame({
    "actual_status": y_test,
    "predicted_status": predictions
}, index=y_test.index)

results.to_csv(out / "order_classification_predictions.csv",
               index_label="original_row")

# Save confusion matrix
matrix_df = pd.DataFrame(
    matrix,
    index=["Actual Cancelled", "Actual Delivered"],
    columns=["Predicted Cancelled", "Predicted Delivered"]
)
matrix_df.to_csv(out / "order_confusion_matrix.csv")

# Save report
with open(out / "order_classification_report.txt",
          "w", encoding="utf-8") as file:
    file.write(f"Accuracy: {accuracy:.4f}\n\n")
    file.write(report)
    file.write("\n\nConfusion Matrix:\n")
    file.write(matrix_df.to_string())

# Save trained model
joblib.dump(model, out / "order_classification_model.joblib")

# 9. Display results
print("\n--- ORDER CLASSIFICATION RESULTS ---")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(report)

print("\nConfusion Matrix:")
print(matrix_df)

print("\nSaved results in:", out)
