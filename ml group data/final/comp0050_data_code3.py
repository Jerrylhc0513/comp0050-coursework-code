import os
import pandas as pd
import numpy as np

# Kaggle API credentials
os.environ["KAGGLE_USERNAME"] = "JerryLi0513"
os.environ["KAGGLE_KEY"] = "KGAT_3b0e58219fbd1115b71520efe161d1b2"

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

DATASET  = "yashdevladdha/uber-ride-analytics-dashboard"
SAVE_DIR = "./data"
os.makedirs(SAVE_DIR, exist_ok=True)

api.dataset_download_files(DATASET, path=SAVE_DIR, unzip=True)

csv_path = os.path.join(SAVE_DIR, "ncr_ride_bookings.csv")
df = pd.read_csv(csv_path)

# Drop ID columns, cancellation detail columns, and columns not available at prediction time
cols_to_drop = [
    "Booking ID",
    "Customer ID",
    "Cancelled Rides by Customer",
    "Reason for cancelling by Customer",
    "Cancelled Rides by Driver",
    "Driver Cancellation Reason",
    "Incomplete Rides",
    "Incomplete Rides Reason",
    "Payment Method",
    "Pickup Location",
    "Drop Location",
    "Booking Status",
    "Customer Rating",
]
df = df.drop(columns=cols_to_drop)

# Only keep completed rides (only these have Driver Ratings)
df = df[df["Driver Ratings"].notna()].copy()

# Target variable: 1 = high rating (>= 4.5), 0 = low rating (< 4.5)
df["high_rating"] = (df["Driver Ratings"] >= 4.5).astype(int)
df = df.drop(columns=["Driver Ratings"])

print("Target distribution:")
print(df["high_rating"].value_counts())
print(f"High rating rate: {df['high_rating'].mean():.2%}")

# Extract time features from Date and Time columns
df["Date"]        = pd.to_datetime(df["Date"])
df["hour"]        = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce").dt.hour
df["day_of_week"] = df["Date"].dt.dayofweek   # 0=Monday, 6=Sunday
df["month"]       = df["Date"].dt.month
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

def get_time_period(hour):
    if 5 <= hour < 10:    return "morning_peak"
    elif 10 <= hour < 16: return "midday"
    elif 16 <= hour < 20: return "evening_peak"
    elif 20 <= hour < 24: return "night"
    else:                 return "late_night"

df["time_period"] = df["hour"].apply(get_time_period)
df = df.drop(columns=["Date", "Time"])

# One-hot encode categorical columns
cat_cols = ["Vehicle Type", "time_period"]
df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

# Save cleaned data
clean_path = os.path.join(SAVE_DIR, "ncr_ride_bookings_clean.csv")
df.to_csv(clean_path, index=False)

# ── Random Forest with GridSearchCV ─────────────────────────
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

X = df.drop(columns=["high_rating"])
y = df["high_rating"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 15],
    "min_samples_leaf": [1, 5],
    "max_features": ["sqrt"],
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
    param_grid,
    cv=5,
    scoring="roc_auc", 
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best CV accuracy:", grid.best_score_)

best_rf = grid.best_estimator_
y_pred = best_rf.predict(X_test)
y_prob = best_rf.predict_proba(X_test)[:, 1]

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
importance = pd.Series(best_rf.feature_importances_, index=X.columns)
print("\nTop 10 Most Important Features:")
print(importance.nlargest(10))

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.show()