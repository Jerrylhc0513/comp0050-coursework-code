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

# Drop ID columns, cancellation detail columns (leak target variable),
# Payment Method, and location columns (too sparse)
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
]
df = df.drop(columns=cols_to_drop)

# Binary target variable: 1 = completed, 0 = not completed
df["is_completed"] = (df["Booking Status"] == "Completed").astype(int)
df = df.drop(columns=["Booking Status"])

# Extract time features from Date and Time columns
df["Date"]        = pd.to_datetime(df["Date"])
df["hour"]        = pd.to_datetime(df["Time"], errors="coerce").dt.hour
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

# Rating nulls are ambiguous: 0 could mean "very bad" or "no rating"
# Add indicator columns so model can distinguish the two cases
df["driver_rating_missing"]   = df["Driver Ratings"].isna().astype(int)
df["customer_rating_missing"] = df["Customer Rating"].isna().astype(int)

df["Driver Ratings"]  = df["Driver Ratings"].fillna(0)
df["Customer Rating"] = df["Customer Rating"].fillna(0)

# All other nulls mean ride didn't complete → 0 is accurate
df["Avg VTAT"]      = df["Avg VTAT"].fillna(0)
df["Avg CTAT"]      = df["Avg CTAT"].fillna(0)
df["Booking Value"] = df["Booking Value"].fillna(0)
df["Ride Distance"] = df["Ride Distance"].fillna(0)

# One-hot encode categorical columns
cat_cols = ["Vehicle Type", "time_period"]
df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)


# Save cleaned data
clean_path = os.path.join(SAVE_DIR, "ncr_ride_bookings_clean.csv")
df.to_csv(clean_path, index=False)



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Split features and target
X = df.drop(columns=["is_completed"])
y = df["is_completed"]

# Train test split (80% train, 20% test, stratify keeps class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
importance = pd.Series(rf.feature_importances_, index=X.columns)
print("\nTop 10 Most Important Features:")
print(importance.nlargest(10))