import pandas as pd
import joblib # py ki library for saving and loading kar na ko kaam aate hai 

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ==========================================================
# AI LOST TIME DETECTOR
# DATA PREPROCESSING
# ==========================================================

print("=" *65)
print("AI LOST TIME DETECTOR - DATA PREPROCESSING")
print("=" *65)

# ============================================================
# 1. DEFINE PATHS
# ============================================================

input_path = Path("data/raw/productivity_data.csv")

processed_dir = Path("data/processed")
models_dir = Path("models")

processed_dir.mkdir(
    parents=True,
    exist_ok=True
)

models_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n[1] Loading dataset...")

df = pd.read_csv(input_path)

print(f"Dataset loaded successfully.")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# 3. CONVERT TIMESTAMP
# ============================================================

print("\n[2] Processing timestamp...")

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

print("Timestamp converted successfully.")


# ============================================================
# 4. CHECK MISSING VALUES
# ============================================================

print("\n[3] Checking missing values...")

missing_values = df.isnull().sum().sum()

print(f"Total missing values: {missing_values}")

if missing_values > 0:

    print("ERROR: Dataset contains missing values.")

    raise ValueError(
        "Missing values detected. "
        "Clean the dataset before continuing."
    )

else:

    print("No missing values found.")


# ============================================================
# 5. DEFINE FEATURES
# ============================================================

print("\n[4] Selecting ML features...")


feature_columns = [

    "typing_speed",

    "mouse_activity",

    "idle_time",

    "app_switch_count",

    "session_duration",

    "break_count",

    "time_of_day",

    "idle_ratio",

    "switch_rate",

    "activity_score"

]


target_column = "productivity_label"


X = df[feature_columns].copy()

y = df[target_column].copy()


print("\nFeatures selected:")

for feature in feature_columns:

    print(f"- {feature}")


print(f"\nTarget column: {target_column}")


# ============================================================
# 6. ENCODE TARGET LABELS
# ============================================================

print("\n[5] Encoding productivity labels...")


label_mapping = {

    "Highly Unproductive": 0,

    "Distracted": 1,

    "Normal": 2,

    "Focused": 3

}


y_encoded = y.map(label_mapping)


# Check for unknown labels

if y_encoded.isnull().any():

    unknown_labels = y[
        y_encoded.isnull()
    ].unique()

    raise ValueError(
        f"Unknown productivity labels found: "
        f"{unknown_labels}"
    )


y_encoded = y_encoded.astype(int)


print("\nLabel mapping:")

for label, number in label_mapping.items():

    print(f"{label} -> {number}")


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

print("\n[6] Splitting dataset...")


X_train, X_test, y_train, y_test = train_test_split(

    X,

    y_encoded,

    test_size=0.20,

    random_state=42,

    stratify=y_encoded

)


print(f"Training records : {len(X_train)}")
print(f"Testing records  : {len(X_test)}")


# ============================================================
# 8. FEATURE SCALING
# ============================================================

print("\n[7] Scaling numerical features...")


scaler = StandardScaler()


# IMPORTANT:
# Scaler is fitted ONLY on training data.

X_train_scaled = scaler.fit_transform(
    X_train
)


# Test data is transformed using
# the scaler learned from training data.

X_test_scaled = scaler.transform(
    X_test
)


print("Feature scaling completed.")


# ============================================================
# 9. CONVERT SCALED DATA BACK TO DATAFRAME
# ============================================================

X_train_scaled_df = pd.DataFrame(

    X_train_scaled,

    columns=feature_columns,

    index=X_train.index

)


X_test_scaled_df = pd.DataFrame(

    X_test_scaled,

    columns=feature_columns,

    index=X_test.index

)


# ============================================================
# 10. SAVE PROCESSED DATA
# ============================================================

print("\n[8] Saving processed datasets...")


X_train_scaled_df.to_csv(

    processed_dir /
    "X_train_scaled.csv",

    index=False

)


X_test_scaled_df.to_csv(

    processed_dir /
    "X_test_scaled.csv",

    index=False

)


y_train.to_csv(

    processed_dir /
    "y_train.csv",

    index=False

)


y_test.to_csv(

    processed_dir /
    "y_test.csv",

    index=False

)


# ============================================================
# 11. SAVE SCALER
# ============================================================

scaler_path = models_dir / "scaler.pkl"


joblib.dump(

    scaler,

    scaler_path

)


# ============================================================
# 12. SAVE LABEL MAPPING
# ============================================================

mapping_path = models_dir / "label_mapping.pkl"


joblib.dump(

    label_mapping,

    mapping_path

)


# ============================================================
# 13. SAVE COMPLETE PROCESSED DATASET
# ============================================================

processed_df = df.copy()

processed_df["productivity_label_encoded"] = y_encoded


processed_df.to_csv(

    processed_dir /
    "processed_productivity_data.csv",

    index=False

)


# ============================================================
# 14. DISPLAY FINAL INFORMATION
# ============================================================

print("\n[9] Preprocessing Summary")

print("-" * 50)

print(
    f"Original records : {len(df)}"
)

print(
    f"Training records : {len(X_train)}"
)

print(
    f"Testing records  : {len(X_test)}"
)

print(
    f"Number of features: {len(feature_columns)}"
)


print("\nTraining class distribution:")

print(
    y_train.value_counts()
    .sort_index()
)


print("\nTesting class distribution:")

print(
    y_test.value_counts()
    .sort_index()
)


# ============================================================
# 15. FINAL STATUS
# ============================================================

print("\n" + "=" * 65)

print("DATA PREPROCESSING COMPLETED SUCCESSFULLY!")

print("=" * 65)

print("\nGenerated files:")

print(
    "data/processed/X_train_scaled.csv"
)

print(
    "data/processed/X_test_scaled.csv"
)

print(
    "data/processed/y_train.csv"
)

print(
    "data/processed/y_test.csv"
)

print(
    "data/processed/processed_productivity_data.csv"
)

print(
    "models/scaler.pkl"
)

print(
    "models/label_mapping.pkl"
)