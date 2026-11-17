import pandas as pd
from pathlib import Path

# ==========================================
# AI LOST TIME DETECTOR
# Dataset Quality Checker
# ==========================================

print("=" * 60)
print("AI LOST TIME DETECTOR - DATASET QUALITY CHECK")
print("=" * 60)

# ------------------------------------------
# 1. Dataset Path
# ------------------------------------------

dataset_path = Path("data/raw/productivity_data.csv")

# Check whether file exists
if not dataset_path.exists():
    print("\nERROR: Dataset file not found!")
    print(f"Expected location: {dataset_path}")
    exit()

print(f"\nDataset found: {dataset_path}")

# ------------------------------------------
# 2. Load Dataset
# ------------------------------------------

df = pd.read_csv(dataset_path)

print("\nDataset loaded successfully.")

# ------------------------------------------
# 3. Basic Shape
# ------------------------------------------

rows, columns = df.shape

print("\n--- Dataset Shape ---")
print(f"Rows    : {rows}")
print(f"Columns : {columns}")

# ------------------------------------------
# 4. Column Names
# ------------------------------------------

print("\n--- Columns ---")

for column in df.columns:
    print(f"- {column}")

# ------------------------------------------
# 5. Missing Values
# ------------------------------------------

print("\n--- Missing Values ---")

missing_values = df.isnull().sum()

print(missing_values)

total_missing = missing_values.sum()

print(f"\nTotal missing values: {total_missing}")

# ------------------------------------------
# 6. Duplicate Rows
# ------------------------------------------

print("\n--- Duplicate Rows ---")

duplicate_count = df.duplicated().sum()

print(f"Duplicate rows: {duplicate_count}")

# ------------------------------------------
# 7. Data Types
# ------------------------------------------

print("\n--- Data Types ---")

print(df.dtypes)

# ------------------------------------------
# 8. Productivity Distribution
# ------------------------------------------

print("\n--- Productivity Distribution ---")

label_counts = df["productivity_label"].value_counts()

print(label_counts)

print("\n--- Productivity Percentage ---")

label_percentages = (
    df["productivity_label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print(label_percentages)

# ------------------------------------------
# 9. Numeric Statistics
# ------------------------------------------

print("\n--- Numeric Statistics ---")

print(df.describe())

# ------------------------------------------
# 10. Invalid Value Checks
# ------------------------------------------

print("\n--- Invalid Value Checks ---")

checks = {
    "Negative typing speed": (df["typing_speed"] < 0).sum(),
    "Negative mouse activity": (df["mouse_activity"] < 0).sum(),
    "Negative idle time": (df["idle_time"] < 0).sum(),
    "Negative app switches": (df["app_switch_count"] < 0).sum(),
    "Negative session duration": (df["session_duration"] < 0).sum(),
    "Negative break count": (df["break_count"] < 0).sum(),
    "Invalid idle ratio": (df["idle_ratio"] < 0).sum(),
    "Invalid switch rate": (df["switch_rate"] < 0).sum(),
}

for check, count in checks.items():
    print(f"{check}: {count}")

# ------------------------------------------
# 11. Final Status
# ------------------------------------------

print("\n" + "=" * 60)

if total_missing == 0 and duplicate_count == 0:
    print("DATASET QUALITY CHECK PASSED!")
else:
    print("DATASET NEEDS CLEANING.")

print("=" * 60)