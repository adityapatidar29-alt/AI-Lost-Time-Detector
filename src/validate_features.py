import pandas as pd
import numpy as np

from pathlib import Path


# ============================================================
# AI LOST TIME DETECTOR
# ENGINEERED FEATURES VALIDATION
# ============================================================

print("=" * 65)
print("AI LOST TIME DETECTOR - FEATURE VALIDATION")
print("=" * 65)


# ============================================================
# 1. FILE PATH
# ============================================================

file_path = Path(
    "data/processed/engineered_features.csv"
)


# ============================================================
# 2. CHECK FILE
# ============================================================

print("\n[1] Checking engineered dataset...")

if not file_path.exists():

    raise FileNotFoundError(
        f"File not found: {file_path}"
    )

print(
    f"✓ Found: {file_path}"
)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n[2] Loading engineered dataset...")

df = pd.read_csv(
    file_path
)

print(
    f"Rows    : {df.shape[0]}"
)

print(
    f"Columns : {df.shape[1]}"
)


# ============================================================
# 4. CHECK SHAPE
# ============================================================

print("\n[3] Checking dataset shape...")

assert df.shape == (10000, 20)

print(
    "✓ Dataset contains exactly 10,000 rows "
    "and 20 columns."
)


# ============================================================
# 5. DEFINE ENGINEERED FEATURES
# ============================================================

engineered_features = [

    "typing_efficiency",

    "mouse_efficiency",

    "break_frequency",

    "distraction_index",

    "engagement_score",

    "interaction_intensity",

    "switching_pressure",

    "active_time_ratio"

]


# ============================================================
# 6. CHECK FEATURE EXISTENCE
# ============================================================

print("\n[4] Checking engineered features...")

for feature in engineered_features:

    assert feature in df.columns

    print(
        f"✓ {feature}"
    )


# ============================================================
# 7. CHECK MISSING VALUES
# ============================================================

print("\n[5] Checking missing values...")

missing_values = (

    df[engineered_features]
    .isnull()
    .sum()
    .sum()

)

print(
    f"Missing values: {missing_values}"
)

assert missing_values == 0


# ============================================================
# 8. CHECK INFINITE VALUES
# ============================================================

print("\n[6] Checking infinite values...")

infinite_values = np.isinf(
    df[engineered_features]
    .select_dtypes(include=np.number)
).sum().sum()


print(
    f"Infinite values: {infinite_values}"
)

assert infinite_values == 0


# ============================================================
# 9. CHECK NUMERIC TYPES
# ============================================================

print("\n[7] Checking data types...")

for feature in engineered_features:

    assert np.issubdtype(
        df[feature].dtype,
        np.number
    )

print(
    "✓ All engineered features are numeric."
)


# ============================================================
# 10. CHECK TARGET
# ============================================================

print("\n[8] Checking target column...")

assert "productivity_label" in df.columns

labels = sorted(
    df["productivity_label"]
    .unique()
)

print(
    f"Productivity labels: {labels}"
)

expected_labels = [

    "Distracted",

    "Focused",

    "Highly Unproductive",

    "Normal"

]

assert labels == expected_labels

print(
    "✓ Target column is valid."
)


# ============================================================
# 11. CHECK VALUE RANGES
# ============================================================

print("\n[9] Checking feature ranges...")


assert (
    df["active_time_ratio"]
    .between(0, 1)
    .all()
)

assert (
    df["engagement_score"]
    .between(0, 100)
    .all()
)

assert (
    df["typing_efficiency"] >= 0
).all()

assert (
    df["mouse_efficiency"] >= 0
).all()

assert (
    df["break_frequency"] >= 0
).all()

assert (
    df["distraction_index"] >= 0
).all()

assert (
    df["interaction_intensity"] >= 0
).all()

assert (
    df["switching_pressure"] >= 0
).all()


print(
    "✓ Feature value ranges are valid."
)


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)

print(
    "ENGINEERED FEATURE VALIDATION PASSED!"
)

print("=" * 65)

print(
    f"\nTotal records : {len(df)}"
)

print(
    f"Total columns : {len(df.columns)}"
)

print(
    f"New features  : {len(engineered_features)}"
)

print(
    "Missing values: 0"
)

print(
    "Infinite values: 0"
)