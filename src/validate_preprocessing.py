import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# AI LOST TIME DETECTOR
# PREPROCESSED DATA VALIDATION
# ============================================================

print("=" * 65)
print("AI LOST TIME DETECTOR - PREPROCESSING VALIDATION")
print("=" * 65)


# ============================================================
# 1. DEFINE FILE PATHS
# ============================================================

processed_dir = Path("data/processed")


X_train_path = processed_dir / "X_train_scaled.csv"
X_test_path = processed_dir / "X_test_scaled.csv"
y_train_path = processed_dir / "y_train.csv"
y_test_path = processed_dir / "y_test.csv"


# ============================================================
# 2. CHECK FILE EXISTENCE
# ============================================================

print("\n[1] Checking processed files...")

files = [
    X_train_path,
    X_test_path,
    y_train_path,
    y_test_path
]


for file_path in files:

    if file_path.exists():

        print(f"✓ Found: {file_path}")

    else:

        raise FileNotFoundError(
            f"Missing file: {file_path}"
        )


# ============================================================
# 3. LOAD PROCESSED DATA
# ============================================================

print("\n[2] Loading processed datasets...")

X_train = pd.read_csv(X_train_path)
X_test = pd.read_csv(X_test_path)

y_train = pd.read_csv(y_train_path)
y_test = pd.read_csv(y_test_path)


print(
    f"X_train shape: {X_train.shape}"
)

print(
    f"X_test shape : {X_test.shape}"
)

print(
    f"y_train shape: {y_train.shape}"
)

print(
    f"y_test shape : {y_test.shape}"
)


# ============================================================
# 4. CHECK SHAPES
# ============================================================

print("\n[3] Checking dataset shapes...")

assert X_train.shape == (8000, 10)
assert X_test.shape == (2000, 10)

assert y_train.shape == (8000, 1)
assert y_test.shape == (2000, 1)

print("✓ Dataset shapes are correct.")


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

print("\n[4] Checking missing values...")

train_missing = X_train.isnull().sum().sum()
test_missing = X_test.isnull().sum().sum()

y_train_missing = y_train.isnull().sum().sum()
y_test_missing = y_test.isnull().sum().sum()


print(f"X_train missing values: {train_missing}")
print(f"X_test missing values : {test_missing}")
print(f"y_train missing values: {y_train_missing}")
print(f"y_test missing values : {y_test_missing}")


assert train_missing == 0
assert test_missing == 0
assert y_train_missing == 0
assert y_test_missing == 0


# ============================================================
# 6. CHECK NUMERIC VALUES
# ============================================================

print("\n[5] Checking numeric data...")

assert all(
    np.issubdtype(dtype, np.number)
    for dtype in X_train.dtypes
)

assert all(
    np.issubdtype(dtype, np.number)
    for dtype in X_test.dtypes
)

print("✓ All feature columns are numeric.")


# ============================================================
# 7. CHECK TARGET VALUES
# ============================================================

print("\n[6] Checking target labels...")

train_labels = sorted(
    y_train.iloc[:, 0].unique()
)

test_labels = sorted(
    y_test.iloc[:, 0].unique()
)


print(
    f"Training labels: {train_labels}"
)

print(
    f"Testing labels : {test_labels}"
)


expected_labels = [0, 1, 2, 3]


assert train_labels == expected_labels
assert test_labels == expected_labels


# ============================================================
# 8. CHECK SCALING
# ============================================================

print("\n[7] Checking feature scaling...")

train_means = X_train.mean()
train_stds = X_train.std()


print("\nTraining feature means:")

print(
    train_means.round(3)
)


print("\nTraining feature standard deviations:")

print(
    train_stds.round(3)
)


# Means should be close to zero
assert np.allclose(
    train_means.values,
    0,
    atol=0.05
)


# Standard deviations should be close to 1
assert np.allclose(
    train_stds.values,
    1,
    atol=0.05
)


print(
    "\n✓ Features are properly standardized."
)


# ============================================================
# 9. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 65)

print(
    "PREPROCESSING VALIDATION PASSED!"
)

print("=" * 65)