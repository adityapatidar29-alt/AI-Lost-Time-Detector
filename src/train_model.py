import pandas as pd
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# AI LOST TIME DETECTOR
# BASELINE MODEL TRAINING
# ============================================================

print("=" * 65)
print("AI LOST TIME DETECTOR - BASELINE MODEL")
print("=" * 65)


# ============================================================
# 1. PATHS
# ============================================================

data_dir = Path("data/processed")
models_dir = Path("models")

models_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD TRAINING DATA
# ============================================================

print("\n[1] Loading training data...")

X_train = pd.read_csv(
    data_dir / "X_train_scaled.csv"
)

X_test = pd.read_csv(
    data_dir / "X_test_scaled.csv"
)

y_train = pd.read_csv(
    data_dir / "y_train.csv"
).squeeze()

y_test = pd.read_csv(
    data_dir / "y_test.csv"
).squeeze()


print(
    f"X_train: {X_train.shape}"
)

print(
    f"X_test : {X_test.shape}"
)

print(
    f"y_train: {y_train.shape}"
)

print(
    f"y_test : {y_test.shape}"
)


# ============================================================
# 3. CREATE BASELINE MODEL
# ============================================================

print("\n[2] Creating Random Forest model...")

model = RandomForestClassifier(

    n_estimators=200,

    max_depth=None,

    min_samples_split=2,

    min_samples_leaf=1,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1

)


# ============================================================
# 4. TRAIN MODEL
# ============================================================

print("\n[3] Training model...")

model.fit(
    X_train,
    y_train
)

print(
    "Model training completed."
)


# ============================================================
# 5. MAKE PREDICTIONS
# ============================================================

print("\n[4] Making predictions...")

y_pred = model.predict(
    X_test
)


# ============================================================
# 6. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n" + "=" * 65)

print(
    f"BASELINE ACCURACY: {accuracy:.4f}"
)

print(
    f"BASELINE ACCURACY: {accuracy * 100:.2f}%"
)


# ============================================================
# 7. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 65)

print("CLASSIFICATION REPORT")

print("=" * 65)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Highly Unproductive",
            "Distracted",
            "Normal",
            "Focused"
        ]
    )
)


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 65)

print("CONFUSION MATRIX")

print("=" * 65)

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ============================================================
# 9. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 65)

print("FEATURE IMPORTANCE")

print("=" * 65)


feature_importance = pd.DataFrame({

    "feature": X_train.columns,

    "importance": model.feature_importances_

})


feature_importance = (
    feature_importance
    .sort_values(
        "importance",
        ascending=False
    )
)


print(
    feature_importance
)


# ============================================================
# 10. SAVE MODEL
# ============================================================

model_path = (
    models_dir /
    "baseline_random_forest.pkl"
)


joblib.dump(
    model,
    model_path
)


print("\nModel saved to:")

print(
    model_path
)


# ============================================================
# 11. SAVE FEATURE IMPORTANCE
# ============================================================

importance_path = (
    Path("reports") /
    "baseline_feature_importance.csv"
)


feature_importance.to_csv(
    importance_path,
    index=False
)


print("\nFeature importance saved to:")

print(
    importance_path
)


# ============================================================
# 12. FINAL STATUS
# ============================================================

print("\n" + "=" * 65)

print(
    "BASELINE MODEL TRAINING COMPLETED!"
)

print("=" * 65)