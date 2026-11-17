import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("=" * 70)
print("AI LOST TIME DETECTOR - ENGINEERED FEATURES MODEL")
print("=" * 70)


# --------------------------------------------------
# 1. Load engineered dataset
# --------------------------------------------------

print("\n[1] Loading engineered dataset...")

data_path = Path("data/processed/engineered_features.csv")

df = pd.read_csv(data_path)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. Prepare features and target
# --------------------------------------------------

print("\n[2] Preparing features and target...")

target_column = "productivity_label"

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
    "activity_score",

    "typing_efficiency",
    "mouse_efficiency",
    "break_frequency",
    "distraction_index",
    "engagement_score",
    "interaction_intensity",
    "switching_pressure",
    "active_time_ratio"
]

X = df[feature_columns]

label_mapping = {
    "Highly Unproductive": 0,
    "Distracted": 1,
    "Normal": 2,
    "Focused": 3
}

y = df[target_column].map(label_mapping)

if y.isna().any():
    raise ValueError("Unknown productivity label found in dataset.")


# --------------------------------------------------
# 3. Train-test split
# --------------------------------------------------

print("\n[3] Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# --------------------------------------------------
# 4. Create Random Forest
# --------------------------------------------------

print("\n[4] Creating Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# 5. Train model
# --------------------------------------------------

print("\n[5] Training model...")

model.fit(X_train, y_train)

print("Model training completed.")


# --------------------------------------------------
# 6. Predictions
# --------------------------------------------------

print("\n[6] Making predictions...")

y_pred = model.predict(X_test)

print("Predictions completed.")


# --------------------------------------------------
# 7. Accuracy
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 70)
print("ENGINEERED MODEL ACCURACY")
print("=" * 70)

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")


# --------------------------------------------------
# 8. Classification Report
# --------------------------------------------------

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

class_names = [
    "Highly Unproductive",
    "Distracted",
    "Normal",
    "Focused"
]

print(
    classification_report(
        y_test,
        y_pred,
        target_names=class_names
    )
)


# --------------------------------------------------
# 9. Confusion Matrix
# --------------------------------------------------

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(y_test, y_pred)

print(cm)


# --------------------------------------------------
# 10. Feature Importance
# --------------------------------------------------

print("\n" + "=" * 70)
print("TOP FEATURE IMPORTANCE")
print("=" * 70)

feature_importance = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False
)

print(feature_importance.to_string(index=False))


# --------------------------------------------------
# 11. Save model
# --------------------------------------------------

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

model_path = models_dir / "engineered_random_forest.pkl"

joblib.dump(model, model_path)

print("\nModel saved to:")
print(model_path)


# --------------------------------------------------
# 12. Save feature importance
# --------------------------------------------------

reports_dir = Path("reports")
reports_dir.mkdir(exist_ok=True)

importance_path = (
    reports_dir / "engineered_feature_importance.csv"
)

feature_importance.to_csv(
    importance_path,
    index=False
)

print("\nFeature importance saved to:")
print(importance_path)


# --------------------------------------------------
# 13. Final
# --------------------------------------------------

print("\n" + "=" * 70)
print("ENGINEERED FEATURES MODEL TRAINING COMPLETED!")
print("=" * 70)
