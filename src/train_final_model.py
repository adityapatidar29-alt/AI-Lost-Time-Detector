import pandas as pd
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


print("=" * 70)
print("AI LOST TIME DETECTOR - FINAL TUNED MODEL")
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

label_mapping = {
    "Highly Unproductive": 0,
    "Distracted": 1,
    "Normal": 2,
    "Focused": 3
}

reverse_mapping = {
    0: "Highly Unproductive",
    1: "Distracted",
    2: "Normal",
    3: "Focused"
}

X = df[feature_columns]

y = df["productivity_label"].map(label_mapping)

if y.isna().any():
    raise ValueError("Unknown productivity label found.")

print("Features:", X.shape)
print("Target  :", y.shape)


# --------------------------------------------------
# 3. Train-test split
# --------------------------------------------------

print("\n[3] Splitting dataset...")

from sklearn.model_selection import train_test_split

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
# 4. Create tuned Random Forest
# --------------------------------------------------

print("\n[4] Creating tuned Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# 5. Train model
# --------------------------------------------------

print("\n[5] Training final tuned model...")

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
print("FINAL MODEL ACCURACY")
print("=" * 70)

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")


# --------------------------------------------------
# 8. Classification Report
# --------------------------------------------------

print("\n" + "=" * 70)
print("FINAL CLASSIFICATION REPORT")
print("=" * 70)

class_names = [
    "Highly Unproductive",
    "Distracted",
    "Normal",
    "Focused"
]

report = classification_report(
    y_test,
    y_pred,
    target_names=class_names
)

print(report)


# --------------------------------------------------
# 9. Confusion Matrix
# --------------------------------------------------

print("\n" + "=" * 70)
print("FINAL CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(y_test, y_pred)

print(cm)


# --------------------------------------------------
# 10. Feature Importance
# --------------------------------------------------

print("\n" + "=" * 70)
print("FINAL FEATURE IMPORTANCE")
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
# 11. Save final model
# --------------------------------------------------

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

model_path = models_dir / "final_productivity_model.pkl"

joblib.dump(model, model_path)

print("\nFinal model saved to:")
print(model_path)


# --------------------------------------------------
# 12. Save final feature importance
# --------------------------------------------------

reports_dir = Path("reports")
reports_dir.mkdir(exist_ok=True)

importance_path = (
    reports_dir / "final_feature_importance.csv"
)

feature_importance.to_csv(
    importance_path,
    index=False
)

print("\nFinal feature importance saved to:")
print(importance_path)


# --------------------------------------------------
# 13. Save test predictions
# --------------------------------------------------

predictions_df = pd.DataFrame({
    "actual_label": y_test.map(reverse_mapping),
    "predicted_label": pd.Series(y_pred, index=y_test.index).map(reverse_mapping)
})

predictions_path = reports_dir / "final_test_predictions.csv"

predictions_df.to_csv(
    predictions_path,
    index=False
)

print("\nTest predictions saved to:")
print(predictions_path)


# --------------------------------------------------
# 14. Final
# --------------------------------------------------

print("\n" + "=" * 70)
print("FINAL MODEL TRAINING AND EVALUATION COMPLETED!")
print("=" * 70)
 