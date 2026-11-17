import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier


print("=" * 70)
print("AI LOST TIME DETECTOR - 5-FOLD CROSS VALIDATION")
print("=" * 70)


# --------------------------------------------------
# 1. Load engineered dataset
# --------------------------------------------------

print("\n[1] Loading engineered dataset...")

data_path = Path("data/processed/engineered_features.csv")

df = pd.read_csv(data_path)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. Prepare features
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

X = df[feature_columns]
y = df["productivity_label"].map(label_mapping)

if y.isna().any():
    raise ValueError("Unknown productivity label found.")

print("Features:", X.shape)
print("Target :", y.shape)


# --------------------------------------------------
# 3. Create Random Forest
# --------------------------------------------------

print("\n[3] Creating Random Forest model...")

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
# 4. Create 5-Fold Stratified Cross Validation
# --------------------------------------------------

print("\n[4] Creating 5-fold stratified cross-validation...")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# --------------------------------------------------
# 5. Cross Validation
# --------------------------------------------------

print("\n[5] Running cross-validation...")

scoring = {
    "accuracy": "accuracy",
    "macro_f1": "f1_macro",
    "weighted_f1": "f1_weighted"
}

results = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1,
    return_train_score=False
)


# --------------------------------------------------
# 6. Display fold results
# --------------------------------------------------

print("\n" + "=" * 70)
print("FOLD RESULTS")
print("=" * 70)

for i in range(5):
    print(
        f"Fold {i + 1}: "
        f"Accuracy = {results['test_accuracy'][i]:.4f}, "
        f"Macro F1 = {results['test_macro_f1'][i]:.4f}, "
        f"Weighted F1 = {results['test_weighted_f1'][i]:.4f}"
    )


# --------------------------------------------------
# 7. Calculate mean and standard deviation
# --------------------------------------------------

accuracy_scores = results["test_accuracy"]
macro_f1_scores = results["test_macro_f1"]
weighted_f1_scores = results["test_weighted_f1"]


accuracy_mean = np.mean(accuracy_scores)
accuracy_std = np.std(accuracy_scores)

macro_f1_mean = np.mean(macro_f1_scores)
macro_f1_std = np.std(macro_f1_scores)

weighted_f1_mean = np.mean(weighted_f1_scores)
weighted_f1_std = np.std(weighted_f1_scores)


# --------------------------------------------------
# 8. Display final results
# --------------------------------------------------

print("\n" + "=" * 70)
print("CROSS-VALIDATION SUMMARY")
print("=" * 70)

print(
    f"Accuracy   : {accuracy_mean:.4f} ± {accuracy_std:.4f}"
)

print(
    f"Macro F1   : {macro_f1_mean:.4f} ± {macro_f1_std:.4f}"
)

print(
    f"Weighted F1: {weighted_f1_mean:.4f} ± {weighted_f1_std:.4f}"
)


# --------------------------------------------------
# 9. Save results
# --------------------------------------------------

reports_dir = Path("reports")
reports_dir.mkdir(exist_ok=True)

cv_results = pd.DataFrame({
    "fold": [1, 2, 3, 4, 5],
    "accuracy": accuracy_scores,
    "macro_f1": macro_f1_scores,
    "weighted_f1": weighted_f1_scores
})

cv_results.loc[len(cv_results)] = [
    "Mean",
    accuracy_mean,
    macro_f1_mean,
    weighted_f1_mean
]

cv_path = reports_dir / "cross_validation_results.csv"

cv_results.to_csv(
    cv_path,
    index=False
)

print("\nCross-validation results saved to:")
print(cv_path)


# --------------------------------------------------
# 10. Final
# --------------------------------------------------

print("\n" + "=" * 70)
print("CROSS-VALIDATION COMPLETED!")
print("=" * 70)
