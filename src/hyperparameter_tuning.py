import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier


print("=" * 75)
print("AI LOST TIME DETECTOR - HYPERPARAMETER TUNING")
print("=" * 75)


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

X = df[feature_columns]

y = df["productivity_label"].map(label_mapping)

if y.isna().any():
    raise ValueError("Unknown productivity label found.")

print("Features:", X.shape)
print("Target  :", y.shape)


# --------------------------------------------------
# 3. Create base Random Forest
# --------------------------------------------------

print("\n[3] Creating Random Forest...")

rf = RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# 4. Define parameter grid
# --------------------------------------------------

print("\n[4] Defining hyperparameter grid...")

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

total_combinations = (
    len(param_grid["n_estimators"])
    * len(param_grid["max_depth"])
    * len(param_grid["min_samples_split"])
    * len(param_grid["min_samples_leaf"])
)

print("Parameter combinations:", total_combinations)
print("Cross-validation folds: 5")


# --------------------------------------------------
# 5. Create Stratified K-Fold
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# --------------------------------------------------
# 6. Grid Search
# --------------------------------------------------

print("\n[5] Starting GridSearchCV...")
print("Primary scoring metric: Macro F1")
print("Please wait...")

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    return_train_score=False
)

grid_search.fit(X, y)


# --------------------------------------------------
# 7. Best parameters
# --------------------------------------------------

print("\n" + "=" * 75)
print("BEST HYPERPARAMETERS")
print("=" * 75)

print("Best parameters:")

for parameter, value in grid_search.best_params_.items():
    print(f"{parameter}: {value}")


# --------------------------------------------------
# 8. Best score
# --------------------------------------------------

best_score = grid_search.best_score_

print("\nBest Cross-Validation Macro F1:")
print(f"{best_score:.4f}")

print(
    f"Best Cross-Validation Macro F1: "
    f"{best_score * 100:.2f}%"
)


# --------------------------------------------------
# 9. Top configurations
# --------------------------------------------------

print("\n" + "=" * 75)
print("TOP 10 PARAMETER CONFIGURATIONS")
print("=" * 75)

results = pd.DataFrame(grid_search.cv_results_)

top_results = results[
    [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "param_n_estimators",
        "param_max_depth",
        "param_min_samples_split",
        "param_min_samples_leaf"
    ]
].sort_values("rank_test_score").head(10)

print(top_results.to_string(index=False))


# --------------------------------------------------
# 10. Save tuning results
# --------------------------------------------------

reports_dir = Path("reports")
reports_dir.mkdir(exist_ok=True)

results_path = reports_dir / "hyperparameter_tuning_results.csv"

results.to_csv(
    results_path,
    index=False
)

print("\nAll tuning results saved to:")
print(results_path)


# --------------------------------------------------
# 11. Save best parameters
# --------------------------------------------------

best_params_path = reports_dir / "best_hyperparameters.txt"

with open(best_params_path, "w") as file:

    file.write("AI LOST TIME DETECTOR\n")
    file.write("BEST RANDOM FOREST HYPERPARAMETERS\n")
    file.write("=" * 50 + "\n\n")

    for parameter, value in grid_search.best_params_.items():
        file.write(f"{parameter}: {value}\n")

    file.write("\n")
    file.write(
        f"Best Cross-Validation Macro F1: "
        f"{best_score:.4f}\n"
    )

    file.write(
        f"Best Cross-Validation Macro F1: "
        f"{best_score * 100:.2f}%\n"
    )


print("\nBest parameters saved to:")
print(best_params_path)


# --------------------------------------------------
# 12. Final
# --------------------------------------------------

print("\n" + "=" * 75)
print("HYPERPARAMETER TUNING COMPLETED!")
print("=" * 75)

