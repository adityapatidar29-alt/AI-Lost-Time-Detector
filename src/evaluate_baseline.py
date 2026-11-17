import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)

print("=" * 65)
print("AI LOST TIME DETECTOR - BASELINE MODEL EVALUATION")
print("=" * 65)

# --------------------------------------------------
# 1. Load test data
# --------------------------------------------------

print("\n[1] Loading test data...")

data_dir = Path("data/processed")
reports_dir = Path("reports")

X_test = pd.read_csv(data_dir / "X_test_scaled.csv")
y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# --------------------------------------------------
# 2. Load saved predictions
# --------------------------------------------------

print("\n[2] Loading baseline model...")

import joblib

model = joblib.load("models/baseline_random_forest.pkl")

print("Baseline model loaded successfully.")


# --------------------------------------------------
# 3. Make predictions
# --------------------------------------------------

print("\n[3] Making predictions...")

y_pred = model.predict(X_test)

print("Predictions completed.")


# --------------------------------------------------
# 4. Accuracy
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 65)
print("BASELINE ACCURACY")
print("=" * 65)

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")


# --------------------------------------------------
# 5. Classification Report
# --------------------------------------------------

class_names = [
    "Highly Unproductive",
    "Distracted",
    "Normal",
    "Focused"
]

report = classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

report_path = reports_dir / "baseline_classification_report.csv"
report_df.to_csv(report_path)

print("\nClassification report saved to:")
print(report_path)


# --------------------------------------------------
# 6. Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 65)
print("CONFUSION MATRIX")
print("=" * 65)

print(cm)


# --------------------------------------------------
# 7. Plot Confusion Matrix
# --------------------------------------------------

print("\n[4] Creating confusion matrix visualization...")

fig, ax = plt.subplots(figsize=(9, 7))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    ax=ax,
    values_format="d"
)

plt.title("Baseline Random Forest - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(rotation=20)
plt.tight_layout()

cm_path = reports_dir / "baseline_confusion_matrix.png"

plt.savefig(cm_path, dpi=300)
plt.close()

print("Confusion matrix saved to:")
print(cm_path)


# --------------------------------------------------
# 8. Final message
# --------------------------------------------------

print("\n" + "=" * 65)
print("BASELINE EVALUATION COMPLETED!")
print("=" * 65)
