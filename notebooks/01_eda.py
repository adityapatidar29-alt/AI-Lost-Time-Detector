import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# AI LOST TIME DETECTOR
# Exploratory Data Analysis
# ==========================================

print("=" * 60)
print("AI LOST TIME DETECTOR - EDA")
print("=" * 60)

# Load dataset
df = pd.read_csv("data/raw/productivity_data.csv")

print("\nDataset loaded successfully!")

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

# ==========================================
# 1. DESCRIPTIVE STATISTICS
# ==========================================

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

print(df.describe())

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Dataset Shape ---")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# ==========================================
# 2. MISSING VALUE ANALYSIS
# ==========================================

print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

missing_values = df.isnull().sum()

print("\nMissing values per column:")
print(missing_values)

print(
    f"\nTotal missing values: {missing_values.sum()}"
)

# ==========================================
# 3. PRODUCTIVITY CLASS DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="productivity_label"
)

plt.title("Productivity Class Distribution")
plt.xlabel("Productivity Level")
plt.ylabel("Number of Records")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    "reports/productivity_distribution.png",
    dpi=300
)

plt.show()

# ==========================================
# 4. TYPING SPEED VS PRODUCTIVITY
# ==========================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="productivity_label",
    y="typing_speed"
)

plt.title("Typing Speed vs Productivity Level")
plt.xlabel("Productivity Level")
plt.ylabel("Typing Speed")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    "reports/typing_speed_vs_productivity.png",
    dpi=300
)

plt.show()

# ==========================================
# 5. IDLE TIME VS PRODUCTIVITY
# ==========================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="productivity_label",
    y="idle_time"
)

plt.title("Idle Time vs Productivity Level")
plt.xlabel("Productivity Level")
plt.ylabel("Idle Time (minutes)")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    "reports/idle_time_vs_productivity.png",
    dpi=300
)

plt.show()

# ==========================================
# 6. APP SWITCHING VS PRODUCTIVITY
# ==========================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="productivity_label",
    y="app_switch_count"
)

plt.title("Application Switching vs Productivity Level")
plt.xlabel("Productivity Level")
plt.ylabel("Application Switch Count")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    "reports/app_switching_vs_productivity.png",
    dpi=300
)

plt.show()

# ==========================================
# 7. CORRELATION HEATMAP
# ==========================================

# Numerical columns only
numeric_df = df.select_dtypes(include="number")

correlation_matrix = numeric_df.corr()

print("\n" + "=" * 60)
print("CORRELATION MATRIX")
print("=" * 60)

print(correlation_matrix.round(2))

plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Behavioral Features")

plt.tight_layout()

plt.savefig(
    "reports/correlation_heatmap.png",
    dpi=300
)

plt.show()

# ==========================================
# 8. TIME OF DAY VS PRODUCTIVITY
# ==========================================

# Productivity ka average activity score
hourly_productivity = (
    df.groupby("time_of_day")["activity_score"]
    .mean()
    .reset_index()
)

print("\n" + "=" * 60)
print("AVERAGE ACTIVITY SCORE BY HOUR")
print("=" * 60)

print(hourly_productivity)

# Plot
plt.figure(figsize=(12, 6))

sns.lineplot(
    data=hourly_productivity,
    x="time_of_day",
    y="activity_score",
    marker="o"
)

plt.title("Average Activity Score by Time of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Average Activity Score")

plt.xticks(range(0, 24))

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "reports/activity_score_by_time.png",
    dpi=300
)

plt.show()

