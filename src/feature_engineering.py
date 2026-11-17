import pandas as pd

from pathlib import Path


# ============================================================
# AI LOST TIME DETECTOR
# FEATURE ENGINEERING
# ============================================================

print("=" * 65)
print("AI LOST TIME DETECTOR - FEATURE ENGINEERING")
print("=" * 65)


# ============================================================
# 1. DEFINE PATHS
# ============================================================

input_path = Path(
    "data/raw/productivity_data.csv"
)

output_path = Path(
    "data/processed/engineered_features.csv"
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n[1] Loading dataset...")

df = pd.read_csv(
    input_path
)

print(
    f"Dataset loaded: {df.shape[0]} records"
)


# ============================================================
# 3. CREATE COPY
# ============================================================

engineered_df = df.copy()


# ============================================================
# 4. FEATURE: TYPING EFFICIENCY
# ============================================================

print("\n[2] Creating typing efficiency...")

engineered_df["typing_efficiency"] = (

    engineered_df["typing_speed"]
    /
    engineered_df["session_duration"].clip(lower=1)

)


# ============================================================
# 5. FEATURE: MOUSE EFFICIENCY
# ============================================================

print("\n[3] Creating mouse efficiency...")

engineered_df["mouse_efficiency"] = (

    engineered_df["mouse_activity"]
    /
    engineered_df["session_duration"].clip(lower=1)

)


# ============================================================
# 6. FEATURE: BREAK FREQUENCY
# ============================================================

print("\n[4] Creating break frequency...")

engineered_df["break_frequency"] = (

    engineered_df["break_count"]
    /
    engineered_df["session_duration"].clip(lower=1)

)


# ============================================================
# 7. FEATURE: DISTRACTION INDEX
# ============================================================

print("\n[5] Creating distraction index...")

engineered_df["distraction_index"] = (

    engineered_df["idle_ratio"] * 0.6
    +
    engineered_df["switch_rate"] * 0.4

)


# ============================================================
# 8. FEATURE: ENGAGEMENT SCORE
# ============================================================

print("\n[6] Creating engagement score...")

normalized_typing = (
    engineered_df["typing_speed"] / 80
)

normalized_mouse = (
    engineered_df["mouse_activity"] / 100
)

normalized_idle = (
    1 - engineered_df["idle_ratio"].clip(0, 1)
)


engineered_df["engagement_score"] = (

    normalized_typing * 0.4
    +
    normalized_mouse * 0.3
    +
    normalized_idle * 0.3

) * 100


# ============================================================
# 9. FEATURE: INTERACTION INTENSITY
# ============================================================

print("\n[7] Creating interaction intensity...")

engineered_df["interaction_intensity"] = (

    (
        engineered_df["typing_speed"]
        +
        engineered_df["mouse_activity"]
    )
    /
    engineered_df["session_duration"].clip(lower=1)

)


# ============================================================
# 10. FEATURE: SWITCHING PRESSURE
# ============================================================

print("\n[8] Creating switching pressure...")

engineered_df["switching_pressure"] = (

    engineered_df["app_switch_count"]
    /
    (
        engineered_df["session_duration"].clip(lower=1)
    )

)


# ============================================================
# 11. FEATURE: ACTIVE_TIME_RATIO
# ============================================================

print("\n[9] Creating active time ratio...")

engineered_df["active_time_ratio"] = (

    1
    -
    engineered_df["idle_ratio"].clip(0, 1)

)


# ============================================================
# 12. ROUND NEW FEATURES
# ============================================================

new_features = [

    "typing_efficiency",
    "mouse_efficiency",
    "break_frequency",
    "distraction_index",
    "engagement_score",
    "interaction_intensity",
    "switching_pressure",
    "active_time_ratio"

]


for feature in new_features:

    engineered_df[feature] = (
        engineered_df[feature]
        .round(4)
    )


# ============================================================
# 13. CHECK NEW FEATURES
# ============================================================

print("\n" + "=" * 65)

print("ENGINEERED FEATURES")

print("=" * 65)


for feature in new_features:

    print(
        f"{feature:25s} "
        f"min={engineered_df[feature].min():.4f} "
        f"max={engineered_df[feature].max():.4f}"
    )


# ============================================================
# 14. CHECK MISSING VALUES
# ============================================================

print("\n[10] Checking missing values...")

missing_values = (
    engineered_df[new_features]
    .isnull()
    .sum()
    .sum()
)


print(
    f"Missing values in new features: "
    f"{missing_values}"
)


if missing_values > 0:

    raise ValueError(
        "Missing values found in engineered features."
    )


# ============================================================
# 15. SAVE ENGINEERED DATASET
# ============================================================

print("\n[11] Saving engineered dataset...")

engineered_df.to_csv(

    output_path,

    index=False

)


# ============================================================
# 16. FINAL INFORMATION
# ============================================================

print("\nOriginal columns:")

print(
    len(df.columns)
)


print("\nNew columns:")

print(
    len(engineered_df.columns)
)


print("\nNew features added:")

for feature in new_features:

    print(
        f"- {feature}"
    )


print("\nOutput file:")

print(
    output_path
)


# ============================================================
# 17. FINAL STATUS
# ============================================================

print("\n" + "=" * 65)

print(
    "FEATURE ENGINEERING COMPLETED SUCCESSFULLY!"
)

print("=" * 65)