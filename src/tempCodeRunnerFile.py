import numpy as np
import pandas as pd

# ============================================================
# AI LOST TIME DETECTOR
# Improved Synthetic Dataset Generator
# ============================================================

# Reproducibility
np.random.seed(42)

# Number of records
N = 10000

print("=" * 65)
print("AI LOST TIME DETECTOR - IMPROVED DATASET GENERATOR")
print("=" * 65)

# ============================================================
# 1. Generate Timestamp
# ============================================================

timestamps = pd.date_range(
    start="2026-01-01 08:00:00",
    periods=N,
    freq="5min"
)

# Hour extracted from timestamp
time_of_day = timestamps.hour

# ============================================================
# 2. Generate Hidden Productivity Tendency
# ============================================================
# Ye directly final label nahi hai.
# Ye user ke underlying working behavior ko represent karta hai.

productivity_tendency = np.random.beta(
    a=4,
    b=2,
    size=N
)

# ============================================================
# 3. Time-of-Day Effect
# ============================================================

time_effect = np.zeros(N)

# Morning
time_effect[(time_of_day >= 8) & (time_of_day < 12)] = 0.10

# Afternoon
time_effect[(time_of_day >= 12) & (time_of_day < 16)] = -0.02

# Evening
time_effect[(time_of_day >= 16) & (time_of_day < 20)] = -0.08

# Night
time_effect[(time_of_day >= 20)] = -0.12

# ============================================================
# 4. Generate Behavioral Features
# ============================================================

# ------------------------------------------------------------
# Typing Speed
# ------------------------------------------------------------

typing_speed = (
    15
    + productivity_tendency * 45
    + time_effect * 15
    + np.random.normal(0, 8, N)
)

typing_speed = np.clip(
    typing_speed,
    0,
    80
)

# ------------------------------------------------------------
# Mouse Activity
# ------------------------------------------------------------

mouse_activity = (
    25
    + productivity_tendency * 60
    + np.random.normal(0, 15, N)
)

mouse_activity = np.clip(
    mouse_activity,
    0,
    100
)

# ------------------------------------------------------------
# Idle Time
# ------------------------------------------------------------

idle_time = (
    15
    - productivity_tendency * 12
    - time_effect * 5
    + np.random.normal(0, 4, N)
)

idle_time = np.clip(
    idle_time,
    0,
    30
)

# ------------------------------------------------------------
# Application Switching
# ------------------------------------------------------------

app_switch_count = (
    2
    + (1 - productivity_tendency) * 10
    + np.random.poisson(2, N)
)

app_switch_count = np.clip(
    app_switch_count.round(),
    0,
    25
).astype(int)

# ------------------------------------------------------------
# Session Duration
# ------------------------------------------------------------

session_duration = (
    20
    + productivity_tendency * 55
    + np.random.normal(0, 15, N)
)

session_duration = np.clip(
    session_duration,
    5,
    120
)

# ------------------------------------------------------------
# Break Count
# ------------------------------------------------------------

break_count = (
    1
    + (1 - productivity_tendency) * 3
    + np.random.poisson(1, N)
)

break_count = np.clip(
    break_count.round(),
    0,
    10
).astype(int)

# ============================================================
# 5. Derived Features
# ============================================================

# Idle ratio
idle_ratio = idle_time / session_duration

# Application switching rate
switch_rate = app_switch_count / session_duration

# ============================================================
# 6. Activity Score
# ============================================================

activity_score = (
    (typing_speed / 80) * 40
    + (mouse_activity / 100) * 30
    + (1 - idle_ratio) * 30
)

activity_score = np.clip(
    activity_score,
    0,
    100
)

# ============================================================
# 7. Create Productivity Score for Label Generation
# ============================================================
# Important:
# Label is NOT generated from activity_score alone.
# Multiple behavioral factors are considered.

behavior_score = (
    productivity_tendency * 60
    + (typing_speed / 80) * 10
    + (mouse_activity / 100) * 10
    + (1 - idle_ratio) * 10
    + (1 - np.minimum(switch_rate / 0.4, 1)) * 10
)

# Add natural randomness
behavior_score += np.random.normal(
    0,
    8,
    N
)

behavior_score = np.clip(
    behavior_score,
    0,
    100
)

# ============================================================
# 8. Productivity Labels
# ============================================================

productivity_label = []

for score in behavior_score:

    if score >= 75:
        label = "Focused"

    elif score >= 55:
        label = "Normal"

    elif score >= 35:
        label = "Distracted"

    else:
        label = "Highly Unproductive"

    productivity_label.append(label)

# ============================================================
# 9. Create DataFrame
# ============================================================

df = pd.DataFrame({

    "timestamp": timestamps,

    "typing_speed": np.round(
        typing_speed,
        2
    ),

    "mouse_activity": np.round(
        mouse_activity,
        2
    ),

    "idle_time": np.round(
        idle_time,
        2
    ),

    "app_switch_count": app_switch_count,

    "session_duration": np.round(
        session_duration,
        2
    ),

    "break_count": break_count,

    "time_of_day": time_of_day,

    "idle_ratio": np.round(
        idle_ratio,
        4
    ),

    "switch_rate": np.round(
        switch_rate,
        4
    ),

    "activity_score": np.round(
        activity_score,
        2
    ),

    "productivity_label": productivity_label
})

# ============================================================
# 10. Save Dataset
# ============================================================

output_path = "data/raw/productivity_data.csv"

df.to_csv(
    output_path,
    index=False
)

# ============================================================
# 11. Display Dataset Information
# ============================================================

print("\nDataset created successfully!")

print(f"Total records : {len(df)}")
print(f"Total columns : {len(df.columns)}")

print("\nDataset location:")
print(output_path)

print("\nProductivity Distribution:")
print(
    df["productivity_label"]
    .value_counts()
)

print("\nProductivity Percentage:")
print(
    df["productivity_label"]
    .value_counts(
        normalize=True
    )
    .mul(100)
    .round(2)
)

print("\nFirst 5 records:")
print(
    df.head()
)

print("\nDataset Information:")
df.info()

print("\n" + "=" * 65)
print("IMPROVED DATASET GENERATION COMPLETED!")
print("=" * 65)