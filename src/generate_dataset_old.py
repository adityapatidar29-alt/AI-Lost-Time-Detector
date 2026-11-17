import numpy as np
import pandas as pd

# ==========================================
# AI LOST TIME DETECTOR
# Synthetic Dataset Generator
# ==========================================

# Reproducibility ke liye random seed
np.random.seed(42)

# Kitne records generate karne hain
N = 10000

# ------------------------------------------
# 1. Timestamp Generate Karna
# ------------------------------------------

timestamps = pd.date_range(
    start="2026-01-01 08:00:00",
    periods=N,
    freq="5min"
)

# ------------------------------------------
# 2. Time of Day
# ------------------------------------------

time_of_day = timestamps.hour

# ------------------------------------------
# 3. Basic Behavioral Features
# ------------------------------------------

# Typing speed: words/keystroke activity ka approximate measure
typing_speed = np.clip(
    np.random.normal(40, 15, N),
    0,
    80
)

# Mouse activity: movement/click activity ka approximate score
mouse_activity = np.clip(
    np.random.normal(55, 20, N),
    0,
    100
)

# Idle time in minutes
idle_time = np.clip(
    np.random.exponential(5, N),
    0,
    30
)

# Application switching count
app_switch_count = np.random.poisson(5, N)

# Session duration in minutes
session_duration = np.clip(
    np.random.normal(45, 20, N),
    5,
    120
)

# Break count
break_count = np.random.poisson(2, N)

# ------------------------------------------
# 4. Derived Features
# ------------------------------------------

idle_ratio = idle_time / session_duration

switch_rate = app_switch_count / session_duration

# Overall activity score
activity_score = (
    (typing_speed / 80) * 40
    + (mouse_activity / 100) * 30
    + (1 - idle_ratio) * 30
)

activity_score = np.clip(activity_score, 0, 100)

# ------------------------------------------
# 5. Productivity Label
# ------------------------------------------

productivity_label = []

for i in range(N):

    score = activity_score[i]
    idle = idle_ratio[i]
    switches = switch_rate[i]

    # Highly Unproductive
    if score < 30 or idle > 0.45 or switches > 0.35:
        label = "Highly Unproductive"

    # Distracted
    elif score < 50 or idle > 0.25 or switches > 0.20:
        label = "Distracted"

    # Focused
    elif score >= 70 and idle < 0.15 and switches < 0.15:
        label = "Focused"

    # Normal
    else:
        label = "Normal"

    productivity_label.append(label)

# ------------------------------------------
# 6. Create DataFrame
# ------------------------------------------

df = pd.DataFrame({
    "timestamp": timestamps,
    "typing_speed": np.round(typing_speed, 2),
    "mouse_activity": np.round(mouse_activity, 2),
    "idle_time": np.round(idle_time, 2),
    "app_switch_count": app_switch_count,
    "session_duration": np.round(session_duration, 2),
    "break_count": break_count,
    "time_of_day": time_of_day,
    "idle_ratio": np.round(idle_ratio, 4),
    "switch_rate": np.round(switch_rate, 4),
    "activity_score": np.round(activity_score, 2),
    "productivity_label": productivity_label
})

# ------------------------------------------
# 7. Save Dataset
# ------------------------------------------

output_path = "data/raw/productivity_data.csv"

df.to_csv(
    output_path,
    index=False
)

# ------------------------------------------
# 8. Display Results
# ------------------------------------------

print("=" * 60)
print("AI LOST TIME DETECTOR - DATASET GENERATOR")
print("=" * 60)

print(f"\nDataset created successfully!")
print(f"Total records: {len(df)}")
print(f"Total columns: {len(df.columns)}")

print("\nDataset location:")
print(output_path)

print("\nProductivity Distribution:")
print(df["productivity_label"].value_counts())

print("\nFirst 5 records:")
print(df.head())

print("\nDataset Information:")
print(df.info())