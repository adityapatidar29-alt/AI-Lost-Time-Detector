import streamlit as st
import sys
from pathlib import Path


# --------------------------------------------------
# Project path
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(BASE_DIR))


# --------------------------------------------------
# Import prediction functions
# --------------------------------------------------

from src.prediction import (
    predict_productivity,
    get_lost_time_risk
)


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Lost Time Detector",
    page_icon="🧠",
    layout="wide"
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🧠 AI Lost Time Detector")

st.markdown(
    """
    **AI-powered productivity analysis system**

    Analyze behavioral activity patterns to identify
    productivity levels and detect potential lost-time periods.
    """
)

st.divider()


# ==================================================
# INPUT SECTION
# ==================================================

st.header("📊 Behavioral Activity Input")

st.info(
    "Enter the user's current activity information "
    "and click **Predict Productivity**."
)


col1, col2 = st.columns(2)


# --------------------------------------------------
# Column 1
# --------------------------------------------------

with col1:

    st.subheader("⌨️ Activity")

    typing_speed = st.number_input(
        "Typing Speed",
        min_value=0.0,
        max_value=150.0,
        value=35.0,
        step=1.0
    )

    mouse_activity = st.number_input(
        "Mouse Activity",
        min_value=0.0,
        max_value=200.0,
        value=45.0,
        step=1.0
    )

    idle_time = st.number_input(
        "Idle Time (minutes)",
        min_value=0.0,
        max_value=120.0,
        value=5.0,
        step=1.0
    )

    app_switch_count = st.number_input(
        "App Switch Count",
        min_value=0,
        max_value=100,
        value=4,
        step=1
    )

    session_duration = st.number_input(
        "Session Duration (minutes)",
        min_value=1.0,
        max_value=480.0,
        value=60.0,
        step=1.0
    )


# --------------------------------------------------
# Column 2
# --------------------------------------------------

with col2:

    st.subheader("⏱️ Session Information")

    break_count = st.number_input(
        "Break Count",
        min_value=0,
        max_value=30,
        value=2,
        step=1
    )

    time_of_day = st.slider(
        "Time of Day",
        min_value=0,
        max_value=23,
        value=14
    )

    idle_ratio = st.slider(
        "Idle Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.08,
        step=0.01
    )

    switch_rate = st.number_input(
        "Switch Rate",
        min_value=0.0,
        max_value=10.0,
        value=0.07,
        step=0.01
    )

    activity_score = st.slider(
        "Activity Score",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=1.0
    )


# ==================================================
# PREDICTION BUTTON
# ==================================================

st.divider()

predict_button = st.button(
    "🔍 Predict Productivity",
    type="primary",
    use_container_width=True
)


# ==================================================
# PREDICTION RESULT
# ==================================================

if predict_button:

    input_data = {
        "typing_speed": typing_speed,
        "mouse_activity": mouse_activity,
        "idle_time": idle_time,
        "app_switch_count": app_switch_count,
        "session_duration": session_duration,
        "break_count": break_count,
        "time_of_day": time_of_day,
        "idle_ratio": idle_ratio,
        "switch_rate": switch_rate,
        "activity_score": activity_score
    }

    result = predict_productivity(input_data)

    risk = get_lost_time_risk(
        result["prediction"],
        result["confidence"]
    )


    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    st.divider()

    st.header("🎯 Prediction Result")


    result_col1, result_col2, result_col3 = st.columns(3)


    with result_col1:

        st.metric(
            "Productivity State",
            result["prediction"]
        )


    with result_col2:

        st.metric(
            "Confidence",
            f"{result['confidence'] * 100:.2f}%"
        )


    with result_col3:

        st.metric(
            "Lost-Time Risk",
            risk["risk_level"]
        )


    # --------------------------------------------------
    # Confidence
    # --------------------------------------------------

    st.subheader("📈 Model Confidence")

    st.progress(
        result["confidence"]
    )

    st.caption(
        f"The model is {result['confidence'] * 100:.2f}% "
        "confident in this prediction."
    )


    # --------------------------------------------------
    # Risk
    # --------------------------------------------------

    st.subheader("🚨 Lost-Time Risk Analysis")

    if risk["risk_level"] == "HIGH":

        st.error(risk["message"])

    elif risk["risk_level"] == "MEDIUM":

        st.warning(risk["message"])

    elif risk["risk_level"] == "LOW":

        st.info(risk["message"])

    else:

        st.success(risk["message"])


    # --------------------------------------------------
    # Recommendation
    # --------------------------------------------------

    st.subheader("💡 Recommendation")

    recommendations = {
        "Highly Unproductive":
            "Consider taking a short break and reducing "
            "distractions before continuing work.",

        "Distracted":
            "Try closing unnecessary applications and "
            "focus on one task at a time.",

        "Normal":
            "Productivity is currently normal. Continue "
            "maintaining the current work pattern.",

        "Focused":
            "Good focus detected. Continue the current "
            "working pattern."
    }

    st.write(
        recommendations[result["prediction"]]
    )


    # --------------------------------------------------
    # Input Summary
    # --------------------------------------------------

    st.subheader("📋 Activity Summary")

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:

        st.write(
            f"**Typing Speed:** {typing_speed}"
        )

        st.write(
            f"**Mouse Activity:** {mouse_activity}"
        )

        st.write(
            f"**Idle Time:** {idle_time} minutes"
        )

        st.write(
            f"**App Switches:** {app_switch_count}"
        )

        st.write(
            f"**Session Duration:** {session_duration} minutes"
        )

    with summary_col2:

        st.write(
            f"**Break Count:** {break_count}"
        )

        st.write(
            f"**Time of Day:** {time_of_day}:00"
        )

        st.write(
            f"**Idle Ratio:** {idle_ratio:.2f}"
        )

        st.write(
            f"**Switch Rate:** {switch_rate:.2f}"
        )

        st.write(
            f"**Activity Score:** {activity_score:.0f}/100"
        )


# ==================================================
# MODEL INFORMATION
# ==================================================

st.divider()

st.header("🤖 Model Information")

model_col1, model_col2, model_col3 = st.columns(3)

with model_col1:
    st.metric(
        "Algorithm",
        "Random Forest"
    )

with model_col2:
    st.metric(
        "Features",
        "18"
    )

with model_col3:
    st.metric(
        "Test Accuracy",
        "66.50%"
    )


st.caption(
    "AI Lost Time Detector | Final Tuned Random Forest Model"
)






"""

import streamlit as st
import sys
from pathlib import Path


# --------------------------------------------------
# Project path
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(BASE_DIR))


# --------------------------------------------------
# Import prediction functions
# --------------------------------------------------

from src.prediction import (
    predict_productivity,
    get_lost_time_risk
)


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Lost Time Detector",
    page_icon="🧠",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🧠 AI Lost Time Detector")

st.markdown(
    '''
    ### AI-Based Productivity Analysis System

    This application analyzes behavioral activity patterns
    and predicts the current productivity state.
    '''
)

st.divider()


# --------------------------------------------------
# Input section
# --------------------------------------------------

st.subheader("📊 Enter Behavioral Data")

col1, col2 = st.columns(2)


# --------------------------------------------------
# Column 1
# --------------------------------------------------

with col1:

    typing_speed = st.number_input(
        "Typing Speed",
        min_value=0.0,
        max_value=150.0,
        value=35.0,
        step=1.0
    )

    mouse_activity = st.number_input(
        "Mouse Activity",
        min_value=0.0,
        max_value=200.0,
        value=45.0,
        step=1.0
    )

    idle_time = st.number_input(
        "Idle Time (minutes)",
        min_value=0.0,
        max_value=120.0,
        value=5.0,
        step=1.0
    )

    app_switch_count = st.number_input(
        "App Switch Count",
        min_value=0,
        max_value=100,
        value=4,
        step=1
    )

    session_duration = st.number_input(
        "Session Duration (minutes)",
        min_value=1.0,
        max_value=480.0,
        value=60.0,
        step=1.0
    )


# --------------------------------------------------
# Column 2
# --------------------------------------------------

with col2:

    break_count = st.number_input(
        "Break Count",
        min_value=0,
        max_value=30,
        value=2,
        step=1
    )

    time_of_day = st.slider(
        "Time of Day",
        min_value=0,
        max_value=23,
        value=14
    )

    idle_ratio = st.slider(
        "Idle Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.08,
        step=0.01
    )

    switch_rate = st.number_input(
        "Switch Rate",
        min_value=0.0,
        max_value=10.0,
        value=0.07,
        step=0.01
    )

    activity_score = st.slider(
        "Activity Score",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=1.0
    )


st.divider()


# --------------------------------------------------
# Prediction button
# --------------------------------------------------

predict_button = st.button(
    "🔍 Predict Productivity",
    type="primary",
    use_container_width=True
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if predict_button:

    input_data = {
        "typing_speed": typing_speed,
        "mouse_activity": mouse_activity,
        "idle_time": idle_time,
        "app_switch_count": app_switch_count,
        "session_duration": session_duration,
        "break_count": break_count,
        "time_of_day": time_of_day,
        "idle_ratio": idle_ratio,
        "switch_rate": switch_rate,
        "activity_score": activity_score
    }

    result = predict_productivity(input_data)

    risk = get_lost_time_risk(
        result["prediction"],
        result["confidence"]
    )


    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    st.divider()

    st.subheader("🎯 Prediction Result")


    result_col1, result_col2, result_col3 = st.columns(3)


    with result_col1:

        st.metric(
            "Productivity State",
            result["prediction"]
        )


    with result_col2:

        st.metric(
            "Confidence",
            f"{result['confidence'] * 100:.2f}%"
        )


    with result_col3:

        st.metric(
            "Lost-Time Risk",
            risk["risk_level"]
        )


    # --------------------------------------------------
    # Recommendation
    # --------------------------------------------------

    st.subheader("💡 Recommendation")

    if risk["risk_level"] == "HIGH":

        st.error(risk["message"])

    elif risk["risk_level"] == "MEDIUM":

        st.warning(risk["message"])

    elif risk["risk_level"] == "LOW":

        st.info(risk["message"])

    else:

        st.success(risk["message"])


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "AI Lost Time Detector | Machine Learning Project"
)


"""