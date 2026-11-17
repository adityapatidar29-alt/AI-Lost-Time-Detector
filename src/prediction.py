import pandas as pd
import joblib
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "final_productivity_model.pkl"


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Label mapping
# --------------------------------------------------

LABEL_MAPPING = {
    0: "Highly Unproductive",
    1: "Distracted",
    2: "Normal",
    3: "Focused"
}


# --------------------------------------------------
# Feature engineering function
# --------------------------------------------------

def create_features(data):
    """
    Create engineered features from raw behavior data.
    """

    df = pd.DataFrame([data])

    # Avoid division by zero
    session = df["session_duration"].replace(0, 1)

    df["typing_efficiency"] = (
        df["typing_speed"] / session
    )

    df["mouse_efficiency"] = (
        df["mouse_activity"] / session
    )

    df["break_frequency"] = (
        df["break_count"] / session
    )

    df["distraction_index"] = (
        df["idle_ratio"] * df["switch_rate"]
    )

    df["engagement_score"] = (
        df["activity_score"]
        * (1 - df["idle_ratio"])
    )

    df["interaction_intensity"] = (
        df["typing_speed"]
        + df["mouse_activity"]
    ) / session

    df["switching_pressure"] = (
        df["app_switch_count"] / session
    )

    df["active_time_ratio"] = (
        1 - df["idle_ratio"]
    )

    return df


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_productivity(data):
    """
    Predict productivity state from behavior data.
    """

    df = create_features(data)

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

    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]

    confidence = probabilities[prediction]

    label = LABEL_MAPPING[prediction]

    return {
        "prediction": label,
        "confidence": float(confidence)
    }


# --------------------------------------------------
# Lost-time risk function
# --------------------------------------------------

def get_lost_time_risk(prediction, confidence):
    """
    Determine lost-time risk based on predicted
    productivity state and model confidence.
    """

    if prediction == "Highly Unproductive":

        return {
            "risk_level": "HIGH",
            "message": (
                "High lost-time risk detected. "
                "The current behavior indicates highly "
                "unproductive activity."
            )
        }

    elif prediction == "Distracted":

        return {
            "risk_level": "MEDIUM",
            "message": (
                "Moderate lost-time risk detected. "
                "The current behavior suggests distraction."
            )
        }

    elif prediction == "Normal":

        return {
            "risk_level": "LOW",
            "message": (
                "Low lost-time risk. "
                "Productivity appears normal."
            )
        }

    else:

        return {
            "risk_level": "MINIMAL",
            "message": (
                "Minimal lost-time risk. "
                "The current behavior indicates focused work."
            )
        }


# --------------------------------------------------
# Test prediction
# --------------------------------------------------

if __name__ == "__main__":

    sample_data = {
        "typing_speed": 35,
        "mouse_activity": 45,
        "idle_time": 5,
        "app_switch_count": 4,
        "session_duration": 60,
        "break_count": 2,
        "time_of_day": 14,
        "idle_ratio": 0.08,
        "switch_rate": 0.07,
        "activity_score": 80
    }

    result = predict_productivity(sample_data)

    risk = get_lost_time_risk(
        result["prediction"],
        result["confidence"]
    )

    print("=" * 60)
    print("AI LOST TIME DETECTOR - TEST PREDICTION")
    print("=" * 60)

    print("\nPrediction:")
    print(result["prediction"])

    print("\nConfidence:")
    print(f"{result['confidence'] * 100:.2f}%")

    print("\nLost-Time Risk:")
    print(risk["risk_level"])

    print("\nRecommendation:")
    print(risk["message"])

    print("\nPrediction pipeline working successfully!")