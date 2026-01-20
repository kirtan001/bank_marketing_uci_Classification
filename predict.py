import pandas as pd
import joblib


# -----------------------------
# Paths to saved artifacts
# -----------------------------
MODEL_PATH = "artifacts/best_model.pkl"
FEATURES_PATH = "artifacts/feature_columns.pkl"


def load_artifacts():
    """
    Load trained model and feature column list.
    """
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return model, feature_columns


def preprocess_input(input_df, feature_columns):
    """
    Apply one-hot encoding and align input features
    with the training feature space.
    """
    # One-hot encode categorical features
    encoded_df = pd.get_dummies(input_df, drop_first=True)

    # Add missing columns seen during training
    for col in feature_columns:
        if col not in encoded_df.columns:
            encoded_df[col] = 0

    # Remove extra columns and reorder
    encoded_df = encoded_df[feature_columns]

    return encoded_df


def predict(input_data):
    """
    Generate prediction and probability for a single customer.
    """
    model, feature_columns = load_artifacts()

    # Convert input dict to DataFrame
    input_df = pd.DataFrame([input_data])

    # Preprocess
    processed_input = preprocess_input(input_df, feature_columns)

    # Predict
    probability = model.predict_proba(processed_input)[0][1]
    prediction = 1 if probability >= 0.5 else 0

    return {
        "prediction": "yes" if prediction == 1 else "no",
        "probability": round(probability, 4)
    }


# --------------------------------
# Manual test (CLI execution)
# --------------------------------
if __name__ == "__main__":

    sample_input = {
        "age": 35,
        "marital": "married",
        "education": "secondary",
        "default": "no",
        "balance": 1200,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": 5,
        "month": "may",
        "campaign": 2,
        "pdays": -1,
        "previous": 0,
        "poutcome": "no_previous_campaign"
    }

    result = predict(sample_input)
    print("Prediction Result:", result)
