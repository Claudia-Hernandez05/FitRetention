import joblib
import pandas as pd


MODEL_PATH = "models/churn_model.pkl"


def load_model():
    return joblib.load(MODEL_PATH)


def predict_churn(input_data):
    model = load_model()
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    if probability < 0.35:
        risk_level = "Low"
        recommendation = "Maintain current engagement and continue regular communication."
    elif probability < 0.65:
        risk_level = "Medium"
        recommendation = "Offer a personalized check-in, class recommendation or small discount."
    else:
        risk_level = "High"
        recommendation = "Contact the member directly and propose a retention plan."

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "risk_level": risk_level,
        "recommendation": recommendation
    }
