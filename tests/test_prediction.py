import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.train_model import train_model
from src.predict import predict_churn


def test_prediction_output():
    train_model()

    input_data = {
        "age": 30,
        "membership_months": 12,
        "weekly_visits": 3,
        "days_since_last_visit": 7,
        "monthly_fee": 50,
        "satisfaction_score": 8,
        "personal_trainer": "Yes",
        "group_classes": "Yes",
        "membership_type": "Premium"
    }

    result = predict_churn(input_data)

    assert "probability" in result
    assert "risk_level" in result
    assert "recommendation" in result
    assert 0 <= result["probability"] <= 1
