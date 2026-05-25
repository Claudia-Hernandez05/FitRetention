import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42


def generate_gym_dataset(n_samples=1000):
    np.random.seed(RANDOM_STATE)

    age = np.random.randint(18, 70, n_samples)
    membership_months = np.random.randint(1, 60, n_samples)
    weekly_visits = np.random.randint(0, 7, n_samples)
    days_since_last_visit = np.random.randint(0, 60, n_samples)
    monthly_fee = np.random.randint(20, 120, n_samples)
    satisfaction_score = np.random.randint(1, 11, n_samples)
    personal_trainer = np.random.choice(["Yes", "No"], n_samples)
    group_classes = np.random.choice(["Yes", "No"], n_samples)
    membership_type = np.random.choice(["Basic", "Standard", "Premium"], n_samples)

    churn_probability = (
        0.15
        + 0.25 * (weekly_visits <= 1)
        + 0.25 * (days_since_last_visit > 20)
        + 0.20 * (satisfaction_score <= 5)
        + 0.10 * (monthly_fee > 80)
        + 0.10 * (membership_months < 3)
        - 0.10 * (personal_trainer == "Yes")
        - 0.10 * (group_classes == "Yes")
    )

    churn_probability = np.clip(churn_probability, 0, 1)
    churn = np.random.binomial(1, churn_probability)

    return pd.DataFrame({
        "age": age,
        "membership_months": membership_months,
        "weekly_visits": weekly_visits,
        "days_since_last_visit": days_since_last_visit,
        "monthly_fee": monthly_fee,
        "satisfaction_score": satisfaction_score,
        "personal_trainer": personal_trainer,
        "group_classes": group_classes,
        "membership_type": membership_type,
        "churn": churn
    })


def train_model():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    data = generate_gym_dataset()
    data.to_csv("data/gym_churn.csv", index=False)

    X = data.drop("churn", axis=1)
    y = data["churn"]

    numeric_features = [
        "age",
        "membership_months",
        "weekly_visits",
        "days_since_last_visit",
        "monthly_fee",
        "satisfaction_score"
    ]

    categorical_features = [
        "personal_trainer",
        "group_classes",
        "membership_type"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 3),
        "precision": round(precision_score(y_test, y_pred), 3),
        "recall": round(recall_score(y_test, y_pred), 3),
        "f1_score": round(f1_score(y_test, y_pred), 3)
    }

    joblib.dump(pipeline, "models/churn_model.pkl")
    joblib.dump(metrics, "models/model_metrics.pkl")

    print("Model trained successfully.")
    print(metrics)


if __name__ == "__main__":
    train_model()
