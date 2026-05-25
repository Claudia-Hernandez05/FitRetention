import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42


def generate_gym_dataset(n_samples=1500):
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
        0.12
        + 0.28 * (weekly_visits <= 1)
        + 0.28 * (days_since_last_visit > 20)
        + 0.22 * (satisfaction_score <= 5)
        + 0.10 * (monthly_fee > 80)
        + 0.10 * (membership_months < 3)
        - 0.10 * (personal_trainer == "Yes")
        - 0.10 * (group_classes == "Yes")
        - 0.05 * (membership_type == "Premium")
    )

    churn_probability = np.clip(churn_probability, 0, 1)
    churn = np.random.binomial(1, churn_probability)

    return pd.DataFrame(
        {
            "age": age,
            "membership_months": membership_months,
            "weekly_visits": weekly_visits,
            "days_since_last_visit": days_since_last_visit,
            "monthly_fee": monthly_fee,
            "satisfaction_score": satisfaction_score,
            "personal_trainer": personal_trainer,
            "group_classes": group_classes,
            "membership_type": membership_type,
            "churn": churn,
        }
    )


def build_preprocessor():
    numeric_features = [
        "age",
        "membership_months",
        "weekly_visits",
        "days_since_last_visit",
        "monthly_fee",
        "satisfaction_score",
    ]

    categorical_features = [
        "personal_trainer",
        "group_classes",
        "membership_type",
    ]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 3),
        "precision": round(precision_score(y_test, y_pred), 3),
        "recall": round(recall_score(y_test, y_pred), 3),
        "f1_score": round(f1_score(y_test, y_pred), 3),
    }


def save_feature_importance(best_model, feature_names):
    model_step = best_model.named_steps["model"]

    if not hasattr(model_step, "feature_importances_"):
        return

    importances = model_step.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values(by="importance", ascending=False)

    importance_df.to_csv("models/feature_importance.csv", index=False)


def get_feature_names(preprocessor):
    numeric_features = [
        "age",
        "membership_months",
        "weekly_visits",
        "days_since_last_visit",
        "monthly_fee",
        "satisfaction_score",
    ]

    categorical_features = list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(
            ["personal_trainer", "group_classes", "membership_type"]
        )
    )

    return numeric_features + categorical_features


def train_model():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    data = generate_gym_dataset()
    data.to_csv("data/gym_churn.csv", index=False)

    X = data.drop("churn", axis=1)
    y = data["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    candidate_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
    }

    results = []

    best_model = None
    best_model_name = None
    best_f1_score = -1

    for model_name, classifier in candidate_models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", classifier),
            ]
        )

        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)

        result_row = {
            "model_name": model_name,
            **metrics,
        }

        results.append(result_row)

        if metrics["f1_score"] > best_f1_score:
            best_f1_score = metrics["f1_score"]
            best_model = pipeline
            best_model_name = model_name

    results_df = pd.DataFrame(results).sort_values(by="f1_score", ascending=False)
    results_df.to_csv("models/model_comparison.csv", index=False)

    best_metrics = results_df.iloc[0].to_dict()

    joblib.dump(best_model, "models/churn_model.pkl")
    joblib.dump(best_metrics, "models/model_metrics.pkl")
    joblib.dump(best_model_name, "models/best_model_name.pkl")

    preprocessor = best_model.named_steps["preprocessor"]
    feature_names = get_feature_names(preprocessor)
    save_feature_importance(best_model, feature_names)

    print("Model training completed successfully.")
    print(f"Best model: {best_model_name}")
    print(best_metrics)


if __name__ == "__main__":
    train_model()