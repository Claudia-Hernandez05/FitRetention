import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.predict import predict_churn


st.set_page_config(
    page_title="FitRetention",
    page_icon="🏋️",
    layout="wide"
)

st.title("🏋️ FitRetention")
st.subheader("Gym Member Churn Prediction using Machine Learning and MLOps")

st.markdown(
    """
    FitRetention is a Machine Learning application that predicts whether a gym member
    is likely to cancel their membership. The project demonstrates an end-to-end MLOps
    workflow including model training, model persistence, Streamlit frontend, Docker,
    CI/CD and cloud infrastructure preparation.
    """
)

st.divider()

tab1, tab2, tab3 = st.tabs(["Prediction", "Model Performance", "MLOps Pipeline"])

with tab1:
    st.header("Predict Gym Member Churn")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 70, 30)
        membership_months = st.slider("Membership duration in months", 1, 60, 12)
        weekly_visits = st.slider("Weekly visits", 0, 7, 3)

    with col2:
        days_since_last_visit = st.slider("Days since last visit", 0, 60, 7)
        monthly_fee = st.slider("Monthly fee", 20, 120, 50)
        satisfaction_score = st.slider("Satisfaction score", 1, 10, 7)

    with col3:
        personal_trainer = st.selectbox("Personal trainer", ["Yes", "No"])
        group_classes = st.selectbox("Group classes", ["Yes", "No"])
        membership_type = st.selectbox("Membership type", ["Basic", "Standard", "Premium"])

    input_data = {
        "age": age,
        "membership_months": membership_months,
        "weekly_visits": weekly_visits,
        "days_since_last_visit": days_since_last_visit,
        "monthly_fee": monthly_fee,
        "satisfaction_score": satisfaction_score,
        "personal_trainer": personal_trainer,
        "group_classes": group_classes,
        "membership_type": membership_type
    }

    if st.button("Predict churn risk"):
        result = predict_churn(input_data)
        probability = result["probability"]

        st.subheader("Prediction Result")

        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Churn probability", f"{probability:.1%}")
        metric_col2.metric("Risk level", result["risk_level"])

        if result["risk_level"] == "Low":
            st.success("Low churn risk")
        elif result["risk_level"] == "Medium":
            st.warning("Medium churn risk")
        else:
            st.error("High churn risk")

        st.info(f"Recommended action: {result['recommendation']}")

        st.write("Input data used for prediction:")
        st.dataframe(pd.DataFrame([input_data]))

with tab2:
    st.header("Model Performance")

    metrics_path = ROOT_DIR / "models" / "model_metrics.pkl"

    if metrics_path.exists():
        metrics = joblib.load(metrics_path)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", metrics["accuracy"])
        col2.metric("Precision", metrics["precision"])
        col3.metric("Recall", metrics["recall"])
        col4.metric("F1-score", metrics["f1_score"])

        st.markdown(
            """
            The model is a Random Forest classifier trained on simulated gym membership data.
            The dataset includes behavioral, subscription and satisfaction variables.
            """
        )
    else:
        st.warning("Model metrics file not found. Please run `python src/train_model.py` first.")

with tab3:
    st.header("MLOps Pipeline")

    st.markdown(
        """
        This project follows a simple MLOps workflow:

        1. Data generation and preprocessing.
        2. Model training using Scikit-learn.
        3. Model persistence using Joblib.
        4. Streamlit web application for user interaction.
        5. Docker containerization.
        6. Automated testing with Pytest.
        7. CI/CD pipeline using GitHub Actions.
        8. Cloud infrastructure preparation using Terraform.
        """
    )

    st.success("The application is ready to be containerized and deployed.")