import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.predict import predict_churn


DATA_PATH = ROOT_DIR / "data" / "gym_churn.csv"
METRICS_PATH = ROOT_DIR / "models" / "model_metrics.pkl"
LOG_PATH = ROOT_DIR / "logs" / "predictions_log.csv"
MODEL_COMPARISON_PATH = ROOT_DIR / "models" / "model_comparison.csv"
BEST_MODEL_NAME_PATH = ROOT_DIR / "models" / "best_model_name.pkl"


st.set_page_config(
    page_title="FitRetention",
    page_icon="🏋️",
    layout="wide",
)


st.sidebar.title("🏋️ FitRetention")
st.sidebar.caption("Gym churn prediction with MLOps")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Prediction",
        "Data Insights",
        "Model Performance",
        "Prediction Monitoring",
        "MLOps Pipeline",
    ],
)


def load_dataset():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


def show_project_badges():
    col1, col2, col3, col4 = st.columns(4)
    col1.success("ML Model")
    col2.success("Streamlit App")
    col3.success("Docker")
    col4.success("CI/CD")


if page == "Home":
    st.title("🏋️ FitRetention")
    st.subheader("Professional Gym Member Churn Prediction Platform")

    st.markdown(
        """
        **FitRetention** is a Machine Learning and MLOps web application designed to help gyms
        identify members who are likely to cancel their membership.

        The application predicts churn risk based on behavioral, subscription and satisfaction
        variables, then provides a practical retention recommendation.
        """
    )

    show_project_badges()

    st.divider()

    st.header("Business Problem")

    st.markdown(
        """
        Gyms often lose members because of low attendance, low satisfaction, lack of engagement
        or price sensitivity. Detecting high-risk members early allows the business to act before
        cancellation happens.

        This application supports retention actions such as:

        - contacting inactive members,
        - offering personalized training plans,
        - recommending group classes,
        - giving targeted discounts,
        - improving customer engagement.
        """
    )

    st.header("Project Value")

    col1, col2, col3 = st.columns(3)
    col1.metric("Use Case", "Churn Prediction")
    col2.metric("Interface", "Web App")
    col3.metric("Workflow", "MLOps")

    st.info(
        "The project demonstrates the full lifecycle from model training to a dockerized web "
        "application with testing, CI/CD and infrastructure-as-code preparation."
    )


elif page == "Prediction":
    st.title("Predict Gym Member Churn")

    st.markdown(
        """
        Enter the details of a gym member. The model will estimate the probability that this member
        cancels their subscription.
        """
    )

    with st.form("prediction_form"):
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
            membership_type = st.selectbox(
                "Membership type",
                ["Basic", "Standard", "Premium"],
            )

        submitted = st.form_submit_button("Predict churn risk")

    input_data = {
        "age": age,
        "membership_months": membership_months,
        "weekly_visits": weekly_visits,
        "days_since_last_visit": days_since_last_visit,
        "monthly_fee": monthly_fee,
        "satisfaction_score": satisfaction_score,
        "personal_trainer": personal_trainer,
        "group_classes": group_classes,
        "membership_type": membership_type,
    }

    if submitted:
        result = predict_churn(input_data)
        probability = result["probability"]

        st.divider()
        st.subheader("Prediction Result")

        col1, col2, col3 = st.columns(3)
        col1.metric("Churn probability", f"{probability:.1%}")
        col2.metric("Risk level", result["risk_level"])
        col3.metric(
            "Prediction",
            "Churn" if result["prediction"] == 1 else "No churn",
        )

        if result["risk_level"] == "Low":
            st.success("Low churn risk. The member appears to be engaged.")
        elif result["risk_level"] == "Medium":
            st.warning("Medium churn risk. The member may need additional engagement.")
        else:
            st.error("High churn risk. A retention action is recommended.")

        st.info(f"Recommended action: {result['recommendation']}")

        st.markdown("### Input data used for prediction")
        st.dataframe(pd.DataFrame([input_data]), use_container_width=True)


elif page == "Data Insights":
    st.title("Data Insights")

    df = load_dataset()

    if df is None:
        st.warning("Dataset not found. Please run `python src/train_model.py` first.")
    else:
        st.markdown("### Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.markdown("### Dataset Overview")

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Churn rate", f"{df['churn'].mean():.1%}")

        st.divider()

        st.markdown("### Churn Distribution")
        churn_counts = df["churn"].value_counts().sort_index()

        fig, ax = plt.subplots()
        ax.bar(["No churn", "Churn"], churn_counts.values)
        ax.set_ylabel("Number of members")
        ax.set_title("Churn distribution")
        st.pyplot(fig)

        st.markdown("### Average Behavior by Churn Status")

        summary = df.groupby("churn")[
            [
                "weekly_visits",
                "days_since_last_visit",
                "monthly_fee",
                "satisfaction_score",
                "membership_months",
            ]
        ].mean()

        st.dataframe(summary, use_container_width=True)

        st.markdown(
            """
            These insights help explain the model logic. Members with fewer weekly visits,
            more days since the last visit and lower satisfaction tend to have a higher churn risk.
            """
        )


elif page == "Model Performance":
    st.title("Model Performance")

    if not METRICS_PATH.exists():
        st.warning("Model metrics file not found. Please run `python src/train_model.py` first.")
    else:
        metrics = joblib.load(METRICS_PATH)

        best_model_name = "Random Forest"
        if BEST_MODEL_NAME_PATH.exists():
            best_model_name = joblib.load(BEST_MODEL_NAME_PATH)

        st.markdown(f"### Selected model: **{best_model_name}**")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", metrics.get("accuracy", "N/A"))
        col2.metric("Precision", metrics.get("precision", "N/A"))
        col3.metric("Recall", metrics.get("recall", "N/A"))
        col4.metric("F1-score", metrics.get("f1_score", "N/A"))

        st.markdown(
            """
            The training pipeline evaluates model performance using classification metrics.
            The F1-score is especially useful because it balances precision and recall.
            """
        )

        if MODEL_COMPARISON_PATH.exists():
            comparison_df = pd.read_csv(MODEL_COMPARISON_PATH)

            st.markdown("### Model Comparison")
            st.dataframe(comparison_df, use_container_width=True)

            st.markdown("### F1-score by Model")

            fig, ax = plt.subplots()
            ax.bar(comparison_df["model_name"], comparison_df["f1_score"])
            ax.set_ylabel("F1-score")
            ax.set_title("Model comparison by F1-score")
            ax.tick_params(axis="x", rotation=20)
            st.pyplot(fig)
        else:
            st.info(
                "Model comparison file not found yet. The current app can still use the saved model."
            )

        st.markdown("### Business Interpretation")

        st.markdown(
            """
            The strongest expected churn indicators are:

            - many days since last visit,
            - low weekly attendance,
            - low satisfaction score,
            - high monthly fee,
            - short membership duration.

            These variables reflect engagement, price sensitivity and customer satisfaction.
            """
        )


elif page == "Prediction Monitoring":
    st.title("Prediction Monitoring")

    st.markdown(
        """
        This section shows the prediction logs generated by the application.
        Each prediction is stored to support basic monitoring, traceability and future model auditing.
        """
    )

    if LOG_PATH.exists():
        logs = pd.read_csv(LOG_PATH)

        st.metric("Total predictions logged", len(logs))

        st.markdown("### Recent Predictions")
        st.dataframe(logs.tail(20), use_container_width=True)

        if "risk_level" in logs.columns:
            st.markdown("### Risk Level Distribution")

            risk_counts = logs["risk_level"].value_counts()

            fig, ax = plt.subplots()
            ax.bar(risk_counts.index, risk_counts.values)
            ax.set_ylabel("Number of predictions")
            ax.set_title("Logged risk levels")
            st.pyplot(fig)
    else:
        st.info("No predictions have been logged yet. Go to the Prediction page and make a prediction.")


elif page == "MLOps Pipeline":
    st.title("MLOps Pipeline")

    st.markdown(
        """
        FitRetention follows a complete academic MLOps workflow:

        ### 1. Data Layer
        Synthetic gym member data is generated for educational purposes.

        ### 2. Model Training
        A Scikit-learn model is trained to predict member churn.

        ### 3. Model Persistence
        The trained model and metrics are stored using Joblib.

        ### 4. Application Layer
        Streamlit provides an interactive web application.

        ### 5. Testing
        Pytest validates the prediction function.

        ### 6. CI/CD
        GitHub Actions automatically installs dependencies, trains the model and runs tests.

        ### 7. Containerization
        Docker and Docker Compose are used to run the app in a reproducible environment.

        ### 8. Monitoring
        Prediction logs are stored in a CSV file to support traceability and future monitoring.

        ### 9. Infrastructure as Code
        Terraform files are included as a starting point for cloud deployment.
        """
    )

    st.success(
        "The application is prepared for local execution, Docker execution and future cloud deployment."
    )