import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.predict import predict_churn


DATA_PATH = ROOT_DIR / "data" / "gym_churn.csv"
METRICS_PATH = ROOT_DIR / "models" / "model_metrics.pkl"
LOG_PATH = ROOT_DIR / "logs" / "predictions_log.csv"
MODEL_COMPARISON_PATH = ROOT_DIR / "models" / "model_comparison.csv"
BEST_MODEL_NAME_PATH = ROOT_DIR / "models" / "best_model_name.pkl"
FEATURE_IMPORTANCE_PATH = ROOT_DIR / "models" / "feature_importance.csv"


st.set_page_config(
    page_title="FitRetention",
    page_icon="🏋️",
    layout="wide",
)


st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #1F2937 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    [data-testid="stSidebar"] * {
        color: #F9FAFB;
    }

    h1 {
        font-size: 3rem !important;
        font-weight: 850 !important;
        color: #F9FAFB !important;
        letter-spacing: -0.04em;
    }

    h2, h3 {
        color: #F9FAFB !important;
        font-weight: 750 !important;
        letter-spacing: -0.025em;
    }

    p, li, label, div {
        color: #E5E7EB;
    }

    .hero-card {
        padding: 2.2rem;
        border-radius: 28px;
        background:
            radial-gradient(circle at top left, rgba(16,185,129,0.22), transparent 35%),
            linear-gradient(135deg, #1F2937 0%, #111827 45%, #064E3B 100%);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 24px 55px rgba(0,0,0,0.28);
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 3.3rem;
        font-weight: 900;
        color: white;
        margin-bottom: 0.3rem;
        letter-spacing: -0.05em;
    }

    .hero-subtitle {
        font-size: 1.35rem;
        color: #D1FAE5;
        margin-bottom: 1.2rem;
    }

    .card {
        padding: 1.45rem;
        border-radius: 20px;
        background-color: #111827;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 12px 30px rgba(0,0,0,0.20);
        margin-bottom: 1rem;
    }

    .metric-card {
        padding: 1.25rem;
        border-radius: 20px;
        background:
            radial-gradient(circle at top right, rgba(110,231,183,0.18), transparent 38%),
            linear-gradient(135deg, #064E3B 0%, #065F46 100%);
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 12px 28px rgba(0,0,0,0.18);
    }

    .metric-title {
        font-size: 0.95rem;
        color: #D1FAE5;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 1.65rem;
        font-weight: 850;
        color: white;
    }

    .badge {
        display: inline-block;
        padding: 0.45rem 0.8rem;
        margin: 0.22rem;
        border-radius: 999px;
        background-color: rgba(16,185,129,0.13);
        color: #6EE7B7;
        border: 1px solid rgba(110,231,183,0.30);
        font-weight: 650;
    }

    .warning-card {
        padding: 1.15rem;
        border-radius: 18px;
        background-color: rgba(245,158,11,0.12);
        border: 1px solid rgba(245,158,11,0.35);
    }

    .success-card {
        padding: 1.15rem;
        border-radius: 18px;
        background-color: rgba(16,185,129,0.12);
        border: 1px solid rgba(16,185,129,0.35);
    }

    .danger-card {
        padding: 1.15rem;
        border-radius: 18px;
        background-color: rgba(239,68,68,0.12);
        border: 1px solid rgba(239,68,68,0.35);
    }

    div[data-testid="stMetric"] {
        background-color: #111827;
        padding: 1rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 20px rgba(0,0,0,0.14);
    }

    div[data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.06);
    }

    .stButton > button {
        border-radius: 999px;
        padding: 0.7rem 1.4rem;
        font-weight: 750;
        background: linear-gradient(90deg, #10B981 0%, #059669 100%);
        color: white;
        border: none;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #059669 0%, #047857 100%);
        color: white;
    }

    [data-testid="stSidebar"] .stButton > button {
        background-color: transparent;
        color: #E5E7EB;
        border: none;
        text-align: left;
        justify-content: flex-start;
        padding: 0.68rem 0.9rem;
        border-radius: 13px;
        font-weight: 650;
        width: 100%;
        box-shadow: none;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: rgba(16,185,129,0.16);
        color: #6EE7B7;
    }

    [data-testid="stSidebar"] .stButton > button:focus {
        background-color: rgba(16,185,129,0.22);
        color: #6EE7B7;
        border: none;
        box-shadow: none;
    }

    .nav-active {
        padding: 0.65rem 0.9rem;
        border-radius: 13px;
        background-color: rgba(16,185,129,0.22);
        color: #6EE7B7;
        font-weight: 750;
        margin-bottom: 0.25rem;
    }

    .small-muted {
        color: #9CA3AF;
        font-size: 0.98rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_dataset():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


def section_header(title, subtitle=None):
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(
            f"<p class='small-muted'>{subtitle}</p>",
            unsafe_allow_html=True,
        )


def custom_metric(title, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_badges():
    st.markdown(
        """
        <span class="badge">Machine Learning</span>
        <span class="badge">Streamlit</span>
        <span class="badge">Docker</span>
        <span class="badge">GitHub Actions</span>
        <span class="badge">Terraform</span>
        <span class="badge">Monitoring</span>
        <span class="badge">Cloud Deployment</span>
        """,
        unsafe_allow_html=True,
    )


def log_prediction(input_data, result):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    row = dict(input_data)
    row["timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    row["churn_probability"] = result.get("churn_probability")
    row["prediction"] = result.get("prediction")
    row["risk_level"] = result.get("risk_level")
    row["recommended_action"] = result.get("recommended_action")

    log_df = pd.DataFrame([row])

    if LOG_PATH.exists():
        existing_logs = pd.read_csv(LOG_PATH)
        log_df = pd.concat([existing_logs, log_df], ignore_index=True)

    log_df.to_csv(LOG_PATH, index=False)


def apply_plotly_dark_theme(fig):
    fig.update_layout(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#111827",
        font=dict(color="#F9FAFB", size=13),
        title=dict(
            font=dict(size=18, color="#F9FAFB"),
            x=0.02,
        ),
        xaxis=dict(
            gridcolor="#374151",
            zerolinecolor="#374151",
            linecolor="#374151",
            tickfont=dict(color="#D1D5DB"),
            title=dict(font=dict(color="#D1D5DB")),
        ),
        yaxis=dict(
            gridcolor="#374151",
            zerolinecolor="#374151",
            linecolor="#374151",
            tickfont=dict(color="#D1D5DB"),
            title=dict(font=dict(color="#D1D5DB")),
        ),
        margin=dict(l=30, r=30, t=70, b=45),
        hoverlabel=dict(
            bgcolor="#111827",
            font_size=13,
            font_color="#F9FAFB",
            bordercolor="#10B981",
        ),
    )
    return fig


def plotly_bar_chart(data, x, y, title, labels=None):
    fig = px.bar(
        data,
        x=x,
        y=y,
        title=title,
        text=y,
        labels=labels,
        color_discrete_sequence=["#10B981"],
    )

    fig.update_traces(
        texttemplate="%{text}",
        textposition="outside",
        marker_line_color="#6EE7B7",
        marker_line_width=1.2,
        hovertemplate="<b>%{x}</b><br>%{y}<extra></extra>",
    )

    fig = apply_plotly_dark_theme(fig)

    fig.update_layout(
        bargap=0.35,
        showlegend=False,
    )

    return fig


def plotly_horizontal_bar_chart(data, x, y, title, labels=None):
    fig = px.bar(
        data,
        x=x,
        y=y,
        orientation="h",
        title=title,
        text=x,
        labels=labels,
        color_discrete_sequence=["#10B981"],
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
        marker_line_color="#6EE7B7",
        marker_line_width=1.2,
        hovertemplate="<b>%{y}</b><br>%{x:.4f}<extra></extra>",
    )

    fig = apply_plotly_dark_theme(fig)

    fig.update_layout(
        yaxis=dict(
            categoryorder="total ascending",
            gridcolor="#374151",
            zerolinecolor="#374151",
            linecolor="#374151",
            tickfont=dict(color="#D1D5DB"),
            title=dict(font=dict(color="#D1D5DB")),
        ),
        showlegend=False,
    )

    return fig


st.sidebar.markdown("## 🏋️ FitRetention")
st.sidebar.caption("Gym churn prediction with MLOps")
st.sidebar.divider()

if "page" not in st.session_state:
    st.session_state.page = "Home"


def set_page(page_name):
    st.session_state.page = page_name


st.sidebar.markdown("### Navigation")

nav_items = [
    "Home",
    "Executive Summary",
    "Prediction",
    "Data Insights",
    "Model Performance",
    "Prediction Monitoring",
    "MLOps Pipeline",
]

for item in nav_items:
    if st.session_state.page == item:
        st.sidebar.markdown(
            f"<div class='nav-active'>{item}</div>",
            unsafe_allow_html=True,
        )
    else:
        if st.sidebar.button(item, key=f"nav_{item}", use_container_width=True):
            set_page(item)
            st.rerun()

page = st.session_state.page

st.sidebar.divider()
st.sidebar.markdown("### Project Stack")
st.sidebar.markdown(
    """
    - Python  
    - Scikit-learn  
    - Streamlit  
    - Plotly  
    - Docker  
    - GitHub Actions  
    - Terraform  
    - Streamlit Cloud  
    """
)


if page == "Home":
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">🏋️ FitRetention</div>
            <div class="hero-subtitle">Professional Gym Member Churn Prediction Platform</div>
            <p>
                FitRetention helps gyms identify members who are likely to cancel their membership.
                The platform combines Machine Learning and MLOps practices to provide predictions,
                business recommendations and monitoring capabilities.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_badges()

    st.divider()

    section_header(
        "Business Problem",
        "Gyms lose revenue when members cancel their subscriptions. Early detection allows better retention strategies.",
    )

    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown(
            """
            <div class="card">
                <h3>Why churn prediction matters</h3>
                <p>
                    Gym members often leave because of low attendance, low satisfaction,
                    lack of engagement, high monthly fees or lack of personalized support.
                    By predicting churn risk, the business can contact members before they cancel.
                </p>
                <p>
                    The goal is not only to predict risk, but also to recommend actions that improve retention.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        custom_metric("Use Case", "Churn")
        custom_metric("Interface", "Web App")
        custom_metric("Workflow", "MLOps")

    st.divider()

    section_header("What the app provides")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="card">
                <h3>Prediction</h3>
                <p>Estimate the probability that a member cancels their membership.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="card">
                <h3>Recommendation</h3>
                <p>Suggest a practical retention action based on the risk level.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="card">
                <h3>Monitoring</h3>
                <p>Store prediction logs to support traceability and future model monitoring.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


elif page == "Executive Summary":
    section_header(
        "Executive Summary",
        "High-level overview of the FitRetention application, model and MLOps workflow.",
    )

    df = load_dataset()

    best_model_name = "Not available"
    if BEST_MODEL_NAME_PATH.exists():
        best_model_name = joblib.load(BEST_MODEL_NAME_PATH)

    f1_score = "N/A"
    if METRICS_PATH.exists():
        metrics = joblib.load(METRICS_PATH)
        f1_score = metrics.get("f1_score", "N/A")

    total_predictions = 0
    high_risk_predictions = 0
    high_risk_rate = "N/A"

    if LOG_PATH.exists():
        logs = pd.read_csv(LOG_PATH)
        total_predictions = len(logs)

        if total_predictions > 0 and "risk_level" in logs.columns:
            high_risk_predictions = (logs["risk_level"] == "High").sum()
            high_risk_rate = f"{high_risk_predictions / total_predictions:.1%}"

    churn_rate = "N/A"
    dataset_rows = "N/A"

    if df is not None:
        dataset_rows = df.shape[0]
        churn_rate = f"{df['churn'].mean():.1%}"

    st.markdown(
        """
        <div class="card">
            <h3>Business Overview</h3>
            <p>
                FitRetention is designed for gym managers and retention teams.
                It helps identify members with a higher probability of cancelling their membership,
                enabling proactive retention actions before churn occurs.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Dataset size", dataset_rows)
    col2.metric("Dataset churn rate", churn_rate)
    col3.metric("Best model", best_model_name)
    col4.metric("F1-score", f1_score)

    st.divider()

    section_header(
        "Operational Monitoring",
        "Summary of predictions generated by the application.",
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Predictions logged", total_predictions)
    col2.metric("High-risk predictions", high_risk_predictions)
    col3.metric("High-risk rate", high_risk_rate)

    st.markdown(
        """
        <div class="card">
            <h3>Manager Interpretation</h3>
            <p>
                A high-risk prediction indicates that the member should receive a retention action,
                such as a personalized follow-up, class recommendation, personal training offer
                or membership discount.
            </p>
            <p>
                The monitoring section allows the business to track prediction activity and review
                recently analyzed members.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    section_header(
        "MLOps Readiness",
        "Summary of the engineering and deployment components implemented in the project.",
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="success-card">
                <h3>Model Pipeline</h3>
                <p>Model training, model comparison, automatic selection and model persistence are implemented.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="success-card">
                <h3>Automation</h3>
                <p>GitHub Actions runs dependency installation, model training and tests automatically.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="success-card">
                <h3>Deployment</h3>
                <p>The application is deployed publicly using Streamlit Community Cloud and is prepared for Docker execution.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Public application")
    st.code("https://fitretention-gym-churn.streamlit.app", language="text")


elif page == "Prediction":
    section_header(
        "Churn Prediction",
        "Estimate the probability that a gym member cancels their membership.",
    )

    st.markdown(
        """
        <div class="card">
            <h3>Member Information</h3>
            <p>
                Enter the member profile below. The model will estimate the churn probability,
                assign a risk level and suggest a retention action.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_options = load_dataset()
    if df_options is not None and "membership_type" in df_options.columns:
        membership_options = sorted(df_options["membership_type"].dropna().unique().tolist())
    else:
        membership_options = ["Basic", "Standard", "Premium"]

    membership_labels = {
        0: "Basic",
        1: "Standard",
        2: "Premium",
        "Basic": "Basic",
        "Standard": "Standard",
        "Premium": "Premium",
        "basic": "Basic",
        "standard": "Standard",
        "premium": "Premium",
    }

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider("Age", 16, 80, 30)
            membership_months = st.slider("Membership months", 1, 120, 12)
            weekly_visits = st.slider("Weekly visits", 0, 7, 2)

        with col2:
            days_since_last_visit = st.slider("Days since last visit", 0, 60, 10)
            monthly_fee = st.slider("Monthly fee", 10, 150, 50)
            satisfaction_score = st.slider("Satisfaction score", 1, 10, 5)

        with col3:
            personal_trainer_label = st.selectbox("Personal trainer", ["No", "Yes"])
            group_classes_label = st.selectbox("Group classes", ["No", "Yes"])
            membership_type = st.selectbox(
                "Membership type",
                membership_options,
                format_func=lambda x: membership_labels.get(x, str(x)),
            )

        submitted = st.form_submit_button("Predict churn risk", use_container_width=True)

    if submitted:
        input_data = {
            "age": age,
            "membership_months": membership_months,
            "weekly_visits": weekly_visits,
            "days_since_last_visit": days_since_last_visit,
            "monthly_fee": monthly_fee,
            "satisfaction_score": satisfaction_score,
            "personal_trainer": personal_trainer_label,
            "group_classes": group_classes_label,
            "membership_type": membership_type,
        }

        try:
            result = predict_churn(input_data)
        except Exception as error:
            st.error("The prediction could not be generated. Please retrain the model and try again.")
            st.exception(error)
            st.stop()

        probability = result.get("churn_probability", result.get("probability"))

        if probability is None:
            st.error("Prediction probability was not found in the model output.")
            st.write("Model output:", result)
            st.stop()

        prediction = result.get("prediction", result.get("churn_prediction"))

        if prediction is None:
            prediction = 1 if probability >= 0.5 else 0

        risk_level = result.get("risk_level")

        if risk_level is None:
            if probability >= 0.7:
                risk_level = "High"
            elif probability >= 0.4:
                risk_level = "Medium"
            else:
                risk_level = "Low"

        recommended_action = result.get("recommended_action")

        if recommended_action is None:
            if risk_level == "High":
                recommended_action = "Contact the member within 48 hours and offer a personalized retention plan."
            elif risk_level == "Medium":
                recommended_action = "Send a personalized check-in message and encourage participation in group activities."
            else:
                recommended_action = "Maintain regular engagement and promote loyalty benefits."

        log_prediction(input_data, result)

        st.divider()

        section_header(
            "Prediction Result",
            "Model output and recommended retention action.",
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Churn probability", f"{probability:.1%}")
        col2.metric("Risk level", risk_level)
        col3.metric("Final prediction", "Churn" if prediction == 1 else "No churn")

        st.progress(float(probability))

        if risk_level == "High":
            card_class = "danger-card"
            risk_message = "This member has a high probability of cancelling and should receive a proactive retention action."
        elif risk_level == "Medium":
            card_class = "warning-card"
            risk_message = "This member shows some churn risk signals and should be monitored."
        else:
            card_class = "success-card"
            risk_message = "This member currently shows a low churn risk."

        st.markdown(
            f"""
            <div class="{card_class}">
                <h3>{risk_level} Risk</h3>
                <p>{risk_message}</p>
                <p><strong>Recommended action:</strong> {recommended_action}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        section_header(
            "Member Profile Summary",
            "Main profile values used for this prediction.",
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            custom_metric("Age", age)
            custom_metric("Membership months", membership_months)
            custom_metric("Weekly visits", weekly_visits)

        with col2:
            custom_metric("Days since last visit", days_since_last_visit)
            custom_metric("Monthly fee", f"€{monthly_fee}")
            custom_metric("Satisfaction", f"{satisfaction_score}/10")

        with col3:
            custom_metric("Personal trainer", personal_trainer_label)
            custom_metric("Group classes", group_classes_label)
            custom_metric("Membership type", membership_labels.get(membership_type, str(membership_type)))


elif page == "Data Insights":
    section_header(
        "Data Insights",
        "Explore the generated gym member dataset used for model training.",
    )

    df = load_dataset()

    if df is None:
        st.warning("Dataset not found. Please run `python src/train_model.py` first.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Churn rate", f"{df['churn'].mean():.1%}")

        st.markdown("### Dataset preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Churn distribution")
            churn_counts = df["churn"].value_counts().sort_index()

            churn_chart_df = pd.DataFrame(
                {
                    "Status": ["No churn", "Churn"],
                    "Members": churn_counts.values,
                }
            )

            fig = plotly_bar_chart(
                churn_chart_df,
                x="Status",
                y="Members",
                title="Churn distribution",
                labels={
                    "Status": "Churn status",
                    "Members": "Number of members",
                },
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Average satisfaction by churn")
            satisfaction = df.groupby("churn")["satisfaction_score"].mean()

            satisfaction_chart_df = pd.DataFrame(
                {
                    "Status": ["No churn", "Churn"],
                    "Average satisfaction": satisfaction.values,
                }
            )

            fig = plotly_bar_chart(
                satisfaction_chart_df,
                x="Status",
                y="Average satisfaction",
                title="Average satisfaction by churn",
                labels={
                    "Status": "Churn status",
                    "Average satisfaction": "Average satisfaction",
                },
            )

            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Average behavior by churn status")

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


elif page == "Model Performance":
    section_header(
        "Model Performance",
        "Compare candidate models and review the selected model.",
    )

    if not METRICS_PATH.exists():
        st.warning("Model metrics file not found. Please run `python src/train_model.py` first.")
    else:
        metrics = joblib.load(METRICS_PATH)

        best_model_name = "Unknown"
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
            The model training pipeline compares several candidate algorithms and selects
            the best one using the F1-score.
            """
        )

        if MODEL_COMPARISON_PATH.exists():
            comparison_df = pd.read_csv(MODEL_COMPARISON_PATH)

            st.markdown("### Model comparison")
            st.dataframe(comparison_df, use_container_width=True)

            fig = plotly_bar_chart(
                comparison_df,
                x="model_name",
                y="f1_score",
                title="Model comparison by F1-score",
                labels={
                    "model_name": "Model",
                    "f1_score": "F1-score",
                },
            )

            st.plotly_chart(fig, use_container_width=True)

        if FEATURE_IMPORTANCE_PATH.exists():
            importance_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)

            st.markdown("### Feature importance")
            top_features = importance_df.head(10)

            fig = plotly_horizontal_bar_chart(
                top_features,
                x="importance",
                y="feature",
                title="Top 10 most important features",
                labels={
                    "importance": "Importance",
                    "feature": "Feature",
                },
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance is available when the selected model supports it.")


elif page == "Prediction Monitoring":
    section_header(
        "Prediction Monitoring",
        "Track predictions generated by the application.",
    )

    if LOG_PATH.exists():
        logs = pd.read_csv(LOG_PATH)

        st.metric("Total predictions logged", len(logs))

        st.markdown("### Recent predictions")
        st.dataframe(logs.tail(20), use_container_width=True)

        if "risk_level" in logs.columns:
            st.markdown("### Risk level distribution")
            risk_counts = logs["risk_level"].value_counts()

            risk_chart_df = pd.DataFrame(
                {
                    "Risk level": risk_counts.index,
                    "Predictions": risk_counts.values,
                }
            )

            fig = plotly_bar_chart(
                risk_chart_df,
                x="Risk level",
                y="Predictions",
                title="Logged risk levels",
                labels={
                    "Risk level": "Risk level",
                    "Predictions": "Number of predictions",
                },
            )

            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No predictions have been logged yet. Go to the Prediction page and make a prediction.")


elif page == "MLOps Pipeline":
    section_header(
        "MLOps Pipeline",
        "Overview of the technical workflow implemented in the project.",
    )

    st.markdown(
        """
        <div class="card">
            <h3>End-to-end workflow</h3>
            <p><b>1. Data layer:</b> synthetic gym member data generation.</p>
            <p><b>2. Model training:</b> multiple Scikit-learn models are compared.</p>
            <p><b>3. Model selection:</b> the best model is selected using F1-score.</p>
            <p><b>4. Model persistence:</b> the trained model is saved with Joblib.</p>
            <p><b>5. Application:</b> Streamlit provides the web interface.</p>
            <p><b>6. Testing:</b> Pytest validates the prediction function.</p>
            <p><b>7. CI/CD:</b> GitHub Actions runs training and tests automatically.</p>
            <p><b>8. Containerization:</b> Docker runs the app in a reproducible environment.</p>
            <p><b>9. Monitoring:</b> predictions are logged for future analysis.</p>
            <p><b>10. Cloud deployment:</b> the app is deployed with Streamlit Community Cloud.</p>
            <p><b>11. Infrastructure:</b> Terraform prepares the project for cloud deployment.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.success(
        "The application is ready for local execution, Docker execution and cloud access."
    )
