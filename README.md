# FitRetention

**FitRetention** is a Machine Learning and MLOps web application designed to predict gym member churn.

The application helps gym managers and retention teams identify members who are likely to cancel their membership and take preventive actions before churn happens.

The deployed application is available at:

```text
https://fitretention-gym-churn.streamlit.app
```

---

## Project Overview

FitRetention transforms a Machine Learning model into a professional web application through an end-to-end MLOps workflow.

The project combines:

- Machine Learning model training
- Automatic model comparison and model selection
- Streamlit web application
- Interactive data visualizations with Plotly
- Prediction logging for basic monitoring
- Automated testing with Pytest
- CI/CD with GitHub Actions
- Docker containerization
- Docker Compose configuration
- Cloud deployment with Streamlit Community Cloud
- Terraform infrastructure-as-code preparation

The main goal is not only to train a model, but also to demonstrate how a Machine Learning solution can be packaged, tested, deployed and monitored as part of a complete MLOps pipeline.

---

## Business Problem

Customer churn is an important business problem for gyms and fitness centers.

Gyms often lose members because of:

- Low attendance
- Low satisfaction
- Lack of engagement
- High monthly fees
- Long periods without visiting the gym
- Lack of personalized support

By predicting churn risk early, a gym can take preventive actions such as contacting inactive members, offering personalized plans, recommending group classes or providing targeted discounts.

FitRetention is designed as a decision-support tool for gym managers. The members are the analyzed customers, while the application is used by the business to support retention decisions.

---

## Application Features

The Streamlit application is organized into several sections.

### Home

The home page introduces the business problem, project value and technology stack.

### Prediction

The prediction page allows the user to enter information about a gym member and receive:

- Churn probability
- Risk level: Low, Medium or High
- Final prediction: Churn or No churn
- Recommended retention action

### Data Insights

The data insights page provides an overview of the dataset, including:

- Dataset preview
- Number of rows and columns
- Churn rate
- Churn distribution
- Average satisfaction by churn status
- Average behavior comparison

The charts are interactive and built with Plotly.

### Model Performance

The model performance page shows:

- Selected best model
- Accuracy
- Precision
- Recall
- F1-score
- Model comparison table
- Interactive F1-score comparison chart
- Feature importance when available

### Prediction Monitoring

The prediction monitoring page stores and displays predictions made through the application.

Each prediction includes:

- Timestamp
- Input values
- Churn probability
- Risk level
- Recommended action

This provides a basic monitoring and traceability mechanism.

### MLOps Pipeline

The MLOps pipeline page explains the technical workflow implemented in the project, including model training, testing, CI/CD, Docker, cloud deployment and infrastructure-as-code preparation.

---

## Machine Learning Approach

The project treats churn prediction as a binary classification problem.

The target variable is:

```text
churn
```

where:

- `0` means the member is not expected to churn
- `1` means the member is expected to churn

The project trains and compares multiple classification models:

- Logistic Regression
- Random Forest
- Gradient Boosting

The best model is selected automatically based on the **F1-score**.

The F1-score is used because churn prediction requires balancing:

- **Precision**: how many predicted churn cases are actually churn
- **Recall**: how many real churn cases are correctly detected

This is important because failing to detect a high-risk member may result in a lost customer, while incorrectly flagging a member may lead to unnecessary retention actions.

The final trained model is saved with Joblib and used by the Streamlit application for real-time predictions.

---

## Dataset

For this academic project, the dataset is synthetically generated to simulate realistic gym member behavior.

The dataset includes:

| Feature | Description |
|---|---|
| age | Age of the gym member |
| membership_months | Number of months since the member joined |
| weekly_visits | Average number of weekly gym visits |
| days_since_last_visit | Number of days since the last visit |
| monthly_fee | Monthly membership fee |
| satisfaction_score | Satisfaction level from 1 to 10 |
| personal_trainer | Whether the member uses a personal trainer |
| group_classes | Whether the member attends group classes |
| membership_type | Type of membership |
| churn | Target variable indicating cancellation |

Synthetic data is used because real gym membership data may contain sensitive personal information and is usually protected by privacy regulations.

---

## Project Structure

```text
FitRetention/
├── app/
│   └── streamlit_app.py
├── data/
│   └── gym_churn.csv
├── logs/
│   └── predictions_log.csv
├── models/
│   ├── churn_model.pkl
│   ├── model_metrics.pkl
│   ├── model_comparison.csv
│   ├── best_model_name.pkl
│   └── feature_importance.csv
├── src/
│   ├── train_model.py
│   ├── predict.py
│   └── preprocessing.py
├── tests/
│   └── test_prediction.py
├── terraform/
│   ├── main.tf
│   ├── variables.tf
│   └── outputs.tf
├── .github/
│   └── workflows/
│       └── ci.yml
├── .streamlit/
│   └── config.toml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── PROJECT_REPORT.md
└── README.md
```

---

## Cloud Deployment

The application is deployed using **Streamlit Community Cloud** and is publicly accessible at:

```text
https://fitretention-gym-churn.streamlit.app
```

This deployment allows users to access the Machine Learning application directly from a browser without running it locally.

The cloud deployment is connected to the GitHub repository, so the application can be updated from the `main` branch.

---

## How to Run Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python src/train_model.py
```

### 3. Run the Streamlit application

```bash
streamlit run app/streamlit_app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## Run Tests

```bash
pytest
```

The tests validate that the prediction function returns the expected output format.

---

## Run with Docker

### Build the Docker image

```bash
docker build -t fitretention .
```

### Run the container

```bash
docker run -p 8501:8501 fitretention
```

Then open:

```text
http://localhost:8501
```

---

## Run with Docker Compose

```bash
docker compose up --build
```

---

## CI/CD Pipeline

The project includes a GitHub Actions workflow located in:

```text
.github/workflows/ci.yml
```

The workflow runs automatically on each push or pull request to the `main` branch.

It performs the following steps:

1. Checks out the repository
2. Sets up Python
3. Installs dependencies
4. Trains the model
5. Runs tests with Pytest

This ensures that the project remains reproducible and that the prediction logic is continuously validated.

---

## Docker

The application is containerized using Docker.

Docker allows the project to run in a reproducible environment, independent of the local machine configuration.

The Dockerfile:

- Uses a Python base image
- Installs project dependencies
- Copies the application code
- Trains the model
- Starts the Streamlit application

This demonstrates that the application can be packaged and executed consistently across different environments.

---

## Terraform

The project includes Terraform configuration files in:

```text
terraform/
```

Terraform is used as the infrastructure-as-code component of the project.

The current configuration defines a structured base for future cloud infrastructure deployment and can be extended to provision resources such as:

- Azure resource groups
- Container registries
- App services
- Storage accounts
- Monitoring resources

This demonstrates how the application could be moved toward a more advanced production cloud deployment.

---

## MLOps Workflow

The project demonstrates the following MLOps lifecycle:

1. Data generation
2. Data preprocessing
3. Model training
4. Model comparison
5. Model selection
6. Model persistence
7. Web application layer
8. Prediction logging
9. Automated testing
10. CI/CD automation
11. Docker containerization
12. Cloud deployment
13. Infrastructure-as-code preparation

This workflow shows how a Machine Learning model can be transformed into a usable, reproducible and deployable application.

---

## Requirements Coverage

| Requirement | Implementation in FitRetention |
|---|---|
| Machine Learning model | Churn prediction model trained with Scikit-learn |
| Web application | Interactive Streamlit application |
| Model evaluation | Accuracy, precision, recall and F1-score |
| Model comparison | Logistic Regression, Random Forest and Gradient Boosting |
| Automated testing | Pytest test suite |
| CI/CD | GitHub Actions workflow |
| Containerization | Dockerfile and Docker Compose |
| Cloud deployment | Streamlit Community Cloud |
| Infrastructure as Code | Terraform configuration |
| Monitoring | Prediction logging and monitoring page |
| Documentation | README and PROJECT_REPORT.md |

---

## Limitations

This project uses synthetic data for academic purposes. In a real production scenario, the model should be trained with real gym membership data and monitored continuously after deployment.

The current monitoring system is based on prediction logs. In a production environment, this could be extended with advanced monitoring dashboards, model drift detection, data drift detection and automated retraining.

The predictions should be used as decision support, not as the only basis for business decisions.

---

## Future Work

Possible future improvements include:

- Training the model with real gym membership data
- Adding user authentication
- Storing prediction logs in a database
- Adding advanced monitoring dashboards
- Implementing model drift detection
- Implementing scheduled retraining
- Adding explainability tools such as SHAP
- Deploying the Docker container to a cloud platform such as Azure
- Expanding the Terraform configuration for full cloud infrastructure deployment

---

## Additional Documentation

A detailed technical report is available in:

```text
PROJECT_REPORT.md
```

This report explains the business context, Machine Learning methodology, MLOps workflow, system architecture, Docker setup, CI/CD pipeline, cloud deployment and limitations of the project.

---

## Authors


- Claudia Hernández Miranda
- Carlos Ortega Bernardos
- Michael Bellido



Developed as part of a Machine Learning Operations course project.