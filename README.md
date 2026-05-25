# FitRetention

**FitRetention** is a Machine Learning and MLOps web application designed to predict gym member churn.

The goal of the project is to help a fitness center identify members who are likely to cancel their membership and recommend retention actions before churn happens.

The deployed application is available here:

```text
https://fitretention-gym-churn.streamlit.app
```

---

## Project Overview

FitRetention transforms a Machine Learning model into a usable web application through an end-to-end MLOps workflow.

The project includes:

- Machine Learning model training
- Automatic model comparison and model selection
- Streamlit web application
- Professional user interface
- Prediction logging for basic monitoring
- Automated testing with Pytest
- CI/CD with GitHub Actions
- Docker containerization
- Docker Compose configuration
- Cloud deployment with Streamlit Community Cloud
- Terraform infrastructure-as-code preparation

---

## Business Problem

Gyms often lose members because of:

- Low attendance
- Low satisfaction
- Lack of engagement
- High monthly fees
- Long periods without visiting the gym

By predicting churn risk early, a gym can take preventive actions such as contacting inactive members, offering personalized plans, recommending group classes or providing targeted discounts.

---

## Application Features

The Streamlit application includes the following sections:

### Home

General explanation of the business problem, project value and technology stack.

### Prediction

A form where the user enters gym member information and receives:

- Churn probability
- Risk level: Low, Medium or High
- Final prediction: Churn or No churn
- Recommended retention action

### Data Insights

Dataset exploration, including:

- Dataset preview
- Number of rows and columns
- Churn rate
- Churn distribution chart
- Average satisfaction by churn status
- Average behavior comparison

### Model Performance

Model evaluation section showing:

- Selected best model
- Accuracy
- Precision
- Recall
- F1-score
- Model comparison table
- F1-score comparison chart
- Feature importance when available

### Prediction Monitoring

Basic monitoring section that logs predictions made through the application.

Each prediction stores:

- Timestamp
- Input data
- Churn probability
- Risk level
- Recommended action

### MLOps Pipeline

Explanation of the technical MLOps workflow implemented in the project.

---

## Machine Learning Approach

The project trains and compares multiple classification models:

- Logistic Regression
- Random Forest
- Gradient Boosting

The best model is selected automatically based on the **F1-score**, because this metric balances precision and recall.

The final trained model is saved with Joblib and used by the Streamlit application for real-time predictions.

---

## Dataset

For this academic project, the dataset is synthetically generated to simulate realistic gym member behavior.

The dataset includes:

- Age
- Membership duration in months
- Weekly visits
- Days since last visit
- Monthly fee
- Satisfaction score
- Personal trainer usage
- Group class participation
- Membership type
- Churn label

Synthetic data is used because real gym membership data is usually private and may be protected by data protection regulations.

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
│   └── main.tf
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
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

The app will be available at:

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

The workflow runs automatically on each push or pull request to the main branch.

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

---

## Terraform

The project includes a basic Terraform configuration in:

```text
terraform/main.tf
```

This file represents the infrastructure-as-code stage of the MLOps workflow.

It can be extended to deploy the application to cloud platforms such as:

- Microsoft Azure
- AWS
- Google Cloud Platform

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

---

## Current Status

Completed:

- Machine Learning model
- Automatic model comparison
- Streamlit web application
- Professional UI
- Prediction logging
- Pytest tests
- GitHub Actions CI pipeline
- Dockerfile
- Docker Compose
- Cloud deployment
- Terraform placeholder
- GitHub repository documentation

Pending / Future improvements:

- Authentication
- Real production dataset
- Advanced monitoring
- Model drift detection
- Scheduled retraining
- Full production deployment with Azure resources

---

## Limitations

This project uses synthetic data for academic purposes. In a real production scenario, the model should be trained with real gym membership data and monitored continuously after deployment.

The predictions should be used as decision support, not as the only basis for business decisions.

---

## Authors

Developed as part of a Machine Learning Operations course project.