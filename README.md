# FitRetention

FitRetention is a Machine Learning and MLOps project focused on predicting gym member churn.

The application helps a fitness center identify members who may cancel their membership based on behavioral and subscription-related factors such as attendance frequency, membership duration, last visit, monthly fee and satisfaction level.

## Project Objective

The main objective is to build an end-to-end Machine Learning application following an MLOps workflow.

The project includes:

- Machine Learning model training with Scikit-learn
- Model persistence with Joblib
- Streamlit web application
- Docker containerization
- Automated testing with Pytest
- CI/CD pipeline with GitHub Actions
- Basic Terraform configuration for cloud infrastructure preparation

## Application Features

The Streamlit app allows the user to enter gym member information and returns:

- Churn probability
- Risk level: Low, Medium or High
- Recommended retention action
- Model performance metrics
- MLOps pipeline explanation

## Project Structure

```text
FitRetention/
├── app/
│   └── streamlit_app.py
├── data/
│   └── gym_churn.csv
├── models/
│   ├── churn_model.pkl
│   └── model_metrics.pkl
├── src/
│   ├── train_model.py
│   └── predict.py
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