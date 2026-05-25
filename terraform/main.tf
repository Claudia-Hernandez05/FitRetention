terraform {
  required_version = ">= 1.5.0"
}

# Basic Terraform placeholder for the cloud deployment stage.
# In a real production environment, this file could define:
# - Resource group
# - Container registry
# - App service or container app
# - Networking
# - Environment variables

output "project_name" {
  value = "FitRetention"
}

output "description" {
  value = "MLOps infrastructure placeholder for Streamlit gym churn prediction app"
}