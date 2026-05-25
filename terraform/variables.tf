variable "project_name" {
  description = "Name of the MLOps project"
  type        = string
  default     = "FitRetention"
}

variable "resource_group_name" {
  description = "Name of the Azure resource group"
  type        = string
  default     = "rg-fitretention-mlops"
}

variable "location" {
  description = "Azure region where resources would be deployed"
  type        = string
  default     = "West Europe"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "academic"
}