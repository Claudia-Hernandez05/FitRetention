terraform {
  required_version = ">= 1.5.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.100"
    }
  }
}

provider "azurerm" {
  features {}
}

# This Terraform configuration is prepared for a future Azure deployment.
# It defines the base resource group that could host the containerized Streamlit application.

resource "azurerm_resource_group" "fitretention_rg" {
  name     = var.resource_group_name
  location = var.location

  tags = {
    project     = var.project_name
    environment = var.environment
    managed_by  = "terraform"
  }
}