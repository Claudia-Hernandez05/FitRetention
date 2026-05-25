output "project_name" {
  description = "Name of the project"
  value       = var.project_name
}

output "resource_group_name" {
  description = "Name of the Azure resource group"
  value       = azurerm_resource_group.fitretention_rg.name
}

output "location" {
  description = "Azure deployment location"
  value       = azurerm_resource_group.fitretention_rg.location
}

output "environment" {
  description = "Deployment environment"
  value       = var.environment
}