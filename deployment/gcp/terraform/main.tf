# Configure the Google Cloud Provider
terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

# Configure the Google Cloud Provider
provider "google" {
  project = var.project_id
  region  = var.region
}

# Variables
variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region"
  type        = string
  default     = "us-central1"
}

variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = ""
}

# Enable required APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "containerregistry.googleapis.com",
    "storage.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "iam.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com"
  ])
  
  project = var.project_id
  service = each.value
  
  disable_dependent_services = true
}

# Cloud Storage bucket for static assets and data
resource "google_storage_bucket" "app_data" {
  name     = "${var.project_id}-shl-app-data"
  location = var.region
  
  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }
}

# Cloud Storage bucket for frontend hosting (alternative to Cloud Run)
resource "google_storage_bucket" "frontend" {
  name     = "${var.project_id}-shl-frontend"
  location = var.region
  
  uniform_bucket_level_access = true
  
  website {
    main_page_suffix = "index.html"
    not_found_page   = "index.html"
  }
}

# Make frontend bucket publicly readable
resource "google_storage_bucket_iam_member" "frontend_public" {
  bucket = google_storage_bucket.frontend.name
  role   = "roles/storage.objectViewer"
  member = "allUsers"
}

# Service account for the application
resource "google_service_account" "app_service_account" {
  account_id   = "shl-app-service"
  display_name = "SHL Application Service Account"
  description  = "Service account for SHL recommendation system"
}

# Grant necessary permissions to the service account
resource "google_project_iam_member" "app_permissions" {
  for_each = toset([
    "roles/storage.objectViewer",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.app_service_account.email}"
}

# Cloud Run service for API (will be created by Cloud Build)
# This is just for reference and IAM setup
resource "google_cloud_run_service_iam_member" "api_public" {
  location = var.region
  project  = var.project_id
  service  = "shl-api"
  role     = "roles/run.invoker"
  member   = "allUsers"
  
  depends_on = [google_project_service.apis]
}

resource "google_cloud_run_service_iam_member" "frontend_public" {
  location = var.region
  project  = var.project_id
  service  = "shl-frontend"
  role     = "roles/run.invoker"
  member   = "allUsers"
  
  depends_on = [google_project_service.apis]
}

# Outputs
output "project_id" {
  value = var.project_id
}

output "region" {
  value = var.region
}

output "app_data_bucket" {
  value = google_storage_bucket.app_data.name
}

output "frontend_bucket" {
  value = google_storage_bucket.frontend.name
}

output "service_account_email" {
  value = google_service_account.app_service_account.email
}