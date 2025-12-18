#!/bin/bash

# Main deployment script for SHL Assessment Recommendation System
# Usage: ./deploy.sh [PROJECT_ID] [REGION] [DOMAIN_NAME]

set -e

# Configuration
PROJECT_ID=${1:-$GOOGLE_CLOUD_PROJECT}
REGION=${2:-us-central1}
DOMAIN_NAME=${3}

if [ -z "$PROJECT_ID" ]; then
    echo "Error: PROJECT_ID is required. Set GOOGLE_CLOUD_PROJECT or pass as first argument."
    echo "Usage: ./deploy.sh PROJECT_ID [REGION] [DOMAIN_NAME]"
    exit 1
fi

echo "🚀 Deploying SHL Assessment Recommendation System"
echo "================================================"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Domain: ${DOMAIN_NAME:-"Not configured"}"
echo ""

# Check prerequisites
echo "🔍 Checking prerequisites..."

# Check if gcloud is installed and authenticated
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI is not installed. Please install it first."
    exit 1
fi

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install it first."
    exit 1
fi

# Check if user is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Not authenticated with gcloud. Please run 'gcloud auth login' first."
    exit 1
fi

# Set the project
gcloud config set project $PROJECT_ID

echo "✅ Prerequisites check passed"
echo ""

# Step 1: Setup infrastructure with Terraform (optional)
echo "📋 Step 1: Setting up infrastructure..."
cd deployment/gcp/terraform

if [ -f "terraform.tfvars" ]; then
    echo "Found terraform.tfvars, running Terraform..."
    terraform init
    terraform plan -var="project_id=$PROJECT_ID" -var="region=$REGION"
    
    read -p "Apply Terraform configuration? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        terraform apply -var="project_id=$PROJECT_ID" -var="region=$REGION" -auto-approve
    fi
else
    echo "⚠️  No terraform.tfvars found. Skipping Terraform setup."
    echo "   You can manually enable required APIs and create resources."
fi

cd ../../..

# Step 2: Deploy API
echo ""
echo "🔧 Step 2: Deploying API to Cloud Run..."
./deployment/gcp/deploy-api.sh $PROJECT_ID $REGION

# Get API URL for frontend configuration
API_URL=$(gcloud run services describe shl-api --region $REGION --project $PROJECT_ID --format 'value(status.url)')
echo "API deployed at: $API_URL"

# Step 3: Deploy Frontend
echo ""
echo "🎨 Step 3: Deploying Frontend..."
./deployment/gcp/deploy-frontend.sh $PROJECT_ID $REGION $API_URL

# Get Frontend URL
FRONTEND_URL=$(gcloud run services describe shl-frontend --region $REGION --project $PROJECT_ID --format 'value(status.url)')
echo "Frontend deployed at: $FRONTEND_URL"

# Step 4: Setup monitoring
echo ""
echo "📊 Step 4: Setting up monitoring and logging..."
./deployment/gcp/monitoring/setup-monitoring.sh $PROJECT_ID $API_URL $FRONTEND_URL

# Step 5: Setup custom domain (if provided)
if [ -n "$DOMAIN_NAME" ]; then
    echo ""
    echo "🌐 Step 5: Setting up custom domain..."
    ./deployment/gcp/setup-domain.sh $PROJECT_ID $DOMAIN_NAME $REGION
fi

# Final summary
echo ""
echo "🎉 Deployment Complete!"
echo "======================="
echo ""
echo "📍 Service URLs:"
echo "   API: $API_URL"
echo "   Frontend: $FRONTEND_URL"
echo "   Health Check: $API_URL/health"
echo ""

if [ -n "$DOMAIN_NAME" ]; then
    echo "🌐 Custom Domains (configure DNS as instructed above):"
    echo "   API: https://api.$DOMAIN_NAME"
    echo "   Frontend: https://app.$DOMAIN_NAME"
    echo ""
fi

echo "📊 Monitoring:"
echo "   Dashboard: https://console.cloud.google.com/monitoring/dashboards?project=$PROJECT_ID"
echo "   Logs: https://console.cloud.google.com/logs?project=$PROJECT_ID"
echo ""

echo "🧪 Test your deployment:"
echo "   curl $API_URL/health"
echo "   open $FRONTEND_URL"
echo ""

echo "✅ All services are now running in production!"