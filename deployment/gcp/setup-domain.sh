#!/bin/bash

# Setup custom domain and SSL certificates for SHL application
# Usage: ./setup-domain.sh [PROJECT_ID] [DOMAIN_NAME] [REGION]

set -e

# Configuration
PROJECT_ID=${1:-$GOOGLE_CLOUD_PROJECT}
DOMAIN_NAME=${2}
REGION=${3:-us-central1}

if [ -z "$PROJECT_ID" ] || [ -z "$DOMAIN_NAME" ]; then
    echo "Error: PROJECT_ID and DOMAIN_NAME are required."
    echo "Usage: ./setup-domain.sh PROJECT_ID DOMAIN_NAME [REGION]"
    exit 1
fi

API_DOMAIN="api.$DOMAIN_NAME"
APP_DOMAIN="app.$DOMAIN_NAME"

echo "Setting up custom domain for SHL application..."
echo "Project: $PROJECT_ID"
echo "Domain: $DOMAIN_NAME"
echo "API Domain: $API_DOMAIN"
echo "App Domain: $APP_DOMAIN"

# Create domain mappings for Cloud Run services
echo "Creating domain mapping for API..."
gcloud run domain-mappings create \
    --service shl-api \
    --domain $API_DOMAIN \
    --region $REGION \
    --project $PROJECT_ID

echo "Creating domain mapping for frontend..."
gcloud run domain-mappings create \
    --service shl-frontend \
    --domain $APP_DOMAIN \
    --region $REGION \
    --project $PROJECT_ID

# Get the required DNS records
echo "Getting DNS configuration..."
API_DNS_RECORDS=$(gcloud run domain-mappings describe $API_DOMAIN --region $REGION --project $PROJECT_ID --format="value(status.resourceRecords[].name,status.resourceRecords[].rrdata)" | tr '\t' ' ')
APP_DNS_RECORDS=$(gcloud run domain-mappings describe $APP_DOMAIN --region $REGION --project $PROJECT_ID --format="value(status.resourceRecords[].name,status.resourceRecords[].rrdata)" | tr '\t' ' ')

echo ""
echo "🔧 DNS Configuration Required:"
echo "================================"
echo ""
echo "Add the following DNS records to your domain registrar:"
echo ""
echo "For API ($API_DOMAIN):"
echo "$API_DNS_RECORDS"
echo ""
echo "For App ($APP_DOMAIN):"
echo "$APP_DNS_RECORDS"
echo ""
echo "📝 Note: SSL certificates will be automatically provisioned by Google Cloud"
echo "after DNS records are properly configured and propagated (may take up to 24 hours)."
echo ""
echo "🔍 Check certificate status with:"
echo "gcloud run domain-mappings describe $API_DOMAIN --region $REGION --project $PROJECT_ID"
echo "gcloud run domain-mappings describe $APP_DOMAIN --region $REGION --project $PROJECT_ID"