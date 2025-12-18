#!/bin/bash

# Deploy SHL API to Google Cloud Run
# Usage: ./deploy-api.sh [PROJECT_ID] [REGION]

set -e

# Configuration
PROJECT_ID=${1:-$GOOGLE_CLOUD_PROJECT}
REGION=${2:-us-central1}
SERVICE_NAME="shl-api"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

if [ -z "$PROJECT_ID" ]; then
    echo "Error: PROJECT_ID is required. Set GOOGLE_CLOUD_PROJECT or pass as first argument."
    exit 1
fi

echo "Deploying SHL API to Cloud Run..."
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"

# Build and push the container image
echo "Building container image..."
docker build -t $IMAGE_NAME:latest .

echo "Pushing container image..."
docker push $IMAGE_NAME:latest

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME:latest \
    --platform managed \
    --region $REGION \
    --project $PROJECT_ID \
    --allow-unauthenticated \
    --port 8000 \
    --memory 2Gi \
    --cpu 1 \
    --max-instances 10 \
    --timeout 300 \
    --set-env-vars "ENVIRONMENT=production,LOG_LEVEL=info,GOOGLE_CLOUD_PROJECT=$PROJECT_ID" \
    --service-account "shl-app-service@$PROJECT_ID.iam.gserviceaccount.com"

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --project $PROJECT_ID --format 'value(status.url)')

echo "Deployment complete!"
echo "API URL: $SERVICE_URL"
echo "Health check: $SERVICE_URL/health"

# Test the deployment
echo "Testing deployment..."
curl -f "$SERVICE_URL/health" && echo "✅ Health check passed" || echo "❌ Health check failed"