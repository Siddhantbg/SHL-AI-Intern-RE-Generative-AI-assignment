#!/bin/bash

# Deploy SHL Frontend to Google Cloud Run or Cloud Storage
# Usage: ./deploy-frontend.sh [PROJECT_ID] [REGION] [API_URL]

set -e

# Configuration
PROJECT_ID=${1:-$GOOGLE_CLOUD_PROJECT}
REGION=${2:-us-central1}
API_URL=${3:-"https://shl-api-$PROJECT_ID.a.run.app"}
SERVICE_NAME="shl-frontend"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"
BUCKET_NAME="$PROJECT_ID-shl-frontend"

if [ -z "$PROJECT_ID" ]; then
    echo "Error: PROJECT_ID is required. Set GOOGLE_CLOUD_PROJECT or pass as first argument."
    exit 1
fi

echo "Deploying SHL Frontend..."
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "API URL: $API_URL"

# Option 1: Deploy to Cloud Run (containerized)
deploy_to_cloud_run() {
    echo "Building frontend container image..."
    cd frontend
    
    # Update environment variables
    echo "REACT_APP_API_URL=$API_URL" > .env.production
    
    docker build -t $IMAGE_NAME:latest .
    
    echo "Pushing container image..."
    docker push $IMAGE_NAME:latest
    
    echo "Deploying to Cloud Run..."
    gcloud run deploy $SERVICE_NAME \
        --image $IMAGE_NAME:latest \
        --platform managed \
        --region $REGION \
        --project $PROJECT_ID \
        --allow-unauthenticated \
        --port 80 \
        --memory 512Mi \
        --cpu 1 \
        --max-instances 5 \
        --set-env-vars "REACT_APP_API_URL=$API_URL"
    
    cd ..
}

# Option 2: Deploy to Cloud Storage (static hosting)
deploy_to_storage() {
    echo "Building frontend for production..."
    cd frontend
    
    # Update environment variables
    echo "REACT_APP_API_URL=$API_URL" > .env.production
    
    # Build the React app
    npm run build
    
    echo "Uploading to Cloud Storage..."
    gsutil -m rsync -r -d build/ gs://$BUCKET_NAME/
    
    # Set proper content types
    gsutil -m setmeta -h "Content-Type:text/html" -h "Cache-Control:no-cache" gs://$BUCKET_NAME/index.html
    gsutil -m setmeta -h "Content-Type:application/javascript" gs://$BUCKET_NAME/static/js/*.js
    gsutil -m setmeta -h "Content-Type:text/css" gs://$BUCKET_NAME/static/css/*.css
    
    cd ..
    
    echo "Frontend deployed to: https://storage.googleapis.com/$BUCKET_NAME/index.html"
}

# Choose deployment method (default to Cloud Run for simplicity)
DEPLOYMENT_METHOD=${DEPLOYMENT_METHOD:-"cloud_run"}

if [ "$DEPLOYMENT_METHOD" = "storage" ]; then
    deploy_to_storage
else
    deploy_to_cloud_run
    
    # Get the service URL
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --project $PROJECT_ID --format 'value(status.url)')
    echo "Frontend URL: $SERVICE_URL"
fi

echo "Frontend deployment complete!"