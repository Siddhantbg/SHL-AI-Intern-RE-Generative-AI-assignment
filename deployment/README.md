# SHL Assessment Recommendation System - Deployment Guide

This guide covers deploying the SHL Assessment Recommendation System to Google Cloud Platform (GCP) using Cloud Run for containerized services.

## Prerequisites

1. **Google Cloud Platform Account**: Free tier is sufficient
2. **gcloud CLI**: [Install gcloud](https://cloud.google.com/sdk/docs/install)
3. **Docker**: [Install Docker](https://docs.docker.com/get-docker/)
4. **Domain Name** (optional): For custom domain setup

## Quick Deployment

### 1. Setup GCP Project

```bash
# Create a new project (or use existing)
gcloud projects create your-project-id --name="SHL Recommendation System"

# Set the project
gcloud config set project your-project-id

# Enable billing (required for Cloud Run)
# Do this through the GCP Console: https://console.cloud.google.com/billing
```

### 2. Authenticate and Configure

```bash
# Authenticate with Google Cloud
gcloud auth login

# Configure Docker to use gcloud as credential helper
gcloud auth configure-docker
```

### 3. Deploy Everything

```bash
# Make scripts executable (Linux/Mac)
chmod +x deployment/gcp/*.sh
chmod +x deployment/gcp/monitoring/*.sh

# Run the main deployment script
./deployment/gcp/deploy.sh your-project-id us-central1

# Or with custom domain
./deployment/gcp/deploy.sh your-project-id us-central1 yourdomain.com
```

## Manual Deployment Steps

If you prefer to deploy step by step:

### 1. Infrastructure Setup (Optional - using Terraform)

```bash
cd deployment/gcp/terraform

# Copy and edit the variables file
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your project details

# Initialize and apply Terraform
terraform init
terraform plan
terraform apply
```

### 2. Deploy API

```bash
./deployment/gcp/deploy-api.sh your-project-id us-central1
```

### 3. Deploy Frontend

```bash
# Get the API URL from previous step
API_URL="https://shl-api-your-project-id.a.run.app"
./deployment/gcp/deploy-frontend.sh your-project-id us-central1 $API_URL
```

### 4. Setup Monitoring

```bash
API_URL="https://shl-api-your-project-id.a.run.app"
FRONTEND_URL="https://shl-frontend-your-project-id.a.run.app"
./deployment/gcp/monitoring/setup-monitoring.sh your-project-id $API_URL $FRONTEND_URL
```

### 5. Custom Domain (Optional)

```bash
./deployment/gcp/setup-domain.sh your-project-id yourdomain.com us-central1
```

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │      API        │
│  (Cloud Run)    │────│  (Cloud Run)    │
│                 │    │                 │
└─────────────────┘    └─────────────────┘
         │                       │
         │              ┌─────────────────┐
         │              │  Cloud Storage  │
         │              │  (Data & Models)│
         │              └─────────────────┘
         │
┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │    │    Logging      │
│ (Cloud Monitor) │    │ (Cloud Logging) │
└─────────────────┘    └─────────────────┘
```

## Service Configuration

### API Service (Cloud Run)
- **Memory**: 2GB
- **CPU**: 1 vCPU
- **Max Instances**: 10
- **Port**: 8000
- **Timeout**: 300s

### Frontend Service (Cloud Run)
- **Memory**: 512MB
- **CPU**: 1 vCPU
- **Max Instances**: 5
- **Port**: 80

## Environment Variables

### API Service
- `ENVIRONMENT=production`
- `LOG_LEVEL=info`
- `GOOGLE_CLOUD_PROJECT=your-project-id`

### Frontend Service
- `REACT_APP_API_URL=https://your-api-url`

## Monitoring and Logging

The deployment includes:

1. **Uptime Checks**: Monitor service availability
2. **Error Rate Alerts**: Alert on high error rates (>5%)
3. **Response Time Alerts**: Alert on slow responses (>5s)
4. **Custom Dashboard**: View key metrics
5. **Log-based Metrics**: Track API performance

Access monitoring:
- Dashboard: https://console.cloud.google.com/monitoring/dashboards
- Alerts: https://console.cloud.google.com/monitoring/alerting
- Logs: https://console.cloud.google.com/logs

## Custom Domain Setup

If you have a custom domain:

1. Run the domain setup script:
   ```bash
   ./deployment/gcp/setup-domain.sh your-project-id yourdomain.com
   ```

2. Add the DNS records shown in the output to your domain registrar

3. Wait for DNS propagation (up to 24 hours)

4. SSL certificates will be automatically provisioned

## Troubleshooting

### Common Issues

1. **Permission Denied**: Ensure you have the necessary IAM roles
   ```bash
   gcloud projects add-iam-policy-binding your-project-id \
     --member="user:your-email@gmail.com" \
     --role="roles/owner"
   ```

2. **API Not Enabled**: Enable required APIs
   ```bash
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com
   ```

3. **Docker Push Failed**: Configure Docker authentication
   ```bash
   gcloud auth configure-docker
   ```

4. **Build Timeout**: Increase Cloud Build timeout
   ```bash
   gcloud config set builds/timeout 1200
   ```

### Health Checks

Test your deployment:

```bash
# API health check
curl https://your-api-url/health

# Frontend check
curl https://your-frontend-url/

# Test recommendation endpoint
curl -X POST https://your-api-url/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "software engineer python"}'
```

## Cost Optimization

For free tier usage:
- Services auto-scale to zero when not in use
- First 2 million requests per month are free
- 180,000 vCPU-seconds and 360,000 GiB-seconds per month are free

## Security Considerations

1. **Service Accounts**: Uses least-privilege service accounts
2. **HTTPS Only**: All traffic encrypted in transit
3. **Container Security**: Non-root user in containers
4. **Network Security**: Services communicate over Google's private network
5. **Input Validation**: API validates all inputs

## Backup and Recovery

- **Code**: Stored in Git repository
- **Data**: Stored in Cloud Storage with versioning
- **Configuration**: Infrastructure as Code with Terraform
- **Monitoring**: Automated alerts for service issues

## Scaling

The system automatically scales based on traffic:
- **Horizontal Scaling**: More instances during high load
- **Vertical Scaling**: Adjust memory/CPU as needed
- **Global Distribution**: Deploy to multiple regions if needed

For high-traffic scenarios, consider:
- Increasing max instances
- Using Cloud CDN for frontend
- Implementing caching layers
- Database optimization

## Support

For deployment issues:
1. Check the logs: `gcloud logs read --project=your-project-id`
2. Review monitoring dashboard
3. Verify service configuration
4. Check IAM permissions

## Next Steps

After successful deployment:
1. Test all endpoints thoroughly
2. Set up CI/CD pipeline for automated deployments
3. Configure additional monitoring and alerting
4. Implement backup strategies
5. Plan for scaling and optimization