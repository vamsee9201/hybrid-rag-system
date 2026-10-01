#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-ai-lab-502500}"
REGION="${REGION:-us-central1}"
REPOSITORY="${REPOSITORY:-hybrid-rag}"
SERVICE="${SERVICE:-govlens}"
RUNTIME_SERVICE_ACCOUNT="hybrid-rag-run@${PROJECT_ID}.iam.gserviceaccount.com"
ARTIFACT_GCS_URI="${ARTIFACT_GCS_URI:?Set ARTIFACT_GCS_URI to the versioned runtime.tar.gz object}"
ARTIFACT_SHA256="${ARTIFACT_SHA256:?Set ARTIFACT_SHA256 to the local archive checksum}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/govlens:$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)"
AUTH_FLAG="--no-allow-unauthenticated"
if [[ "${PUBLIC_ACCESS:-false}" == "true" ]]; then
  AUTH_FLAG="--allow-unauthenticated"
fi

if [[ "${IP_HASH_SALT_SECRET:-hybrid-rag-ip-salt}" != "hybrid-rag-ip-salt" ]]; then
  echo "Unexpected IP hash salt secret name" >&2
  exit 2
fi

gcloud builds submit \
  --project="${PROJECT_ID}" \
  --tag="${IMAGE}"

gcloud run deploy "${SERVICE}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="${IMAGE}" \
  --service-account="${RUNTIME_SERVICE_ACCOUNT}" \
  --cpu=2 \
  --memory=4Gi \
  --concurrency=4 \
  --min=0 \
  --max=3 \
  --cpu-throttling \
  --timeout=300 \
  "${AUTH_FLAG}" \
  --invoker-iam-check \
  --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GCP_REGION=${REGION},GEMINI_LOCATION=global,GENERATION_MODEL=gemini-3.8-flash,EMBEDDING_MODEL=gemini-embedding-001,EMBEDDING_DIMENSIONS=768,ARTIFACT_GCS_URI=${ARTIFACT_GCS_URI},ARTIFACT_SHA256=${ARTIFACT_SHA256},ARTIFACT_DIR=/tmp/govlens-artifact,DAILY_BUDGET_USD=0.50,ALLOW_LOCAL_QUOTA_FALLBACK=false" \
  --set-secrets="IP_HASH_SALT=hybrid-rag-ip-salt:latest"

gcloud run services describe "${SERVICE}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --format='value(status.url)'
