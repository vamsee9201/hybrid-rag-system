#!/usr/bin/env bash
set -euo pipefail

ARCHIVE="${1:?Usage: scripts/upload_artifact.sh data/artifacts/VERSION.tar.gz}"
PROJECT_ID="${PROJECT_ID:-ai-lab-502500}"
BUCKET="${BUCKET:-ai-lab-502500-hybrid-rag}"
VERSION="$(basename "${ARCHIVE}" .tar.gz)"

if [[ ! -f "${ARCHIVE}" ]]; then
  echo "Artifact does not exist: ${ARCHIVE}" >&2
  exit 2
fi

gcloud storage cp "${ARCHIVE}" "gs://${BUCKET}/artifacts/${VERSION}/runtime.tar.gz" \
  --project="${PROJECT_ID}" \
  --if-generation-match=0

echo "ARTIFACT_GCS_URI=gs://${BUCKET}/artifacts/${VERSION}/runtime.tar.gz"
echo "ARTIFACT_SHA256=$(shasum -a 256 "${ARCHIVE}" | awk '{print $1}')"
