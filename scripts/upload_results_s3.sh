#!/usr/bin/env bash
# Upload results and manifests to S3.
# Usage: bash scripts/upload_results_s3.sh s3://bucket/prefix/
set -euo pipefail

S3_PREFIX="${1:?Usage: $0 s3://bucket/prefix/}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "Syncing results (JSON only, no score CSVs) …"
aws s3 sync "${REPO_ROOT}/results/"   "${S3_PREFIX}/results/"   --exclude "*.csv"
echo "Syncing manifests …"
aws s3 sync "${REPO_ROOT}/manifests/" "${S3_PREFIX}/manifests/"
echo "Done → ${S3_PREFIX}"
