#!/usr/bin/env bash
# Run ESMC-300M D2Deep benchmark on H100 / CUDA (p5.4xlarge).
#
# Usage:
#   bash scripts/run_d2deep_cuda.sh [batch_size]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

D2DEEP_CSV="${D2DEEP_CSV:-${REPO_ROOT}/data/compiled_data.csv}"
BATCH_SIZE="${1:-16}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results/cuda}"
HOURLY_COST="${HOURLY_COST:-10.18}"

SLUG="esmc_300m_h100_bs${BATCH_SIZE}_d2deep"

docker run --rm --gpus all \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "${REPO_ROOT}:/workspace/esmc-neuronx" \
    -v "/data:/data" \
    nvcr.io/nvidia/pytorch:25.03-py3 \
    bash -c "
        cd /workspace/esmc-neuronx
        pip install -r requirements-cuda.txt -q 2>&1 | tail -2
        pip install -e . -q
        python experiments/esmc_d2deep_benchmark.py \
            --device cuda \
            --d2deep-csv '${D2DEEP_CSV}' \
            --batch-size ${BATCH_SIZE} \
            --no-compile \
            --hourly-cost ${HOURLY_COST} \
            --output-csv '${RESULTS_DIR}/${SLUG}_scores.csv' \
            --summary-json '${RESULTS_DIR}/${SLUG}_summary.json' \
            --log-every 100
    "

echo "Done. Results at ${RESULTS_DIR}/${SLUG}_summary.json"
