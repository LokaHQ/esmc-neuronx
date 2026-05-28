#!/usr/bin/env bash
# Run ESMC-300M D2Deep benchmark on Trainium2.
#
# Launches 4 workers — one per NeuronCore (cores 0-3) — each processing
# a 1/4 shard of the D2Deep dataset in parallel.
#
# Prerequisite: prepare shards first:
#   bash scripts/prepare_d2deep.sh data/compiled_data.csv
#
# Usage: bash scripts/run_d2deep_neuron.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SHARD_DIR="${SHARD_DIR:-${REPO_ROOT}/data/d2deep_splits4}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results/neuron}"
BATCH_SIZE="${BATCH_SIZE:-1}"
COMPILE_STRATEGY="${COMPILE_STRATEGY:-regional}"
DLC="${DLC:-torch-neuronx-loka:native-pytorch-neuron}"

if [ ! -f "${SHARD_DIR}/part0.csv" ]; then
    echo "ERROR: shards not found at ${SHARD_DIR}/"
    echo "Run: bash scripts/prepare_d2deep.sh data/compiled_data.csv"
    exit 1
fi

mkdir -p "${RESULTS_DIR}"
start_time=$(date +%s)
pids=()

for core in 0 1 2 3; do
    SLUG="esmc_300m_trn2_bs${BATCH_SIZE}_core${core}_d2deep"
    docker run --rm --privileged \
        -e NEURON_CC_FLAGS="--target trn2 --model-type transformer" \
        -e TORCH_NEURONX_ENABLE_HOST_CC=1 \
        -e NEURON_RT_VISIBLE_CORES="${core}" \
        -e NEURON_RT_NUM_CORES=1 \
        -e NEURON_LAUNCH_BLOCKING=1 \
        -v "${REPO_ROOT}:/workspace/esmc-neuronx" \
        "${DLC}" \
        bash -c "
            cd /workspace/esmc-neuronx
            pip install -r requirements-neuron.txt -q 2>&1 | tail -2
            pip install -e . -q
            python experiments/esmc_d2deep_benchmark.py \
                --device neuron \
                --d2deep-csv '/workspace/esmc-neuronx/data/d2deep_splits4/part${core}.csv' \
                --batch-size ${BATCH_SIZE} \
                --compile-strategy ${COMPILE_STRATEGY} \
                --output-csv '/workspace/esmc-neuronx/results/neuron/${SLUG}_scores.csv' \
                --summary-json '/workspace/esmc-neuronx/results/neuron/${SLUG}_summary.json' \
                --log-every 100
        " > "${RESULTS_DIR}/${SLUG}.log" 2>&1 &
    pids+=($!)
    echo "Worker core=${core} started (PID=$!)"
done

for pid in "${pids[@]}"; do
    wait "${pid}" \
        && echo "Worker PID=${pid} completed" \
        || echo "Worker PID=${pid} FAILED — check ${RESULTS_DIR}/*.log"
done

wall_s=$(( $(date +%s) - start_time ))
echo "All workers done. Wall time: ${wall_s}s"
echo "${wall_s}" > "${RESULTS_DIR}/wall_time_neuron.txt"
