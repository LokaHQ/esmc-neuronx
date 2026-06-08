#!/usr/bin/env bash
# Prepare D2Deep dataset: compile from Zenodo (if needed) and split into 4 shards
# for the 4-worker Trainium2 benchmark.
#
# Usage:
#   bash scripts/prepare_d2deep.sh [path-to-compiled_data.csv]
#
# If compiled_data.csv is absent, it is compiled automatically from public
# Zenodo sources via scripts/d2deep_data/compile_d2deep_data.py (~2-5 min).
#
# To skip compilation and supply a pre-built CSV (e.g. from the Loka S3 bucket):
#   mkdir -p data/compiled_data
#   aws s3 cp \
#     s3://torch-neuronx-loka-datasets-846430536449-sa-east-1/d2deep_data/compiled_data/compiled_data.csv \
#     data/compiled_data/compiled_data.csv
#
# Output:
#   data/d2deep_splits4/part{0,1,2,3}.csv
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_CSV="${REPO_ROOT}/data/compiled_data/compiled_data.csv"
INPUT_CSV="${1:-${DEFAULT_CSV}}"
OUTPUT_DIR="${REPO_ROOT}/data/d2deep_splits4"

if [ ! -f "${INPUT_CSV}" ]; then
    echo "Compiled CSV not found at ${INPUT_CSV}."
    echo "Compiling D2Deep dataset from Zenodo (this may take a few minutes)..."
    python3 "${SCRIPT_DIR}/d2deep_data/compile_d2deep_data.py" \
        --data_dir "${REPO_ROOT}/data"
fi

python3 - "${INPUT_CSV}" "${OUTPUT_DIR}" << 'PYEOF'
import sys
import pathlib
import pandas as pd

csv_path = pathlib.Path(sys.argv[1])
out_dir  = pathlib.Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)
n  = len(df)
k  = 4
chunk_size = n // k

print(f"Splitting {n} rows into {k} shards (~{chunk_size} rows each)")

for i in range(k):
    start = i * chunk_size
    end   = start + chunk_size if i < k - 1 else n
    shard = df.iloc[start:end]
    out   = out_dir / f"part{i}.csv"
    shard.to_csv(out, index=False)
    print(f"  part{i}.csv  {len(shard)} rows -> {out}")

print(f"Done. Shards written to {out_dir}/")
PYEOF
