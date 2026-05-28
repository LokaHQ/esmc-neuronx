#!/usr/bin/env bash
# Prepare D2Deep dataset: split compiled_data.csv into 4 equal shards
# for the 4-worker Trainium2 benchmark.
#
# Usage:
#   bash scripts/prepare_d2deep.sh <path-to-compiled_data.csv>
#
# Example:
#   bash scripts/prepare_d2deep.sh data/compiled_data.csv
#
# Output:
#   data/d2deep_splits4/part{0,1,2,3}.csv
set -euo pipefail

INPUT_CSV="${1:?Usage: $0 <path-to-compiled_data.csv>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/data/d2deep_splits4"

if [ ! -f "${INPUT_CSV}" ]; then
    echo "ERROR: input file not found: ${INPUT_CSV}"
    echo "Download it first:"
    echo "  aws s3 cp s3://torch-neuronx-loka-datasets-846430536449-sa-east-1/d2deep_data/compiled_data/compiled_data.csv data/compiled_data.csv"
    exit 1
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
