#!/usr/bin/env bash
# Bootstrap the ESMC-Neuronx environment on a Trainium2 instance.
#
# Creates a Python virtual environment, installs the Neuron SDK and
# the ESM library, then prints activation instructions.
#
# Usage:
#   bash scripts/bootstrap.sh
#
# Environment variables (with defaults):
#   PYTHON_BIN   python3.10
#   VENV_DIR     .venv-esmc-neuron

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_DIR="${VENV_DIR:-.venv-esmc-neuron}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "==> Creating virtual environment at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel -q

echo "==> Installing AWS Neuron SDK packages"
pip config set global.extra-index-url \
  "https://pip.repos.neuron.amazonaws.com"
pip install -r "${REPO_ROOT}/requirements-neuron.txt" -q

echo "==> Installing esm_neuron (editable)"
pip install -e "${REPO_ROOT}" -q

echo ""
echo "Done. Activate with:"
echo "  source ${VENV_DIR}/bin/activate"
