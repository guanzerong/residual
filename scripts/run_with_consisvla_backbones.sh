#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "Usage: scripts/run_with_consisvla_backbones.sh <command> [args...]" >&2
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"
DINO_DIR="${THIRD_PARTY_DIR}/dinov2"
VGGT_DIR="${THIRD_PARTY_DIR}/vggt"

if [ ! -d "${DINO_DIR}" ] || [ ! -d "${VGGT_DIR}" ]; then
    echo "Missing third_party sources. Run: bash scripts/setup_consisvla_backbones.sh" >&2
    exit 1
fi

PYTHONPATH_ENTRIES="${DINO_DIR}:${VGGT_DIR}"
if [ -n "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="${PYTHONPATH_ENTRIES}:${PYTHONPATH}"
else
    export PYTHONPATH="${PYTHONPATH_ENTRIES}"
fi

if [ -n "${CONDA_PREFIX:-}" ]; then
    if [ "$1" = "python" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
        shift
        exec "${CONDA_PREFIX}/bin/python" "$@"
    fi
    if [ "$1" = "pip" ] && [ -x "${CONDA_PREFIX}/bin/pip" ]; then
        shift
        exec "${CONDA_PREFIX}/bin/pip" "$@"
    fi
fi

exec "$@"
