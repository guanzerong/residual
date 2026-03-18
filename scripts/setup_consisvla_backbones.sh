#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"

CLONE_REPOS=1
INSTALL_DEPS=1
UPDATE_REPOS=0

if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
    PYTHON_BIN="$(command -v python)"
fi

if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/pip" ]; then
    PIP_BIN="${CONDA_PREFIX}/bin/pip"
else
    PIP_BIN=""
fi

usage() {
    cat <<'EOF'
Usage: bash scripts/setup_consisvla_backbones.sh [--clone-only] [--deps-only] [--update]

Options:
  --clone-only  Clone repositories but do not install any extra Python packages.
  --deps-only   Install missing utility packages but do not clone repositories.
  --update      If a repository already exists, run `git pull --ff-only`.
  -h, --help    Show this help message.

This script intentionally does not install DINOv2's pinned torch/xformers
stack. It is designed to work with the existing `residual1` environment.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --clone-only)
            INSTALL_DEPS=0
            ;;
        --deps-only)
            CLONE_REPOS=0
            ;;
        --update)
            UPDATE_REPOS=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

clone_repo() {
    local name="$1"
    local url="$2"
    local target="${THIRD_PARTY_DIR}/${name}"

    if [ -d "${target}/.git" ]; then
        echo "[setup] ${name} already exists at ${target}"
        if [ "${UPDATE_REPOS}" -eq 1 ]; then
            echo "[setup] Updating ${name}"
            git -C "${target}" pull --ff-only
        fi
        return 0
    fi

    echo "[setup] Cloning ${name} into ${target}"
    git clone --depth 1 "${url}" "${target}"
}

if [ "${CLONE_REPOS}" -eq 1 ]; then
    mkdir -p "${THIRD_PARTY_DIR}"
    clone_repo "dinov2" "https://github.com/facebookresearch/dinov2.git"
    clone_repo "vggt" "https://github.com/facebookresearch/vggt.git"
fi

if [ "${INSTALL_DEPS}" -eq 1 ]; then
    echo "[setup] Checking for minimal missing Python packages"
    mapfile -t missing_packages < <(
        "${PYTHON_BIN}" - <<'PY'
import importlib.util

checks = [
    ("fvcore", "fvcore"),
    ("iopath", "iopath"),
    ("submitit", "submitit"),
]

for import_name, pip_name in checks:
    if importlib.util.find_spec(import_name) is None:
        print(pip_name)
PY
    )

    if [ "${#missing_packages[@]}" -gt 0 ]; then
        echo "[setup] Installing: ${missing_packages[*]}"
        if [ -n "${PIP_BIN}" ]; then
            "${PIP_BIN}" install "${missing_packages[@]}"
        else
            "${PYTHON_BIN}" -m pip install "${missing_packages[@]}"
        fi
    else
        echo "[setup] No extra Python packages needed"
    fi
fi

cat <<EOF
[setup] Done.

Next step:
  scripts/run_with_consisvla_backbones.sh python -c "from dinov2.hub.backbones import dinov2_vitl14_reg; from vggt.models.vggt import VGGT; print('ok')"
EOF
