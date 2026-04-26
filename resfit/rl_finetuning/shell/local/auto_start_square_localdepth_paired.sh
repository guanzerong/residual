#!/bin/bash

set -euo pipefail

ROOT_DIR="/data_all/gzr1/code/residual-offpolicy-rl-macrocls-change-xiugai"
LAUNCHER="${ROOT_DIR}/resfit/rl_finetuning/shell/local/run_square_localdepth_paired_3gpu.sh"

RUN_NODEPTH="${RUN_NODEPTH:-1}"
PARALLEL_NODEPTH="${PARALLEL_NODEPTH:-0}"
if [ "${PARALLEL_NODEPTH}" = "1" ] && [ "${RUN_NODEPTH}" = "1" ]; then
    NUM_GPUS="${NUM_GPUS:-6}"
else
    NUM_GPUS="${NUM_GPUS:-3}"
fi
MAX_MEM_USED_MB="${MAX_MEM_USED_MB:-4000}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-20}"
ALLOW_BUSY_FALLBACK="${ALLOW_BUSY_FALLBACK:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/square_localdepth_auto_${RUN_ID}}"
WANDB_GROUP="${WANDB_GROUP:-square_macro4710_localdepth_paired_${RUN_ID}}"

cd "${ROOT_DIR}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found. Run this script on the GPU server." >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"

nvidia-smi >"${LOG_DIR}/nvidia_smi_before.txt"

mapfile -t gpu_rows < <(
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits \
    | awk -F',' '{
        for (i = 1; i <= NF; i++) {
            gsub(/^ +| +$/, "", $i)
        }
        free = $3 - $2
        printf "%s %s %s %s %s\n", $1, $2, $3, $4, free
    }' \
    | sort -k2,2n -k4,4n -k1,1n
)

if [ "${#gpu_rows[@]}" -eq 0 ]; then
    echo "No GPU rows returned by nvidia-smi." >&2
    exit 1
fi

selected=()
: >"${LOG_DIR}/gpu_selection.txt"
echo "idx used_mb total_mb util_pct free_mb status" | tee -a "${LOG_DIR}/gpu_selection.txt"
for row in "${gpu_rows[@]}"; do
    read -r idx used total util free <<<"${row}"
    status="skip"
    if [ "${used}" -le "${MAX_MEM_USED_MB}" ] && [ "${util}" -le "${MAX_GPU_UTIL}" ]; then
        status="candidate"
        if [ "${#selected[@]}" -lt "${NUM_GPUS}" ]; then
            selected+=("${idx}")
            status="selected"
        fi
    fi
    echo "${idx} ${used} ${total} ${util} ${free} ${status}" | tee -a "${LOG_DIR}/gpu_selection.txt"
done

if [ "${#selected[@]}" -lt "${NUM_GPUS}" ]; then
    if [ "${ALLOW_BUSY_FALLBACK}" = "1" ]; then
        selected=()
        for row in "${gpu_rows[@]}"; do
            read -r idx _ <<<"${row}"
            selected+=("${idx}")
            [ "${#selected[@]}" -ge "${NUM_GPUS}" ] && break
        done
        echo "Not enough GPUs matched thresholds; ALLOW_BUSY_FALLBACK=1, using best available: ${selected[*]}" \
            | tee -a "${LOG_DIR}/gpu_selection.txt"
    else
        echo "Only selected ${#selected[@]} GPU(s), need ${NUM_GPUS}." >&2
        echo "Relax thresholds, for example:" >&2
        echo "  MAX_MEM_USED_MB=8000 MAX_GPU_UTIL=50 bash $0" >&2
        echo "Or force best available:" >&2
        echo "  ALLOW_BUSY_FALLBACK=1 bash $0" >&2
        exit 1
    fi
fi

echo "${selected[*]}" >"${LOG_DIR}/selected_gpus.txt"

export LOG_DIR
export WANDB_GROUP
export RUN_NODEPTH
export PARALLEL_NODEPTH
export ALLOW_GPU_OVERSUBSCRIBE="${ALLOW_GPU_OVERSUBSCRIBE:-0}"

echo "Selected GPUs: ${selected[*]}"
echo "W&B group: ${WANDB_GROUP}"
echo "Log dir: ${LOG_DIR}"
echo "RUN_NODEPTH=${RUN_NODEPTH} PARALLEL_NODEPTH=${PARALLEL_NODEPTH}"

nohup bash "${LAUNCHER}" "${selected[@]}" >"${LOG_DIR}/master.out" 2>&1 &
master_pid="$!"
echo "${master_pid}" >"${LOG_DIR}/master.pid"

echo "Started launcher PID: ${master_pid}"
echo
echo "Monitor:"
echo "  tail -f ${LOG_DIR}/master.out"
echo "  tail -f ${LOG_DIR}/launcher.log"
echo "  watch -n 10 nvidia-smi"
echo
echo "Stop all runs from this launcher if needed:"
echo "  pkill -P ${master_pid}"
